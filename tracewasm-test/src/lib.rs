//! Shared plumbing for the TraceWasm test suite.
//!
//! ## Layout
//!
//! * [`NoImports`] — an import registry for guests that declare no imports.
//! * [`call_typed`] / [`call_i32`] / [`try_call`] — compile → instantiate → call
//!   in one step, for fixture-style tests that assert on a single export.
//! * [`Guest`] — a `guests/*.rs` program compiled to wasm by `build.rs`,
//!   instantiated once so a test file with 40 cases does not pay 40
//!   compilations.
//! * [`metrics`] — timing and reporting used by `tests/metrics.rs`.
//!
//! ## Why guests are built from source
//!
//! Every guest is compiled to wasm by `build.rs` *and* `include!`d natively by
//! `tests/differential.rs`. Expected values are therefore computed by rustc's own
//! backend from the same source rather than written down by hand. A checked-in
//! `.wasm` plus a hand-written expected value can be wrong together, and nobody
//! can re-derive the number a year later.
//!
//! ## What differential comparison cannot cover
//!
//! Three classes of result legitimately differ between native and wasm. Guests
//! must avoid them or the suite will flake:
//!
//! * **Pointer and `usize` width.** `wasm32-unknown-unknown` is a **32-bit**
//!   target; the native oracle is 64-bit. `size_of::<usize>()`,
//!   `size_of::<Option<&T>>()`, `size_of::<Box<T>>()` and any `as usize`
//!   round-trip therefore differ by design — use a fixed-width type (`u32`,
//!   `u64`) instead. Collection *lengths* are fine; a count is a count.
//! * **Transcendentals.** `sin`, `cos`, `exp`, `ln`, `powf` come from the host
//!   libm natively and from a compiled-in libm under wasm, and can disagree in
//!   the last bit. Safe: the arithmetic operators, `sqrt` (IEEE 754 defines it
//!   exactly), `abs`, `floor`, `ceil`, `trunc`, `round`, `min`/`max`, `copysign`.
//! * **Hash iteration order.** `HashMap`/`HashSet` seed differently in the two
//!   environments. Aggregate with an order-independent reduction (sum, xor,
//!   count) or use a `BTreeMap`; never compare a `{:?}` of a hash container.
//!
//! All three were found the hard way. When a guest fails with two
//! plausible-looking but unequal numbers, check this list before suspecting the
//! interpreter.

use std::time::{Duration, Instant};

use tracewasm_core::error::{FuncCallError, TraceWasmError};
use tracewasm_core::instance::config::Config;
use tracewasm_core::instance::traits::{ImportRegistry, ImportSignature, ResultVals, Results, Val};
use tracewasm_core::memory::{MemoryView, linear::LinearMemory};
use tracewasm_core::module::Module;

// ---------------------------------------------------------------------------
// Import registry
// ---------------------------------------------------------------------------

/// A registry for modules that declare no imports at all.
///
/// Every method is unreachable rather than returning a default, so a guest that
/// unexpectedly grows an import fails loudly instead of silently receiving zeros.
pub struct NoImports;

impl ImportRegistry for NoImports {
    fn execute<V: MemoryView>(
        &mut self,
        module_name: &str,
        func_name: &str,
        _params: &[Val],
        _memory_view: &mut V,
    ) -> Result<ResultVals, TraceWasmError> {
        unreachable!("guest declares no imports, but called `{module_name}::{func_name}`")
    }

    fn signature(&self, _module_name: &str, _func_name: &str) -> Option<ImportSignature> {
        None
    }

    fn func_count(&self) -> u32 {
        0
    }

    fn global_count(&self) -> u32 {
        0
    }

    fn get_global(&self, module_name: &str, global_name: &str) -> Result<Val, TraceWasmError> {
        unreachable!("guest declares no globals, but read `{module_name}::{global_name}`")
    }
}

// ---------------------------------------------------------------------------
// One-shot call helpers (fixture style)
// ---------------------------------------------------------------------------

/// Compiles `wasm`, instantiates it, and calls the `() -> R` export `name`.
///
/// Panics on any failure. For the trapping cases, where the error *is* the thing
/// under test, use [`try_call`].
pub fn call_typed<R: Results>(wasm: &[u8], name: &str) -> R {
    match try_call::<R>(wasm, name) {
        Ok(v) => v,
        Err(e) => panic!("calling `{name}` failed: {e}"),
    }
}

/// [`call_typed`] specialised to the very common `() -> i32` shape.
pub fn call_i32(wasm: &[u8], name: &str) -> i32 {
    let (v,) = call_typed::<(i32,)>(wasm, name);
    v
}

/// As [`call_typed`], but surfaces the failure instead of panicking.
// Returns `TypedFunc::call`'s own error type unchanged; boxing it here would
// diverge from the signature under test.
#[allow(clippy::result_large_err)]
pub fn try_call<R: Results>(wasm: &[u8], name: &str) -> Result<R, FuncCallError> {
    let module = Module::compile(wasm).expect("module should compile");

    let func = module
        .get_typed_func::<(), R>(name)
        .unwrap_or_else(|e| panic!("export `{name}`: {e}"));

    let mut instance = module
        .instantiate::<LinearMemory, _>(NoImports, None)
        .expect("module should instantiate");

    func.call((), &mut instance)
}

// ---------------------------------------------------------------------------
// Guest wasm blobs
// ---------------------------------------------------------------------------

/// Whether the guest programs were built. `false` when the wasm target is absent.
pub const GUESTS_AVAILABLE: bool = !cfg!(no_guest_wasm);

/// The `.wasm` blobs built from `guests/*.rs`.
///
/// `build.rs` writes these into `OUT_DIR`; `include_bytes!` pulls them into the
/// test binary so a run never depends on the build directory still existing.
#[cfg(not(no_guest_wasm))]
pub mod guests {
    /// From `guests/arithmetic.rs`.
    pub const ARITHMETIC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/arithmetic.wasm"));
    /// From `guests/control_flow.rs`.
    pub const CONTROL_FLOW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/control_flow.wasm"));
    /// From `guests/heap.rs`.
    pub const HEAP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/heap.wasm"));
    /// From `guests/memory.rs`.
    pub const MEMORY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/memory.wasm"));
    /// From `guests/frames.rs`.
    pub const FRAMES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/frames.wasm"));
    /// From `guests/exotic.rs`.
    pub const EXOTIC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/exotic.wasm"));
}

// ---------------------------------------------------------------------------
// Guest
// ---------------------------------------------------------------------------

/// A compiled, instantiated guest ready to be called repeatedly.
///
/// Compilation and instantiation are the expensive part — a guest linking `std`
/// is well over a megabyte of wasm — so a test file checking many exports should
/// build one of these and reuse it.
pub struct Guest {
    module: std::sync::Arc<Module>,
    instance: tracewasm_core::instance::Instance<LinearMemory, NoImports>,
}

impl Guest {
    /// Compiles and instantiates `wasm` with the default configuration.
    pub fn new(wasm: &[u8]) -> Self {
        Self::with_config(wasm, None)
    }

    /// Compiles and instantiates `wasm` with an explicit [`Config`].
    ///
    /// The frame tests need this to raise `max_call_stack_depth` above its
    /// conservative default; otherwise the guard fires long before the native
    /// stack is the binding constraint, and a "how deep can we go" measurement
    /// just reports the config value back.
    pub fn with_config(wasm: &[u8], config: Option<Config>) -> Self {
        let module = Module::compile(wasm).expect("guest should compile");
        let instance = module
            .instantiate::<LinearMemory, _>(NoImports, config)
            .expect("guest should instantiate");

        Guest { module, instance }
    }

    /// Calls the `(i32) -> i32` export `name`.
    pub fn i32_i32(&mut self, name: &str, arg: i32) -> i32 {
        self.try_i32_i32(name, arg)
            .unwrap_or_else(|e| panic!("calling `{name}({arg})` failed: {e}"))
    }

    /// Calls the `(i32) -> i64` export `name`.
    pub fn i32_i64(&mut self, name: &str, arg: i32) -> i64 {
        self.try_i32_i64(name, arg)
            .unwrap_or_else(|e| panic!("calling `{name}({arg})` failed: {e}"))
    }

    /// Calls the `(i32) -> f64` export `name`.
    ///
    /// The float guests take an `i32` like every other export so one argument
    /// table can drive the whole suite; only the *result* is floating point.
    pub fn i32_f64(&mut self, name: &str, arg: i32) -> f64 {
        let f = self
            .module
            .get_typed_func::<(i32,), (f64,)>(name)
            .unwrap_or_else(|e| panic!("export `{name}` as (i32)->f64: {e}"));

        match f.call((arg,), &mut self.instance) {
            Ok((v,)) => v,
            Err(e) => panic!("calling `{name}({arg})` failed: {e}"),
        }
    }

    /// Calls the `(f64) -> f64` export `name`.
    pub fn f64_f64(&mut self, name: &str, arg: f64) -> f64 {
        let f = self
            .module
            .get_typed_func::<(f64,), (f64,)>(name)
            .unwrap_or_else(|e| panic!("export `{name}` as (f64)->f64: {e}"));

        match f.call((arg,), &mut self.instance) {
            Ok((v,)) => v,
            Err(e) => panic!("calling `{name}({arg})` failed: {e}"),
        }
    }

    /// Calls the 5-parameter `(i32, i32, i32, i32, i32) -> i64` export `name`.
    ///
    /// Five is the maximum arity `Params` is implemented for, so this doubles as a
    /// check that the typed-call boundary handles its widest supported shape.
    pub fn i32x5_i64(&mut self, name: &str, args: (i32, i32, i32, i32, i32)) -> i64 {
        let f = self
            .module
            .get_typed_func::<(i32, i32, i32, i32, i32), (i64,)>(name)
            .unwrap_or_else(|e| panic!("export `{name}` as (i32 x5)->i64: {e}"));

        match f.call(args, &mut self.instance) {
            Ok((v,)) => v,
            Err(e) => panic!("calling `{name}{args:?}` failed: {e}"),
        }
    }

    /// [`Self::i32_i32`], surfacing the trap instead of panicking.
    #[allow(clippy::result_large_err)]
    pub fn try_i32_i32(&mut self, name: &str, arg: i32) -> Result<i32, FuncCallError> {
        let f = self
            .module
            .get_typed_func::<(i32,), (i32,)>(name)
            .unwrap_or_else(|e| panic!("export `{name}` as (i32)->i32: {e}"));

        f.call((arg,), &mut self.instance).map(|(v,)| v)
    }

    /// [`Self::i32_i64`], surfacing the trap instead of panicking.
    #[allow(clippy::result_large_err)]
    pub fn try_i32_i64(&mut self, name: &str, arg: i32) -> Result<i64, FuncCallError> {
        let f = self
            .module
            .get_typed_func::<(i32,), (i64,)>(name)
            .unwrap_or_else(|e| panic!("export `{name}` as (i32)->i64: {e}"));

        f.call((arg,), &mut self.instance).map(|(v,)| v)
    }
}

// ---------------------------------------------------------------------------
// Running recursion tests safely
// ---------------------------------------------------------------------------

/// Deepest recursion the tests should attempt, scaled to the build profile.
///
/// Every active wasm frame costs a native Rust frame, and a **debug** build of the
/// interpreter spends roughly **30 KB** of native stack per frame against about
/// **1.3 KB** in release — a ~20x difference, because debug spills everything and
/// reuses no slots. A depth that is comfortable under `--release` therefore
/// overflows under a plain `cargo test`.
///
/// Testing a deeper stack in debug proves nothing extra about the interpreter, so
/// the depth is simply scaled instead.
pub const MAX_TEST_RECURSION: i32 = if cfg!(debug_assertions) { 150 } else { 3_000 };

/// Native stack given to [`with_large_stack`].
///
/// Sized for `MAX_TEST_RECURSION` at the debug per-frame cost, with headroom. The
/// pages are faulted lazily, so reserving this is cheap.
const LARGE_STACK_BYTES: usize = 256 * 1024 * 1024;

/// Runs `body` on a thread with an explicitly large stack, propagating panics.
///
/// **Any test that drives wasm recursion must use this.** libtest runs each test
/// on a spawned thread whose default stack (~2 MiB) fits only a few dozen frames
/// in a debug build, and a Rust stack overflow is `SIGABRT` — it takes the whole
/// test binary down rather than failing one test, so the symptom is an
/// inscrutable abort rather than a useful assertion failure.
pub fn with_large_stack<T, F>(body: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let handle = std::thread::Builder::new()
        .stack_size(LARGE_STACK_BYTES)
        .name("large-stack".into())
        .spawn(body)
        .expect("spawning a large-stack thread should succeed");

    match handle.join() {
        Ok(v) => v,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Timing and reporting for the tests that produce numbers rather than verdicts.
///
/// These deliberately *print* measurements instead of asserting on wall-clock
/// time: a throughput threshold that fails on a loaded CI box is worse than no
/// test at all. Where something is asserted it is a correctness-adjacent
/// invariant — a ratio, an ordering, a count — not an absolute duration.
pub mod metrics {
    use super::{Duration, Instant};

    /// One measured row of a metric table.
    pub struct Row {
        pub label: String,
        pub iterations: u64,
        pub elapsed: Duration,
        /// Guest-level operations per iteration, where the test knows it. `None`
        /// when only wall-clock per call is meaningful.
        pub ops_per_iter: Option<u64>,
    }

    impl Row {
        /// Wall-clock per iteration.
        pub fn per_iter(&self) -> Duration {
            self.elapsed / self.iterations.max(1) as u32
        }

        /// Nanoseconds per guest operation, when the operation count is known.
        pub fn ns_per_op(&self) -> Option<f64> {
            let ops = self.ops_per_iter?.checked_mul(self.iterations)?;

            if ops == 0 {
                return None;
            }

            Some(self.elapsed.as_secs_f64() * 1e9 / ops as f64)
        }
    }

    /// A named collection of measurements, printed by [`Table::report`].
    pub struct Table {
        title: String,
        rows: Vec<Row>,
    }

    impl Table {
        pub fn new(title: impl Into<String>) -> Self {
            Table {
                title: title.into(),
                rows: Vec::new(),
            }
        }

        /// Times `body` over `iterations`, recording the result.
        ///
        /// One warm-up iteration runs first and is not counted: the first call
        /// into a fresh instance pays for lazily-faulted memory pages, which
        /// would otherwise dominate a short measurement.
        pub fn measure(
            &mut self,
            label: impl Into<String>,
            iterations: u64,
            ops_per_iter: Option<u64>,
            mut body: impl FnMut(),
        ) {
            body();

            let start = Instant::now();

            for _ in 0..iterations {
                body();
            }

            let elapsed = start.elapsed();

            self.rows.push(Row {
                label: label.into(),
                iterations,
                elapsed,
                ops_per_iter,
            });
        }

        pub fn rows(&self) -> &[Row] {
            &self.rows
        }

        /// Looks a row up by label, for the few assertions that compare rows.
        pub fn row(&self, label: &str) -> Option<&Row> {
            self.rows.iter().find(|r| r.label == label)
        }

        /// Prints the table. Visible under `cargo test -- --nocapture`.
        pub fn report(&self) {
            let width = 58;

            println!();
            println!("  {}", self.title);
            println!("  {}", "-".repeat(width));
            println!("  {:<32} {:>12} {:>10}", "measurement", "per call", "ns/op");

            for row in &self.rows {
                let per = row.per_iter();

                let per_str = if per.as_micros() >= 1000 {
                    format!("{:.2} ms", per.as_secs_f64() * 1e3)
                } else if per.as_nanos() >= 1000 {
                    format!("{:.2} us", per.as_secs_f64() * 1e6)
                } else {
                    format!("{} ns", per.as_nanos())
                };

                let ns_op = row
                    .ns_per_op()
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_else(|| "-".to_string());

                println!("  {:<32} {:>12} {:>10}", row.label, per_str, ns_op);
            }
        }
    }
}
