//! Runs lowered wasm in-process under the JIT, and checks every call against the
//! interpreter.
//!
//! [`check`] takes a module, an export and a list of argument lists. For each call it
//! instantiates the module in the interpreter, builds a [`RuntimeInstance`] from that
//! instance's freshly initialised globals and tables, and runs the call both ways:
//! once through the interpreter, and once through the lowered code at each
//! [`Level`]. The two must agree on the results, on every global's final value, and,
//! when the interpreter traps, on the [`TrapCode`].
//!
//! Running at `O2` as well as unoptimised is deliberate. The optimiser exploits
//! poison and undefined behaviour, so a lowering that leans on either — an unmasked
//! shift, a raw `srem` of `MIN` by `-1` — tends to pass at one level and fail at the
//! other.
//!
//! # Traps run in a child process
//!
//! Generated code cannot come back from a trap: the call to the trap handler is
//! followed by `unreachable`, and an `extern "C"` handler cannot unwind through JIT'd
//! frames. So a call the interpreter says traps is re-run in a child: this same test
//! binary, filtered to the current test, with [`CHILD_VAR`] naming the one call to
//! make. The child's handler reports the trap code on stderr and exits. A trap in the
//! *parent* is therefore always one nobody expected, and the handler says which call
//! raised it before aborting.
//!
//! # Running it
//!
//! It links libLLVM, so it sits behind the `jit` feature:
//!
//! ```text
//! cargo test -p tracewasm-core --features jit --lib jit_harness
//! ```

use crate::{
    Stack,
    error::{CallIndirectError, InstructionExecutionError},
    instance::{
        Instance,
        traits::{ImportRegistry, ImportSignature, ResultVals, Val},
    },
    instruction::stack::llvm::ctx::{GlobalVal, RuntimeContext, RuntimeInstance, TrapCode},
    memory::{MemoryView, linear::LinearMemory, mmap::MmapMemory},
    module::{Export, FuncIndex, Module, ValType},
    runtime::{TraceVM, value::TableVal},
};
use llvm_sys::target_machine::LLVMCodeGenOptLevel;
use std::{
    cell::{Cell, RefCell},
    fmt::Write as _,
    io::Read as _,
    process::{Command, Stdio},
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};
use tracewasm_llvm::{
    cfg::emit::IREmitter,
    jit::{JITHandler, OptLevel},
};

/// Builds a call's argument list from Rust literals: `vals![1, 2i64, 1.5f32]`.
/// An unsuffixed integer is an `i32`.
macro_rules! vals {
    ($($v:expr),* $(,)?) => {
        vec![$($crate::instance::traits::WasmTy::to_val(&$v)),*]
    };
}

mod cases;

/// How long one call may run, on either side, before it is taken to loop forever.
/// Generous: a case is a handful of instructions.
const LIMIT: Duration = Duration::from_secs(10);

/// Set in a child process to `<check>:<call>:<level>`, the one call it should make.
const CHILD_VAR: &str = "TRACEWASM_JIT_CHILD";

/// What a child prints on stderr when its call traps, followed by the code.
const TRAP_MARKER: &str = "tracewasm-jit-trap=";

/// What a child prints on stderr when its call returns instead, followed by the
/// results.
const RETURNED_MARKER: &str = "tracewasm-jit-returned=";

/// The child's exit status after a trap, so a crash is not mistaken for one.
const TRAP_EXIT: i32 = 86;

/// The trampoline appended to every lowered module; see [`trampoline`].
const ENTRY: &str = "harness.entry";

/// The optimisation levels every call is run at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    Unoptimised,
    O2,
}

const LEVELS: [Level; 2] = [Level::Unoptimised, Level::O2];

thread_local! {
    /// The call the JIT is running right now, for the trap handler to name.
    static CURRENT_CALL: RefCell<String> = const { RefCell::new(String::new()) };

    /// How many [`check`]s this test has started, so a child can find its own.
    static CHECKS_STARTED: Cell<usize> = const { Cell::new(0) };
}

/// Lowers `wat`, then runs `export` once per argument list, through the JIT at every
/// [`Level`] and through the interpreter, asserting they agree.
///
/// Returns the lowered IR (with the trampoline appended), so a case can go on to
/// assert on its shape.
///
/// # Panics
///
/// On any disagreement, naming the call, its arguments and the level. Also if `wat`
/// does not validate, imports anything, or `export` takes or returns a reference:
/// the harness passes numbers only.
pub(super) fn check(wat: &str, export: &str, calls: &[Vec<Val>]) -> String {
    let check_no = CHECKS_STARTED.with(|c| {
        let n = c.get();

        c.set(n + 1);

        n
    });

    let lowered = lower(wat, export);

    for args in calls {
        lowered.assert_args_fit(args);
    }

    if let Ok(spec) = std::env::var(CHILD_VAR) {
        let (target, call_no, level) = parse_child_spec(&spec);

        if target == check_no {
            run_in_child(&lowered, &calls[call_no], level);
        }

        // Some other check in this test is the child's target; this one was run
        // already, by the parent.
        return lowered.ir;
    }

    for (call_no, args) in calls.iter().enumerate() {
        let (mut instance, start) = instantiate(&lowered.module);

        let what = format!("`{export}`{args:?}");
        let outcome = with_watchdog(&format!("{what} in the interpreter"), || {
            TraceVM::run(lowered.func_index, args, &mut instance, &lowered.module)
        });

        match outcome {
            Ok(results) => {
                let want: Vec<Val> = results.into_iter().collect();

                for level in LEVELS {
                    let what = format!("{what} at {level:?}");

                    CURRENT_CALL.with(|c| *c.borrow_mut() = what.clone());

                    let got =
                        with_watchdog(&what, || run_jit(&lowered, level, start.clone(), args));

                    lowered.assert_results(&what, &got.results, &want);
                    lowered.assert_globals(&what, &got.globals, &instance.global_vals);
                }
            }
            Err(err) => {
                let want = expected_trap(err.cause());

                for level in LEVELS {
                    let what = format!("{what} at {level:?}");

                    match run_child(check_no, call_no, level, &what) {
                        ChildOutcome::Trapped(code) => assert_eq!(
                            TrapCode::from_u32(code),
                            Some(want),
                            "{what}: the JIT trapped with code {code}, the interpreter with \
                             {want:?} (`{err}`)\n--- IR ---\n{}",
                            lowered.ir
                        ),
                        ChildOutcome::Returned(results) => panic!(
                            "{what}: the interpreter trapped with {want:?} (`{err}`), but the \
                             JIT returned {results}\n--- IR ---\n{}",
                            lowered.ir
                        ),
                    }
                }
            }
        }
    }

    lowered.ir
}

// ---------------------------------------------------------------------------
// Lowering

/// A module lowered for the JIT, and what the harness needs to call into it.
struct Lowered {
    module: Arc<Module<Stack>>,
    /// The emitted IR with [`trampoline`] appended.
    ir: String,
    func_index: FuncIndex,
    params: Vec<ValType>,
    results: Vec<ValType>,
}

fn lower(wat: &str, export: &str) -> Lowered {
    let bytes = wat::parse_str(wat).expect("invalid wat");

    wasmparser::Validator::new()
        .validate_all(&bytes)
        .expect("wat does not validate — the case would prove nothing");

    let module = Module::<Stack>::compile(&bytes).expect("module compiles");

    // Nothing links an import's `fn{N}` symbol yet, and `NoImports` provides none.
    assert_eq!(
        (module.imported_func_count, module.imported_global_count),
        (0, 0),
        "the harness links no imports"
    );

    let func_index = match module.export(export) {
        Some(Export::Func(index)) => index,
        _ => panic!("the module exports no function named `{export}`"),
    };

    let ty = &module.types[module.func_decls[func_index.0 as usize].ty.0 as usize];
    let params = ty.params.to_vec();
    let results = ty.results.to_vec();

    for ty in params.iter().chain(&results) {
        assert!(
            matches!(
                ty,
                ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64
            ),
            "the harness passes numbers only, and `{export}` has a {ty:?}"
        );
    }

    let cfg = module.build_cfg().expect("cfg builds");
    let mut ir = IREmitter::emit(cfg).expect("ir emits");

    ir.push_str(&trampoline(func_index, &params, &results));

    Lowered {
        module,
        ir,
        func_index,
        params,
        results,
    }
}

/// One entry point for every signature: `void (ptr %args, ptr %results, ptr %ctx)`.
///
/// Arguments and results cross as `i64` slots holding each value's bits, low bits
/// first — the encoding [`GlobalVal`] uses. That keeps the Rust side to a single
/// [`get_func`](tracewasm_llvm::jit::JITCompiledInstance::get_func) type, and keeps
/// struct returns, which the JIT's typed calls do not support, on the IR side.
fn trampoline(func_index: FuncIndex, params: &[ValType], results: &[ValType]) -> String {
    let mut f = format!("\ndefine void @{ENTRY}(ptr %args, ptr %results, ptr %ctx) {{\nentry:\n");
    let mut actuals = Vec::with_capacity(params.len() + 1);

    for (i, ty) in params.iter().enumerate() {
        writeln!(
            f,
            "  %a{i}.slot = getelementptr inbounds i64, ptr %args, i64 {i}"
        )
        .unwrap();
        writeln!(f, "  %a{i}.bits = load i64, ptr %a{i}.slot").unwrap();

        let reg = match ty {
            ValType::I64 => format!("%a{i}.bits"),
            ValType::I32 => {
                writeln!(f, "  %a{i} = trunc i64 %a{i}.bits to i32").unwrap();
                format!("%a{i}")
            }
            ValType::F32 => {
                writeln!(f, "  %a{i}.i32 = trunc i64 %a{i}.bits to i32").unwrap();
                writeln!(f, "  %a{i} = bitcast i32 %a{i}.i32 to float").unwrap();
                format!("%a{i}")
            }
            ValType::F64 => {
                writeln!(f, "  %a{i} = bitcast i64 %a{i}.bits to double").unwrap();
                format!("%a{i}")
            }
            _ => unreachable!("rejected by `lower`"),
        };

        actuals.push(format!("{} {reg}", ir_ty(ty)));
    }

    actuals.push("ptr %ctx".to_string());

    let ret = ret_ty(results);
    let call = format!("call {ret} @fn{}({})", func_index.0, actuals.join(", "));

    if results.is_empty() {
        writeln!(f, "  {call}").unwrap();
    } else {
        writeln!(f, "  %r = {call}").unwrap();
    }

    for (k, ty) in results.iter().enumerate() {
        let val = if results.len() == 1 {
            "%r".to_string()
        } else {
            writeln!(f, "  %r{k} = extractvalue {ret} %r, {k}").unwrap();
            format!("%r{k}")
        };

        let bits = match ty {
            ValType::I64 => val,
            ValType::I32 => {
                writeln!(f, "  %r{k}.bits = zext i32 {val} to i64").unwrap();
                format!("%r{k}.bits")
            }
            ValType::F32 => {
                writeln!(f, "  %r{k}.i32 = bitcast float {val} to i32").unwrap();
                writeln!(f, "  %r{k}.bits = zext i32 %r{k}.i32 to i64").unwrap();
                format!("%r{k}.bits")
            }
            ValType::F64 => {
                writeln!(f, "  %r{k}.bits = bitcast double {val} to i64").unwrap();
                format!("%r{k}.bits")
            }
            _ => unreachable!("rejected by `lower`"),
        };

        writeln!(
            f,
            "  %r{k}.slot = getelementptr inbounds i64, ptr %results, i64 {k}"
        )
        .unwrap();
        writeln!(f, "  store i64 {bits}, ptr %r{k}.slot").unwrap();
    }

    f.push_str("  ret void\n}\n");

    f
}

fn ir_ty(ty: &ValType) -> &'static str {
    match ty {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        ValType::F32 => "float",
        ValType::F64 => "double",
        _ => unreachable!("rejected by `lower`"),
    }
}

/// How the pass returns `results`: `void`, the one type, or a literal struct.
fn ret_ty(results: &[ValType]) -> String {
    match results {
        [] => "void".to_string(),
        [one] => ir_ty(one).to_string(),
        many => format!(
            "{{ {} }}",
            many.iter().map(ir_ty).collect::<Vec<_>>().join(", ")
        ),
    }
}

// ---------------------------------------------------------------------------
// The reference

/// A registry for guests that import nothing, which [`lower`] insists on.
struct NoImports;

impl ImportRegistry for NoImports {
    fn execute<V: MemoryView>(
        &mut self,
        module_name: &str,
        func_name: &str,
        _params: &[Val],
        _memory_view: &mut V,
    ) -> Result<ResultVals, anyhow::Error> {
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

    fn get_global(&self, module_name: &str, global_name: &str) -> Result<Val, anyhow::Error> {
        unreachable!("guest declares no globals, but read `{module_name}::{global_name}`")
    }
}

/// A fresh instance's globals and tables, before any call has touched them —
/// constant expressions, element segments and the start function already applied.
struct Start {
    globals: Box<[Val]>,
    tables: Box<[TableVal]>,
}

impl Clone for Start {
    fn clone(&self) -> Self {
        Start {
            globals: self.globals.clone(),
            tables: copy_tables(&self.tables),
        }
    }
}

/// `TableVal` is not `Clone`; the harness copies tables to give each run its own.
fn copy_tables(tables: &[TableVal]) -> Box<[TableVal]> {
    tables
        .iter()
        .map(|t| TableVal {
            table: t.table.clone(),
            maximum: t.maximum,
        })
        .collect()
}

/// Instantiates the module in the interpreter, and copies out the state the JIT side
/// starts from, so both begin each call identically.
fn instantiate(module: &Arc<Module<Stack>>) -> (Instance<LinearMemory, NoImports, Stack>, Start) {
    let instance = module
        .instantiate::<LinearMemory, _>(NoImports, None)
        .expect("module instantiates");

    let start = Start {
        globals: instance.global_vals.clone(),
        tables: copy_tables(&instance.table_vals),
    };

    (instance, start)
}

/// The code the lowered code should trap with, given why the interpreter trapped.
fn expected_trap(cause: &InstructionExecutionError) -> TrapCode {
    use InstructionExecutionError as E;

    match cause {
        E::Unreachable => TrapCode::Unreachable,
        E::CallIndirect(_, CallIndirectError::TableSlotOutOfBounds) => TrapCode::TableOutOfBounds,
        E::CallIndirect(_, CallIndirectError::NullElementInTable) => TrapCode::NullTableElement,
        E::CallIndirect(_, CallIndirectError::FunctionSignatureMismatch(..)) => {
            TrapCode::IndirectCallTypeMismatch
        }
        E::Division { deno, .. } if deno == "0" => TrapCode::IntegerDivideByZero,
        E::Division { .. } => TrapCode::IntegerOverflow,
        // `MIN % -1` is `0`, so a remainder only ever traps on a zero divisor.
        E::Remainder { .. } => TrapCode::IntegerDivideByZero,
        E::FloatToIntTruncation(operand, _) if operand == "NaN" => {
            TrapCode::InvalidConversionToInteger
        }
        // An infinity, or a finite value out of range.
        E::FloatToIntTruncation(..) => TrapCode::IntegerOverflow,
        E::Memory(_) => TrapCode::MemoryOutOfBounds,
        other => panic!("the harness has no trap code for the interpreter's `{other}`"),
    }
}

// ---------------------------------------------------------------------------
// The JIT side

/// What one JIT'd call produced.
struct JitRun {
    results: Vec<Val>,
    /// Every global's slot after the call, raw.
    globals: Vec<u64>,
}

fn run_jit(lowered: &Lowered, level: Level, start: Start, args: &[Val]) -> JitRun {
    let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).expect("JIT starts");

    let mut module = jit.parse_module("case", &lowered.ir).unwrap_or_else(|e| {
        panic!(
            "the JIT rejects the lowered IR: {e}\n--- IR ---\n{}",
            lowered.ir
        )
    });

    module
        .link_host_func(
            TrapCode::TRAP_FUNC_NAME,
            harness_trap as extern "C" fn(*mut u8, u32),
        )
        .expect("the trap handler links");

    if level == Level::O2 {
        module.optimize(OptLevel::O2).expect("O2 runs");
    }

    // SAFETY: what the harness is here to find out. A lowering bug can make this
    // code misbehave, which is exactly what a failing case reports.
    let compiled = unsafe { module.compile() }.expect("the lowered IR compiles");

    let entry = compiled
        .get_func::<(*mut u64, *mut u64, *mut u8), ()>(ENTRY)
        .expect("the trampoline is exported");

    let mut runtime = RuntimeInstance::new(start.globals, MmapMemory {}, start.tables);
    let mut ctx = runtime.ctx();
    let ctx_ptr = (&mut ctx as *mut RuntimeContext).cast::<u8>();

    let mut arg_bits: Vec<u64> = args.iter().map(to_bits).collect();
    let mut result_bits = vec![0u64; lowered.results.len()];

    entry.call(arg_bits.as_mut_ptr(), result_bits.as_mut_ptr(), ctx_ptr);

    // SAFETY: `ctx` is still live, and its layout is the one generated code reads.
    let globals = unsafe { read_globals(ctx_ptr) };

    JitRun {
        results: result_bits
            .iter()
            .zip(&lowered.results)
            .map(|(bits, ty)| from_bits(*bits, ty))
            .collect(),
        globals,
    }
}

/// Every global slot, read the way generated code reads them: `globals_ptr` at
/// offset 0, `globals_len` at 8.
///
/// Through the raw layout because the fields are private to `ctx.rs`. That layout is
/// pinned by `the_slot_layouts_are_what_the_llvm_side_must_match`, so this reads
/// exactly what the lowered code wrote.
///
/// # Safety
///
/// `ctx` must point to a live [`RuntimeContext`].
unsafe fn read_globals(ctx: *const u8) -> Vec<u64> {
    unsafe {
        let ptr = *(ctx as *const *const u64);
        let len = *(ctx.add(8) as *const u32) as usize;

        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

/// Linked in place of [`tracewasm_trap`](super::ctx::tracewasm_trap), which aborts
/// without saying which call trapped.
///
/// It must not return: generated code follows the call with `unreachable`.
extern "C" fn harness_trap(_ctx: *mut u8, code: u32) {
    if std::env::var_os(CHILD_VAR).is_some() {
        uncaptured(&format!("{TRAP_MARKER}{code}"));

        std::process::exit(TRAP_EXIT);
    }

    let what = CURRENT_CALL.with(|c| c.borrow().clone());
    let message = TrapCode::from_u32(code).map_or("an unknown trap code", |t| t.message());

    uncaptured(&format!(
        "{what}: the JIT trapped with `{message}` (code {code}), but the interpreter returned"
    ));

    std::process::abort()
}

/// Writes a line straight to stderr. `eprintln!` inside a test goes to libtest's
/// capture buffer, which a process about to abort or exit never prints.
fn uncaptured(line: &str) {
    use std::io::Write as _;

    let _ = writeln!(std::io::stderr(), "{line}");
}

// ---------------------------------------------------------------------------
// The child process

enum ChildOutcome {
    Trapped(u32),
    /// The results the child printed, as text.
    Returned(String),
}

fn parse_child_spec(spec: &str) -> (usize, usize, Level) {
    let parts: Vec<usize> = spec
        .split(':')
        .map(|p| p.parse().expect("a number"))
        .collect();

    let [check_no, call_no, level] = parts[..] else {
        panic!("malformed {CHILD_VAR}=`{spec}`")
    };

    (check_no, call_no, LEVELS[level])
}

/// The child's side: makes the one call, which should trap inside [`harness_trap`].
/// Getting past it means it returned, which is reported for the parent to fail on.
fn run_in_child(lowered: &Lowered, args: &[Val], level: Level) -> ! {
    let (_, start) = instantiate(&lowered.module);
    let got = run_jit(lowered, level, start, args);

    uncaptured(&format!("{RETURNED_MARKER}{:?}", got.results));

    std::process::exit(0)
}

/// The parent's side: re-runs this test binary, filtered to the current test, to
/// make call `call_no` of check `check_no` at `level`.
fn run_child(check_no: usize, call_no: usize, level: Level, what: &str) -> ChildOutcome {
    // libtest runs each test on a thread named after it.
    let test = thread::current()
        .name()
        .filter(|name| *name != "main")
        .expect("a trap case runs inside a libtest test")
        .to_owned();

    let level_no = LEVELS.iter().position(|l| *l == level).unwrap();

    let mut child = Command::new(std::env::current_exe().expect("the test binary"))
        .args([test.as_str(), "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_VAR, format!("{check_no}:{call_no}:{level_no}"))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the child");

    // Drained on its own thread, so a chatty child cannot fill the pipe and stall.
    let mut pipe = child.stderr.take().expect("piped stderr");
    let reader = thread::spawn(move || {
        let mut stderr = String::new();

        pipe.read_to_string(&mut stderr)
            .expect("read the child's stderr");

        stderr
    });

    let started = Instant::now();

    let status = loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => break status,
            None if started.elapsed() > LIMIT => {
                let _ = child.kill();
                let _ = child.wait();

                panic!(
                    "{what}: did not finish within {LIMIT:?} — the code most likely loops forever"
                );
            }
            None => thread::sleep(Duration::from_millis(10)),
        }
    };

    let stderr = reader.join().expect("the stderr reader");

    if let Some(code) = stderr.lines().find_map(|l| l.strip_prefix(TRAP_MARKER)) {
        assert_eq!(
            status.code(),
            Some(TRAP_EXIT),
            "{what}: the child reported a trap but exited with {status}:\n{stderr}"
        );

        return ChildOutcome::Trapped(code.trim().parse().expect("a trap code"));
    }

    if let Some(results) = stderr.lines().find_map(|l| l.strip_prefix(RETURNED_MARKER)) {
        return ChildOutcome::Returned(results.to_string());
    }

    panic!("{what}: the child neither trapped nor returned ({status}):\n{stderr}")
}

// ---------------------------------------------------------------------------
// Values

/// A value's bits in an `i64` slot, low bits first.
fn to_bits(v: &Val) -> u64 {
    match *v {
        Val::I32(x) => x as u32 as u64,
        Val::I64(x) => x as u64,
        Val::F32(x) => x.to_bits() as u64,
        Val::F64(x) => x.to_bits(),
        Val::Ref(_) => unreachable!("rejected by `lower`"),
    }
}

fn from_bits(bits: u64, ty: &ValType) -> Val {
    match ty {
        ValType::I32 => Val::I32(bits as u32 as i32),
        ValType::I64 => Val::I64(bits as i64),
        ValType::F32 => Val::F32(f32::from_bits(bits as u32)),
        ValType::F64 => Val::F64(f64::from_bits(bits)),
        _ => unreachable!("rejected by `lower`"),
    }
}

/// Equal as wasm values: floats by bit pattern, so `-0.0` and `0.0` differ, except
/// that any two NaNs match — wasm leaves an arithmetic NaN's payload open.
fn same(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::I32(x), Val::I32(y)) => x == y,
        (Val::I64(x), Val::I64(y)) => x == y,
        (Val::F32(x), Val::F32(y)) => (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits(),
        (Val::F64(x), Val::F64(y)) => (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits(),
        (Val::Ref(x), Val::Ref(y)) => x == y,
        _ => false,
    }
}

fn fits(v: &Val, ty: &ValType) -> bool {
    matches!(
        (v, ty),
        (Val::I32(_), ValType::I32)
            | (Val::I64(_), ValType::I64)
            | (Val::F32(_), ValType::F32)
            | (Val::F64(_), ValType::F64)
    )
}

impl Lowered {
    fn assert_args_fit(&self, args: &[Val]) {
        assert!(
            args.len() == self.params.len()
                && args.iter().zip(&self.params).all(|(a, t)| fits(a, t)),
            "{args:?} does not fit the parameters {:?}",
            self.params
        );
    }

    fn assert_results(&self, what: &str, got: &[Val], want: &[Val]) {
        assert!(
            got.len() == want.len() && got.iter().zip(want).all(|(g, w)| same(g, w)),
            "{what}: the JIT returned {got:?}, the interpreter {want:?}\n--- IR ---\n{}",
            self.ir
        );
    }

    /// Every global agrees with the interpreter's after the call. A reference global
    /// is compared in its slot encoding, since that is all the JIT side has.
    fn assert_globals(&self, what: &str, got: &[u64], want: &[Val]) {
        assert_eq!(got.len(), want.len(), "{what}: the global counts differ");

        for (i, (raw, want)) in got.iter().zip(want).enumerate() {
            let ty = self.module.globals[i].ty.content_type();

            let matches = match ty {
                ValType::Ref(_) => GlobalVal::from(*want).bits() == *raw,
                _ => same(&from_bits(*raw, &ty), want),
            };

            assert!(
                matches,
                "{what}: global {i} holds {raw:#x} after the JIT's call, {want:?} after the \
                 interpreter's\n--- IR ---\n{}",
                self.ir
            );
        }
    }
}

/// Runs `f`, aborting with `what` if it takes longer than [`LIMIT`]: a hung case
/// otherwise hangs the whole run instead of failing.
fn with_watchdog<T>(what: &str, f: impl FnOnce() -> T) -> T {
    let (done, wait) = mpsc::channel::<()>();
    let what = what.to_owned();

    let dog = thread::spawn(move || {
        if wait.recv_timeout(LIMIT) == Err(mpsc::RecvTimeoutError::Timeout) {
            uncaptured(&format!(
                "{what}: did not finish within {LIMIT:?} — the code most likely loops forever"
            ));

            std::process::abort();
        }
    });

    let out = f();

    drop(done);
    dog.join().expect("the watchdog");

    out
}
