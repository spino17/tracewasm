//! A single consolidated performance report for the VM, across every guest.
//!
//! Run it with:
//!
//! ```sh
//! cargo test --release -p tracewasm-test --test perf_report -- --nocapture --test-threads=1
//! ```
//!
//! ## Why this exists alongside `metrics.rs`
//!
//! `metrics.rs` answers "did something catastrophically regress" — it asserts loose
//! bounds and prints means. This file answers "how does the VM perform, and is a
//! change real or noise". It reports **distributions**, not just means, because a
//! mean hides exactly the thing that matters for an interpreter embedded in a
//! request path: the tail.
//!
//! ## Why `--test-threads=1` is not optional
//!
//! Every measurement here is a timing loop. Letting cargo run test functions
//! concurrently inflates the throughput numbers and widens run-to-run spread from
//! ~3% to ~11% — wide enough to swallow any realistic single optimisation. Numbers
//! taken without it are not comparable to numbers taken with it.
//!
//! ## How to read it honestly
//!
//! * **Discard the first run after a rebuild.** A cold row can read well over
//!   half again its warm value.
//! * **Ignore `max`.** §2 calibrates the timer against an empty measurement; its
//!   own max lands in the microseconds because the OS deschedules the thread. Any
//!   `max` at that scale is scheduler noise, not the VM. Trust p99, distrust max.
//! * **Treat movement below the spread column as nothing.** That column exists so
//!   a future change can be judged against measured noise rather than hope.
//! * **Compare this report only against itself.** §3 reads a few percent higher
//!   than the same workloads in `metrics.rs`: that file times N iterations in one
//!   block and divides (a mean), this one times each rep and takes the median, and
//!   §3 runs after §1-2 have already done ~400k invocations. Both are right; they
//!   measure slightly different things. Cross-comparing invents regressions.
//!
//! Nothing here asserts on wall-clock time: a throughput threshold that fails on a
//! loaded machine is worse than no test at all. The only assertions are that the
//! trap workloads in §6 really trap — timing a silent success would be meaningless.

// Every workload here is a compiled guest, so the whole file compiles away when the
// wasm target is missing and `tracewasm_test::guests` does not exist.
#![cfg(not(no_guest_wasm))]

use std::time::Instant;

use tracewasm_core::{
    Stack, instance::config::Config, memory::linear::LinearMemory, module::Module,
};
use tracewasm_test::{Guest, MAX_TEST_RECURSION, NoImports, guests, with_large_stack};

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// Guest-loop iterations per workload call. One "op" below is one pass of a
/// guest's `while i < n` body — on the order of 10-30 wasm instructions, *not*
/// one. Never read these figures as cycles-per-instruction.
const WORK: i32 = 20_000;

/// Repeats behind every median in §3 and §5.
///
/// Sized so §3 agrees with §2's 2,000-sample distribution on the one call they both
/// measure; a handful of reps puts the median one sample away from an outlier and
/// the two sections disagree by several percent.
const REPEATS: usize = if cfg!(debug_assertions) { 5 } else { 25 };

/// Samples for latency distributions. Large enough for a meaningful p99.9.
const LATENCY_SAMPLES: usize = if cfg!(debug_assertions) {
    20_000
} else {
    200_000
};

/// Samples for the cold-start distribution. Compiling ~1.4 MB of wasm is
/// milliseconds, so this stays small.
const STARTUP_SAMPLES: usize = if cfg!(debug_assertions) { 20 } else { 200 };

/// One guest workload.
struct Workload {
    name: &'static str,
    wasm: &'static [u8],
    export: &'static str,
    work: i32,
    /// Whether the guest's cost is proportional to `work`, i.e. whether dividing
    /// by it yields a meaningful per-op figure.
    ///
    /// False for `heap`: its body allocates and its total cost is nearly flat in
    /// `n`, so dividing by `n` yields a figure that changes with the work size
    /// while the wall clock does not. Reporting ns/op there would invite a
    /// comparison against the other rows that means nothing.
    scales_with_work: bool,
}

const GUESTS: &[Workload] = &[
    Workload {
        name: "arithmetic",
        wasm: guests::ARITHMETIC,
        export: "arith_mixed_workload",
        work: WORK,
        scales_with_work: true,
    },
    Workload {
        name: "control_flow",
        wasm: guests::CONTROL_FLOW,
        export: "cf_mixed_workload",
        work: WORK,
        scales_with_work: true,
    },
    Workload {
        name: "memory",
        wasm: guests::MEMORY,
        export: "mem_mixed_workload",
        work: WORK,
        scales_with_work: true,
    },
    Workload {
        name: "frames",
        wasm: guests::FRAMES,
        export: "fr_mixed_workload",
        work: WORK,
        scales_with_work: true,
    },
    Workload {
        name: "heap",
        wasm: guests::HEAP,
        export: "heap_mixed_workload",
        work: 2_000,
        scales_with_work: false,
    },
];

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// A sorted sample set. Constructed once so the percentile reads are cheap.
struct Dist(Vec<f64>);

impl Dist {
    fn new(mut v: Vec<f64>) -> Self {
        v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration sample"));
        Dist(v)
    }

    /// Nearest-rank percentile. `p` in `0.0..=1.0`.
    fn at(&self, p: f64) -> f64 {
        let idx = ((self.0.len() - 1) as f64 * p).round() as usize;
        self.0[idx]
    }

    fn mean(&self) -> f64 {
        self.0.iter().sum::<f64>() / self.0.len() as f64
    }

    fn min(&self) -> f64 {
        self.0[0]
    }

    fn max(&self) -> f64 {
        self.0[self.0.len() - 1]
    }

    /// Spread as a percentage of the median — the "is this signal?" column.
    ///
    /// p99-to-min rather than max-to-min: `max` is contaminated by OS descheduling
    /// (see the timer floor in §2), so a noise-floor column built from it would
    /// contradict this report's own advice to distrust `max`. With a handful of
    /// samples p99 *is* the max; with thousands it is not.
    fn spread_pct(&self) -> f64 {
        let med = self.at(0.5);
        if med == 0.0 {
            return 0.0;
        }
        (self.at(0.99) - self.min()) / med * 100.0
    }
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

const W: usize = 96;

fn banner(title: &str) {
    println!("\n{}", "=".repeat(W));
    println!(" {title}");
    println!("{}", "=".repeat(W));
}

fn section(n: u8, title: &str, note: &str) {
    println!("\n {n}. {title}");
    if !note.is_empty() {
        println!("    {note}");
    }
    println!(" {}", "-".repeat(W - 1));
}

fn dist_header(unit: &str) {
    println!(
        "   {:<26}{:>10}{:>10}{:>10}{:>10}{:>10}{:>11}  {}",
        "measurement", "mean", "p50", "p90", "p99", "p99.9", "max", unit
    );
}

fn dist_row(label: &str, d: &Dist) {
    println!(
        "   {:<26}{:>10.2}{:>10.2}{:>10.2}{:>10.2}{:>10.2}{:>11.2}",
        label,
        d.mean(),
        d.at(0.5),
        d.at(0.90),
        d.at(0.99),
        d.at(0.999),
        d.max()
    );
}

fn spread_header(first: &str, unit: &str) {
    println!(
        "   {:<22}{:>12}{:>11}{:>11}{:>10}  {}",
        first, "median", "min", "p99", "spread", unit
    );
}

fn spread_row(label: &str, d: &Dist) {
    println!(
        "   {:<22}{:>12.2}{:>11.2}{:>11.2}{:>9.1}%",
        label,
        d.at(0.5),
        d.min(),
        d.at(0.99),
        d.spread_pct()
    );
}

fn kb(bytes: usize) -> f64 {
    bytes as f64 / 1024.0
}

// ---------------------------------------------------------------------------
// Process RSS, for the one memory number that includes everything
// ---------------------------------------------------------------------------

/// Resident set size of this process in KiB, or `None` where `ps` is unavailable.
///
/// Deliberately shells out rather than taking a dependency: this is a reporting
/// aid, and a missing `ps` should degrade the report rather than fail the test.
/// The deterministic footprint in §4 is the number to rely on; RSS is context.
fn rss_kib() -> Option<f64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;

    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

/// RSS at each stage of bringing one guest up, captured **before anything else in
/// the report runs**.
///
/// Order matters: once §1 has compiled every guest a few hundred times, the
/// allocator's high-water mark is set and every subsequent delta reads as 0.0 MB.
/// So these are taken first and printed later.
struct RssMilestones {
    start: Option<f64>,
    after_compile: Option<f64>,
    after_instantiate: Option<f64>,
    after_calls: Option<f64>,
}

fn rss_milestones() -> RssMilestones {
    let start = rss_kib();

    let module = Module::<Stack>::compile(guests::HEAP).expect("guest compiles");
    let after_compile = rss_kib();

    let mut instance = module
        .instantiate::<LinearMemory, _>(NoImports, None)
        .expect("guest instantiates");
    let after_instantiate = rss_kib();

    let f = module
        .get_typed_func::<(i32,), (i64,)>("heap_mixed_workload")
        .expect("export exists");
    for _ in 0..50 {
        let _ = f.call((2_000,), &mut instance);
    }
    let after_calls = rss_kib();

    RssMilestones {
        start,
        after_compile,
        after_instantiate,
        after_calls,
    }
}

fn rss_row(label: &str, before: Option<f64>, after: Option<f64>) {
    match (before, after) {
        (Some(b), Some(a)) => println!(
            "   {:<34}{:>12.1} MB{:>14.1} MB{:>13.2} MB",
            label,
            b / 1024.0,
            a / 1024.0,
            (a - b) / 1024.0
        ),
        _ => println!("   {label:<34}  (ps unavailable)"),
    }
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

/// Driven on a large-stack thread: §5 recurses thousands of wasm frames, and
/// libtest's ~2 MiB default would abort the whole binary rather than fail a test.
/// See [`with_large_stack`].
#[test]
fn vm_performance_report() {
    with_large_stack(report);
}

fn report() {
    // Taken first, while the allocator is still cold — see `RssMilestones`.
    let rss = rss_milestones();

    banner("TraceWasm VM performance report");

    let profile = if cfg!(debug_assertions) {
        "debug (numbers ~10x slower; use --release to compare)"
    } else {
        "release"
    };
    println!(
        " profile: {profile}    instruction size: {} B",
        Module::<Stack>::instruction_size()
    );

    let timer = timer_floor();
    println!(
        " timer resolution: ~{:.0} ns (p90 of an empty measurement)",
        timer.at(0.90)
    );

    startup();
    latency(&timer);
    throughput();
    memory(&rss);
    scaling();
    error_path();

    println!("\n{}\n", "=".repeat(W));
}

/// Calibrates the clock so §2 can be read honestly.
fn timer_floor() -> Dist {
    let mut v = Vec::with_capacity(LATENCY_SAMPLES);
    for _ in 0..LATENCY_SAMPLES {
        let t = Instant::now();
        v.push(t.elapsed().as_nanos() as f64);
    }
    Dist::new(v)
}

/// §1 — cold path: how long until the first instruction can run.
///
/// For an embedding that compiles per request this is the entire latency budget,
/// and it is invisible in every throughput row.
fn startup() {
    section(
        1,
        "STARTUP — time to first instruction",
        "compile dominates; the guests are ~1.4 MB each because they link std",
    );
    // MB/s of input would flatter the compiler: these guests are mostly data and
    // debug info, so `arithmetic` lowers only 2,626 instructions out of 1.42 MB.
    // Cost per lowered instruction is the comparable figure.
    println!(
        "   {:<16}{:>9}{:>9}{:>12}{:>10}{:>12}{:>13}",
        "guest", "wasm MB", "instrs", "compile ms", "p99 ms", "ns/instr", "instant. us"
    );

    for w in GUESTS {
        let (name, wasm) = (w.name, w.wasm);
        let mut compile = Vec::with_capacity(STARTUP_SAMPLES);
        for _ in 0..STARTUP_SAMPLES {
            let t = Instant::now();
            let m = Module::<Stack>::compile(wasm).expect("guest compiles");
            compile.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&m);
        }
        let compile = Dist::new(compile);

        // Instantiation is measured separately: it allocates linear memory and
        // runs the start section, and scales with the guest's data segments
        // rather than its code size.
        let module = Module::<Stack>::compile(wasm).expect("guest compiles");
        let mut inst = Vec::with_capacity(STARTUP_SAMPLES);
        for _ in 0..STARTUP_SAMPLES {
            let t = Instant::now();
            let i = module
                .instantiate::<LinearMemory, _>(NoImports, None)
                .expect("guest instantiates");
            inst.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&i);
        }
        let inst = Dist::new(inst);

        let instrs = module.instruction_count();

        println!(
            "   {:<16}{:>9.2}{:>9}{:>12.2}{:>10.2}{:>12.0}{:>13.1}",
            name,
            wasm.len() as f64 / (1024.0 * 1024.0),
            instrs,
            compile.at(0.5),
            compile.at(0.99),
            compile.at(0.5) * 1e6 / instrs as f64,
            inst.at(0.5) * 1e3
        );
    }
}

/// §2 — latency: the distribution of a single call, which is what a request path
/// actually experiences.
fn latency(timer: &Dist) {
    section(
        2,
        "LATENCY — per-call distribution",
        "trust p99; `max` is dominated by OS descheduling (see the timer row)",
    );
    // The two invocation rows are DIFFERENT guests with different bodies, so the
    // gap between them is not the cost of extra parameters — naming the export
    // keeps that from being read as "5 params is cheaper than 1".
    println!("   rows below are distinct exports; the gap is not a param-count cost");
    dist_header("ns");

    // n=1 keeps the guest body to a handful of instructions, isolating the cost
    // of entering and leaving the VM.
    let mut g = Guest::<Stack>::new(guests::ARITHMETIC);
    let mut v = Vec::with_capacity(LATENCY_SAMPLES);
    for _ in 0..LATENCY_SAMPLES {
        let t = Instant::now();
        g.i32_i64("arith_mixed_workload", 1);
        v.push(t.elapsed().as_nanos() as f64);
    }
    dist_row("arith_mixed_workload(1)", &Dist::new(v));

    let mut f = Guest::<Stack>::new(guests::FRAMES);
    let mut v = Vec::with_capacity(LATENCY_SAMPLES);
    for _ in 0..LATENCY_SAMPLES {
        let t = Instant::now();
        f.i32x5_i64("fr_five_params", (1, 2, 3, 4, 5));
        v.push(t.elapsed().as_nanos() as f64);
    }
    dist_row("fr_five_params(1..5)", &Dist::new(v));

    dist_row("(timer floor)", timer);

    // A long call should be *tighter* than a short one in relative terms: the
    // fixed entry cost and any scheduler blip amortise over the work.
    println!();
    dist_header("ms");
    let mut v = Vec::with_capacity(2_000);
    for _ in 0..2_000 {
        let t = Instant::now();
        g.i32_i64("arith_mixed_workload", WORK);
        v.push(t.elapsed().as_secs_f64() * 1e3);
    }
    dist_row("arith_mixed_workload(20k)", &Dist::new(v));
}

/// §3 — throughput: steady-state cost, one row per guest so a regression can be
/// localised to a subsystem instead of showing up as a blended slowdown.
fn throughput() {
    section(
        3,
        "THROUGHPUT — steady state, per subsystem",
        "ns/op is per guest loop iteration (~10-30 wasm instrs), NOT per instruction",
    );
    println!(
        "   {:<16}{:>10}{:>11}{:>10}{:>10}{:>10}",
        "workload", "ms/call", "ns/op", "min", "p99", "spread"
    );

    for w in GUESTS {
        let mut g = Guest::<Stack>::new(w.wasm);
        g.i32_i64(w.export, w.work); // warm-up: fault in the pages, not counted

        let mut v = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let t = Instant::now();
            g.i32_i64(w.export, w.work);
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let d = Dist::new(v);

        // ns/op only where cost is proportional to `work` — see `scales_with_work`.
        let per_op = |ms: f64| ms * 1e6 / w.work as f64;
        if w.scales_with_work {
            println!(
                "   {:<16}{:>10.2}{:>11.2}{:>10.2}{:>10.2}{:>9.1}%",
                w.name,
                d.at(0.5),
                per_op(d.at(0.5)),
                per_op(d.min()),
                per_op(d.at(0.99)),
                d.spread_pct()
            );
        } else {
            println!(
                "   {:<16}{:>10.2}{:>11}{:>10}{:>10}{:>9.1}%",
                w.name,
                d.at(0.5),
                "-",
                "-",
                "-",
                d.spread_pct()
            );
        }
    }
    println!("   (heap: cost is flat in n, so ns/op would not be comparable)");
}

/// §4 — memory: the deterministic part first, because it has no noise at all.
fn memory(rss: &RssMilestones) {
    section(
        4,
        "MEMORY — instruction stream (exact) and process RSS (context)",
        "stream bytes have no measurement noise: any movement here is real",
    );
    println!(
        "   {:<22}{:>12}{:>12}{:>12}{:>12}",
        "guest", "instrs", "stream KB", "offsets KB", "locals KB"
    );

    let isz = Module::<Stack>::instruction_size();
    let (mut ti, mut ts, mut to) = (0usize, 0usize, 0usize);

    for w in GUESTS {
        let (name, wasm) = (w.name, w.wasm);
        let m = Module::<Stack>::compile(wasm).expect("guest compiles");
        let instrs = m.instruction_count();
        let offs = m.instruction_offset_count();
        let locals = m.locals_count();

        ti += instrs;
        ts += instrs * isz;
        to += offs * 4;

        println!(
            "   {:<22}{:>12}{:>12.0}{:>12.0}{:>12.1}",
            name,
            instrs,
            kb(instrs * isz),
            kb(offs * 4),
            kb(locals)
        );
    }
    println!(
        "   {:<22}{:>12}{:>12.0}{:>12.0}",
        "TOTAL",
        ti,
        kb(ts),
        kb(to)
    );

    // RSS captures everything the deterministic columns cannot: the operand-stack
    // reservation, linear memory, and allocator slack. Measured on a cold
    // allocator at the very start of the report — deltas taken here would all read
    // 0.0 MB, because §1 has already established the high-water mark.
    println!();
    println!(
        "   {:<34}{:>15}{:>17}{:>16}",
        "process RSS (heap guest, cold)", "before", "after", "delta"
    );
    rss_row("compile", rss.start, rss.after_compile);
    rss_row("+ instantiate", rss.after_compile, rss.after_instantiate);
    rss_row("+ 50 calls", rss.after_instantiate, rss.after_calls);
}

/// §5 — scaling: does cost stay proportional as work and depth grow?
///
/// A VM that is linear in work and flat in depth is predictable, which matters
/// more than a good number at one size.
fn scaling() {
    section(
        5,
        "SCALING — linearity in work, flatness in depth",
        "ns/op should stay flat across sizes; a rise means per-call cost or cache pressure",
    );

    println!("   {:<22}{:>12}{:>14}", "work size (n)", "ms/call", "ns/op");
    let mut g = Guest::<Stack>::new(guests::ARITHMETIC);
    for n in [100, 1_000, 20_000, 100_000] {
        // Median of several reps, not a single shot: one sample per size made a
        // 160-vs-144 ns/op blip look like a scaling cliff when it was noise.
        let mut v = Vec::with_capacity(REPEATS);
        g.i32_i64("arith_mixed_workload", n); // warm
        for _ in 0..REPEATS {
            let t = Instant::now();
            g.i32_i64("arith_mixed_workload", n);
            v.push(t.elapsed().as_secs_f64());
        }
        let el = Dist::new(v).at(0.5);
        println!(
            "   {:<22}{:>12.3}{:>14.2}",
            n,
            el * 1e3,
            el * 1e9 / n as f64
        );
    }

    println!(
        "\n   {:<22}{:>12}{:>14}",
        "recursion depth", "ms/call", "ns/frame"
    );

    // The default guard trips well below these depths, so raise it — but stay
    // under the native capacity of the large-stack thread, since a Rust stack
    // overflow is SIGABRT and would take the binary down.
    let mut cfg = Config::default();
    cfg.set_max_call_stack_depth(4_000);
    let mut f = Guest::<Stack>::with_config(guests::FRAMES, Some(cfg));

    for depth in [100, 1_000, 3_000]
        .into_iter()
        .filter(|d| *d <= MAX_TEST_RECURSION)
    {
        let mut v = Vec::with_capacity(REPEATS);
        f.i32_i64("fr_recurse_depth", depth); // warm
        for _ in 0..REPEATS {
            let t = Instant::now();
            f.i32_i64("fr_recurse_depth", depth);
            v.push(t.elapsed().as_secs_f64());
        }
        let el = Dist::new(v).at(0.5);
        println!(
            "   {:<22}{:>12.3}{:>14.2}",
            depth,
            el * 1e3,
            el * 1e9 / depth as f64
        );
    }
}

/// §6 — the error path, which for a tracing VM is a feature rather than an edge.
///
/// A trap that costs milliseconds is a denial-of-service surface when guest code
/// is untrusted, so it belongs in a performance report.
fn error_path() {
    section(
        6,
        "ERROR PATH — trap cost",
        "traps are a normal outcome for untrusted guests, not an exceptional one",
    );
    spread_header("trap", "us");

    let traps: &[(&str, &[u8], &str)] = &[
        ("out-of-bounds read", guests::MEMORY, "mem_trap_oob_read"),
        ("divide by zero", guests::MEMORY, "mem_trap_div_zero"),
        ("explicit panic", guests::MEMORY, "mem_trap_explicit_panic"),
    ];

    for (label, wasm, export) in traps {
        let mut g = Guest::<Stack>::new(wasm);

        // Confirm it actually traps: timing a success would be meaningless.
        assert!(
            g.try_i32_i64(export, 0).is_err(),
            "`{export}` was expected to trap but returned a value"
        );

        let mut v = Vec::with_capacity(2_000);
        for _ in 0..2_000 {
            let t = Instant::now();
            let r = g.try_i32_i64(export, 0);
            v.push(t.elapsed().as_secs_f64() * 1e6);
            std::hint::black_box(&r);
        }
        spread_row(label, &Dist::new(v));
    }
}
