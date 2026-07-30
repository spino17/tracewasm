//! Quantitative reports: throughput per workload category, per-call overhead,
//! frame cost, and reachable call depth.
//!
//! ## These print numbers; they mostly do not assert on them
//!
//! A wall-clock threshold fails on a loaded CI box, which trains everyone to
//! ignore the suite. So the timing rows are *reported* — run with
//! `cargo test -p tracewasm-test --release -- --nocapture` to see them — and the
//! only assertions are on things that are true regardless of machine speed:
//! orderings, ratios, and counts.
//!
//! The suite passes in both profiles, but **read the numbers from `--release`
//! only**. A debug interpreter is roughly an order of magnitude slower and spends
//! ~30 KB of native stack per wasm frame against ~1.3 KB, so debug figures measure
//! the build profile rather than the engine. Anything profile-sensitive here
//! (iteration counts, depth caps, the per-call bound) is scaled via
//! `cfg!(debug_assertions)` or `MAX_TEST_RECURSION`.
//!
//! ## Where the baselines live
//!
//! `PERFORMANCE_IMPROVEMENTS.md` at the repo root records the current figures and
//! the reasoning behind each optimisation lever. Update it when these move.

#![cfg(not(no_guest_wasm))]

use tracewasm_core::instance::config::Config;
use tracewasm_test::metrics::Table;
use tracewasm_test::{Guest, MAX_TEST_RECURSION, guests, with_large_stack};

/// Iterations per guest call for the throughput rows.
///
/// Every `*_mixed_workload` export takes a loop count, so one call does `WORK`
/// units of guest work and the reported `ns/op` is per unit.
const WORK: i32 = 20_000;

/// How many times each workload is invoked when timing it.
const REPS: u64 = 5;

/// Invocations used for the per-call overhead rows.
///
/// Fewer in debug, where the interpreter is roughly an order of magnitude slower
/// and 200k calls would make `cargo test` noticeably slow for no extra signal.
const CALL_REPS: u64 = if cfg!(debug_assertions) { 20_000 } else { 200_000 };

/// Loose upper bound on per-call overhead, scaled to the build profile.
///
/// This exists to catch a *catastrophic* regression — an allocation per call, a
/// linear scan of the export table — not to police performance. The release
/// figure at the time of writing was ~350 ns, so the bound is deliberately many
/// times that, and higher again in debug where every operation is slower.
const MAX_PER_CALL_NS: u128 = if cfg!(debug_assertions) { 20_000 } else { 2_000 };

/// Throughput by workload category, which is what localises a regression.
///
/// A change that slows arithmetic but not calls looks identical to one that slows
/// calls but not arithmetic if you only measure a single blended benchmark. These
/// four rows are chosen so each isolates a different interpreter cost: instruction
/// dispatch, indirect calls, linear-memory access, and the locals region.
#[test]
fn report_throughput_by_category() {
    let mut table = Table::new("throughput by workload category (release)");

    let mut arith = Guest::new(guests::ARITHMETIC);
    table.measure("arithmetic (i64 + f64 mix)", REPS, Some(WORK as u64), || {
        arith.i32_i64("arith_mixed_workload", WORK);
    });

    let mut cf = Guest::new(guests::CONTROL_FLOW);
    table.measure("control flow (match + loops)", REPS, Some(WORK as u64), || {
        cf.i32_i64("cf_mixed_workload", WORK);
    });

    let mut mem = Guest::new(guests::MEMORY);
    table.measure("memory (load/store)", REPS, Some(WORK as u64), || {
        mem.i32_i64("mem_mixed_workload", WORK);
    });

    let mut frames = Guest::new(guests::FRAMES);
    table.measure("calls (one indirect per iter)", REPS, Some(WORK as u64), || {
        frames.i32_i64("fr_mixed_workload", WORK);
    });

    let mut heap = Guest::new(guests::HEAP);
    table.measure("heap (alloc + collections)", REPS, None, || {
        heap.i32_i64("heap_mixed_workload", 2_000);
    });

    table.report();

    // Sanity, not performance: every row must have actually run. A zero elapsed
    // time means the workload was optimised away and the row is meaningless.
    for row in table.rows() {
        assert!(
            row.elapsed.as_nanos() > 0,
            "row `{}` measured zero time, so it is not measuring anything",
            row.label
        );
    }
}

/// Per-invocation overhead: the fixed cost of crossing the host boundary,
/// separated from the cost of the work inside.
///
/// Measured by calling a trivial export in a tight loop. This is the number that
/// matters for an embedder making many small calls, and it moves independently of
/// interpreter throughput — the two were 3.4x and 20x off a reference engine
/// respectively at the time of writing, which is a useful thing to know.
#[test]
fn report_per_call_overhead() {
    let mut table = Table::new("host-boundary overhead");

    let mut arith = Guest::new(guests::ARITHMETIC);

    // n = 1 keeps the guest body to a handful of instructions, so this is
    // dominated by call setup and teardown
    table.measure("trivial call, n=1", CALL_REPS, Some(1), || {
        arith.i32_i64("arith_mixed_workload", 1);
    });

    let mut frames = Guest::new(guests::FRAMES);
    table.measure("5-param call", CALL_REPS, Some(1), || {
        frames.i32x5_i64("fr_five_params", (1, 2, 3, 4, 5));
    });

    table.report();

    let trivial = table.row("trivial call, n=1").expect("row was recorded");

    let per_call_ns = trivial.per_iter().as_nanos();

    assert!(
        per_call_ns < MAX_PER_CALL_NS,
        "per-call overhead regressed badly: {per_call_ns} ns/call \
         (bound for this profile is {MAX_PER_CALL_NS})"
    );
}

/// Reachable call depth, and the timing of deep recursion.
///
/// ## Why this does not measure the native stack cliff
///
/// The obvious version of this test raises `max_call_stack_depth` to disable the
/// guard and binary-searches for the depth at which the *native* stack overflows.
/// That cannot be done in-process: a Rust stack overflow raises `SIGABRT` and
/// takes the whole test binary with it, so the search aborts on its first probe
/// above the cliff rather than returning a bound.
///
/// Finding the true per-frame cost therefore needs one process per probe. That
/// rig lives outside the test suite, and its results are recorded in
/// `PERFORMANCE_IMPROVEMENTS.md`.
///
/// What this test *can* establish, safely, is that recursion reaches a depth
/// ordinary Rust needs — a recursive-descent parser or a deep tree walk wants
/// thousands of frames — with the guard set to a value known to sit below the
/// native limit.
#[test]
fn report_reachable_call_depth() {
    // libtest runs each test on a spawned thread whose default stack (~2 MiB)
    // fits only a few dozen wasm frames in a debug build, and a Rust stack
    // overflow is SIGABRT — it kills the whole binary instead of failing one test.
    with_large_stack(deep_recursion_body);
}

fn deep_recursion_body() {
    // Comfortably below the native capacity of the thread above, so the guard
    // stops the recursion before the stack does. Raising this to probe the real
    // cliff would abort the process; see the note on the test.
    const SAFE_LIMIT: u32 = 4_000;

    let mut cfg = Config::default();
    cfg.set_max_call_stack_depth(SAFE_LIMIT);

    let mut g = Guest::with_config(guests::FRAMES, Some(cfg));

    let mut table = Table::new("deep recursion (guard set to 4000)");

    // Depths are capped by `MAX_TEST_RECURSION`, which is far lower in debug
    // builds — see its docs for the ~20x per-frame stack difference.
    let depths: Vec<i32> = [100, 1_000, 3_000]
        .into_iter()
        .filter(|d| *d <= MAX_TEST_RECURSION)
        .collect();

    for depth in depths {
        // one call per rep; ops_per_iter is the frame count, so ns/op is
        // per-frame cost including the work in each frame
        table.measure(
            format!("direct recursion, depth {depth}"),
            20,
            Some(depth as u64),
            || {
                g.i32_i64("fr_recurse_depth", depth);
            },
        );
    }

    table.report();

    // Machine-independent facts, which is all that is asserted here.
    let deep = MAX_TEST_RECURSION;

    assert!(
        g.try_i32_i64("fr_recurse_depth", deep).is_ok(),
        "{deep} frames of direct recursion must succeed under a limit of {SAFE_LIMIT}; \
         if this fails, either the guard is miscounting or per-frame stack cost has \
         regressed badly"
    );

    assert!(
        g.try_i32_i64("fr_recurse_indirect", deep * 2 / 3).is_ok(),
        "indirect recursion must reach {} frames under a limit of {SAFE_LIMIT}",
        deep * 2 / 3
    );

    // And the guard is what stops it, not the stack: just past the limit traps.
    assert!(
        g.try_i32_i64("fr_recurse_depth", SAFE_LIMIT as i32 + 100).is_err(),
        "recursion past the configured limit should trap"
    );
}

/// Compilation and instantiation cost, which an embedder pays per module rather
/// than per call.
///
/// Worth tracking separately: the guests are ~1.5 MB of wasm each because they
/// link `std`, which is representative of a real Rust guest and very different
/// from a hand-written fixture.
#[test]
fn report_compile_and_instantiate_cost() {
    let mut table = Table::new("module load cost (~1.5 MB guests linking std)");

    for (label, wasm) in [
        ("arithmetic", guests::ARITHMETIC),
        ("heap", guests::HEAP),
        ("exotic", guests::EXOTIC),
    ] {
        table.measure(format!("compile + instantiate: {label}"), 5, None, || {
            let _ = Guest::new(wasm);
        });
    }

    table.report();

    println!();
    println!("  guest sizes");
    println!("  {}", "-".repeat(58));

    for (label, wasm) in [
        ("arithmetic", guests::ARITHMETIC),
        ("control_flow", guests::CONTROL_FLOW),
        ("heap", guests::HEAP),
        ("memory", guests::MEMORY),
        ("frames", guests::FRAMES),
        ("exotic", guests::EXOTIC),
    ] {
        println!(
            "  {:<32} {:>12}",
            label,
            format!("{:.2} MB", wasm.len() as f64 / (1024.0 * 1024.0))
        );
    }
}

/// Allocation behaviour, observed from inside the guest.
///
/// The interpreter has no allocation counter, and the guest's allocator lives in
/// its own linear memory, so this reports the *memory footprint the guest asks
/// for* rather than host allocations. That is the number that matters for the
/// `memory.grow` path and for the configured page ceiling.
#[test]
fn report_guest_memory_growth() {
    let mut table = Table::new("guest linear-memory growth");

    let mut mem = Guest::new(guests::MEMORY);

    // each unit is roughly one 64 KiB page, so this walks the grow path
    for pages in [1, 8, 32] {
        table.measure(format!("allocate ~{pages} page(s)"), 3, None, || {
            mem.i32_i64("mem_growth", pages);
        });
    }

    let mut heap = Guest::new(guests::HEAP);
    table.measure("vec growth to 20k elements", 3, None, || {
        heap.i32_i64("heap_vec_growth", 20_000);
    });
    table.measure("4k short-lived allocations", 3, None, || {
        heap.i32_i64("heap_churn", 4_000);
    });

    table.report();

    // Machine-independent: growing to more pages must not be *cheaper* than
    // growing to fewer, which would indicate the work is being skipped.
    let small = table.row("allocate ~1 page(s)").expect("row recorded");
    let large = table.row("allocate ~32 page(s)").expect("row recorded");

    assert!(
        large.per_iter() >= small.per_iter(),
        "growing to 32 pages ({:?}) was faster than growing to 1 ({:?}), \
         which suggests the allocation is being elided",
        large.per_iter(),
        small.per_iter()
    );
}
