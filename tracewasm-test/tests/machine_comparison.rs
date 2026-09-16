//! Stack machine versus register machine, on identical modules.
//!
//! ## Why this file exists separately
//!
//! `tests/metrics.rs` and `tests/perf_report.rs` both instantiate [`Stack`]
//! exclusively, so until this file landed there was no committed harness behind
//! any stack-vs-register claim. A ratio quoted in a README that no test produces
//! is not reproducible, and a reader cannot tell whether it still holds.
//!
//! ## Method
//!
//! The one number this file exists to produce is a *ratio* between two machines
//! measured on one host, which makes it far more sensitive to drift than an
//! absolute throughput figure: anything that changes the machine's speed between
//! the two halves of a comparison lands directly in the result.
//!
//! So, following the methodology notes in `OUTLINING_COLD_PATHS.md`:
//!
//! * **Rounds alternate A-B-A-B**, never AAA-BBB. An earlier measurement in this
//!   repo reported a 14% win that turned out to be partly thermal drift, because
//!   every sample of one variant ran before every sample of the other.
//! * **Which machine goes first flips each round**, so any residual
//!   first-in-round advantage cancels instead of accumulating on one side.
//! * **The reported figure is a median**, not a mean — one descheduled round
//!   otherwise moves the answer.
//! * **`rounds_won` is reported next to the ratio.** A 1.3x median where the
//!   winner took 4 of 7 rounds is noise wearing a ratio's clothing; the same
//!   ratio at 7 of 7 is a result. Read the two together, never the ratio alone.
//! * **One warm-up call per machine is discarded**, since the first call into a
//!   fresh instance faults in memory pages.
//!
//! Run it the way the rest of the perf suite is run — release, single-threaded,
//! output not captured:
//!
//! ```text
//! cargo test --release -p tracewasm-test --test machine_comparison -- --nocapture --test-threads=1
//! ```
//!
//! ## What is asserted
//!
//! Timing is *reported*; the assertions are the ones that hold regardless of how
//! fast or loaded the host is:
//!
//! * both machines return **bit-identical results** for every workload, checked
//!   before the clock starts — a speed comparison between two machines that
//!   disagree would be meaningless;
//! * every row measured non-zero time, so no row was optimised away.
//!
//! There is deliberately **no assertion that the register machine is faster**. It
//! is expected to be, but encoding that as a threshold would make the suite fail
//! on a loaded CI box, which trains everyone to ignore it.

#![cfg(not(no_guest_wasm))]

use std::time::{Duration, Instant};

use tracewasm_core::{Register, Stack};
use tracewasm_test::{Guest, guests};

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// Guest-loop iterations per workload call.
///
/// Matches `tests/metrics.rs` so the per-machine columns here are directly
/// comparable with the throughput table there. As in that file, one "op" is one
/// pass of a guest's `while i < n` body — on the order of 10-30 wasm
/// instructions, *not* one. Never read these as cycles-per-instruction.
const WORK: i32 = 20_000;

/// Iterations for the allocation-heavy guest, which is milliseconds per call.
const HEAP_WORK: i32 = 2_000;

/// Calls per timed sample within one round.
const REPS: u64 = 5;

/// Interleaved rounds behind every median.
///
/// Odd, so the median is a real sample rather than a midpoint of two. Seven is
/// enough for `rounds_won` to be meaningful without making the file slow.
const ROUNDS: usize = if cfg!(debug_assertions) { 3 } else { 7 };

/// Invocations behind the per-call overhead rows.
const CALL_REPS: u64 = if cfg!(debug_assertions) { 5_000 } else { 100_000 };

// ---------------------------------------------------------------------------
// Comparison plumbing
// ---------------------------------------------------------------------------

/// One workload, timed on both machines.
struct Comparison {
    label: &'static str,
    /// Per-round totals, index-aligned with `register` — round `i` of each ran
    /// adjacently, so the pair is comparable even if the host drifted between
    /// rounds.
    stack: Vec<Duration>,
    register: Vec<Duration>,
    /// Guest ops behind one timed sample, where the workload knows it.
    ops_per_sample: Option<u64>,
    /// Calls behind one timed sample, so a row without an op count can still be
    /// reported per call rather than as a raw sample total.
    reps: u64,
}

impl Comparison {
    fn median(mut v: Vec<Duration>) -> Duration {
        v.sort_unstable();
        v[v.len() / 2]
    }

    fn stack_median(&self) -> Duration {
        Self::median(self.stack.clone())
    }

    fn register_median(&self) -> Duration {
        Self::median(self.register.clone())
    }

    /// How many times faster the register machine is. Above 1.0 favours register.
    fn speedup(&self) -> f64 {
        self.stack_median().as_secs_f64() / self.register_median().as_secs_f64()
    }

    /// Rounds in which the register machine beat the stack machine.
    ///
    /// Reported beside the ratio because the two answer different questions: the
    /// ratio says how big the difference is, this says whether it is real.
    fn rounds_won(&self) -> usize {
        self.stack
            .iter()
            .zip(&self.register)
            .filter(|(s, r)| r < s)
            .count()
    }

    fn ns_per_op(&self, total: Duration) -> Option<f64> {
        let ops = self.ops_per_sample?;
        Some(total.as_secs_f64() * 1e9 / ops as f64)
    }
}

/// Times `body` over `REPS` calls, returning the total.
fn sample(reps: u64, mut body: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..reps {
        body();
    }
    start.elapsed()
}

/// Runs one workload on both machines with interleaved, order-flipped rounds.
///
/// Takes one closure per machine rather than one generic closure: `Guest<Stack>`
/// and `Guest<Register>` are distinct types, and unifying them behind a single
/// call site would mean making `Guest`'s call methods object-safe for the sake
/// of a test.
fn compare_with(
    label: &'static str,
    reps: u64,
    ops_per_sample: Option<u64>,
    mut run_stack: impl FnMut(),
    mut run_register: impl FnMut(),
) -> Comparison {
    // Discarded: the first call into a fresh instance faults in memory pages and
    // would otherwise land entirely on whichever machine happened to go first.
    run_stack();
    run_register();

    let mut stack = Vec::with_capacity(ROUNDS);
    let mut register = Vec::with_capacity(ROUNDS);

    for round in 0..ROUNDS {
        // Flip the order each round so any first-in-round advantage cancels
        // rather than accumulating on one machine.
        if round % 2 == 0 {
            stack.push(sample(reps, &mut run_stack));
            register.push(sample(reps, &mut run_register));
        } else {
            register.push(sample(reps, &mut run_register));
            stack.push(sample(reps, &mut run_stack));
        }
    }

    Comparison {
        label,
        stack,
        register,
        ops_per_sample,
        reps,
    }
}

/// Prints a comparison table. Visible under `--nocapture`.
fn report(title: &str, rows: &[Comparison]) {
    let width = 78;

    println!();
    println!("  {title}");
    println!("  {}", "-".repeat(width));
    println!(
        "  {:<30} {:>11} {:>11} {:>9} {:>10}",
        "workload", "stack", "register", "speedup", "rounds won"
    );

    for row in rows {
        let s = row.stack_median();
        let r = row.register_median();

        // ns/op where the op count is known, wall-clock per *call* otherwise —
        // never the raw sample total, which is `reps` calls and would not be
        // comparable with the ns/op rows above it.
        let (s_str, r_str) = match (row.ns_per_op(s), row.ns_per_op(r)) {
            (Some(sn), Some(rn)) => (format!("{sn:.2} ns"), format!("{rn:.2} ns")),
            _ => {
                let n = row.reps.max(1) as u32;
                (fmt_dur(s / n), fmt_dur(r / n))
            }
        };

        println!(
            "  {:<30} {:>11} {:>11} {:>8.2}x {:>7}/{}",
            row.label,
            s_str,
            r_str,
            row.speedup(),
            row.rounds_won(),
            ROUNDS
        );
    }

    println!();
    println!("  speedup > 1.00x means the register machine is faster.");
    println!("  read `rounds won` with the ratio: {ROUNDS}/{ROUNDS} is a result, a bare majority is noise.");
}

fn fmt_dur(d: Duration) -> String {
    if d.as_micros() >= 1000 {
        format!("{:.2} ms", d.as_secs_f64() * 1e3)
    } else if d.as_nanos() >= 1000 {
        format!("{:.2} us", d.as_secs_f64() * 1e6)
    } else {
        format!("{} ns", d.as_nanos())
    }
}

/// Asserts both machines compute the same answer, before anything is timed.
///
/// `differential.rs` already checks both machines against native Rust across the
/// whole guest suite; this is the narrower, local version — if these two ever
/// disagree on the exact export about to be timed, the ratio below is comparing
/// two different computations and is meaningless.
fn assert_agree<T: PartialEq + std::fmt::Debug>(label: &str, stack: T, register: T) {
    assert_eq!(
        stack, register,
        "`{label}`: the two machines disagree, so a timing comparison between \
         them would be meaningless"
    );
}

// ---------------------------------------------------------------------------
// Throughput
// ---------------------------------------------------------------------------

/// Register-vs-stack throughput across the four interpreter cost centres, plus
/// the allocation-heavy guest.
///
/// The categories are the same four `tests/metrics.rs` uses, and for the same
/// reason: a change that helps arithmetic but not calls is invisible in a single
/// blended benchmark. The register machine's wins are expected to be uneven
/// across them — its whole advantage is removing operand traffic, which is most
/// of what arithmetic does and least of what a call does.
#[test]
fn report_register_vs_stack_throughput() {
    let mut rows = Vec::new();

    // --- arithmetic -------------------------------------------------------
    {
        let mut s = Guest::<Stack>::new(guests::ARITHMETIC);
        let mut r = Guest::<Register>::new(guests::ARITHMETIC);

        assert_agree(
            "arith_mixed_workload",
            s.i32_i64("arith_mixed_workload", WORK),
            r.i32_i64("arith_mixed_workload", WORK),
        );

        rows.push(compare_with(
            "arithmetic (i64 + f64 mix)",
            REPS,
            Some(WORK as u64 * REPS),
            || {
                s.i32_i64("arith_mixed_workload", WORK);
            },
            || {
                r.i32_i64("arith_mixed_workload", WORK);
            },
        ));
    }

    // --- control flow -----------------------------------------------------
    {
        let mut s = Guest::<Stack>::new(guests::CONTROL_FLOW);
        let mut r = Guest::<Register>::new(guests::CONTROL_FLOW);

        assert_agree(
            "cf_mixed_workload",
            s.i32_i64("cf_mixed_workload", WORK),
            r.i32_i64("cf_mixed_workload", WORK),
        );

        rows.push(compare_with(
            "control flow (match + loops)",
            REPS,
            Some(WORK as u64 * REPS),
            || {
                s.i32_i64("cf_mixed_workload", WORK);
            },
            || {
                r.i32_i64("cf_mixed_workload", WORK);
            },
        ));
    }

    // --- memory -----------------------------------------------------------
    {
        let mut s = Guest::<Stack>::new(guests::MEMORY);
        let mut r = Guest::<Register>::new(guests::MEMORY);

        assert_agree(
            "mem_mixed_workload",
            s.i32_i64("mem_mixed_workload", WORK),
            r.i32_i64("mem_mixed_workload", WORK),
        );

        rows.push(compare_with(
            "memory (load/store)",
            REPS,
            Some(WORK as u64 * REPS),
            || {
                s.i32_i64("mem_mixed_workload", WORK);
            },
            || {
                r.i32_i64("mem_mixed_workload", WORK);
            },
        ));
    }

    // --- calls ------------------------------------------------------------
    {
        let mut s = Guest::<Stack>::new(guests::FRAMES);
        let mut r = Guest::<Register>::new(guests::FRAMES);

        assert_agree(
            "fr_mixed_workload",
            s.i32_i64("fr_mixed_workload", WORK),
            r.i32_i64("fr_mixed_workload", WORK),
        );

        rows.push(compare_with(
            "calls (one indirect per iter)",
            REPS,
            Some(WORK as u64 * REPS),
            || {
                s.i32_i64("fr_mixed_workload", WORK);
            },
            || {
                r.i32_i64("fr_mixed_workload", WORK);
            },
        ));
    }

    // --- heap -------------------------------------------------------------
    //
    // No ops_per_sample: one call is milliseconds of allocator work, and a
    // per-op figure over a guest loop that allocates, grows and frees would
    // imply a uniformity the workload does not have.
    {
        let mut s = Guest::<Stack>::new(guests::HEAP);
        let mut r = Guest::<Register>::new(guests::HEAP);

        assert_agree(
            "heap_mixed_workload",
            s.i32_i64("heap_mixed_workload", HEAP_WORK),
            r.i32_i64("heap_mixed_workload", HEAP_WORK),
        );

        rows.push(compare_with(
            "heap (alloc + collections)",
            REPS,
            None,
            || {
                s.i32_i64("heap_mixed_workload", HEAP_WORK);
            },
            || {
                r.i32_i64("heap_mixed_workload", HEAP_WORK);
            },
        ));
    }

    report("register vs stack: throughput by category", &rows);

    for row in &rows {
        assert!(
            row.stack_median().as_nanos() > 0 && row.register_median().as_nanos() > 0,
            "row `{}` measured zero time, so it is not measuring anything",
            row.label
        );
    }
}

// ---------------------------------------------------------------------------
// Per-call overhead
// ---------------------------------------------------------------------------

/// Host-boundary cost on both machines.
///
/// Separated from throughput because it moves independently: this is frame
/// setup, parameter marshalling and teardown, none of which the register
/// machine's operand elision touches. A near-1.00x here alongside a clear win
/// above is the expected shape, not a failure.
#[test]
fn report_register_vs_stack_call_overhead() {
    let mut rows = Vec::new();

    {
        let mut s = Guest::<Stack>::new(guests::ARITHMETIC);
        let mut r = Guest::<Register>::new(guests::ARITHMETIC);

        assert_agree(
            "arith_mixed_workload(1)",
            s.i32_i64("arith_mixed_workload", 1),
            r.i32_i64("arith_mixed_workload", 1),
        );

        rows.push(compare_with(
            "trivial call, n=1",
            CALL_REPS,
            Some(CALL_REPS),
            || {
                s.i32_i64("arith_mixed_workload", 1);
            },
            || {
                r.i32_i64("arith_mixed_workload", 1);
            },
        ));
    }

    {
        let mut s = Guest::<Stack>::new(guests::FRAMES);
        let mut r = Guest::<Register>::new(guests::FRAMES);

        assert_agree(
            "fr_five_params",
            s.i32x5_i64("fr_five_params", (1, 2, 3, 4, 5)),
            r.i32x5_i64("fr_five_params", (1, 2, 3, 4, 5)),
        );

        rows.push(compare_with(
            "5-param call",
            CALL_REPS,
            Some(CALL_REPS),
            || {
                s.i32x5_i64("fr_five_params", (1, 2, 3, 4, 5));
            },
            || {
                r.i32x5_i64("fr_five_params", (1, 2, 3, 4, 5));
            },
        ));
    }

    report("register vs stack: host-boundary overhead", &rows);

    for row in &rows {
        assert!(
            row.stack_median().as_nanos() > 0 && row.register_median().as_nanos() > 0,
            "row `{}` measured zero time, so it is not measuring anything",
            row.label
        );
    }
}

// ---------------------------------------------------------------------------
// Footprint
// ---------------------------------------------------------------------------

/// Compiled footprint of the two machines on the same modules.
///
/// Unlike everything above this is exact and deterministic — no timing, no host
/// variance, identical on every machine — so it *is* asserted rather than merely
/// reported. `RegInstruction` is 8 bytes against `StackInstruction`'s 16, and the
/// register lowering also emits fewer instructions per body because `local.get`
/// and `i32.const` become operands rather than instructions.
#[test]
fn register_lowering_is_smaller() {
    let stack_instr = tracewasm_core::module::Module::<Stack>::instruction_size();
    let reg_instr = tracewasm_core::module::Module::<Register>::instruction_size();

    println!();
    println!("  compiled footprint");
    println!("  {}", "-".repeat(78));
    println!(
        "  {:<30} {:>11} {:>11} {:>9}",
        "measurement", "stack", "register", "ratio"
    );
    println!(
        "  {:<30} {:>9} B {:>9} B {:>8.2}x",
        "instruction size",
        stack_instr,
        reg_instr,
        stack_instr as f64 / reg_instr as f64
    );

    let all: [(&str, &[u8]); 6] = [
        ("arithmetic", guests::ARITHMETIC),
        ("control_flow", guests::CONTROL_FLOW),
        ("heap", guests::HEAP),
        ("memory", guests::MEMORY),
        ("frames", guests::FRAMES),
        ("exotic", guests::EXOTIC),
    ];

    let mut total_stack_bytes = 0usize;
    let mut total_reg_bytes = 0usize;

    for (name, wasm) in all {
        let s = tracewasm_core::module::Module::<Stack>::compile(wasm).expect("compiles");
        let r = tracewasm_core::module::Module::<Register>::compile(wasm).expect("compiles");

        let s_count = s.instruction_count();
        let r_count = r.instruction_count();

        let s_bytes = s_count * stack_instr;
        let r_bytes = r_count * reg_instr;

        total_stack_bytes += s_bytes;
        total_reg_bytes += r_bytes;

        println!(
            "  {:<30} {:>9} KB {:>8} KB {:>8.2}x",
            format!("{name} (instr stream)"),
            s_bytes / 1024,
            r_bytes / 1024,
            s_bytes as f64 / r_bytes as f64
        );
    }

    println!(
        "  {:<30} {:>9} KB {:>8} KB {:>8.2}x",
        "total",
        total_stack_bytes / 1024,
        total_reg_bytes / 1024,
        total_stack_bytes as f64 / total_reg_bytes as f64
    );

    // Exact and host-independent, so this one is a real assertion.
    assert!(
        reg_instr < stack_instr,
        "register instruction ({reg_instr} B) should be smaller than stack ({stack_instr} B)"
    );
    assert!(
        total_reg_bytes < total_stack_bytes,
        "register instruction streams ({total_reg_bytes} B) should be smaller \
         than stack ({total_stack_bytes} B) across the guest suite"
    );
}
