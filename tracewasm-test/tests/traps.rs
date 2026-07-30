//! Trap behaviour: the failure paths must produce a *catchable error*, not a
//! panic and not a process abort.
//!
//! This distinction is the whole point of the file. An embedder that runs
//! untrusted guest code has to be able to survive that code misbehaving, so
//! "returns `Err`" is a hard requirement, separate from whether the message is
//! right. A trap that aborts the host is a security problem, not a cosmetic one.
//!
//! The deliberate-trap guest exports live at the bottom of `guests/memory.rs` and
//! `guests/frames.rs`.

#![cfg(not(no_guest_wasm))]

use tracewasm_core::instance::config::Config;
use tracewasm_test::{Guest, MAX_TEST_RECURSION, guests, with_large_stack};

/// Every trapping export returns `Err` rather than panicking or aborting, and the
/// rendered message mentions something recognisable.
///
/// The message assertions are deliberately loose — they pin the *category* of
/// failure, not the exact prose, so improving an error message does not break the
/// suite. `tests/traps.rs` is not the place to freeze wording.
/// Note on what these actually exercise: **every** one of them arrives as
/// `unreachable`, including the two division cases.
///
/// rustc emits its own zero and overflow checks ahead of an integer division and
/// panics on failure, so `a / b` in safe Rust never reaches a raw wasm
/// `i64.div_s` trap. The interpreter's division-trap path is therefore
/// unreachable from compiled Rust and is covered instead by the hand-written
/// fixtures in `tests/numeric.rs`, which can emit the bare instruction.
///
/// That split is the reason both kinds of test are worth keeping: guests cover
/// what rustc really emits, fixtures reach the instructions rustc guards away.
#[test]
fn guest_faults_become_catchable_errors() {
    let mut g = Guest::new(guests::MEMORY);

    let cases: &[(&str, &str)] = &[
        // a Rust bounds-check failure panics, which lowers to `unreachable`
        ("mem_trap_oob_read", "unreachable"),
        ("mem_trap_explicit_panic", "unreachable"),
        // rustc's own divide-by-zero and overflow checks panic before the wasm
        // division instruction is reached, so these are `unreachable` too
        ("mem_trap_div_zero", "unreachable"),
        ("mem_trap_div_overflow", "unreachable"),
    ];

    for (name, expect) in cases {
        let result = g.try_i32_i64(name, 0);

        let err = match result {
            Err(e) => e,
            Ok(v) => panic!("`{name}` returned {v} instead of trapping"),
        };

        let rendered = err.to_string();

        assert!(
            rendered.contains(expect),
            "`{name}` trapped, but the message does not mention `{expect}`:\n{rendered}"
        );
    }
}

/// A trap carries a backtrace of the wasm frames that led to it.
///
/// This is the feature the whole engine exists for, so it gets an explicit test
/// rather than being implied by the others. A panic deep inside `std` (formatting,
/// allocation, bounds checks) should surface with the intervening frames intact.
#[test]
fn a_trap_reports_the_wasm_call_chain() {
    let mut g = Guest::new(guests::MEMORY);

    let err = g
        .try_i32_i64("mem_trap_oob_read", 0)
        .expect_err("indexing past the end of a Vec should trap");

    let rendered = err.to_string();

    // the panic happens several frames below the export, inside the slice
    // bounds-check machinery, so the chain should have real depth
    let frames = rendered.matches("in func(").count();

    assert!(
        frames >= 2,
        "expected a multi-frame backtrace, got {frames} frame(s):\n{rendered}"
    );

    assert!(
        rendered.contains("mem_trap_oob_read"),
        "the backtrace should name the entry point:\n{rendered}"
    );
}

/// Unbounded recursion must trap via the depth guard rather than overflowing the
/// host's native stack.
///
/// Each active wasm frame costs a native Rust frame, so without a ceiling this
/// aborts the process — uncatchable, and fatal to the embedder. The guard turning
/// that into an ordinary error is the single most important safety property here,
/// which is why this test exists even though it is slow.
#[test]
fn runaway_recursion_traps_instead_of_aborting_the_process() {
    with_large_stack(|| {
        let mut cfg = Config::default();

        // deliberately low: the point is to prove the guard fires, and a small limit
        // keeps the test fast. The default is higher.
        cfg.set_max_call_stack_depth(512);

        let mut g = Guest::with_config(guests::FRAMES, Some(cfg));

        let err = g
            .try_i32_i64("fr_trap_infinite_recursion", 0)
            .expect_err("infinite recursion should hit the call-depth guard");

        let rendered = err.to_string();

        assert!(
            rendered.contains("call stack exhausted"),
            "expected a call-depth trap, got:\n{}",
            // the message nests one layer per frame, so truncate it for readability
            &rendered[..rendered.len().min(400)]
        );
    });
}

/// The depth guard counts *depth*, not cumulative calls.
///
/// A guard implemented as a bare counter that is never decremented passes a
/// recursion test and then breaks every real program, because a shallow function
/// called a few thousand times trips the limit. This is a regression test for
/// exactly that bug.
#[test]
fn the_depth_guard_counts_depth_not_total_calls() {
    with_large_stack(|| {
        let mut cfg = Config::default();
        cfg.set_max_call_stack_depth(64);

        let mut g = Guest::with_config(guests::FRAMES, Some(cfg));

        // `fr_call_chain` nests only ~5 deep but makes thousands of calls in total,
        // so it must succeed under a limit of 64
        let deep_but_narrow = g.try_i32_i64("fr_call_chain", 2_000);

        assert!(
            deep_but_narrow.is_ok(),
            "a shallow function called many times must not trip a depth limit of 64: {:?}",
            deep_but_narrow.err().map(|e| e.to_string())
        );

        // and recursion just under the limit still works
        assert!(
            g.try_i32_i64("fr_recurse_depth", 50).is_ok(),
            "recursion to depth 50 must succeed under a limit of 64"
        );
    });
}

/// Raising the configured limit raises the depth actually reachable, so the guard
/// is genuinely driven by configuration rather than a hardcoded constant.
#[test]
fn the_depth_limit_is_configurable() {
    with_large_stack(|| {
        // capped for debug builds, where each frame costs ~30 KB of native stack
        let depth = 300.min(MAX_TEST_RECURSION);

        let mut low = Config::default();
        low.set_max_call_stack_depth((depth as u32) / 3);

        let mut high = Config::default();
        high.set_max_call_stack_depth(4_000);

        let mut g_low = Guest::with_config(guests::FRAMES, Some(low));
        let mut g_high = Guest::with_config(guests::FRAMES, Some(high));

        assert!(
            g_low.try_i32_i64("fr_recurse_depth", depth).is_err(),
            "depth {depth} should exceed a limit of 100"
        );

        assert!(
            g_high.try_i32_i64("fr_recurse_depth", depth).is_ok(),
            "depth {depth} should fit within a limit of 4000"
        );
    });
}

/// An instance survives a trap: the error is not poison, and the same instance can
/// be called again afterwards.
///
/// Without this, an embedder would have to rebuild the world after every guest
/// fault. It also catches a trap path that leaves the operand stack or the depth
/// counter in a corrupt state, which would show up as the *second* call failing.
#[test]
fn an_instance_is_reusable_after_a_trap() {
    let mut g = Guest::new(guests::MEMORY);

    // a healthy call first, to establish a baseline
    let before = g.i32_i64("mem_endianness", 0);

    // then a trap
    g.try_i32_i64("mem_trap_div_zero", 0)
        .expect_err("division by zero should trap");

    // then the same call again: identical result, no leaked state
    let after = g.i32_i64("mem_endianness", 0);

    assert_eq!(
        before, after,
        "an instance produced different results before and after a trap, \
         which means the trap left state behind"
    );

    // and a second trap still traps rather than doing something worse
    g.try_i32_i64("mem_trap_oob_read", 0)
        .expect_err("the second trap should behave like the first");

    let after_two = g.i32_i64("mem_endianness", 0);

    assert_eq!(before, after_two, "state leaked across two traps");
}
