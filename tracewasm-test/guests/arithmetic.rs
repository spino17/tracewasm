//! Integer and float arithmetic, aimed at the operations where a plausible
//! implementation diverges from the spec.
//!
//! Compiled to wasm by `build.rs` and `include!`d natively by
//! `tests/differential.rs`; every export is checked against rustc's own result.
//!
//! Export names are prefixed `arith_` because the differential test includes all
//! guests into one binary and `#[unsafe(no_mangle)]` symbols must not collide.
//!
//! No transcendentals here (`sin`, `exp`, `powf`, …): those come from the host
//! libm natively and a compiled-in libm under wasm, and can differ in the last
//! bit. `sqrt` is exact per IEEE 754 and is fair game.

#![allow(dead_code)]
// `x == x`, `x != x`, `inf - inf`, `0.0 / 0.0`: these are the point. NaN makes
// every ordered comparison false and `!=` true, and the degenerate float
// identities are exactly where a plausible interpreter diverges from the spec.
// Clippy reads them as mistakes; rewriting them would delete the test.
#![allow(clippy::eq_op, clippy::zero_divided_by_zero)]
// A redundant-looking `as i64` keeps a table of cases visually parallel.
#![allow(clippy::unnecessary_cast)]

use std::hint::black_box;

// ---------------------------------------------------------------------------
// Integer edge cases
// ---------------------------------------------------------------------------

/// `i32::MIN / -1` overflows and must trap; `i32::MIN % -1` is defined as 0 and
/// must *not*. Returns a bitfield so one call covers several cases.
#[unsafe(no_mangle)]
pub extern "C" fn arith_div_rem_edges(_: i32) -> i64 {
    let mut acc = 0i64;

    // wrapping_div avoids the trap so we can observe the wrapped value
    acc = acc.wrapping_add(i32::MIN.wrapping_div(-1) as i64);
    // the spec-defined non-trapping case
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(i32::MIN.wrapping_rem(-1) as i64);
    acc = acc.wrapping_mul(31).wrapping_add(i64::MIN.wrapping_rem(-1));
    // signed remainder takes the sign of the dividend
    acc = acc.wrapping_mul(31).wrapping_add((-7i32 % 3) as i64);
    acc = acc.wrapping_mul(31).wrapping_add((7i32 % -3) as i64);
    acc = acc.wrapping_mul(31).wrapping_add((-7i64 / 2) as i64);

    acc
}

/// Shifts and rotates take their count modulo the operand width, so a count of
/// 32/64 or more wraps rather than saturating or trapping.
#[unsafe(no_mangle)]
pub extern "C" fn arith_shift_rotate(n: i32) -> i64 {
    let x = black_box(0x0123_4567_89ab_cdefu64);
    let mut acc = 0u64;

    for i in 0..(n.clamp(1, 2_000) as u32) {
        let k = i & 63;

        acc = acc
            .wrapping_add(x << k)
            .wrapping_add(x >> k)
            .wrapping_add(x.rotate_left(k))
            .wrapping_add(x.rotate_right(k))
            .wrapping_add(((x as i64) >> k) as u64);
    }

    acc as i64
}

/// `clz`/`ctz`/`popcnt` of zero return the full operand width, and `popcnt`
/// counts a negative operand's sign bits too.
#[unsafe(no_mangle)]
pub extern "C" fn arith_bit_counting(_: i32) -> i64 {
    let mut acc = 0i64;

    for v in [0u32, 1, 0x8000_0000, u32::MAX, 0x0f0f_0f0f] {
        let v = black_box(v);

        acc = acc.wrapping_mul(37).wrapping_add(v.leading_zeros() as i64);
        acc = acc.wrapping_mul(37).wrapping_add(v.trailing_zeros() as i64);
        acc = acc.wrapping_mul(37).wrapping_add(v.count_ones() as i64);
    }

    for v in [0u64, 1, 0x8000_0000_0000_0000, u64::MAX] {
        let v = black_box(v);

        acc = acc.wrapping_mul(37).wrapping_add(v.leading_zeros() as i64);
        acc = acc.wrapping_mul(37).wrapping_add(v.trailing_zeros() as i64);
        acc = acc.wrapping_mul(37).wrapping_add(v.count_ones() as i64);
    }

    acc
}

/// Sign- vs zero-extension: `-1i32` widens to `-1i64` signed but `4294967295`
/// unsigned, and the `extendN_s` family reinterprets the low bits.
#[unsafe(no_mangle)]
pub extern "C" fn arith_extend_wrap(_: i32) -> i64 {
    let mut acc = 0i64;

    let neg = black_box(-1i32);

    acc = acc.wrapping_add(neg as i64);
    acc = acc.wrapping_mul(31).wrapping_add((neg as u32) as i64);
    // i32.wrap_i64 discards the high bits rather than saturating
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(black_box(0x1_0000_0000i64) as i32 as i64);
    // extend8_s / extend16_s
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(black_box(0xffu32) as u8 as i8 as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(black_box(0xffffu32) as u16 as i16 as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(black_box(0xffff_ffffu64) as u32 as i32 as i64);

    acc
}

/// Unsigned and signed comparison disagree exactly when an operand's high bit is
/// set. Every comparison yields an `i32` regardless of operand width.
#[unsafe(no_mangle)]
pub extern "C" fn arith_comparisons(_: i32) -> i64 {
    let mut acc = 0i64;
    let pairs32: [(i32, i32); 5] = [
        (-1, 1),
        (i32::MIN, 1),
        (0, 0),
        (i32::MAX, i32::MIN),
        (-5, -6),
    ];

    for (a, b) in pairs32 {
        let (a, b) = (black_box(a), black_box(b));
        for bit in [
            (a < b) as i64,
            (a > b) as i64,
            (a <= b) as i64,
            (a >= b) as i64,
            (a == b) as i64,
            ((a as u32) < (b as u32)) as i64,
            ((a as u32) > (b as u32)) as i64,
        ] {
            acc = acc.wrapping_mul(3).wrapping_add(bit);
        }
    }

    let pairs64: [(i64, i64); 3] = [(-1, 1), (i64::MIN, 1), (i64::MAX, -1)];

    for (a, b) in pairs64 {
        let (a, b) = (black_box(a), black_box(b));
        for bit in [
            (a < b) as i64,
            ((a as u64) < (b as u64)) as i64,
            (a == b) as i64,
        ] {
            acc = acc.wrapping_mul(3).wrapping_add(bit);
        }
    }

    acc
}

/// Overflow behaviour of the checked/wrapping/saturating families. In release
/// builds `+` wraps silently, so these are the explicit forms.
#[unsafe(no_mangle)]
pub extern "C" fn arith_overflow_families(_: i32) -> i64 {
    let mut acc = 0i64;
    let big = black_box(i32::MAX);

    acc = acc.wrapping_add(big.wrapping_add(1) as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(big.saturating_add(1) as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(big.checked_add(1).unwrap_or(-1) as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(big.overflowing_add(1).1 as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(i32::MIN.wrapping_abs() as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(black_box(-128i8).wrapping_neg() as i64);

    acc
}

// ---------------------------------------------------------------------------
// Float edge cases
// ---------------------------------------------------------------------------

/// `min`/`max` must return NaN if *either* operand is NaN, and must break the
/// `-0.0`/`+0.0` tie (which compare equal) toward `-0.0` and `+0.0` respectively.
/// Rust's `f64::min`/`max` do *not* do the first of those, so the guest uses the
/// same operations the interpreter has to implement.
#[unsafe(no_mangle)]
pub extern "C" fn arith_float_minmax(_: i32) -> f64 {
    let mut acc = 0.0f64;

    for (a, b) in [
        (-0.0f64, 0.0f64),
        (0.0, -0.0),
        (1.0, 2.0),
        (f64::INFINITY, f64::MAX),
        (f64::NEG_INFINITY, f64::MIN),
    ] {
        let (a, b) = (black_box(a), black_box(b));

        acc = acc * 3.0 + a.min(b);
        acc = acc * 3.0 + a.max(b);
        // observe the signed-zero tie through its bit pattern, since -0.0 == 0.0
        acc = acc * 3.0 + (a.min(b).is_sign_negative() as i32 as f64);
        acc = acc * 3.0 + (a.max(b).is_sign_negative() as i32 as f64);
    }

    acc
}

/// Rounding: `round` breaks ties away from zero, `round_ties_even` toward even,
/// and `trunc`/`floor`/`ceil` differ on negatives. Signed zero must survive.
#[unsafe(no_mangle)]
pub extern "C" fn arith_float_rounding(_: i32) -> f64 {
    let mut acc = 0.0f64;

    for v in [
        -2.5f64, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 3.5, 1e16, -1e16,
    ] {
        let v = black_box(v);

        acc = acc * 3.0 + v.trunc();
        acc = acc * 3.0 + v.floor();
        acc = acc * 3.0 + v.ceil();
        acc = acc * 3.0 + v.round();
        acc = acc * 3.0 + v.round_ties_even();
        acc = acc * 3.0 + v.abs();
        acc = acc * 3.0 + (-v);
    }

    acc
}

/// NaN and infinity propagation, and the sign-bit operations that are defined for
/// NaN. Compared through `to_bits` by the test, so payloads matter.
#[unsafe(no_mangle)]
pub extern "C" fn arith_float_specials(_: i32) -> i64 {
    let mut acc = 0i64;
    let nan = black_box(f64::NAN);
    let inf = black_box(f64::INFINITY);

    // NaN makes every ordered comparison false and `!=` true
    acc = acc.wrapping_mul(3).wrapping_add((nan < 1.0) as i64);
    acc = acc.wrapping_mul(3).wrapping_add((nan > 1.0) as i64);
    acc = acc.wrapping_mul(3).wrapping_add((nan == nan) as i64);
    acc = acc.wrapping_mul(3).wrapping_add((nan != nan) as i64);
    // sign-bit ops are defined for NaN
    acc = acc.wrapping_mul(3).wrapping_add(nan.abs().is_nan() as i64);
    acc = acc
        .wrapping_mul(3)
        .wrapping_add((-nan).is_sign_negative() as i64);
    acc = acc
        .wrapping_mul(3)
        .wrapping_add(1.0f64.copysign(-0.0).is_sign_negative() as i64);
    // infinity arithmetic
    acc = acc
        .wrapping_mul(3)
        .wrapping_add((inf - inf).is_nan() as i64);
    acc = acc
        .wrapping_mul(3)
        .wrapping_add((1.0 / 0.0f64 == inf) as i64);
    acc = acc
        .wrapping_mul(3)
        .wrapping_add((0.0f64 / 0.0).is_nan() as i64);
    acc = acc
        .wrapping_mul(3)
        .wrapping_add((-1.0f64).sqrt().is_nan() as i64);
    acc = acc
        .wrapping_mul(3)
        .wrapping_add((-0.0f64).sqrt().is_sign_negative() as i64);

    acc
}

/// Float→int conversion. The saturating (`as`) form clamps and maps NaN to 0;
/// the trapping form is exercised separately by `tests/traps.rs`.
#[unsafe(no_mangle)]
pub extern "C" fn arith_float_to_int_saturating(_: i32) -> i64 {
    let mut acc = 0i64;

    for v in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e300,
        -1e300,
        -0.9,
        0.9,
        2147483648.0,
        -2147483649.0,
    ] {
        let v = black_box(v);

        acc = acc.wrapping_mul(7).wrapping_add(v as i32 as i64);
        acc = acc.wrapping_mul(7).wrapping_add(v as u32 as i64);
        acc = acc.wrapping_mul(7).wrapping_add(v as i64);
        acc = acc.wrapping_mul(7).wrapping_add(v as u64 as i64);
    }

    acc
}

/// Int→float conversion is lossy past the significand width, and the signed and
/// unsigned readings of the same bits differ.
#[unsafe(no_mangle)]
pub extern "C" fn arith_int_to_float(_: i32) -> f64 {
    let mut acc = 0.0f64;

    for v in [0i64, 1, -1, i64::MAX, i64::MIN, 1 << 53, (1 << 53) + 1] {
        let v = black_box(v);

        acc = acc * 3.0 + v as f64;
        acc = acc * 3.0 + (v as u64) as f64;
        acc = acc * 3.0 + (v as f32) as f64;
    }

    for v in [0i32, -1, i32::MAX, i32::MIN, 1 << 24, (1 << 24) + 1] {
        let v = black_box(v);

        acc = acc * 3.0 + v as f64;
        acc = acc * 3.0 + (v as u32) as f64;
        acc = acc * 3.0 + (v as f32) as f64;
    }

    acc
}

/// `f32`/`f64` reinterpretation is a pure bit move, so NaN payloads survive and
/// round-tripping either way is the identity.
#[unsafe(no_mangle)]
pub extern "C" fn arith_reinterpret(_: i32) -> i64 {
    let mut acc = 0i64;

    for bits in [0u32, 0x7f80_0000, 0x7fc0_0000, 0xffff_ffff, 0x3fc0_0000] {
        let bits = black_box(bits);
        let f = f32::from_bits(bits);

        acc = acc.wrapping_mul(31).wrapping_add(f.to_bits() as i64);
        acc = acc
            .wrapping_mul(31)
            .wrapping_add(f64::from(f).to_bits() as i64);
    }

    for bits in [0u64, 0x7ff0_0000_0000_0000, 0x7ff8_0000_0000_0000, u64::MAX] {
        let bits = black_box(bits);
        let d = f64::from_bits(bits);

        acc = acc.wrapping_mul(31).wrapping_add(d.to_bits() as i64);
        acc = acc
            .wrapping_mul(31)
            .wrapping_add((d as f32).to_bits() as i64);
    }

    acc
}

/// `f64`→`f32` demotion overflows to infinity rather than clamping, and the
/// threshold is the halfway point to 2^128 rather than `f32::MAX` itself.
#[unsafe(no_mangle)]
pub extern "C" fn arith_demote_promote(_: i32) -> i64 {
    let mut acc = 0i64;

    for v in [
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.0000001,
        1e39,
        -1e39,
        1e-46,
        -1e-46,
        f64::from(0.1f32),
    ] {
        let v = black_box(v);
        let narrowed = v as f32;

        acc = acc.wrapping_mul(31).wrapping_add(narrowed.to_bits() as i64);
        // promotion back is exact, so this round-trips
        acc = acc
            .wrapping_mul(31)
            .wrapping_add(f64::from(narrowed).to_bits() as i64);
    }

    acc
}

/// A mixed arithmetic workload, for the throughput row in `tests/metrics.rs`.
/// `n` scales the iteration count so the metric test can pick a duration.
#[unsafe(no_mangle)]
pub extern "C" fn arith_mixed_workload(n: i32) -> i64 {
    let mut a = 1i64;
    let mut b = 2u64;
    let mut f = 1.5f64;
    let mut i = 0;

    while i < n {
        a = a.wrapping_mul(31).wrapping_add(i as i64) ^ (a >> 7);
        b = b.rotate_left((i & 63) as u32) ^ (b >> 3);
        f = (f * 1.0000001 + 0.5).sqrt() + 1.0;
        i += 1;
    }

    a ^ (b as i64) ^ (f.to_bits() as i64)
}
