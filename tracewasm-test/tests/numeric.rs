//! Numeric instruction coverage, ported from `tracewasm-core/tests/comparisons.rs`.
//!
//! Each case targets a place where a plausible-looking implementation diverges
//! from the spec: signed vs unsigned ordering, comparisons always yielding an
//! `i32` regardless of operand width, IEEE 754 corners (NaN, signed zero,
//! ties-to-even), and the conversions — where the sign/zero-extension split and
//! the trapping `trunc` bounds live.
//!
//! These run against small hand-built `.wasm` fixtures in `fixtures/`, which is
//! complementary to `differential.rs`: a fixture can isolate one instruction with
//! one operand pair, where a compiled Rust guest necessarily drags in a lot of
//! surrounding code. Keep both.

use tracewasm_core::{
    Register, Stack, VirtualMachine, error::InstructionExecutionError, instance::traits::Results,
};
use tracewasm_test::{call_i32, call_typed, try_call};

/// Calls a `() -> i32` export of the integer-comparison fixture.
fn call<V: VirtualMachine>(name: &str) -> i32 {
    call_i32::<V>(include_bytes!("../fixtures/comparisons.wasm"), name)
}

/// Calls a `() -> i32` export of the float-comparison fixture.
fn call_f<V: VirtualMachine>(name: &str) -> i32 {
    call_i32::<V>(include_bytes!("../fixtures/float_comparisons.wasm"), name)
}

// `-1` is `0xFFFF_FFFF` unsigned, so it is the *largest* u32 — the unsigned and
// signed forms must disagree here. Sharing one comparison for both would show up
// as these two returning the same value.
#[test]
fn unsigned_and_signed_ordering_disagree_on_negative_operands() {
    unsigned_and_signed_ordering_disagree_on_negative_operands_on::<Stack>();
    unsigned_and_signed_ordering_disagree_on_negative_operands_on::<Register>();
}

fn unsigned_and_signed_ordering_disagree_on_negative_operands_on<V: VirtualMachine>() {
    assert_eq!(call::<V>("lt_u"), 0, "-1 <u 1 is false: 0xFFFFFFFF > 1");
    assert_eq!(call::<V>("lt_s"), 1, "-1 <s 1 is true");
    assert_eq!(call::<V>("ge_u"), 1, "-1 >=u 1 is true");
}

#[test]
fn i64_comparisons_use_the_right_signedness_too() {
    i64_comparisons_use_the_right_signedness_too_on::<Stack>();
    i64_comparisons_use_the_right_signedness_too_on::<Register>();
}

fn i64_comparisons_use_the_right_signedness_too_on<V: VirtualMachine>() {
    assert_eq!(call::<V>("i64_lt_u"), 0);
    assert_eq!(call::<V>("i64_lt_s"), 1);
}

// Regression guard: an i64 comparison must push an `i32`, not an `i64`. Feeding
// one to `br_if` — which reads its condition as an `i32` — panics the interpreter
// if the wrong variant was pushed, so this fails loudly rather than subtly.
#[test]
fn i64_comparison_result_is_an_i32_usable_as_a_branch_condition() {
    i64_comparison_result_is_an_i32_usable_as_a_branch_condition_on::<Stack>();
    i64_comparison_result_is_an_i32_usable_as_a_branch_condition_on::<Register>();
}

fn i64_comparison_result_is_an_i32_usable_as_a_branch_condition_on<V: VirtualMachine>() {
    assert_eq!(call::<V>("i64_brif"), 111, "5 <s 9 should take the branch");
}

// IEEE 754 ordering, which is *not* the two's-complement ordering the integer
// comparisons use. Rust's operators already behave this way, so these guard
// against someone "fixing" them into something like `a.partial_cmp(b)` handling.
#[test]
fn nan_makes_ordered_comparisons_false_but_ne_true() {
    nan_makes_ordered_comparisons_false_but_ne_true_on::<Stack>();
    nan_makes_ordered_comparisons_false_but_ne_true_on::<Register>();
}

fn nan_makes_ordered_comparisons_false_but_ne_true_on<V: VirtualMachine>() {
    assert_eq!(call_f::<V>("nan_lt"), 0, "NaN < 1 is false");
    assert_eq!(call_f::<V>("nan_ge"), 0, "NaN >= 1 is false");
    assert_eq!(call_f::<V>("nan_eq"), 0, "NaN == NaN is false");
    assert_eq!(call_f::<V>("nan_ne"), 1, "NaN != NaN is true");
    assert_eq!(call_f::<V>("f64_nan_le"), 0, "and the same on the f64 path");
}

#[test]
fn negative_and_positive_zero_compare_equal() {
    negative_and_positive_zero_compare_equal_on::<Stack>();
    negative_and_positive_zero_compare_equal_on::<Register>();
}

fn negative_and_positive_zero_compare_equal_on<V: VirtualMachine>() {
    assert_eq!(call_f::<V>("negzero_eq"), 1, "-0.0 == +0.0");
    assert_eq!(call_f::<V>("negzero_lt"), 0, "-0.0 is not less than +0.0");
}

#[test]
fn infinities_order_normally() {
    infinities_order_normally_on::<Stack>();
    infinities_order_normally_on::<Register>();
}

fn infinities_order_normally_on<V: VirtualMachine>() {
    assert_eq!(call_f::<V>("f64_inf_gt"), 1);
}

// As for the i64 comparisons: the result must be an `i32`, or `br_if` panics.
#[test]
fn float_comparison_result_is_an_i32_usable_as_a_branch_condition() {
    float_comparison_result_is_an_i32_usable_as_a_branch_condition_on::<Stack>();
    float_comparison_result_is_an_i32_usable_as_a_branch_condition_on::<Register>();
}

fn float_comparison_result_is_an_i32_usable_as_a_branch_condition_on<V: VirtualMachine>() {
    assert_eq!(call_f::<V>("f64_brif"), 111, "1 < 2 should take the branch");
}

/// `f32` result of the min/max/copysign fixture.
fn min_max_f32<V: VirtualMachine>(name: &str) -> f32 {
    let (v,) = call_typed::<V, (f32,)>(include_bytes!("../fixtures/float_minmax.wasm"), name);
    v
}

/// `f64` result of the min/max/copysign fixture.
fn min_max_f64<V: VirtualMachine>(name: &str) -> f64 {
    let (v,) = call_typed::<V, (f64,)>(include_bytes!("../fixtures/float_minmax.wasm"), name);
    v
}

// The cases where Rust's `f32::min`/`max` disagree with wasm: they return the
// *non*-NaN operand, and leave the signed-zero tie unspecified. Delegating to
// them would pass every ordinary test and fail only these.
#[test]
fn min_max_return_nan_when_either_operand_is_nan() {
    min_max_return_nan_when_either_operand_is_nan_on::<Stack>();
    min_max_return_nan_when_either_operand_is_nan_on::<Register>();
}

fn min_max_return_nan_when_either_operand_is_nan_on<V: VirtualMachine>() {
    assert!(
        min_max_f32::<V>("min_nan").is_nan(),
        "min(NaN, 1) must be NaN"
    );
    assert!(
        min_max_f32::<V>("max_nan").is_nan(),
        "max(1, NaN) must be NaN"
    );
    assert!(
        min_max_f64::<V>("f64_min_nan").is_nan(),
        "and on the f64 path"
    );
}

// `-0.0 == +0.0`, so the tie is broken by sign: min yields -0.0, max +0.0.
// Checked via `is_sign_negative`, since `==` cannot tell the two zeros apart.
#[test]
fn min_max_break_the_signed_zero_tie_by_sign() {
    min_max_break_the_signed_zero_tie_by_sign_on::<Stack>();
    min_max_break_the_signed_zero_tie_by_sign_on::<Register>();
}

fn min_max_break_the_signed_zero_tie_by_sign_on<V: VirtualMachine>() {
    let min = min_max_f32::<V>("min_zero");
    assert_eq!(min, 0.0);
    assert!(min.is_sign_negative(), "min(+0.0, -0.0) must be -0.0");

    let max = min_max_f32::<V>("max_zero");
    assert_eq!(max, 0.0);
    assert!(max.is_sign_positive(), "max(-0.0, +0.0) must be +0.0");

    assert!(min_max_f64::<V>("f64_max_zero").is_sign_positive());
}

#[test]
fn min_max_order_ordinary_values() {
    min_max_order_ordinary_values_on::<Stack>();
    min_max_order_ordinary_values_on::<Register>();
}

fn min_max_order_ordinary_values_on<V: VirtualMachine>() {
    assert_eq!(min_max_f32::<V>("min_plain"), 3.0);
    assert_eq!(min_max_f32::<V>("max_plain"), 7.0);
}

#[test]
fn copysign_takes_magnitude_from_the_first_and_sign_from_the_second() {
    copysign_takes_magnitude_from_the_first_and_sign_from_the_second_on::<Stack>();
    copysign_takes_magnitude_from_the_first_and_sign_from_the_second_on::<Register>();
}

fn copysign_takes_magnitude_from_the_first_and_sign_from_the_second_on<V: VirtualMachine>() {
    assert_eq!(min_max_f32::<V>("copysign_neg"), -5.0);
    assert_eq!(min_max_f32::<V>("copysign_pos"), 5.0);
    assert_eq!(min_max_f64::<V>("f64_copysign"), -5.0);
}

/// `i32` result of the bit-counting fixture.
fn bits_i32<V: VirtualMachine>(name: &str) -> i32 {
    call_i32::<V>(include_bytes!("../fixtures/bit_counting.wasm"), name)
}

/// `i64` result of the bit-counting fixture.
fn bits_i64<V: VirtualMachine>(name: &str) -> i64 {
    let (v,) = call_typed::<V, (i64,)>(include_bytes!("../fixtures/bit_counting.wasm"), name);
    v
}

// Zero is the edge case: `clz`/`ctz` of `0` are the full operand width, not `0`.
#[test]
fn clz_and_ctz_of_zero_are_the_operand_width() {
    clz_and_ctz_of_zero_are_the_operand_width_on::<Stack>();
    clz_and_ctz_of_zero_are_the_operand_width_on::<Register>();
}

fn clz_and_ctz_of_zero_are_the_operand_width_on<V: VirtualMachine>() {
    assert_eq!(bits_i32::<V>("clz_zero"), 32);
    assert_eq!(bits_i32::<V>("ctz_zero"), 32);
    assert_eq!(bits_i64::<V>("i64_clz_zero"), 64, "i64 width, not 32");
    assert_eq!(bits_i64::<V>("i64_ctz_zero"), 64);
}

#[test]
fn clz_ctz_count_ordinary_values() {
    clz_ctz_count_ordinary_values_on::<Stack>();
    clz_ctz_count_ordinary_values_on::<Register>();
}

fn clz_ctz_count_ordinary_values_on<V: VirtualMachine>() {
    assert_eq!(bits_i32::<V>("clz_one"), 31);
    assert_eq!(bits_i32::<V>("ctz_8"), 3);
    assert_eq!(bits_i32::<V>("clz_neg1"), 0, "-1 has the high bit set");
}

// `popcnt` works on the two's-complement bits, so `-1` is all ones.
#[test]
fn popcnt_counts_sign_bits_of_negative_operands() {
    popcnt_counts_sign_bits_of_negative_operands_on::<Stack>();
    popcnt_counts_sign_bits_of_negative_operands_on::<Register>();
}

fn popcnt_counts_sign_bits_of_negative_operands_on<V: VirtualMachine>() {
    assert_eq!(bits_i32::<V>("popcnt_neg1"), 32);
    assert_eq!(bits_i32::<V>("popcnt_5"), 2, "0b101");
    assert_eq!(bits_i64::<V>("i64_popcnt_neg1"), 64);
}

/// `i32` result of the `eqz` fixture.
fn eqz<V: VirtualMachine>(name: &str) -> i32 {
    call_i32::<V>(include_bytes!("../fixtures/eqz.wasm"), name)
}

#[test]
fn eqz_tests_against_zero() {
    eqz_tests_against_zero_on::<Stack>();
    eqz_tests_against_zero_on::<Register>();
}

fn eqz_tests_against_zero_on<V: VirtualMachine>() {
    assert_eq!(eqz::<V>("eqz_zero"), 1);
    assert_eq!(eqz::<V>("eqz_nonzero"), 0);
    assert_eq!(eqz::<V>("eqz_neg"), 0, "-1 is non-zero");
    assert_eq!(eqz::<V>("i64_eqz_zero"), 1);
    assert_eq!(eqz::<V>("i64_eqz_nonzero"), 0);
}

// `i64.eqz` returns an `i32` despite its `i64` operand — the opposite of the
// bit-counting unaries. Pushing `Val::I64` would panic `br_if`, not just differ.
#[test]
fn i64_eqz_result_is_an_i32_usable_as_a_branch_condition() {
    i64_eqz_result_is_an_i32_usable_as_a_branch_condition_on::<Stack>();
    i64_eqz_result_is_an_i32_usable_as_a_branch_condition_on::<Register>();
}

fn i64_eqz_result_is_an_i32_usable_as_a_branch_condition_on<V: VirtualMachine>() {
    assert_eq!(eqz::<V>("i64_eqz_brif"), 111);
}

/// `f32` result of the float-unary fixture.
fn unary_f32<V: VirtualMachine>(name: &str) -> f32 {
    let (v,) = call_typed::<V, (f32,)>(include_bytes!("../fixtures/float_unary.wasm"), name);
    v
}

/// `f64` result of the float-unary fixture.
fn unary_f64<V: VirtualMachine>(name: &str) -> f64 {
    let (v,) = call_typed::<V, (f64,)>(include_bytes!("../fixtures/float_unary.wasm"), name);
    v
}

// `nearest` breaks ties to **even**, unlike Rust's `round`, which goes away from
// zero. Swapping in `round()` would pass most inputs and fail exactly these.
#[test]
fn nearest_breaks_ties_to_even_not_away_from_zero() {
    nearest_breaks_ties_to_even_not_away_from_zero_on::<Stack>();
    nearest_breaks_ties_to_even_not_away_from_zero_on::<Register>();
}

fn nearest_breaks_ties_to_even_not_away_from_zero_on<V: VirtualMachine>() {
    assert_eq!(unary_f32::<V>("nearest_2_5"), 2.0, "round() would give 3.0");
    assert_eq!(unary_f32::<V>("nearest_3_5"), 4.0, "4 is already even");
    assert_eq!(
        unary_f32::<V>("nearest_neg2_5"),
        -2.0,
        "round() would give -3.0"
    );
    assert_eq!(unary_f64::<V>("f64_nearest_2_5"), 2.0);
}

// `abs`/`neg` are sign-bit operations, so they are observable on `-0.0` even
// though it compares equal to `+0.0`.
#[test]
fn abs_and_neg_act_on_the_sign_bit_of_negative_zero() {
    abs_and_neg_act_on_the_sign_bit_of_negative_zero_on::<Stack>();
    abs_and_neg_act_on_the_sign_bit_of_negative_zero_on::<Register>();
}

fn abs_and_neg_act_on_the_sign_bit_of_negative_zero_on<V: VirtualMachine>() {
    let abs = unary_f32::<V>("abs_negzero");
    assert_eq!(abs, 0.0);
    assert!(abs.is_sign_positive(), "abs(-0.0) must be +0.0");

    let neg = unary_f32::<V>("neg_negzero");
    assert_eq!(neg, 0.0);
    assert!(neg.is_sign_positive(), "-(-0.0) must be +0.0");
}

#[test]
fn ceil_floor_and_trunc_round_in_their_own_directions() {
    ceil_floor_and_trunc_round_in_their_own_directions_on::<Stack>();
    ceil_floor_and_trunc_round_in_their_own_directions_on::<Register>();
}

fn ceil_floor_and_trunc_round_in_their_own_directions_on<V: VirtualMachine>() {
    assert_eq!(unary_f32::<V>("ceil_neg"), -1.0);
    assert_eq!(unary_f32::<V>("floor_neg"), -2.0);
    assert_eq!(
        unary_f32::<V>("trunc_neg"),
        -1.0,
        "toward zero, unlike floor"
    );
}

// A negative operand yields NaN rather than trapping — float unaries are total.
#[test]
fn sqrt_of_a_negative_operand_is_nan() {
    sqrt_of_a_negative_operand_is_nan_on::<Stack>();
    sqrt_of_a_negative_operand_is_nan_on::<Register>();
}

fn sqrt_of_a_negative_operand_is_nan_on<V: VirtualMachine>() {
    assert!(unary_f32::<V>("sqrt_neg").is_nan());
    assert_eq!(unary_f32::<V>("sqrt_4"), 2.0);
    assert_eq!(unary_f64::<V>("f64_abs_neg"), 3.5);
}

/// `i32` result of the sign-extension fixture.
fn extend_i32<V: VirtualMachine>(name: &str) -> i32 {
    call_i32::<V>(include_bytes!("../fixtures/extend.wasm"), name)
}

/// `i64` result of the sign-extension fixture.
fn extend_i64<V: VirtualMachine>(name: &str) -> i64 {
    let (v,) = call_typed::<V, (i64,)>(include_bytes!("../fixtures/extend.wasm"), name);
    v
}

// The narrow sign-extensions read only the low bits of their operand, so the
// high bits have to be dropped first. An implementation that widened the operand
// as-is would pass `e8s_7f` and fail every other case here.
#[test]
fn narrow_sign_extension_discards_the_high_bits() {
    narrow_sign_extension_discards_the_high_bits_on::<Stack>();
    narrow_sign_extension_discards_the_high_bits_on::<Register>();
}

fn narrow_sign_extension_discards_the_high_bits_on<V: VirtualMachine>() {
    assert_eq!(
        extend_i32::<V>("e8s_ff"),
        -1,
        "low byte 0xFF is -1 as an i8"
    );
    assert_eq!(extend_i32::<V>("e8s_7f"), 127, "0x7F stays positive");
    assert_eq!(
        extend_i32::<V>("e8s_high"),
        -128,
        "only the low byte 0x80 is read"
    );
    assert_eq!(extend_i32::<V>("e8s_256"), 0, "0x100 has a zero low byte");
    assert_eq!(extend_i32::<V>("e16s_ffff"), -1);
    assert_eq!(extend_i32::<V>("e16s_65536"), 0);
}

#[test]
fn narrow_sign_extension_to_i64_uses_the_same_rule() {
    narrow_sign_extension_to_i64_uses_the_same_rule_on::<Stack>();
    narrow_sign_extension_to_i64_uses_the_same_rule_on::<Register>();
}

fn narrow_sign_extension_to_i64_uses_the_same_rule_on<V: VirtualMachine>() {
    assert_eq!(extend_i64::<V>("i64_e8s"), -1);
    assert_eq!(extend_i64::<V>("i64_e16s"), -1);
    assert_eq!(
        extend_i64::<V>("i64_e32s"),
        -1,
        "0x1_FFFFFFFF keeps only the low 32 bits, which are all ones"
    );
}

// The one pair that is easy to conflate: both widen an `i32` to an `i64`, but
// `_s` copies the sign bit and `_u` does not. They agree on every non-negative
// operand, so a negative one is the only thing that tells them apart.
#[test]
fn i32_to_i64_signed_and_unsigned_widening_disagree_on_negatives() {
    i32_to_i64_signed_and_unsigned_widening_disagree_on_negatives_on::<Stack>();
    i32_to_i64_signed_and_unsigned_widening_disagree_on_negatives_on::<Register>();
}

fn i32_to_i64_signed_and_unsigned_widening_disagree_on_negatives_on<V: VirtualMachine>() {
    assert_eq!(extend_i64::<V>("i32_to_i64_s_neg"), -1, "sign-extended");
    assert_eq!(
        extend_i64::<V>("i32_to_i64_u_neg"),
        4_294_967_295,
        "zero-extended: 0xFFFFFFFF, not -1"
    );
    assert_eq!(
        extend_i64::<V>("i32_to_i64_s_pos"),
        7,
        "and agree on positives"
    );
    assert_eq!(extend_i64::<V>("i32_to_i64_u_pos"), 7);
}

// `wrap_i64` is the only conversion here that *discards* bits, and it wraps
// instead of trapping on operands that do not fit. Anything that range-checked
// first would reject the first four of these.
#[test]
fn wrap_i64_keeps_the_low_32_bits_without_trapping() {
    wrap_i64_keeps_the_low_32_bits_without_trapping_on::<Stack>();
    wrap_i64_keeps_the_low_32_bits_without_trapping_on::<Register>();
}

fn wrap_i64_keeps_the_low_32_bits_without_trapping_on<V: VirtualMachine>() {
    assert_eq!(
        extend_i32::<V>("wrap_2pow32"),
        0,
        "0x1_0000_0000 has no low bits"
    );
    assert_eq!(extend_i32::<V>("wrap_neg1"), -1);
    assert_eq!(
        extend_i32::<V>("wrap_u32_max"),
        -1,
        "4294967295 does not fit an i32; its low 32 bits are all ones"
    );
    assert_eq!(extend_i32::<V>("wrap_i64_max"), -1, "i64::MAX wraps to -1");
    assert_eq!(
        extend_i32::<V>("wrap_mixed"),
        0x2345_6789,
        "high nibble dropped"
    );
    assert_eq!(
        extend_i32::<V>("wrap_small"),
        42,
        "in-range operands pass through"
    );
}

// `wrap_i64` then `extend_i32_s` is a round trip for operands that fit in an
// i32; `extend_i32_u` would not agree, which is what the pair test above pins.
#[test]
fn wrap_undoes_a_signed_widening() {
    wrap_undoes_a_signed_widening_on::<Stack>();
    wrap_undoes_a_signed_widening_on::<Register>();
}

fn wrap_undoes_a_signed_widening_on<V: VirtualMachine>() {
    assert_eq!(extend_i64::<V>("i32_to_i64_s_neg"), -1);
    assert_eq!(extend_i32::<V>("wrap_neg1"), -1);
}

/// `i32` result of the float-to-int truncation fixture.
fn trunc_i32<V: VirtualMachine>(name: &str) -> i32 {
    call_i32::<V>(include_bytes!("../fixtures/trunc.wasm"), name)
}

/// `i64` result of the float-to-int truncation fixture.
fn trunc_i64<V: VirtualMachine>(name: &str) -> i64 {
    let (v,) = call_typed::<V, (i64,)>(include_bytes!("../fixtures/trunc.wasm"), name);
    v
}

/// Asserts that `name` traps, and that the trap is a truncation failure naming
/// `target` as the type it could not reach.
///
/// `R` is the export's declared result type; the call never produces a value, but
/// it still has to be named for `get_typed_func` to resolve the signature.
fn assert_traps<V: VirtualMachine, R: Results>(name: &str, target: &str) {
    let err = match try_call::<V, R>(include_bytes!("../fixtures/trunc.wasm"), name) {
        Err(e) => e,
        Ok(_) => panic!("`{name}` should have trapped, but returned a value"),
    };

    let msg = err.to_string();
    assert!(
        msg.contains("float truncation"),
        "`{name}` trapped for the wrong reason: {msg}"
    );
    // Also pins the error's own formatting: the target type has to appear, which
    // it will not if the format string interpolates the operand twice.
    assert!(
        msg.contains(target),
        "`{name}` should name `{target}` as the target type: {msg}"
    );
}

// Truncation goes toward zero, so a negative operand rounds *up*. Using `floor`
// would give -4 for these.
#[test]
fn truncation_rounds_toward_zero() {
    truncation_rounds_toward_zero_on::<Stack>();
    truncation_rounds_toward_zero_on::<Register>();
}

fn truncation_rounds_toward_zero_on<V: VirtualMachine>() {
    assert_eq!(trunc_i32::<V>("s_pos"), 3);
    assert_eq!(trunc_i32::<V>("s_neg"), -3, "floor would give -4");
    assert_eq!(trunc_i32::<V>("f64_s_neg"), -3);
}

// `trunc_u` of an operand in (-1, 0) truncates to -0.0, which is a valid zero.
// Rejecting negatives *before* truncating would trap here instead.
#[test]
fn unsigned_truncation_accepts_operands_that_truncate_to_negative_zero() {
    unsigned_truncation_accepts_operands_that_truncate_to_negative_zero_on::<Stack>();
    unsigned_truncation_accepts_operands_that_truncate_to_negative_zero_on::<Register>();
}

fn unsigned_truncation_accepts_operands_that_truncate_to_negative_zero_on<V: VirtualMachine>() {
    assert_eq!(trunc_i32::<V>("u_neg_zero"), 0, "trunc(-0.9) is -0.0, so 0");
    assert_eq!(trunc_i32::<V>("f64_u_neg_zero"), 0);
    assert_eq!(trunc_i64::<V>("i64_u_neg_zero"), 0);
}

#[test]
fn unsigned_truncation_still_rejects_operands_that_truncate_below_zero() {
    unsigned_truncation_still_rejects_operands_that_truncate_below_zero_on::<Stack>();
    unsigned_truncation_still_rejects_operands_that_truncate_below_zero_on::<Register>();
}

fn unsigned_truncation_still_rejects_operands_that_truncate_below_zero_on<V: VirtualMachine>() {
    assert_traps::<V, (i32,)>("u_neg_one_traps", "u32");
}

// The unsigned forms yield the bit pattern, so results above the signed maximum
// read back as negative — the same convention as `i64.extend_i32_u`'s inverse.
#[test]
fn unsigned_truncation_results_are_bit_patterns() {
    unsigned_truncation_results_are_bit_patterns_on::<Stack>();
    unsigned_truncation_results_are_bit_patterns_on::<Register>();
}

fn unsigned_truncation_results_are_bit_patterns_on<V: VirtualMachine>() {
    assert_eq!(trunc_i32::<V>("u_max"), -1, "4294967295 as an i32 is -1");
    assert_eq!(trunc_i64::<V>("i64_u_big"), -8_446_744_073_709_551_616);
}

// The extremes are in range and must *not* trap. `s_min_frac_ok` is the low-side
// counterpart of the negative-zero case: -2147483648.5 truncates to exactly
// `i32::MIN`, so checking the operand before truncating would reject it.
#[test]
fn truncation_accepts_the_range_boundaries() {
    truncation_accepts_the_range_boundaries_on::<Stack>();
    truncation_accepts_the_range_boundaries_on::<Register>();
}

fn truncation_accepts_the_range_boundaries_on<V: VirtualMachine>() {
    assert_eq!(trunc_i32::<V>("s_max_ok"), i32::MAX);
    assert_eq!(trunc_i32::<V>("s_min_ok"), i32::MIN);
    assert_eq!(
        trunc_i32::<V>("s_min_frac_ok"),
        i32::MIN,
        "-2147483648.5 truncates in"
    );
    assert_eq!(trunc_i64::<V>("i64_s_min_ok"), i64::MIN);
}

// The bound has to be exclusive. `i32::MAX as f32` rounds *up* to 2^31 and
// `i64::MAX as f64` rounds up to 2^63, so an implementation comparing against a
// cast maximum would accept exactly these four values as in range.
#[test]
fn truncation_traps_on_the_power_of_two_just_past_the_range() {
    truncation_traps_on_the_power_of_two_just_past_the_range_on::<Stack>();
    truncation_traps_on_the_power_of_two_just_past_the_range_on::<Register>();
}

fn truncation_traps_on_the_power_of_two_just_past_the_range_on<V: VirtualMachine>() {
    assert_traps::<V, (i32,)>("s_2pow31_traps", "i32");
    assert_traps::<V, (i32,)>("s_f32_2pow31_traps", "i32");
    assert_traps::<V, (i64,)>("i64_s_2pow63_traps", "i64");
    assert_traps::<V, (i64,)>("i64_u_2pow64_traps", "u64");
}

// Rust's `as` casts saturate rather than wrapping, so an implementation whose
// range check never fires returns `i32::MAX`/`i32::MIN` here instead of trapping.
#[test]
fn truncation_traps_on_far_out_of_range_operands() {
    truncation_traps_on_far_out_of_range_operands_on::<Stack>();
    truncation_traps_on_far_out_of_range_operands_on::<Register>();
}

fn truncation_traps_on_far_out_of_range_operands_on<V: VirtualMachine>() {
    assert_traps::<V, (i32,)>("s_big_traps", "i32");
    assert_traps::<V, (i32,)>("s_negbig_traps", "i32");
    assert_traps::<V, (i64,)>("i64_s_big_traps", "i64");
}

#[test]
fn truncation_traps_on_nan_and_infinity() {
    truncation_traps_on_nan_and_infinity_on::<Stack>();
    truncation_traps_on_nan_and_infinity_on::<Register>();
}

fn truncation_traps_on_nan_and_infinity_on<V: VirtualMachine>() {
    assert_traps::<V, (i32,)>("nan_traps", "i32");
    assert_traps::<V, (i32,)>("inf_traps", "i32");
    assert_traps::<V, (i32,)>("neg_inf_traps", "i32");
    assert_traps::<V, (i64,)>("i64_nan_traps", "u64");
}

/// `i32` result of the saturating-truncation fixture.
fn sat_i32<V: VirtualMachine>(name: &str) -> i32 {
    call_i32::<V>(include_bytes!("../fixtures/trunc_sat.wasm"), name)
}

/// `i64` result of the saturating-truncation fixture.
fn sat_i64<V: VirtualMachine>(name: &str) -> i64 {
    let (v,) = call_typed::<V, (i64,)>(include_bytes!("../fixtures/trunc_sat.wasm"), name);
    v
}

// NaN converts to `0`, *not* to the target's minimum. Treating it as an
// out-of-range value and clamping it — the intuitive reading of "saturate" — is
// the standard way to get this wrong.
#[test]
fn saturating_truncation_maps_nan_to_zero() {
    saturating_truncation_maps_nan_to_zero_on::<Stack>();
    saturating_truncation_maps_nan_to_zero_on::<Register>();
}

fn saturating_truncation_maps_nan_to_zero_on<V: VirtualMachine>() {
    assert_eq!(sat_i32::<V>("nan_s"), 0, "not i32::MIN");
    assert_eq!(sat_i32::<V>("nan_u"), 0);
    assert_eq!(sat_i32::<V>("f64_nan_s"), 0);
    assert_eq!(sat_i64::<V>("i64_nan_s"), 0, "not i64::MIN");
    assert_eq!(sat_i64::<V>("i64_nan_u"), 0);
}

// The infinities, by contrast, *do* clamp — they sit at the ends of the ordering.
#[test]
fn saturating_truncation_clamps_the_infinities_to_the_bounds() {
    saturating_truncation_clamps_the_infinities_to_the_bounds_on::<Stack>();
    saturating_truncation_clamps_the_infinities_to_the_bounds_on::<Register>();
}

fn saturating_truncation_clamps_the_infinities_to_the_bounds_on<V: VirtualMachine>() {
    assert_eq!(sat_i32::<V>("inf_s"), i32::MAX);
    assert_eq!(sat_i32::<V>("neg_inf_s"), i32::MIN);
    assert_eq!(sat_i32::<V>("inf_u"), -1, "u32::MAX as an i32");
    assert_eq!(sat_i32::<V>("neg_inf_u"), 0, "unsigned clamps at zero");
    assert_eq!(sat_i64::<V>("i64_inf_s"), i64::MAX);
    assert_eq!(sat_i64::<V>("i64_neg_inf_s"), i64::MIN);
    assert_eq!(sat_i64::<V>("i64_inf_u"), -1, "u64::MAX as an i64");
}

// The whole point of the family: these operands trap under `trunc` and clamp here.
#[test]
fn saturating_truncation_clamps_where_trunc_would_trap() {
    saturating_truncation_clamps_where_trunc_would_trap_on::<Stack>();
    saturating_truncation_clamps_where_trunc_would_trap_on::<Register>();
}

fn saturating_truncation_clamps_where_trunc_would_trap_on<V: VirtualMachine>() {
    assert_eq!(sat_i32::<V>("big_s"), i32::MAX);
    assert_eq!(sat_i32::<V>("negbig_s"), i32::MIN);
    assert_eq!(
        sat_i32::<V>("neg_u"),
        0,
        "-1.5 clamps to 0 instead of trapping"
    );
    assert_eq!(sat_i32::<V>("at_2pow31_s"), i32::MAX);
    assert_eq!(sat_i32::<V>("at_u32_max"), -1);
}

// Saturation has to clamp at the *target* integer's bound. Deriving the bound
// from the source float's width instead is right only when the two widths happen
// to agree, so it shows up on exactly these cross-width unsigned conversions:
// 1e10 exceeds u32 but fits comfortably in u64.
#[test]
fn saturating_truncation_clamps_at_the_target_width_not_the_source_width() {
    saturating_truncation_clamps_at_the_target_width_not_the_source_width_on::<Stack>();
    saturating_truncation_clamps_at_the_target_width_not_the_source_width_on::<Register>();
}

fn saturating_truncation_clamps_at_the_target_width_not_the_source_width_on<V: VirtualMachine>() {
    assert_eq!(
        sat_i32::<V>("cross_i32_u"),
        -1,
        "1e10 must clamp to u32::MAX; going via u64 would wrap to 1410065408"
    );
    assert_eq!(
        sat_i64::<V>("cross_i64_u"),
        10_000_000_000,
        "1e10 fits a u64, so clamping at u32::MAX here would lose it"
    );
    assert_eq!(
        sat_i64::<V>("cross_i64_u_huge"),
        -1,
        "1e30 is past u64 too, so it clamps to u64::MAX"
    );
}

#[test]
fn saturating_truncation_leaves_in_range_operands_alone() {
    saturating_truncation_leaves_in_range_operands_alone_on::<Stack>();
    saturating_truncation_leaves_in_range_operands_alone_on::<Register>();
}

fn saturating_truncation_leaves_in_range_operands_alone_on<V: VirtualMachine>() {
    assert_eq!(sat_i32::<V>("plain_s"), 3);
    assert_eq!(
        sat_i32::<V>("plain_neg_s"),
        -3,
        "still toward zero, not floor"
    );
    assert_eq!(sat_i32::<V>("plain_u"), 0, "trunc(-0.9) is -0.0");
    assert_eq!(sat_i64::<V>("i64_plain_s"), -3);
}

/// `f32` result of the integer-to-float conversion fixture.
fn conv_f32<V: VirtualMachine>(name: &str) -> f32 {
    let (v,) = call_typed::<V, (f32,)>(include_bytes!("../fixtures/convert.wasm"), name);
    v
}

/// `f64` result of the integer-to-float conversion fixture.
fn conv_f64<V: VirtualMachine>(name: &str) -> f64 {
    let (v,) = call_typed::<V, (f64,)>(include_bytes!("../fixtures/convert.wasm"), name);
    v
}

// `-1` is `0xFFFFFFFF`, so the unsigned forms convert the largest u32 while the
// signed forms convert a negative number. Reading the operand at the wrong
// signedness is the one mistake here that changes the result outright rather than
// by a rounding step.
#[test]
fn conversion_signedness_decides_how_the_operand_is_read() {
    conversion_signedness_decides_how_the_operand_is_read_on::<Stack>();
    conversion_signedness_decides_how_the_operand_is_read_on::<Register>();
}

fn conversion_signedness_decides_how_the_operand_is_read_on<V: VirtualMachine>() {
    assert_eq!(conv_f32::<V>("s_neg1"), -1.0);
    assert_eq!(
        conv_f32::<V>("u_neg1"),
        4294967296.0,
        "0xFFFFFFFF, rounded to f32"
    );
    assert_eq!(conv_f64::<V>("f64_s_neg1"), -1.0);
    assert_eq!(conv_f64::<V>("f64_u_neg1"), 4294967295.0, "exact in f64");
    assert_eq!(conv_f64::<V>("i64_s_neg1"), -1.0);
    assert_eq!(conv_f64::<V>("i64_u_neg1"), 18446744073709551616.0, "2^64");
    assert_eq!(conv_f32::<V>("f32_i64_u_neg1"), 18446744073709551616.0);
}

// Conversion is not exact: an operand needing more significand bits than the
// target has gets rounded. `f32` runs out at 2^24, so `i32::MAX` lands one *above*
// itself — while `f64`, with 53 bits, keeps every `i32` intact.
#[test]
fn conversion_to_f32_rounds_where_conversion_to_f64_stays_exact() {
    conversion_to_f32_rounds_where_conversion_to_f64_stays_exact_on::<Stack>();
    conversion_to_f32_rounds_where_conversion_to_f64_stays_exact_on::<Register>();
}

fn conversion_to_f32_rounds_where_conversion_to_f64_stays_exact_on<V: VirtualMachine>() {
    assert_eq!(
        conv_f32::<V>("s_i32_max"),
        2147483648.0,
        "2^31, above i32::MAX"
    );
    assert_eq!(conv_f64::<V>("f64_s_i32_max"), 2147483647.0, "exact in f64");
    assert_eq!(
        conv_f32::<V>("s_i32_min"),
        -2147483648.0,
        "i32::MIN is a power of two, so it survives"
    );
    assert_eq!(
        conv_f64::<V>("f64_s_i64_max"),
        9223372036854775808.0,
        "2^63"
    );
}

// Rounding is to nearest with ties to even, matching the spec's roundTiesToEven.
// 16777217 is 2^24+1, the first integer `f32` cannot hold; it and 16777219 are
// both ties, and they resolve in opposite directions because only one neighbour
// of each is even.
#[test]
fn conversion_breaks_rounding_ties_to_even() {
    conversion_breaks_rounding_ties_to_even_on::<Stack>();
    conversion_breaks_rounding_ties_to_even_on::<Register>();
}

fn conversion_breaks_rounding_ties_to_even_on<V: VirtualMachine>() {
    assert_eq!(conv_f32::<V>("tie_down"), 16777216.0, "2^24+1 rounds down");
    assert_eq!(conv_f32::<V>("tie_up"), 16777220.0, "2^24+3 rounds up");
}

// `i64` -> `f32` has to round exactly once. Widening through `f64` on the way is
// harmless for `i32` operands, which `f64` holds exactly, but for `i64` both steps
// round and the two can disagree: this operand sits one above the `f32` tie at
// 2^62 + 2^38, so the `f64` step rounds it back *onto* the tie and ties-to-even
// then rounds down, landing 2^39 short.
#[test]
fn conversion_from_i64_to_f32_rounds_only_once() {
    conversion_from_i64_to_f32_rounds_only_once_on::<Stack>();
    conversion_from_i64_to_f32_rounds_only_once_on::<Register>();
}

fn conversion_from_i64_to_f32_rounds_only_once_on<V: VirtualMachine>() {
    assert_eq!(
        conv_f32::<V>("double_round_s"),
        4611686568183201792.0,
        "rounding via f64 would give 4611686018427387904.0"
    );
    assert_eq!(conv_f32::<V>("double_round_u"), 4611686568183201792.0);
}

#[test]
fn conversion_is_exact_for_small_operands() {
    conversion_is_exact_for_small_operands_on::<Stack>();
    conversion_is_exact_for_small_operands_on::<Register>();
}

fn conversion_is_exact_for_small_operands_on<V: VirtualMachine>() {
    assert_eq!(conv_f32::<V>("exact_s"), 3.0);
    assert_eq!(conv_f32::<V>("exact_neg_s"), -3.0);
    assert_eq!(conv_f64::<V>("f64_exact_s"), 42.0);
    assert_eq!(conv_f32::<V>("exact_zero"), 0.0);
}

/// `f32` result of the demote/promote fixture.
fn dp_f32<V: VirtualMachine>(name: &str) -> f32 {
    let (v,) = call_typed::<V, (f32,)>(include_bytes!("../fixtures/demote_promote.wasm"), name);
    v
}

/// `f64` result of the demote/promote fixture.
fn dp_f64<V: VirtualMachine>(name: &str) -> f64 {
    let (v,) = call_typed::<V, (f64,)>(include_bytes!("../fixtures/demote_promote.wasm"), name);
    v
}

// Demote overflows to an infinity. Clamping to `f32::MAX` — the saturating
// behaviour of `trunc_sat`, and a tempting thing to reuse — would be wrong here.
#[test]
fn demote_overflows_to_infinity_rather_than_clamping() {
    demote_overflows_to_infinity_rather_than_clamping_on::<Stack>();
    demote_overflows_to_infinity_rather_than_clamping_on::<Register>();
}

fn demote_overflows_to_infinity_rather_than_clamping_on<V: VirtualMachine>() {
    assert_eq!(dp_f32::<V>("over_pos"), f32::INFINITY, "not f32::MAX");
    assert_eq!(dp_f32::<V>("over_neg"), f32::NEG_INFINITY);
}

// The overflow threshold is not `f32::MAX` but the halfway point between it and
// 2^128, so an operand past `f32::MAX` can still round back down to it. A check
// shaped like `if v > f32::MAX as f64 { inf }` would wrongly overflow the first
// of these.
#[test]
fn demote_overflows_at_the_rounding_threshold_not_at_f32_max() {
    demote_overflows_at_the_rounding_threshold_not_at_f32_max_on::<Stack>();
    demote_overflows_at_the_rounding_threshold_not_at_f32_max_on::<Register>();
}

fn demote_overflows_at_the_rounding_threshold_not_at_f32_max_on<V: VirtualMachine>() {
    assert_eq!(
        dp_f32::<V>("below_threshold"),
        f32::MAX,
        "a quarter-ULP past f32::MAX still rounds down to it"
    );
    assert_eq!(
        dp_f32::<V>("at_threshold"),
        f32::INFINITY,
        "the exact tie rounds away from f32::MAX, whose significand is all ones"
    );
    assert_eq!(dp_f32::<V>("above_threshold"), f32::INFINITY);
    assert_eq!(dp_f32::<V>("at_f32_max"), f32::MAX);
}

// Underflow collapses to a zero, and the sign survives even though `-0.0 == 0.0`.
#[test]
fn demote_underflows_to_a_signed_zero() {
    demote_underflows_to_a_signed_zero_on::<Stack>();
    demote_underflows_to_a_signed_zero_on::<Register>();
}

fn demote_underflows_to_a_signed_zero_on<V: VirtualMachine>() {
    let pos = dp_f32::<V>("under_pos");
    assert_eq!(pos, 0.0);
    assert!(pos.is_sign_positive());

    let neg = dp_f32::<V>("under_neg");
    assert_eq!(neg, 0.0);
    assert!(neg.is_sign_negative(), "-1e-300 must underflow to -0.0");
}

// Ties to even again, now on the float-narrowing path: at 1.0 the `f32` step is
// 2^-23, so 1+2^-24 is exactly a tie and resolves downward.
#[test]
fn demote_breaks_rounding_ties_to_even() {
    demote_breaks_rounding_ties_to_even_on::<Stack>();
    demote_breaks_rounding_ties_to_even_on::<Register>();
}

fn demote_breaks_rounding_ties_to_even_on<V: VirtualMachine>() {
    assert_eq!(dp_f32::<V>("tie_down"), 1.0, "1+2^-24 rounds down to 1.0");
    assert_eq!(dp_f32::<V>("tie_up"), 1.0000002, "1+3*2^-24 rounds up");
}

#[test]
fn demote_passes_the_specials_through() {
    demote_passes_the_specials_through_on::<Stack>();
    demote_passes_the_specials_through_on::<Register>();
}

fn demote_passes_the_specials_through_on<V: VirtualMachine>() {
    assert!(dp_f32::<V>("demote_nan").is_nan());
    assert_eq!(dp_f32::<V>("demote_inf"), f32::INFINITY);

    let neg_zero = dp_f32::<V>("demote_neg_zero");
    assert_eq!(neg_zero, 0.0);
    assert!(neg_zero.is_sign_negative());
}

// Promote is exact, so demoting a promoted value returns the original `f32`
// bit-for-bit — including at `f32::MAX`, where demote is otherwise near overflow.
#[test]
fn promote_is_exact_so_the_round_trip_is_the_identity() {
    promote_is_exact_so_the_round_trip_is_the_identity_on::<Stack>();
    promote_is_exact_so_the_round_trip_is_the_identity_on::<Register>();
}

fn promote_is_exact_so_the_round_trip_is_the_identity_on<V: VirtualMachine>() {
    // The literal is `f32::consts::PI` — same bit pattern, named so clippy's
    // `approx_constant` lint does not flag it.
    assert_eq!(dp_f32::<V>("round_trip"), std::f32::consts::PI);
    assert_eq!(dp_f32::<V>("round_trip_max"), f32::MAX);
}

// Promote widens the value the `f32` actually held, which is not the decimal that
// produced it: `0.1f32` is 0.100000001490116119384765625, and that is what lands
// in the `f64`. Asserting `0.1` here would fail.
#[test]
fn promote_widens_the_f32_value_not_the_source_decimal() {
    promote_widens_the_f32_value_not_the_source_decimal_on::<Stack>();
    promote_widens_the_f32_value_not_the_source_decimal_on::<Register>();
}

fn promote_widens_the_f32_value_not_the_source_decimal_on<V: VirtualMachine>() {
    assert_eq!(dp_f64::<V>("promote_tenth"), 0.1f32 as f64);
    assert_ne!(
        dp_f64::<V>("promote_tenth"),
        0.1f64,
        "not the f64 nearest to 0.1"
    );
    assert_eq!(dp_f64::<V>("promote_plain"), 3.5, "exact in both widths");
}

#[test]
fn promote_passes_the_specials_and_subnormals_through() {
    promote_passes_the_specials_and_subnormals_through_on::<Stack>();
    promote_passes_the_specials_and_subnormals_through_on::<Register>();
}

fn promote_passes_the_specials_and_subnormals_through_on<V: VirtualMachine>() {
    assert!(dp_f64::<V>("promote_nan").is_nan());
    assert_eq!(dp_f64::<V>("promote_neg_inf"), f64::NEG_INFINITY);

    let neg_zero = dp_f64::<V>("promote_neg_zero");
    assert_eq!(neg_zero, 0.0);
    assert!(neg_zero.is_sign_negative());

    // A subnormal `f32` is a perfectly ordinary `f64`, so it widens exactly rather
    // than being flushed to zero.
    assert_eq!(dp_f64::<V>("promote_subnormal"), f32::from_bits(1) as f64);
    assert!(dp_f64::<V>("promote_subnormal") > 0.0);
}

/// `i32` result of the reinterpretation fixture.
fn rein_i32<V: VirtualMachine>(name: &str) -> i32 {
    call_i32::<V>(include_bytes!("../fixtures/reinterpret.wasm"), name)
}

/// `i64` result of the reinterpretation fixture.
fn rein_i64<V: VirtualMachine>(name: &str) -> i64 {
    let (v,) = call_typed::<V, (i64,)>(include_bytes!("../fixtures/reinterpret.wasm"), name);
    v
}

/// `f32` result of the reinterpretation fixture.
fn rein_f32<V: VirtualMachine>(name: &str) -> f32 {
    let (v,) = call_typed::<V, (f32,)>(include_bytes!("../fixtures/reinterpret.wasm"), name);
    v
}

/// `f64` result of the reinterpretation fixture.
fn rein_f64<V: VirtualMachine>(name: &str) -> f64 {
    let (v,) = call_typed::<V, (f64,)>(include_bytes!("../fixtures/reinterpret.wasm"), name);
    v
}

// Reinterpretation moves bits; it does not convert values. Implementing it with an
// `as` cast — the mistake the whole family invites, since the signatures match —
// would return `1` for all three of these instead of the bit patterns.
#[test]
fn reinterpretation_moves_bits_rather_than_converting_values() {
    reinterpretation_moves_bits_rather_than_converting_values_on::<Stack>();
    reinterpretation_moves_bits_rather_than_converting_values_on::<Register>();
}

fn reinterpretation_moves_bits_rather_than_converting_values_on<V: VirtualMachine>() {
    assert_eq!(
        rein_i32::<V>("f32_bits_1_5"),
        1069547520,
        "0x3FC00000, not 1"
    );
    assert_eq!(rein_i32::<V>("f32_bits_1_0"), 1065353216, "0x3F800000");
    assert_eq!(rein_i64::<V>("f64_bits_1_5"), 4609434218613702656);
}

// The float's sign bit becomes the integer's sign bit. `-0.0` is the sharpest case:
// it compares equal to `+0.0` and every *conversion* sends it to `0`, but
// reinterpreting it gives the minimum integer.
#[test]
fn reinterpretation_carries_the_sign_bit_across() {
    reinterpretation_carries_the_sign_bit_across_on::<Stack>();
    reinterpretation_carries_the_sign_bit_across_on::<Register>();
}

fn reinterpretation_carries_the_sign_bit_across_on<V: VirtualMachine>() {
    assert_eq!(rein_i32::<V>("neg_zero_f32"), i32::MIN, "0x80000000, not 0");
    assert_eq!(rein_i64::<V>("neg_zero_f64"), i64::MIN);
    assert_eq!(rein_i32::<V>("pos_zero"), 0, "+0.0 really is all-zero bits");
    assert_eq!(rein_i32::<V>("neg_1_5_f32"), -1077936128);
}

// Every bit pattern is a valid float, so neither direction can fail and a round
// trip is the identity — including for patterns that are NaNs or subnormals.
#[test]
fn reinterpretation_round_trips_in_both_directions() {
    reinterpretation_round_trips_in_both_directions_on::<Stack>();
    reinterpretation_round_trips_in_both_directions_on::<Register>();
}

fn reinterpretation_round_trips_in_both_directions_on<V: VirtualMachine>() {
    assert_eq!(rein_i32::<V>("round_trip_i32"), 1069547520);
    assert_eq!(rein_i64::<V>("round_trip_i64"), 4609434218613702656);
    assert_eq!(
        rein_f32::<V>("i32_to_f32"),
        1.5,
        "0x3FC00000 read as a float"
    );
    assert_eq!(rein_f64::<V>("i64_to_f64"), 1.5);
    assert_eq!(
        rein_i32::<V>("subnormal"),
        1,
        "the smallest subnormal survives"
    );
    assert_eq!(rein_i32::<V>("infinity"), 2139095040, "0x7F800000");
    assert_eq!(
        rein_i32::<V>("neg_one_to_f32"),
        -1,
        "0xFFFFFFFF is a NaN, and returns"
    );
}

// NaN payloads must survive verbatim. Reinterpretation is not an arithmetic
// operation, so it has no licence to canonicalise them — and that includes the
// signalling patterns, which some float paths quieten by setting the high
// significand bit.
#[test]
fn reinterpretation_preserves_nan_payloads_without_canonicalising() {
    reinterpretation_preserves_nan_payloads_without_canonicalising_on::<Stack>();
    reinterpretation_preserves_nan_payloads_without_canonicalising_on::<Register>();
}

fn reinterpretation_preserves_nan_payloads_without_canonicalising_on<V: VirtualMachine>() {
    assert_eq!(
        rein_i32::<V>("nan_payload"),
        2143289345,
        "0x7FC00001 keeps its payload"
    );
    assert_eq!(
        rein_i32::<V>("nan_signalling"),
        2139095041,
        "0x7F800001 is a signalling NaN and must not be quietened to 0x7FC00001"
    );
    assert_eq!(
        rein_i32::<V>("nan_negative"),
        -4194304,
        "0xFFC00000, sign bit intact"
    );
    assert_eq!(rein_i64::<V>("nan_payload_f64"), 9218868437227405313);
}

/// `i32` result of the `ref.is_null` fixture.
fn ref_null<V: VirtualMachine>(name: &str) -> i32 {
    call_i32::<V>(include_bytes!("../fixtures/ref_is_null.wasm"), name)
}

#[test]
fn ref_is_null_distinguishes_a_null_reference_from_a_function() {
    ref_is_null_distinguishes_a_null_reference_from_a_function_on::<Stack>();
    ref_is_null_distinguishes_a_null_reference_from_a_function_on::<Register>();
}

fn ref_is_null_distinguishes_a_null_reference_from_a_function_on<V: VirtualMachine>() {
    assert_eq!(ref_null::<V>("null_is_null"), 1);
    assert_eq!(ref_null::<V>("func_is_not_null"), 0);
}

// `ref.is_null` is a predicate, so its result is an `i32` and not a reference.
// Pushing a `Val::Ref` would panic the interpreter when `br_if` reads its
// condition as an `i32` — a loud failure rather than a wrong answer, which is why
// this is worth pinning separately from the value checks above.
#[test]
fn ref_is_null_result_is_an_i32_usable_as_a_branch_condition() {
    ref_is_null_result_is_an_i32_usable_as_a_branch_condition_on::<Stack>();
    ref_is_null_result_is_an_i32_usable_as_a_branch_condition_on::<Register>();
}

fn ref_is_null_result_is_an_i32_usable_as_a_branch_condition_on<V: VirtualMachine>() {
    assert_eq!(ref_null::<V>("brif_on_null"), 111, "null takes the branch");
    assert_eq!(
        ref_null::<V>("brif_on_func"),
        222,
        "a real funcref does not"
    );
}

// The reference is consumed, not peeked at: dropping the `i32` result leaves the
// stack back at the sentinel, with no reference stranded above it.
#[test]
fn ref_is_null_consumes_the_reference() {
    ref_is_null_consumes_the_reference_on::<Stack>();
    ref_is_null_consumes_the_reference_on::<Register>();
}

fn ref_is_null_consumes_the_reference_on<V: VirtualMachine>() {
    assert_eq!(ref_null::<V>("consumes_the_ref"), 555);
}

// ---------------------------------------------------------------------------
// Integer division and remainder traps
// ---------------------------------------------------------------------------
//
// These are the fixtures `traps.rs` defers to. rustc emits its own zero and
// overflow checks ahead of every integer division, so a compiled guest panics
// before the wasm instruction runs and the interpreter's own trap is unreachable
// from Rust. Assembled from `wat` to emit the bare instruction.

/// Wraps `body` — a sequence of operators leaving one `i32` — in a `() -> i32`
/// export named `f`, and calls it.
fn call_i32_body<V: VirtualMachine>(
    body: &str,
) -> Result<i32, tracewasm_core::error::FuncCallError> {
    let wat = format!("(module (func (export \"f\") (result i32) {body}))");
    let bytes = wat::parse_str(&wat).expect("invalid wat");

    try_call::<V, (i32,)>(&bytes, "f").map(|(v,)| v)
}

/// As [`call_i32_body`], for a body leaving one `i64`.
fn call_i64_body<V: VirtualMachine>(
    body: &str,
) -> Result<i64, tracewasm_core::error::FuncCallError> {
    let wat = format!("(module (func (export \"f\") (result i64) {body}))");
    let bytes = wat::parse_str(&wat).expect("invalid wat");

    try_call::<V, (i64,)>(&bytes, "f").map(|(v,)| v)
}

#[test]
fn integer_division_by_zero_traps() {
    integer_division_by_zero_traps_on::<Stack>();
    integer_division_by_zero_traps_on::<Register>();
}

fn integer_division_by_zero_traps_on<V: VirtualMachine>() {
    for op in ["i32.div_s", "i32.div_u", "i32.rem_s", "i32.rem_u"] {
        let err = call_i32_body::<V>(&format!("i32.const 1 i32.const 0 {op}"))
            .expect_err(&format!("{op} by zero must trap"));

        assert!(
            matches!(
                err.cause(),
                InstructionExecutionError::Division { .. }
                    | InstructionExecutionError::Remainder { .. }
            ),
            "{op} gave {:?}",
            err.cause()
        );
    }

    for op in ["i64.div_s", "i64.div_u", "i64.rem_s", "i64.rem_u"] {
        assert!(
            call_i64_body::<V>(&format!("i64.const 1 i64.const 0 {op}")).is_err(),
            "{op} by zero must trap"
        );
    }
}

// `MIN / -1` has no representable result, so `div_s` traps — while `div_u` reads
// the same bits as a large positive number and is perfectly well defined. Sharing
// one division for both signednesses would show up right here.
#[test]
fn signed_division_overflow_traps_but_unsigned_does_not() {
    signed_division_overflow_traps_but_unsigned_does_not_on::<Stack>();
    signed_division_overflow_traps_but_unsigned_does_not_on::<Register>();
}

fn signed_division_overflow_traps_but_unsigned_does_not_on<V: VirtualMachine>() {
    let err = call_i32_body::<V>("i32.const -2147483648 i32.const -1 i32.div_s")
        .expect_err("i32::MIN / -1 must trap");

    assert!(
        matches!(err.cause(), InstructionExecutionError::Division { .. }),
        "got {:?}",
        err.cause()
    );

    assert!(
        call_i64_body::<V>("i64.const -9223372036854775808 i64.const -1 i64.div_s").is_err(),
        "i64::MIN / -1 must trap too"
    );

    // 0x8000_0000 /u 0xFFFF_FFFF is 0, not a trap.
    assert_eq!(
        call_i32_body::<V>("i32.const -2147483648 i32.const -1 i32.div_u").unwrap(),
        0,
        "the unsigned form of the same bits is defined"
    );
}

// `MIN % -1` is `0` by definition, *not* the overflow trap its division sibling
// takes — the one asymmetry between the two operators.
#[test]
fn signed_remainder_of_min_by_minus_one_is_zero_not_a_trap() {
    signed_remainder_of_min_by_minus_one_is_zero_not_a_trap_on::<Stack>();
    signed_remainder_of_min_by_minus_one_is_zero_not_a_trap_on::<Register>();
}

fn signed_remainder_of_min_by_minus_one_is_zero_not_a_trap_on<V: VirtualMachine>() {
    assert_eq!(
        call_i32_body::<V>("i32.const -2147483648 i32.const -1 i32.rem_s").unwrap(),
        0
    );
    assert_eq!(
        call_i64_body::<V>("i64.const -9223372036854775808 i64.const -1 i64.rem_s").unwrap(),
        0
    );
}

// Truncation is toward zero, so a negative signed quotient rounds *up*. Rust's
// `/` already does this; the guard is against someone reaching for a flooring
// division, which would give -3 here.
#[test]
fn signed_division_truncates_toward_zero() {
    signed_division_truncates_toward_zero_on::<Stack>();
    signed_division_truncates_toward_zero_on::<Register>();
}

fn signed_division_truncates_toward_zero_on<V: VirtualMachine>() {
    assert_eq!(
        call_i32_body::<V>("i32.const -7 i32.const 3 i32.div_s").unwrap(),
        -2
    );
    assert_eq!(
        call_i32_body::<V>("i32.const -7 i32.const 3 i32.rem_s").unwrap(),
        -1
    );
}
