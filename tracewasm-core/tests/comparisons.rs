//! End-to-end checks for the comparison instructions, covering the two things
//! that are easy to get wrong: signed vs unsigned ordering, and the fact that
//! *every* comparison yields an `i32` regardless of operand width.

use tracewasm_core::{
    instance::traits::{ImportRegistry, ResultVals, Results, Val},
    memory::{MemoryView, linear::LinearMemory},
    module::Module,
};

/// A module with no imports.
struct NoImports;

impl ImportRegistry for NoImports {
    fn execute<V: MemoryView>(
        &mut self,
        _module_name: &str,
        _func_name: &str,
        _params: &[Val],
        _memory_view: &mut V,
    ) -> Result<ResultVals, tracewasm_core::error::TraceWasmError> {
        unreachable!("the fixture declares no imports")
    }

    fn signature(
        &self,
        _module_name: &str,
        _func_name: &str,
    ) -> Option<tracewasm_core::instance::traits::ImportSignature> {
        None
    }

    fn func_count(&self) -> u32 {
        0
    }

    fn global_count(&self) -> u32 {
        0
    }

    fn get_global(
        &self,
        _module_name: &str,
        _global_name: &str,
    ) -> Result<Val, tracewasm_core::error::TraceWasmError> {
        unreachable!("the fixture declares no globals")
    }
}

/// Calls a `() -> i32` export of the integer-comparison fixture.
fn call(name: &str) -> i32 {
    call_in(include_bytes!("fixtures/comparisons.wasm"), name)
}

/// Calls a `() -> i32` export of the float-comparison fixture.
fn call_f(name: &str) -> i32 {
    call_in(include_bytes!("fixtures/float_comparisons.wasm"), name)
}

/// Compiles `wasm`, instantiates it, and calls its `() -> i32` export `name`.
fn call_in(wasm: &[u8], name: &str) -> i32 {
    let (v,) = call_typed::<(i32,)>(wasm, name);
    v
}

/// Compiles `wasm`, instantiates it, and calls its `() -> R` export `name`.
fn call_typed<R: Results>(wasm: &[u8], name: &str) -> R {
    let module = Module::compile(wasm).expect("fixture should compile");

    let func = module
        .get_typed_func::<(), R>(name)
        .unwrap_or_else(|e| panic!("export `{name}`: {e}"));

    let mut instance = module
        .instantiate::<LinearMemory, _>(NoImports, None)
        .expect("fixture should instantiate");

    match func.call((), &mut instance) {
        Ok(v) => v,
        Err(e) => panic!("calling `{name}` failed: {e}"),
    }
}

// `-1` is `0xFFFF_FFFF` unsigned, so it is the *largest* u32 — the unsigned and
// signed forms must disagree here. Sharing one comparison for both would show up
// as these two returning the same value.
#[test]
fn unsigned_and_signed_ordering_disagree_on_negative_operands() {
    assert_eq!(call("lt_u"), 0, "-1 <u 1 is false: 0xFFFFFFFF > 1");
    assert_eq!(call("lt_s"), 1, "-1 <s 1 is true");
    assert_eq!(call("ge_u"), 1, "-1 >=u 1 is true");
}

#[test]
fn i64_comparisons_use_the_right_signedness_too() {
    assert_eq!(call("i64_lt_u"), 0);
    assert_eq!(call("i64_lt_s"), 1);
}

// Regression guard: an i64 comparison must push an `i32`, not an `i64`. Feeding
// one to `br_if` — which reads its condition as an `i32` — panics the interpreter
// if the wrong variant was pushed, so this fails loudly rather than subtly.
#[test]
fn i64_comparison_result_is_an_i32_usable_as_a_branch_condition() {
    assert_eq!(call("i64_brif"), 111, "5 <s 9 should take the branch");
}

// IEEE 754 ordering, which is *not* the two's-complement ordering the integer
// comparisons use. Rust's operators already behave this way, so these guard
// against someone "fixing" them into something like `a.partial_cmp(b)` handling.
#[test]
fn nan_makes_ordered_comparisons_false_but_ne_true() {
    assert_eq!(call_f("nan_lt"), 0, "NaN < 1 is false");
    assert_eq!(call_f("nan_ge"), 0, "NaN >= 1 is false");
    assert_eq!(call_f("nan_eq"), 0, "NaN == NaN is false");
    assert_eq!(call_f("nan_ne"), 1, "NaN != NaN is true");
    assert_eq!(call_f("f64_nan_le"), 0, "and the same on the f64 path");
}

#[test]
fn negative_and_positive_zero_compare_equal() {
    assert_eq!(call_f("negzero_eq"), 1, "-0.0 == +0.0");
    assert_eq!(call_f("negzero_lt"), 0, "-0.0 is not less than +0.0");
}

#[test]
fn infinities_order_normally() {
    assert_eq!(call_f("f64_inf_gt"), 1);
}

// As for the i64 comparisons: the result must be an `i32`, or `br_if` panics.
#[test]
fn float_comparison_result_is_an_i32_usable_as_a_branch_condition() {
    assert_eq!(call_f("f64_brif"), 111, "1 < 2 should take the branch");
}

/// `f32` result of the min/max/copysign fixture.
fn min_max_f32(name: &str) -> f32 {
    let (v,) = call_typed::<(f32,)>(include_bytes!("fixtures/float_minmax.wasm"), name);
    v
}

/// `f64` result of the min/max/copysign fixture.
fn min_max_f64(name: &str) -> f64 {
    let (v,) = call_typed::<(f64,)>(include_bytes!("fixtures/float_minmax.wasm"), name);
    v
}

// The cases where Rust's `f32::min`/`max` disagree with wasm: they return the
// *non*-NaN operand, and leave the signed-zero tie unspecified. Delegating to
// them would pass every ordinary test and fail only these.
#[test]
fn min_max_return_nan_when_either_operand_is_nan() {
    assert!(min_max_f32("min_nan").is_nan(), "min(NaN, 1) must be NaN");
    assert!(min_max_f32("max_nan").is_nan(), "max(1, NaN) must be NaN");
    assert!(min_max_f64("f64_min_nan").is_nan(), "and on the f64 path");
}

// `-0.0 == +0.0`, so the tie is broken by sign: min yields -0.0, max +0.0.
// Checked via `is_sign_negative`, since `==` cannot tell the two zeros apart.
#[test]
fn min_max_break_the_signed_zero_tie_by_sign() {
    let min = min_max_f32("min_zero");
    assert_eq!(min, 0.0);
    assert!(min.is_sign_negative(), "min(+0.0, -0.0) must be -0.0");

    let max = min_max_f32("max_zero");
    assert_eq!(max, 0.0);
    assert!(max.is_sign_positive(), "max(-0.0, +0.0) must be +0.0");

    assert!(min_max_f64("f64_max_zero").is_sign_positive());
}

#[test]
fn min_max_order_ordinary_values() {
    assert_eq!(min_max_f32("min_plain"), 3.0);
    assert_eq!(min_max_f32("max_plain"), 7.0);
}

#[test]
fn copysign_takes_magnitude_from_the_first_and_sign_from_the_second() {
    assert_eq!(min_max_f32("copysign_neg"), -5.0);
    assert_eq!(min_max_f32("copysign_pos"), 5.0);
    assert_eq!(min_max_f64("f64_copysign"), -5.0);
}

/// `i32` result of the bit-counting fixture.
fn bits_i32(name: &str) -> i32 {
    call_in(include_bytes!("fixtures/bit_counting.wasm"), name)
}

/// `i64` result of the bit-counting fixture.
fn bits_i64(name: &str) -> i64 {
    let (v,) = call_typed::<(i64,)>(include_bytes!("fixtures/bit_counting.wasm"), name);
    v
}

// Zero is the edge case: `clz`/`ctz` of `0` are the full operand width, not `0`.
#[test]
fn clz_and_ctz_of_zero_are_the_operand_width() {
    assert_eq!(bits_i32("clz_zero"), 32);
    assert_eq!(bits_i32("ctz_zero"), 32);
    assert_eq!(bits_i64("i64_clz_zero"), 64, "i64 width, not 32");
    assert_eq!(bits_i64("i64_ctz_zero"), 64);
}

#[test]
fn clz_ctz_count_ordinary_values() {
    assert_eq!(bits_i32("clz_one"), 31);
    assert_eq!(bits_i32("ctz_8"), 3);
    assert_eq!(bits_i32("clz_neg1"), 0, "-1 has the high bit set");
}

// `popcnt` works on the two's-complement bits, so `-1` is all ones.
#[test]
fn popcnt_counts_sign_bits_of_negative_operands() {
    assert_eq!(bits_i32("popcnt_neg1"), 32);
    assert_eq!(bits_i32("popcnt_5"), 2, "0b101");
    assert_eq!(bits_i64("i64_popcnt_neg1"), 64);
}

/// `i32` result of the `eqz` fixture.
fn eqz(name: &str) -> i32 {
    call_in(include_bytes!("fixtures/eqz.wasm"), name)
}

#[test]
fn eqz_tests_against_zero() {
    assert_eq!(eqz("eqz_zero"), 1);
    assert_eq!(eqz("eqz_nonzero"), 0);
    assert_eq!(eqz("eqz_neg"), 0, "-1 is non-zero");
    assert_eq!(eqz("i64_eqz_zero"), 1);
    assert_eq!(eqz("i64_eqz_nonzero"), 0);
}

// `i64.eqz` returns an `i32` despite its `i64` operand — the opposite of the
// bit-counting unaries. Pushing `Val::I64` would panic `br_if`, not just differ.
#[test]
fn i64_eqz_result_is_an_i32_usable_as_a_branch_condition() {
    assert_eq!(eqz("i64_eqz_brif"), 111);
}

/// `f32` result of the float-unary fixture.
fn unary_f32(name: &str) -> f32 {
    let (v,) = call_typed::<(f32,)>(include_bytes!("fixtures/float_unary.wasm"), name);
    v
}

/// `f64` result of the float-unary fixture.
fn unary_f64(name: &str) -> f64 {
    let (v,) = call_typed::<(f64,)>(include_bytes!("fixtures/float_unary.wasm"), name);
    v
}

// `nearest` breaks ties to **even**, unlike Rust's `round`, which goes away from
// zero. Swapping in `round()` would pass most inputs and fail exactly these.
#[test]
fn nearest_breaks_ties_to_even_not_away_from_zero() {
    assert_eq!(unary_f32("nearest_2_5"), 2.0, "round() would give 3.0");
    assert_eq!(unary_f32("nearest_3_5"), 4.0, "4 is already even");
    assert_eq!(unary_f32("nearest_neg2_5"), -2.0, "round() would give -3.0");
    assert_eq!(unary_f64("f64_nearest_2_5"), 2.0);
}

// `abs`/`neg` are sign-bit operations, so they are observable on `-0.0` even
// though it compares equal to `+0.0`.
#[test]
fn abs_and_neg_act_on_the_sign_bit_of_negative_zero() {
    let abs = unary_f32("abs_negzero");
    assert_eq!(abs, 0.0);
    assert!(abs.is_sign_positive(), "abs(-0.0) must be +0.0");

    let neg = unary_f32("neg_negzero");
    assert_eq!(neg, 0.0);
    assert!(neg.is_sign_positive(), "-(-0.0) must be +0.0");
}

#[test]
fn ceil_floor_and_trunc_round_in_their_own_directions() {
    assert_eq!(unary_f32("ceil_neg"), -1.0);
    assert_eq!(unary_f32("floor_neg"), -2.0);
    assert_eq!(unary_f32("trunc_neg"), -1.0, "toward zero, unlike floor");
}

// A negative operand yields NaN rather than trapping — float unaries are total.
#[test]
fn sqrt_of_a_negative_operand_is_nan() {
    assert!(unary_f32("sqrt_neg").is_nan());
    assert_eq!(unary_f32("sqrt_4"), 2.0);
    assert_eq!(unary_f64("f64_abs_neg"), 3.5);
}

/// `i32` result of the sign-extension fixture.
fn extend_i32(name: &str) -> i32 {
    call_in(include_bytes!("fixtures/extend.wasm"), name)
}

/// `i64` result of the sign-extension fixture.
fn extend_i64(name: &str) -> i64 {
    let (v,) = call_typed::<(i64,)>(include_bytes!("fixtures/extend.wasm"), name);
    v
}

// The narrow sign-extensions read only the low bits of their operand, so the
// high bits have to be dropped first. An implementation that widened the operand
// as-is would pass `e8s_7f` and fail every other case here.
#[test]
fn narrow_sign_extension_discards_the_high_bits() {
    assert_eq!(extend_i32("e8s_ff"), -1, "low byte 0xFF is -1 as an i8");
    assert_eq!(extend_i32("e8s_7f"), 127, "0x7F stays positive");
    assert_eq!(
        extend_i32("e8s_high"),
        -128,
        "only the low byte 0x80 is read"
    );
    assert_eq!(extend_i32("e8s_256"), 0, "0x100 has a zero low byte");
    assert_eq!(extend_i32("e16s_ffff"), -1);
    assert_eq!(extend_i32("e16s_65536"), 0);
}

#[test]
fn narrow_sign_extension_to_i64_uses_the_same_rule() {
    assert_eq!(extend_i64("i64_e8s"), -1);
    assert_eq!(extend_i64("i64_e16s"), -1);
    assert_eq!(
        extend_i64("i64_e32s"),
        -1,
        "0x1_FFFFFFFF keeps only the low 32 bits, which are all ones"
    );
}

// The one pair that is easy to conflate: both widen an `i32` to an `i64`, but
// `_s` copies the sign bit and `_u` does not. They agree on every non-negative
// operand, so a negative one is the only thing that tells them apart.
#[test]
fn i32_to_i64_signed_and_unsigned_widening_disagree_on_negatives() {
    assert_eq!(extend_i64("i32_to_i64_s_neg"), -1, "sign-extended");
    assert_eq!(
        extend_i64("i32_to_i64_u_neg"),
        4_294_967_295,
        "zero-extended: 0xFFFFFFFF, not -1"
    );
    assert_eq!(extend_i64("i32_to_i64_s_pos"), 7, "and agree on positives");
    assert_eq!(extend_i64("i32_to_i64_u_pos"), 7);
}

// `wrap_i64` is the only conversion here that *discards* bits, and it wraps
// instead of trapping on operands that do not fit. Anything that range-checked
// first would reject the first four of these.
#[test]
fn wrap_i64_keeps_the_low_32_bits_without_trapping() {
    assert_eq!(extend_i32("wrap_2pow32"), 0, "0x1_0000_0000 has no low bits");
    assert_eq!(extend_i32("wrap_neg1"), -1);
    assert_eq!(
        extend_i32("wrap_u32_max"),
        -1,
        "4294967295 does not fit an i32; its low 32 bits are all ones"
    );
    assert_eq!(extend_i32("wrap_i64_max"), -1, "i64::MAX wraps to -1");
    assert_eq!(extend_i32("wrap_mixed"), 0x2345_6789, "high nibble dropped");
    assert_eq!(extend_i32("wrap_small"), 42, "in-range operands pass through");
}

// `wrap_i64` then `extend_i32_s` is a round trip for operands that fit in an
// i32; `extend_i32_u` would not agree, which is what the pair test above pins.
#[test]
fn wrap_undoes_a_signed_widening() {
    assert_eq!(extend_i64("i32_to_i64_s_neg"), -1);
    assert_eq!(extend_i32("wrap_neg1"), -1);
}
