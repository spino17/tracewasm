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
