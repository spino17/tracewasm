//! End-to-end checks for the comparison instructions, covering the two things
//! that are easy to get wrong: signed vs unsigned ordering, and the fact that
//! *every* comparison yields an `i32` regardless of operand width.

use tracewasm_core::{
    instance::traits::{ImportRegistry, ResultVals, Val},
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
    let module = Module::compile(wasm).expect("fixture should compile");

    let func = module
        .get_typed_func::<(), (i32,)>(name)
        .unwrap_or_else(|e| panic!("export `{name}`: {e}"));

    let mut instance = module
        .instantiate::<LinearMemory, _>(NoImports, None)
        .expect("fixture should instantiate");

    match func.call((), &mut instance) {
        Ok((v,)) => v,
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
