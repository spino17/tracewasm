//! Differential testing: every guest export is run under the interpreter and
//! natively, and the two results must agree bit for bit.
//!
//! ## Why this is the most valuable file in the suite
//!
//! There are no expected values written down here. Each guest source is compiled
//! twice — to wasm by `build.rs`, and natively by `include!`ing it below — so the
//! oracle is rustc's own backend. That means:
//!
//! * a wrong answer cannot be papered over by "fixing" a magic constant;
//! * the tests keep working when a guest is edited;
//! * coverage is whatever the guest actually does, including all the library code
//!   it drags in (allocator, formatting, collections), which is far more of the
//!   instruction space than a hand-written fixture reaches.
//!
//! During development this file caught two real interpreter bugs that the entire
//! 147-test fixture suite passed straight through — both in the locals-on-stack
//! frame layout, and both invisible to any single-function test because the
//! fixtures have no declared locals.
//!
//! ## The two things guests must not do
//!
//! * **Transcendentals** (`sin`, `exp`, `powf`, …) come from the host libm
//!   natively and a compiled-in libm under wasm, and may differ in the last bit.
//! * **Hash iteration order** differs between the two environments.
//!
//! Both are documented on the guest modules; if a new guest flakes, check these
//! first.

#![cfg(not(no_guest_wasm))]

use tracewasm_core::{Register, Stack, VirtualMachine};
use tracewasm_test::{Guest, MAX_TEST_RECURSION, guests, with_large_stack};

// ---------------------------------------------------------------------------
// Native copies of the guests
// ---------------------------------------------------------------------------
//
// Each guest is included into its own module so that rustc compiles it for the
// host. `#[unsafe(no_mangle)]` symbols must be unique across the whole binary,
// which is why every guest prefixes its exports (`arith_`, `cf_`, `heap_`, …).

// `#[path]` rather than `include!`: a path-module is a real module *file*, so the
// guests' `//!` docs and `#![allow(dead_code)]` are legal, where an `include!`
// expansion would reject both.
//
// These are declared at file scope, not nested in a `mod native { .. }`: a
// `#[path]` inside an inline module resolves relative to that module's own
// directory (`tests/native/`), which does not exist, and `..` cannot traverse a
// directory that is not there.
#[path = "../guests/arithmetic.rs"]
mod g_arithmetic;
#[path = "../guests/control_flow.rs"]
mod g_control_flow;
#[path = "../guests/exotic.rs"]
mod g_exotic;
#[path = "../guests/frames.rs"]
mod g_frames;
#[path = "../guests/heap.rs"]
mod g_heap;
#[path = "../guests/memory.rs"]
mod g_memory;

/// Argument values every `(i32) -> i64` export is checked against.
///
/// Includes 0 and 1 (guests clamp, so these hit the degenerate paths), a few
/// midsize values, and the signed extremes to catch anything that multiplies the
/// argument without wrapping.
const ARGS: &[i32] = &[
    0,
    1,
    2,
    3,
    7,
    16,
    31,
    64,
    100,
    255,
    1_000,
    -1,
    -7,
    i32::MIN,
    i32::MAX,
];

/// Runs one `(i32) -> i64` export under both engines at every argument.
///
/// Failures report the argument and both values, because "differential mismatch"
/// without the input is not actionable.
fn check_i64<V: VirtualMachine>(
    guest: &mut Guest<V>,
    name: &str,
    native: extern "C" fn(i32) -> i64,
    args: &[i32],
) {
    for &arg in args {
        let expected = native(arg);
        let actual = guest.i32_i64(name, arg);

        assert_eq!(
            actual, expected,
            "`{name}({arg})`: interpreter returned {actual}, native rustc returned {expected}"
        );
    }
}

/// As [`check_i64`] for a `(i32) -> f64` export, compared **by bit pattern**.
///
/// `==` would treat `+0.0` and `-0.0` as equal and every NaN as unequal, so it
/// cannot verify the sign-of-zero and NaN-payload behaviour these guests exist to
/// pin down.
fn check_f64<V: VirtualMachine>(
    guest: &mut Guest<V>,
    name: &str,
    native: extern "C" fn(i32) -> f64,
    args: &[i32],
) {
    for &arg in args {
        let expected = native(arg);
        let actual = guest.i32_f64(name, arg);

        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "`{name}({arg})`: interpreter returned {actual} ({:#018x}), \
             native rustc returned {expected} ({:#018x})",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

#[test]
fn arithmetic_integer_edges_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| arithmetic_integer_edges_match_native_on::<Stack>());
    with_large_stack(|| arithmetic_integer_edges_match_native_on::<Register>());
}

fn arithmetic_integer_edges_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::ARITHMETIC);

    check_i64(
        &mut g,
        "arith_div_rem_edges",
        g_arithmetic::arith_div_rem_edges,
        ARGS,
    );
    check_i64(
        &mut g,
        "arith_bit_counting",
        g_arithmetic::arith_bit_counting,
        ARGS,
    );
    check_i64(
        &mut g,
        "arith_extend_wrap",
        g_arithmetic::arith_extend_wrap,
        ARGS,
    );
    check_i64(
        &mut g,
        "arith_comparisons",
        g_arithmetic::arith_comparisons,
        ARGS,
    );
    check_i64(
        &mut g,
        "arith_overflow_families",
        g_arithmetic::arith_overflow_families,
        ARGS,
    );
}

#[test]
fn arithmetic_shifts_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| arithmetic_shifts_match_native_on::<Stack>());
    with_large_stack(|| arithmetic_shifts_match_native_on::<Register>());
}

fn arithmetic_shifts_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::ARITHMETIC);

    // shift counts are taken modulo the width, so small positive arguments are
    // the interesting ones here
    check_i64(
        &mut g,
        "arith_shift_rotate",
        g_arithmetic::arith_shift_rotate,
        &[1, 2, 8, 31, 32, 33, 63, 64, 65, 100],
    );
}

#[test]
fn arithmetic_float_edges_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| arithmetic_float_edges_match_native_on::<Stack>());
    with_large_stack(|| arithmetic_float_edges_match_native_on::<Register>());
}

fn arithmetic_float_edges_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::ARITHMETIC);

    check_f64(
        &mut g,
        "arith_float_minmax",
        g_arithmetic::arith_float_minmax,
        ARGS,
    );
    check_f64(
        &mut g,
        "arith_float_rounding",
        g_arithmetic::arith_float_rounding,
        ARGS,
    );
    check_f64(
        &mut g,
        "arith_int_to_float",
        g_arithmetic::arith_int_to_float,
        ARGS,
    );
}

#[test]
fn arithmetic_float_specials_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| arithmetic_float_specials_match_native_on::<Stack>());
    with_large_stack(|| arithmetic_float_specials_match_native_on::<Register>());
}

fn arithmetic_float_specials_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::ARITHMETIC);

    check_i64(
        &mut g,
        "arith_float_specials",
        g_arithmetic::arith_float_specials,
        ARGS,
    );
    check_i64(
        &mut g,
        "arith_float_to_int_saturating",
        g_arithmetic::arith_float_to_int_saturating,
        ARGS,
    );
    check_i64(
        &mut g,
        "arith_reinterpret",
        g_arithmetic::arith_reinterpret,
        ARGS,
    );
    check_i64(
        &mut g,
        "arith_demote_promote",
        g_arithmetic::arith_demote_promote,
        ARGS,
    );
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

#[test]
fn control_flow_loops_and_branches_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| control_flow_loops_and_branches_match_native_on::<Stack>());
    with_large_stack(|| control_flow_loops_and_branches_match_native_on::<Register>());
}

fn control_flow_loops_and_branches_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::CONTROL_FLOW);

    // nested triple loops are O(n^3), so keep the arguments small
    check_i64(
        &mut g,
        "cf_nested_loops",
        g_control_flow::cf_nested_loops,
        &[0, 1, 2, 3, 8, 12],
    );
    check_i64(
        &mut g,
        "cf_br_table_dense",
        g_control_flow::cf_br_table_dense,
        ARGS,
    );
    check_i64(
        &mut g,
        "cf_br_table_sparse",
        g_control_flow::cf_br_table_sparse,
        ARGS,
    );
    check_i64(
        &mut g,
        "cf_if_else_chain",
        g_control_flow::cf_if_else_chain,
        ARGS,
    );
}

#[test]
fn control_flow_exits_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| control_flow_exits_match_native_on::<Stack>());
    with_large_stack(|| control_flow_exits_match_native_on::<Register>());
}

fn control_flow_exits_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::CONTROL_FLOW);

    check_i64(
        &mut g,
        "cf_early_return",
        g_control_flow::cf_early_return,
        ARGS,
    );
    check_i64(
        &mut g,
        "cf_unreachable_regions",
        g_control_flow::cf_unreachable_regions,
        ARGS,
    );
    check_i64(
        &mut g,
        "cf_loop_with_value",
        g_control_flow::cf_loop_with_value,
        ARGS,
    );
    check_i64(
        &mut g,
        "cf_labelled_block",
        g_control_flow::cf_labelled_block,
        ARGS,
    );
    check_i64(
        &mut g,
        "cf_select_and_eqz",
        g_control_flow::cf_select_and_eqz,
        ARGS,
    );
}

// Recursion tests run on an explicitly large stack: a debug build spends ~30 KB
// of native stack per wasm frame, which overflows libtest's ~2 MiB thread and
// aborts the whole binary rather than failing one test.
#[test]
fn control_flow_recursion_matches_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| control_flow_recursion_matches_native_on::<Stack>());
    with_large_stack(|| control_flow_recursion_matches_native_on::<Register>());
}

fn control_flow_recursion_matches_native_on<V: VirtualMachine>() {
    with_large_stack(|| {
        let mut g = Guest::<V>::new(guests::CONTROL_FLOW);

        check_i64(
            &mut g,
            "cf_recursion_direct",
            g_control_flow::cf_recursion_direct,
            ARGS,
        );
        check_i64(
            &mut g,
            "cf_recursion_mutual",
            g_control_flow::cf_recursion_mutual,
            ARGS,
        );
    });
}

#[test]
fn control_flow_iterators_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| control_flow_iterators_match_native_on::<Stack>());
    with_large_stack(|| control_flow_iterators_match_native_on::<Register>());
}

fn control_flow_iterators_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::CONTROL_FLOW);

    check_i64(
        &mut g,
        "cf_iterator_chains",
        g_control_flow::cf_iterator_chains,
        ARGS,
    );
    check_i64(
        &mut g,
        "cf_question_mark",
        g_control_flow::cf_question_mark,
        ARGS,
    );
}

// ---------------------------------------------------------------------------
// Heap
// ---------------------------------------------------------------------------

#[test]
fn heap_collections_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| heap_collections_match_native_on::<Stack>());
    with_large_stack(|| heap_collections_match_native_on::<Register>());
}

fn heap_collections_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::HEAP);

    check_i64(&mut g, "heap_vec_growth", g_heap::heap_vec_growth, ARGS);
    check_i64(&mut g, "heap_sort_dedup", g_heap::heap_sort_dedup, ARGS);
    check_i64(&mut g, "heap_btreemap", g_heap::heap_btreemap, ARGS);
    check_i64(
        &mut g,
        "heap_deque_and_binheap",
        g_heap::heap_deque_and_binheap,
        ARGS,
    );
}

#[test]
fn heap_hashmap_matches_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| heap_hashmap_matches_native_on::<Stack>());
    with_large_stack(|| heap_hashmap_matches_native_on::<Register>());
}

fn heap_hashmap_matches_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::HEAP);

    // the guest reduces order-independently, so this is stable despite the two
    // environments seeding their hashers differently
    check_i64(&mut g, "heap_hashmap", g_heap::heap_hashmap, ARGS);
}

#[test]
fn heap_strings_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| heap_strings_match_native_on::<Stack>());
    with_large_stack(|| heap_strings_match_native_on::<Register>());
}

fn heap_strings_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::HEAP);

    check_i64(&mut g, "heap_strings", g_heap::heap_strings, ARGS);
    check_i64(
        &mut g,
        "heap_nested_collections",
        g_heap::heap_nested_collections,
        ARGS,
    );
}

#[test]
fn heap_pointers_and_drops_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| heap_pointers_and_drops_match_native_on::<Stack>());
    with_large_stack(|| heap_pointers_and_drops_match_native_on::<Register>());
}

fn heap_pointers_and_drops_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::HEAP);

    check_i64(
        &mut g,
        "heap_smart_pointers",
        g_heap::heap_smart_pointers,
        ARGS,
    );
    check_i64(&mut g, "heap_drop_order", g_heap::heap_drop_order, ARGS);
    check_i64(&mut g, "heap_churn", g_heap::heap_churn, ARGS);
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

#[test]
fn memory_loads_and_stores_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| memory_loads_and_stores_match_native_on::<Stack>());
    with_large_stack(|| memory_loads_and_stores_match_native_on::<Register>());
}

fn memory_loads_and_stores_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::MEMORY);

    check_i64(
        &mut g,
        "mem_narrow_load_store",
        g_memory::mem_narrow_load_store,
        ARGS,
    );
    check_i64(&mut g, "mem_endianness", g_memory::mem_endianness, ARGS);
    check_i64(&mut g, "mem_unaligned", g_memory::mem_unaligned, ARGS);
}

#[test]
fn memory_bulk_operations_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| memory_bulk_operations_match_native_on::<Stack>());
    with_large_stack(|| memory_bulk_operations_match_native_on::<Register>());
}

fn memory_bulk_operations_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::MEMORY);

    check_i64(
        &mut g,
        "mem_overlapping_copy",
        g_memory::mem_overlapping_copy,
        ARGS,
    );
    check_i64(&mut g, "mem_fill", g_memory::mem_fill, ARGS);
    check_i64(&mut g, "mem_slice_ops", g_memory::mem_slice_ops, ARGS);
}

#[test]
fn memory_large_buffers_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| memory_large_buffers_match_native_on::<Stack>());
    with_large_stack(|| memory_large_buffers_match_native_on::<Register>());
}

fn memory_large_buffers_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::MEMORY);

    check_i64(
        &mut g,
        "mem_strided_walk",
        g_memory::mem_strided_walk,
        &[1, 100, 10_000],
    );
    check_i64(&mut g, "mem_growth", g_memory::mem_growth, &[1, 2, 8, 16]);
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

#[test]
fn frames_locals_and_calls_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| frames_locals_and_calls_match_native_on::<Stack>());
    with_large_stack(|| frames_locals_and_calls_match_native_on::<Register>());
}

fn frames_locals_and_calls_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::FRAMES);

    check_i64(&mut g, "fr_many_locals", g_frames::fr_many_locals, ARGS);
    check_i64(&mut g, "fr_call_chain", g_frames::fr_call_chain, ARGS);
    check_i64(&mut g, "fr_monomorphised", g_frames::fr_monomorphised, ARGS);
}

#[test]
fn frames_recursion_matches_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| frames_recursion_matches_native_on::<Stack>());
    with_large_stack(|| frames_recursion_matches_native_on::<Register>());
}

fn frames_recursion_matches_native_on<V: VirtualMachine>() {
    with_large_stack(|| {
        let mut g = Guest::<V>::new(guests::FRAMES);

        // Capped by `MAX_TEST_RECURSION`, which is much lower in debug builds —
        // see its docs. Also stays under the default `max_call_stack_depth`.
        let depths: Vec<i32> = [0, 1, 2, 5, 20, 100, 400]
            .into_iter()
            .filter(|d| *d <= MAX_TEST_RECURSION)
            .collect();

        check_i64(
            &mut g,
            "fr_recurse_depth",
            g_frames::fr_recurse_depth,
            &depths,
        );
        check_i64(
            &mut g,
            "fr_recurse_indirect",
            g_frames::fr_recurse_indirect,
            &depths,
        );
        check_i64(&mut g, "fr_wide_frames", g_frames::fr_wide_frames, &depths);
    });
}

#[test]
fn frames_indirect_dispatch_matches_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| frames_indirect_dispatch_matches_native_on::<Stack>());
    with_large_stack(|| frames_indirect_dispatch_matches_native_on::<Register>());
}

fn frames_indirect_dispatch_matches_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::FRAMES);

    check_i64(&mut g, "fr_closure_nest", g_frames::fr_closure_nest, ARGS);
    check_i64(&mut g, "fr_dyn_dispatch", g_frames::fr_dyn_dispatch, ARGS);
}

// ---------------------------------------------------------------------------
// Exotic
// ---------------------------------------------------------------------------

#[test]
fn exotic_enums_and_options_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| exotic_enums_and_options_match_native_on::<Stack>());
    with_large_stack(|| exotic_enums_and_options_match_native_on::<Register>());
}

fn exotic_enums_and_options_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::EXOTIC);

    check_i64(&mut g, "ex_enum_payloads", g_exotic::ex_enum_payloads, ARGS);
    check_i64(
        &mut g,
        "ex_option_result_chains",
        g_exotic::ex_option_result_chains,
        ARGS,
    );
    check_i64(
        &mut g,
        "ex_slice_patterns",
        g_exotic::ex_slice_patterns,
        ARGS,
    );
}

#[test]
fn exotic_text_and_casts_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| exotic_text_and_casts_match_native_on::<Stack>());
    with_large_stack(|| exotic_text_and_casts_match_native_on::<Register>());
}

fn exotic_text_and_casts_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::EXOTIC);

    check_i64(&mut g, "ex_utf8", g_exotic::ex_utf8, ARGS);
    check_i64(&mut g, "ex_casts", g_exotic::ex_casts, ARGS);
}

#[test]
fn exotic_layouts_and_consts_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| exotic_layouts_and_consts_match_native_on::<Stack>());
    with_large_stack(|| exotic_layouts_and_consts_match_native_on::<Register>());
}

fn exotic_layouts_and_consts_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::EXOTIC);

    check_i64(&mut g, "ex_repr_layouts", g_exotic::ex_repr_layouts, ARGS);
    check_i64(
        &mut g,
        "ex_const_and_static",
        g_exotic::ex_const_and_static,
        ARGS,
    );
    check_i64(
        &mut g,
        "ex_shadowing_blocks",
        g_exotic::ex_shadowing_blocks,
        ARGS,
    );
}

#[test]
fn exotic_traits_and_ordering_match_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| exotic_traits_and_ordering_match_native_on::<Stack>());
    with_large_stack(|| exotic_traits_and_ordering_match_native_on::<Register>());
}

fn exotic_traits_and_ordering_match_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::EXOTIC);

    check_i64(&mut g, "ex_traits", g_exotic::ex_traits, ARGS);
    check_i64(&mut g, "ex_custom_ord", g_exotic::ex_custom_ord, ARGS);
}

#[test]
fn exotic_realistic_program_matches_native() {
    // On a large stack for both machines: the register machine's debug frame
    // does not fit a default test thread, which holds only a few dozen.
    with_large_stack(|| exotic_realistic_program_matches_native_on::<Stack>());
    with_large_stack(|| exotic_realistic_program_matches_native_on::<Register>());
}

fn exotic_realistic_program_matches_native_on<V: VirtualMachine>() {
    let mut g = Guest::<V>::new(guests::EXOTIC);

    check_i64(
        &mut g,
        "ex_realistic_program",
        g_exotic::ex_realistic_program,
        ARGS,
    );
}
