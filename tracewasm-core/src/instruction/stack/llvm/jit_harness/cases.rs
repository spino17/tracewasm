//! Cases for the instructions the pass lowers so far.
//!
//! None of them can use a constant or arithmetic, which are not lowered yet, so
//! values come in as parameters and every computation is data movement or control
//! flow. A new instruction's cases go here the same way: a module, an export, and
//! the argument lists that reach each of its edges.

use super::check;

/// Every value type passes through unchanged, including the float bit patterns an
/// accidental conversion would lose: a NaN payload, `-0.0`, a subnormal.
#[test]
fn locals_carry_every_type_through_unchanged() {
    check(
        r#"(module
            (func (export "f") (param i32 i64 f32 f64) (result f64 f32 i64 i32)
                local.get 3
                local.get 2
                local.get 1
                local.get 0))"#,
        "f",
        &[
            vals![7, -1i64, 1.5f32, -0.0f64],
            vals![
                i32::MIN,
                i64::MAX,
                f32::from_bits(0x7fc0_beef),
                f64::MIN_POSITIVE / 2.0
            ],
            vals![-1, i64::MIN, -0.0f32, f64::NAN],
        ],
    );
}

/// `local.tee` leaves its operand on the stack and `local.set` consumes it, so the
/// result is whichever parameter the chain ends on.
#[test]
fn tee_and_set_move_a_value_between_locals() {
    check(
        r#"(module
            (func (export "f") (param i32 i32) (result i32) (local i32)
                local.get 0
                local.tee 2
                local.set 1
                local.get 2
                local.get 1
                local.set 0
                local.set 1
                local.get 1))"#,
        "f",
        &[vals![1, 2], vals![-5, 9], vals![i32::MAX, i32::MIN]],
    );
}

/// Reads return the initial values, and writes land in the slots the host reads
/// back — compared global by global against the interpreter's after the call.
#[test]
fn globals_are_read_and_written_through_the_context() {
    check(
        r#"(module
            (global $i (mut i32) (i32.const -7))
            (global $l (mut i64) (i64.const 0x1234_5678_9abc_def0))
            (global $s (mut f32) (f32.const -0.0))
            (global $d (mut f64) (f64.const nan:0x4_0000_dead))
            (global $k i32 (i32.const 42))
            (func (export "f") (param i32 i64 f32 f64) (result i32 i64 f32 f64 i32)
                global.get $i
                global.get $l
                global.get $s
                global.get $d
                global.get $k
                local.get 0
                global.set $i
                local.get 1
                global.set $l
                local.get 2
                global.set $s
                local.get 3
                global.set $d))"#,
        "f",
        &[
            vals![1, 2i64, 3.0f32, 4.0f64],
            vals![-1, -1i64, f32::NAN, -0.0f64],
        ],
    );
}

/// `br_if` with a value: taken, it carries the block's result out; not taken, the
/// value stays on the stack for what follows.
#[test]
fn br_if_carries_a_value_out_of_a_block() {
    check(
        r#"(module
            (func (export "f") (param i32 i64 i64) (result i64)
                (block (result i64)
                    local.get 1
                    local.get 0
                    br_if 0
                    local.set 1
                    local.get 2)))"#,
        "f",
        &[
            vals![1, 10i64, 20i64],
            vals![0, 10i64, 20i64],
            vals![-1, 10i64, 20i64],
        ],
    );
}

/// A loop taken round its back edge from inside an `if`, at most once: the counter
/// is reset from a local that is never written, so it reads zero.
#[test]
fn a_loop_goes_round_its_back_edge() {
    check(
        r#"(module
            (func (export "f") (param i32 i32 i32) (result i32) (local $zero i32)
                (loop $again
                    local.get 0
                    (if
                        (then
                            local.get $zero
                            local.set 0
                            local.get 2
                            local.set 1
                            br $again)))
                local.get 1))"#,
        "f",
        &[vals![0, 1, 2], vals![1, 1, 2], vals![i32::MIN, 1, 2]],
    );
}

/// Every arm of a `br_table`, the default, and indices a signed compare would
/// mistake for in range.
#[test]
fn br_table_picks_an_arm_or_the_default() {
    check(
        r#"(module
            (func (export "f") (param i32 i32 i32 i32) (result i32)
                (block
                    (block
                        (block
                            local.get 0
                            br_table 0 1 2)
                        local.get 1
                        return)
                    local.get 2
                    return)
                local.get 3))"#,
        "f",
        &[
            vals![0, 10, 20, 30],
            vals![1, 10, 20, 30],
            vals![2, 10, 20, 30],
            vals![3, 10, 20, 30],
            vals![-1, 10, 20, 30],
            vals![i32::MIN, 10, 20, 30],
        ],
    );
}

/// Both arms of an `if` with a result, and a condition that is non-zero without
/// being one.
#[test]
fn if_else_yields_the_taken_arms_value() {
    check(
        r#"(module
            (func (export "f") (param i32 f64 f64) (result f64)
                local.get 0
                (if (result f64)
                    (then local.get 1)
                    (else local.get 2))))"#,
        "f",
        &[
            vals![1, 1.5f64, -2.5f64],
            vals![0, 1.5f64, -2.5f64],
            vals![-8, 1.5f64, -2.5f64],
        ],
    );
}

/// A direct call to a multi-value function, whose results come back in order.
#[test]
fn a_call_returns_multiple_values_in_order() {
    check(
        r#"(module
            (func $swap (param i32 i64) (result i64 i32)
                local.get 1
                local.get 0)
            (func (export "f") (param i32 i64) (result i64 i32)
                (call $swap (local.get 0) (local.get 1))))"#,
        "f",
        &[vals![1, 2i64], vals![-1, i64::MIN]],
    );
}

/// A hit through each table, and each of the three traps — which the interpreter
/// raises too, so these run through the child process.
///
/// Two `check`s in one test on purpose: a trap in the second makes the child skip
/// the first, which is the path that has to find the right call.
#[test]
fn call_indirect_hits_or_traps_like_the_interpreter() {
    const WAT: &str = r#"(module
            (type $bin (func (param i32 i32) (result i32)))
            (type $un (func (param i32) (result i32)))
            (func $first (type $bin) local.get 0)
            (func $second (type $bin) local.get 1)
            (func $unary (type $un) local.get 0)
            (table $t0 5 funcref)
            (table $t1 2 funcref)
            (elem (table $t0) (i32.const 0) func $first $second $unary)
            (elem (table $t1) (i32.const 0) func $second $first)
            (func (export "f") (param i32 i32 i32) (result i32)
                local.get 0
                local.get 1
                local.get 2
                call_indirect $t0 (type $bin))
            (func (export "g") (param i32 i32 i32) (result i32)
                local.get 0
                local.get 1
                local.get 2
                call_indirect $t1 (type $bin)))"#;

    check(
        WAT,
        "f",
        &[
            vals![10, 20, 0],
            vals![10, 20, 1],
            vals![10, 20, 2],        // wrong type
            vals![10, 20, 3],        // null slot
            vals![10, 20, 5],        // one past the end
            vals![10, 20, -1],       // out of bounds, unsigned
            vals![10, 20, i32::MIN], // likewise
        ],
    );

    // The second table: its own entries, and its own, shorter, length.
    check(
        WAT,
        "g",
        &[vals![10, 20, 0], vals![10, 20, 1], vals![10, 20, 2]],
    );
}
