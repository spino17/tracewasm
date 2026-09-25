//! The wasm → LLVM lowering, checked three ways.
//!
//! Every case in [`CASES`] is (1) lowered, (2) assembled and put through LLVM's own
//! verifier, and (3) executed under `lli` and compared against the **interpreter**
//! running the same module. The third is what makes this a lowering suite rather
//! than a codegen snapshot: the interpreter is the reference implementation, so
//! "the two machines agree" is the property worth holding.
//!
//! The tests after the table assert on the shape of the IR, for the properties
//! neither verification nor execution can see — a result paired with the wrong phi
//! when both share a type, dead code kept alive, a block left unreachable.
//!
//! # Scope
//!
//! Only what the pass lowers today: locals, `if`/`else`, `loop`, `br`, `return`.
//! There is no arithmetic yet, so a case computes by *selecting* among its
//! arguments rather than by combining them — which is enough to tell control flow
//! that works from control flow that does not.

mod lowering_harness;

use lowering_harness::*;

/// The differential table.
///
/// Arguments are chosen to reach both sides of every branch; a case that only ever
/// took one arm would pass with the other arm miscompiled.
const CASES: &[Case] = &[
    Case {
        name: "identity",
        wat: r#"(module (func (export "f") (param i32) (result i32) (local.get 0)))"#,
        calls: &[&[0], &[7], &[-1]],
        interpret: i32_to_i32,
    },
    Case {
        name: "if_else_selects",
        wat: r#"(module (func (export "f") (param i32) (param i32) (result i32)
            (if (result i32) (local.get 0) (then (local.get 1)) (else (local.get 0)))))"#,
        calls: &[&[1, 42], &[0, 42], &[-5, 9]],
        interpret: i32x2_to_i32,
    },
    Case {
        name: "if_without_else",
        // `[i32] -> [i32]`: the false edge carries the block's param through as its
        // result, which is the edge the pass has to record itself.
        wat: r#"(module (func (export "f") (param i32) (param i32) (result i32)
            (local.get 1)
            (if (param i32) (result i32) (local.get 0)
              (then (local.set 1) (local.get 0)))))"#,
        calls: &[&[1, 42], &[0, 42]],
        interpret: i32x2_to_i32,
    },
    Case {
        name: "local_set_then_get",
        wat: r#"(module (func (export "f") (param i32) (param i32) (result i32) (local i32)
            (local.set 2 (local.get 1))
            (local.get 2)))"#,
        calls: &[&[0, 11], &[3, -4]],
        interpret: i32x2_to_i32,
    },
    Case {
        name: "local_tee_keeps_the_value",
        wat: r#"(module (func (export "f") (param i32) (param i32) (result i32) (local i32)
            (local.tee 2 (local.get 1))))"#,
        calls: &[&[0, 11], &[0, -4]],
        interpret: i32x2_to_i32,
    },
    Case {
        name: "early_return",
        wat: r#"(module (func (export "f") (param i32) (param i32) (result i32)
            (if (local.get 0) (then (local.get 1) (return)))
            (local.get 0)))"#,
        calls: &[&[1, 42], &[0, 42]],
        interpret: i32x2_to_i32,
    },
    Case {
        name: "return_is_the_whole_body",
        // The shape that used to emit no `ret` at all: the body's last instruction
        // is `return`, so it skips its own `end`.
        wat: r#"(module (func (export "f") (param i32) (result i32)
            (local.get 0) (return)))"#,
        calls: &[&[0], &[7]],
        interpret: i32_to_i32,
    },
    Case {
        name: "nested_if",
        wat: r#"(module (func (export "f") (param i32) (param i32) (result i32)
            (if (result i32) (local.get 0)
              (then (if (result i32) (local.get 1)
                      (then (local.get 1))
                      (else (local.get 0))))
              (else (local.get 1)))))"#,
        calls: &[&[1, 5], &[1, 0], &[0, 9]],
        interpret: i32x2_to_i32,
    },
    Case {
        name: "loop_falls_through",
        wat: r#"(module (func (export "f") (param i32) (result i32)
            (loop (result i32) (local.get 0))))"#,
        calls: &[&[0], &[7]],
        interpret: i32_to_i32,
    },
    Case {
        name: "loop_with_a_taken_back_edge",
        // Terminates because the back-edge overwrites local 0 with local 1, and the
        // callers below always pass 0 there — a second iteration always falls out.
        wat: r#"(module (func (export "f") (param i32) (param i32) (result i32)
            (loop
              (if (local.get 0)
                (then (local.set 0 (local.get 1)) (br 1))))
            (local.get 0)))"#,
        calls: &[&[1, 0], &[0, 0]],
        interpret: i32x2_to_i32,
    },
    Case {
        name: "return_from_inside_a_loop",
        wat: r#"(module (func (export "f") (param i32) (param i32) (result i32)
            (loop (result i32)
              (if (local.get 0) (then (local.get 1) (return)))
              (local.get 0))))"#,
        calls: &[&[1, 42], &[0, 42]],
        interpret: i32x2_to_i32,
    },
    Case {
        name: "multi_value_return",
        // Two results of *different* types, so a swapped pair would not even
        // type-check — and two that differ in value, so a same-typed swap shows up.
        wat: r#"(module (func (export "f") (param i32) (param i32) (result i32 i64)
            (local i64)
            (local.get 0)
            (local.get 2)))"#,
        calls: &[&[3, 0], &[-1, 0]],
        interpret: i32x2_to_i32i64,
    },
    Case {
        name: "void_function",
        wat: r#"(module (func (export "f") (param i32) (local.set 0 (local.get 0))))"#,
        calls: &[&[0], &[5]],
        interpret: i32_to_void,
    },
    Case {
        name: "void_with_early_return",
        wat: r#"(module (func (export "f") (param i32)
            (if (local.get 0) (then (return)))
            (local.set 0 (local.get 0))))"#,
        calls: &[&[0], &[1]],
        interpret: i32_to_void,
    },
];

/// Every case lowers to IR that verifies and agrees with the interpreter.
///
/// One test rather than one per case so a failure names the case in its message and
/// the table stays the only thing to edit when the pass grows.
#[test]
fn lowered_code_verifies_and_matches_the_interpreter() {
    for case in CASES {
        // Caught and re-raised so the message names the case: an unimplemented
        // operator panics deep inside the pass, where nothing knows which wat it
        // came from.
        let result = std::panic::catch_unwind(|| check(case));

        if let Err(payload) = result {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic>".to_string());

            panic!("case `{}` failed: {msg}", case.name);
        }
    }
}

// ---------------------------------------------------------------------------
// What the IR has to *look* like
//
// Verification and execution between them miss a whole class of defect: IR that
// assembles, returns the right answer, and is still wrong in a way that bites
// later. These assert on the text.

/// Locals are `alloca`s **in the entry block**, params stored on the way in and
/// declared locals zeroed.
///
/// The entry block is not a detail: `mem2reg` promotes `alloca`s only there, so a
/// local emitted into any other block stays in memory for the life of the function.
/// The fold at the end is what proves it — `ret i32 %param0` only appears if every
/// local was promoted away.
#[test]
fn locals_are_entry_block_allocas_that_promote_away() {
    let ir = lower(
        r#"(module (func (export "f") (param i32) (param i64) (result i32) (local f32) (local i64)
            (local.get 0)))"#,
    );

    let entry: Vec<String> = ir
        .lines()
        .skip_while(|l| l.trim() != "entry:")
        .skip(1)
        .take_while(|l| !l.trim_end().ends_with(':'))
        .map(|l| l.trim().to_string())
        .collect();

    assert_eq!(
        entry.iter().filter(|l| l.contains("alloca")).count(),
        4,
        "one alloca per local — two params and two declared — all in `entry`\n{ir}"
    );

    assert!(
        entry.iter().any(|l| l.contains("store i32 %param0")),
        "a parameter is stored into its slot on the way in\n{ir}"
    );
    assert!(
        entry.iter().any(|l| l.contains("store float 0")),
        "a declared local is zeroed at its own type\n{ir}"
    );
    assert!(
        entry.iter().any(|l| l.contains("store i64 0")),
        "a declared local is zeroed at its own type\n{ir}"
    );

    // The runtime pointer is a parameter but not a local: it must not get a slot.
    assert!(
        !entry.iter().any(|l| l.contains("alloca ptr")),
        "the appended runtime pointer is not a wasm local and takes no slot\n{ir}"
    );

    if let Some(opt) = optimise("locals", &ir, "mem2reg,instcombine,simplifycfg") {
        assert_eq!(
            function_body(&opt),
            ["entry:", "ret i32 %param0"],
            "every local should promote away, leaving one block that returns a \
             parameter\n{opt}"
        );
    }
}

/// Unreachable operators never reach the IR at all.
///
/// The lowering drops them rather than marking them, so there is nothing after a
/// `br` for the pass to skip. Execution cannot see this — dead code does not run
/// either way — and neither can the verifier.
#[test]
fn dead_code_is_absent_rather_than_unreachable() {
    let ir = lower(
        r#"(module (func (export "f") (param i32) (result i32)
            (local.get 0)
            (return)
            (local.set 0 (local.get 0))
            (local.get 0)))"#,
    );

    assert_verifies("dead_code", &ir);

    // The dead `local.set`/`local.get` would each be a `store`/`load`. Only the
    // entry prologue's stores and the one live `local.get` should survive.
    assert_eq!(
        ir.matches("store i32").count(),
        1,
        "only the parameter prologue stores; the dead `local.set` is gone\n{ir}"
    );
    assert_eq!(
        ir.matches("load i32").count(),
        1,
        "only the live `local.get` loads\n{ir}"
    );
}

/// A `loop` gets two blocks, not one: a header the back-edge targets, and an end
/// that falling out of the body lands in.
///
/// They carry different aritys — the header's phis take the loop's *params*, the
/// end's its *results* — which is why one block could not serve both.
#[test]
fn a_loop_has_a_separate_header_and_end() {
    let ir = lower(
        r#"(module (func (export "f") (param i32) (param i32) (result i32)
            (loop
              (if (local.get 0)
                (then (local.set 0 (local.get 1)) (br 1))))
            (local.get 0)))"#,
    );

    assert_verifies("loop_blocks", &ir);

    let labels: Vec<&str> = ir
        .lines()
        .filter_map(|l| l.trim().strip_suffix(':'))
        .collect();

    assert!(
        labels
            .iter()
            .any(|l| l.starts_with("loop") && !l.ends_with("_end")),
        "the loop has a header block\n{ir}"
    );
    assert!(
        labels
            .iter()
            .any(|l| l.starts_with("loop") && l.ends_with("_end")),
        "and a separate end block\n{ir}"
    );

    // The back-edge is a real edge: something branches back to the header.
    let header = labels
        .iter()
        .find(|l| l.starts_with("loop") && !l.ends_with("_end"))
        .expect("a header");

    assert!(
        ir.matches(&format!("br label %{header}")).count() >= 2,
        "the header is entered from outside *and* branched back to\n{ir}"
    );
}

/// Multi-value results are assembled field for field in declaration order.
///
/// Execution catches a swap only when the two values differ; this catches it
/// always, which matters because the phi order and the field order are established
/// in two different places.
#[test]
fn multi_value_results_pair_field_with_phi() {
    let ir = lower(
        r#"(module (func (export "f") (param i32) (param i64) (result i32 i64)
            (local.get 0)
            (local.get 1)))"#,
    );

    assert_verifies("multi_value", &ir);

    let body = ir.replace('\n', " ");
    let field0 = body.find("result0_ptr").expect("field 0 is stored");
    let field1 = body.find("result1_ptr").expect("field 1 is stored");

    assert!(
        field0 < field1,
        "fields are stored in declaration order\n{ir}"
    );

    // Field 0 takes the *deepest* result, which is the first phi.
    assert!(
        ir.contains("getelementptr inbounds { i32, i64 }, ptr %fn_return_ptr, i32 0, i32 0"),
        "field 0 is reached with a constant i32 index\n{ir}"
    );
    assert!(
        ir.contains("store i32 %end") && ir.contains("store i64 %end"),
        "each field is stored from the phi of its own type\n{ir}"
    );

    if let Some(opt) = optimise("multi_value", &ir, "mem2reg,sroa,instcombine,simplifycfg") {
        assert!(
            opt.contains("insertvalue") && !opt.contains("alloca"),
            "the struct should end up built in registers, not memory\n{opt}"
        );
    }
}

// ---------------------------------------------------------------------------
// Regressions
//
// One per defect this lowering has actually had. Each reproduces the exact shape
// that was wrong, so a repeat is caught by the case that first found it.

/// A body whose last instruction is `return` still gets a `ret`.
///
/// It used to get none: `return` resumes past the function's own `end`, so a `ret`
/// emitted by the `End` arm was skipped exactly when the function always returned
/// early. It is emitted by `compile_func` now, which runs once however the body
/// finished.
#[test]
fn regression_a_body_ending_in_return_still_returns() {
    let ir =
        lower(r#"(module (func (export "f") (param i32) (result i32) (local.get 0) (return)))"#);

    assert!(ir.contains("ret i32"), "the function returns\n{ir}");
    assert_verifies("return_at_end", &ir);
}

/// A `br` to an *outer* label does not strand the label it leaves.
///
/// The inner label's end used to keep phis nothing ever fed, which LLVM tolerates
/// only while the block has no predecessors at all.
#[test]
fn regression_br_to_an_outer_label_leaves_no_empty_phi() {
    let ir = lower(
        r#"(module (func (export "f") (param i32) (result i32)
            (if (result i32) (local.get 0)
              (then (if (local.get 0) (then (local.get 0) (br 1)))
                    (local.get 0))
              (else (local.get 0)))))"#,
    );

    assert_verifies("br_outer", &ir);

    // `phi <ty>` with nothing after it is the empty-phi shape.
    for line in ir.lines() {
        let line = line.trim();

        if let Some(rest) = line.split_once("= phi ") {
            assert!(
                rest.1.contains('['),
                "a phi was left with no incoming values\n{ir}"
            );
        }
    }
}

/// An `if` with no `else` records its false edge.
///
/// The edge jumps straight to the `end` carrying the block's params through as its
/// results, and nothing else records it — so the phi there used to be short a
/// predecessor, which only shows up when the label carries a value.
#[test]
fn regression_if_without_else_records_the_false_edge() {
    let ir = lower(
        r#"(module (func (export "f") (param i32) (param i32) (result i32)
            (local.get 1)
            (if (param i32) (result i32) (local.get 0)
              (then (local.set 1) (local.get 0)))))"#,
    );

    assert_verifies("if_no_else", &ir);

    // The *if*'s end, not the function's — the latter is printed first, because its
    // block is created first, and legitimately has the single incoming edge.
    let if_end_phi = ir
        .lines()
        .skip_while(|l| !l.trim().starts_with("if") || !l.trim().ends_with("_end:"))
        .find(|l| l.contains("= phi"))
        .expect("the if's end joins its two paths");

    assert_eq!(
        if_end_phi.matches('[').count(),
        2,
        "both the then-arm and the false edge feed the if's end\n{ir}"
    );
}

/// The first *declared* local is not the appended runtime pointer.
///
/// `FuncBody::locals` runs params-then-declared, while the LLVM parameter list has
/// the runtime pointer appended — so indexing one by the length of the other used
/// to hand the first declared local a `ptr` holding the instance.
#[test]
fn regression_the_first_declared_local_is_not_the_runtime_pointer() {
    let ir =
        lower(r#"(module (func (export "f") (param i32) (result i32) (local i32) (local.get 1)))"#);

    assert_verifies("declared_local", &ir);

    assert!(
        !ir.contains("alloca ptr"),
        "no local is given the runtime pointer's type\n{ir}"
    );
    assert!(
        ir.contains("store i32 0"),
        "the declared local is zeroed, not seeded from a parameter\n{ir}"
    );
}

// ---------------------------------------------------------------------------
// `block`
//
// The plainest label there is: entering one does nothing at runtime, so the only
// thing it needs is somewhere for a branch to land. It still has to register that
// `end` — the arm that forgot to was how `loop` and `block` each failed first.

/// Every shape of `block` lowers, verifies, and agrees with the interpreter.
#[test]
fn blocks_lower_and_match_the_interpreter() {
    const BLOCK_CASES: &[Case] = &[
        Case {
            name: "block_falls_out",
            wat: r#"(module (func (export "f") (param i32) (result i32)
                (block (result i32) (local.get 0))))"#,
            calls: &[&[0], &[7]],
            interpret: i32_to_i32,
        },
        Case {
            name: "br_out_of_a_block",
            wat: r#"(module (func (export "f") (param i32) (result i32)
                (block (result i32) (local.get 0) (br 0))))"#,
            calls: &[&[0], &[7]],
            interpret: i32_to_i32,
        },
        Case {
            name: "br_to_an_outer_block",
            wat: r#"(module (func (export "f") (param i32) (result i32)
                (block (result i32)
                  (block (result i32) (local.get 0) (br 1)))))"#,
            calls: &[&[0], &[-3]],
            interpret: i32_to_i32,
        },
        Case {
            name: "block_wrapping_an_if",
            wat: r#"(module (func (export "f") (param i32) (param i32) (result i32)
                (block (result i32)
                  (if (result i32) (local.get 0)
                    (then (local.get 1) (br 1))
                    (else (local.get 0))))))"#,
            calls: &[&[1, 42], &[0, 42]],
            interpret: i32x2_to_i32,
        },
    ];

    for case in BLOCK_CASES {
        let result = std::panic::catch_unwind(|| check(case));

        if let Err(payload) = result {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic>".to_string());

            panic!("case `{}` failed: {msg}", case.name);
        }
    }
}
