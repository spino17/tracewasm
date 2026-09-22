//! Unnamed registers are numbered by **position in the printed function**, not by
//! when they were created.
//!
//! LLVM requires the numbers to ascend down the page, and reports
//! *"instruction expected to be numbered '%2' or greater"* when they do not. A
//! frontend does not build in that order — it creates the block an `if` merges into
//! before either arm exists, and fills it last — so numbering at construction hands
//! `%0` to something printed after `%1`.
//!
//! Assignment therefore happens in [`Builder::build`], once every definition's
//! position is settled. These build deliberately out of order and check that the
//! output does not care.
//!
//! Through the public API only, like `frontend_loop`: `Value::from_register` is
//! crate-private, so a downstream crate can only obtain registers the three ways
//! exercised here — parameters, instruction results, and phis.

use tracewasm_llvm::{
    cfg::{
        context::Context,
        emit::IREmitter,
        module::{DataLayout, Triple},
    },
    instruction::{
        IBinOp, ICond,
        cursor::{OperandTy, RegName},
    },
};

fn ctx() -> Context {
    Context::new(
        Triple::new("arm64".into(), "apple".into(), "macosx".into(), None),
        DataLayout::default(),
    )
}

/// The `%N` in each `%N = ...`, in the order they are printed.
fn defined_numbers(ir: &str) -> Vec<u32> {
    ir.lines()
        .filter_map(|line| line.trim().strip_prefix('%'))
        .filter_map(|rest| rest.split_once(" = "))
        .filter_map(|(name, _)| name.parse().ok())
        .collect()
}

/// Every block filled in the *reverse* of the order it will be printed in, with
/// nothing named — the shape that motivated moving assignment into `build`.
///
/// ```text
/// define i32 @f(i32 %0, i32 %1) {
/// entry:  %2 = icmp sgt i32 %0, 0
///         br i1 %2, label %then, label %else
/// then:   %3 = mul i32 %0, %1
///         br label %merge
/// else:   %4 = sub i32 %0, %1
///         br label %merge
/// merge:  %5 = phi i32 [ %4, %else ], [ %3, %then ]
///         %6 = add i32 %5, %5
///         ret i32 %6
/// }
/// ```
///
/// Verified against the real assembler: `llvm-as` accepts this and
/// `opt -passes=verify` is clean.
#[test]
fn numbering_follows_the_printed_order_not_the_build_order() {
    let mut builder = ctx().builder();
    let i32_ty = builder.i32_ty();

    // Both parameters unnamed, so they take the first numbers.
    let f = builder
        .define_function(
            "f",
            &[(i32_ty, RegName::Unnamed), (i32_ty, RegName::Unnamed)],
            i32_ty,
        )
        .unwrap();

    // All four blocks up front. `merge` exists long before either arm does.
    let entry = f.add_basic_block("entry", &mut builder).unwrap();
    let then_b = f.add_basic_block("then", &mut builder).unwrap();
    let else_b = f.add_basic_block("else", &mut builder).unwrap();
    let merge = f.add_basic_block("merge", &mut builder).unwrap();

    let p0 = f.nth_param(0, &builder).unwrap();
    let p1 = f.nth_param(1, &builder).unwrap();

    // Filled back to front: merge, else, then, entry.
    let mut in_merge = builder.cursor_at_block(merge);
    let (phi, merged) = in_merge
        .build_phi(&[], OperandTy::Asserted(i32_ty), RegName::Unnamed)
        .unwrap();
    let doubled = in_merge
        .build_ibinop(
            IBinOp::Add,
            OperandTy::Inferred,
            merged,
            merged,
            RegName::Unnamed,
        )
        .unwrap();
    in_merge
        .build_ret(Some(doubled), OperandTy::Inferred)
        .unwrap();

    let mut in_else = builder.cursor_at_block(else_b);
    let e = in_else
        .build_ibinop(IBinOp::Sub, OperandTy::Inferred, p0, p1, RegName::Unnamed)
        .unwrap();
    phi.add_branch((else_b, e), &mut in_else).unwrap();
    in_else.build_unconditional_br(merge).unwrap();

    let mut in_then = builder.cursor_at_block(then_b);
    let t = in_then
        .build_ibinop(IBinOp::Mul, OperandTy::Inferred, p0, p1, RegName::Unnamed)
        .unwrap();
    phi.add_branch((then_b, t), &mut in_then).unwrap();
    in_then.build_unconditional_br(merge).unwrap();

    let mut in_entry = builder.cursor_at_block(entry);
    let zero = in_entry.const_value(0i32, OperandTy::Inferred).unwrap();
    let cond = in_entry
        .build_icmp(ICond::Sgt, OperandTy::Inferred, p0, zero, RegName::Unnamed)
        .unwrap();
    in_entry.build_conditional_br(cond, then_b, else_b).unwrap();

    let ir = IREmitter::emit(builder.build()).unwrap();

    assert!(
        ir.contains("define i32 @f(i32 %0, i32 %1) {"),
        "unnamed parameters take the first numbers\n{ir}"
    );

    // The invariant itself, independent of the exact instructions: the numbers a
    // reader meets going down the page are 0, 1, 2, … with none out of sequence.
    // Creation order was merge → else → then → entry, so anything keyed off that
    // would come out shuffled here.
    let defined = defined_numbers(&ir);

    assert_eq!(
        defined,
        (2..=6).collect::<Vec<_>>(),
        "instruction numbers must ascend in printed order, continuing past the \
         parameters' 0 and 1\n{ir}"
    );

    assert!(
        ir.contains("%5 = phi i32 [ %4, %else ], [ %3, %then ]"),
        "the phi is numbered where it is printed, though it was built first\n{ir}"
    );

    // The placeholder a register carries until `build` names it is not a legal LLVM
    // local, so a leak would be caught by `llvm-as` rather than passing as a name —
    // but it should not reach the output at all.
    assert!(
        !ir.contains("unnamed"),
        "the pre-numbering placeholder leaked into the IR\n{ir}"
    );
}

/// A named register keeps its name and consumes no number, so the unnamed ones
/// around it are numbered as if it were not there.
///
/// `llvm-as` is explicit about this: in `define i32 @f(i32 %n, i32)` the unnamed
/// parameter is `%0`, and numbering an instruction `%0` is then refused.
#[test]
fn named_registers_consume_no_numbers() {
    let mut builder = ctx().builder();
    let i32_ty = builder.i32_ty();

    let f = builder
        .define_function(
            "f",
            &[(i32_ty, "n".into()), (i32_ty, RegName::Unnamed)],
            i32_ty,
        )
        .unwrap();

    let entry = f.add_basic_block("entry", &mut builder).unwrap();
    let p0 = f.nth_param(0, &builder).unwrap();
    let p1 = f.nth_param(1, &builder).unwrap();

    let mut cursor = builder.cursor_at_block(entry);

    // Named, unnamed, named, unnamed — only the unnamed ones draw from the counter.
    let a = cursor
        .build_ibinop(IBinOp::Add, OperandTy::Inferred, p0, p1, "sum".into())
        .unwrap();
    let b = cursor
        .build_ibinop(IBinOp::Mul, OperandTy::Inferred, a, p1, RegName::Unnamed)
        .unwrap();
    let c = cursor
        .build_ibinop(IBinOp::Sub, OperandTy::Inferred, b, p0, "diff".into())
        .unwrap();
    let d = cursor
        .build_ibinop(IBinOp::Add, OperandTy::Inferred, c, c, RegName::Unnamed)
        .unwrap();

    cursor.build_ret(Some(d), OperandTy::Inferred).unwrap();

    let ir = IREmitter::emit(builder.build()).unwrap();

    assert!(
        ir.contains("define i32 @f(i32 %n, i32 %0) {"),
        "the named parameter keeps its name; only the unnamed one is numbered\n{ir}"
    );
    assert!(ir.contains("%sum = add"), "{ir}");
    assert!(ir.contains("%diff = sub"), "{ir}");

    assert_eq!(
        defined_numbers(&ir),
        vec![1, 2],
        "the two unnamed instructions continue from the parameter's 0; the two named \
         ones in between take nothing\n{ir}"
    );
}

/// A declaration has no registers to number.
///
/// `declare` takes types alone — `declare i32 @g(i32)` names nothing — so imports
/// cannot carry a register that the numbering walk (which only visits *defined*
/// functions) would miss. This pins that, because the walk's reach is what makes the
/// placeholder unable to leak.
#[test]
fn a_declaration_has_nothing_to_number() {
    let mut builder = ctx().builder();
    let i32_ty = builder.i32_ty();

    builder
        .declare_function("g", &[i32_ty, i32_ty], i32_ty)
        .unwrap();

    let f = builder
        .define_function("f", &[(i32_ty, RegName::Unnamed)], i32_ty)
        .unwrap();
    let entry = f.add_basic_block("entry", &mut builder).unwrap();
    let p0 = f.nth_param(0, &builder).unwrap();

    let mut cursor = builder.cursor_at_block(entry);
    let doubled = cursor
        .build_ibinop(IBinOp::Add, OperandTy::Inferred, p0, p0, RegName::Unnamed)
        .unwrap();
    cursor
        .build_ret(Some(doubled), OperandTy::Inferred)
        .unwrap();

    let ir = IREmitter::emit(builder.build()).unwrap();

    assert!(
        ir.contains("declare i32 @g(i32, i32)"),
        "a declaration spells types only\n{ir}"
    );
    assert!(
        !ir.contains("unnamed"),
        "the declaration must not leave a placeholder behind\n{ir}"
    );

    // The defined function still starts from zero: the declaration took no numbers.
    assert!(ir.contains("define i32 @f(i32 %0) {"), "{ir}");
    assert_eq!(defined_numbers(&ir), vec![1], "{ir}");
}
