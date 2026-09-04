//! Builds a loop the way a compiler frontend would, through the public API only.
//!
//! The unit tests reach into `pub(crate)` helpers; this one cannot, so it exercises
//! exactly the surface a downstream crate has. The shape is the one that stresses a
//! cursor-based builder hardest: blocks are created before they are filled, values
//! cross between them, and a phi names a block whose contents do not exist yet.

use tracewasm_llvm::{
    cfg::{
        context::Context,
        emit::IREmitter,
        module::{DataLayout, Triple},
    },
    instruction::{IArithmeticOp, ICond, cursor::OperandTy},
};

/// ```text
/// define i32 @sum_to(i32 %n) {
/// entry:  br label %loop
/// loop:   %acc  = phi i32 [ 0, %entry ], [ %next, %loop ]
///         %i    = phi i32 [ %n, %entry ], [ %dec, %loop ]
///         %next = add i32 %acc, %i
///         %dec  = sub i32 %i, 1
///         %cond = icmp sgt i32 %dec, 0
///         br i1 %cond, label %loop, label %exit
/// exit:   ret i32 %next
/// }
/// ```
#[test]
fn a_frontend_can_build_a_loop_with_a_back_edge() {
    let mut ctx = Context::new(
        Triple::new(
            "arm64".to_string(),
            "apple".to_string(),
            "macosx".to_string(),
            None,
        ),
        DataLayout::default(),
    );

    let i32_ty = ctx.i32_ty();
    let mut builder = ctx.builder();

    let f = builder
        .define_function(
            "sum_to".to_string(),
            &[(i32_ty, Some("n".to_string()))],
            i32_ty,
        )
        .unwrap();

    // All three blocks up front, before any is filled — a frontend knows the shape of
    // a loop before it knows the instructions.
    let entry = f
        .add_basic_block("entry".to_string(), &mut builder)
        .unwrap();
    let loop_b = f.add_basic_block("loop".to_string(), &mut builder).unwrap();
    let exit = f.add_basic_block("exit".to_string(), &mut builder).unwrap();

    let n = f.nth_param(0, &builder).unwrap().clone();

    builder
        .cursor_at_block(entry)
        .build_unconditional_br(loop_b)
        .unwrap();

    let mut in_loop = builder.cursor_at_block(loop_b);

    // Constants are made through the cursor, which derefs to the context.
    let zero = in_loop.const_value(0i32, OperandTy::Inferred).unwrap();
    let one = in_loop.const_value(1i32, OperandTy::Inferred).unwrap();

    // Both phis are built knowing only the `entry` edge. The `loop` edge carries
    // values that do not exist yet, so it is added further down.
    let (acc_phi, acc) = in_loop
        .build_phi(&[(entry, zero.clone())], "acc".into())
        .unwrap();
    let (i_phi, i) = in_loop.build_phi(&[(entry, n)], "i".into()).unwrap();

    let next = in_loop
        .build_iarithmetic(
            IArithmeticOp::Add,
            OperandTy::Inferred,
            &acc,
            &i,
            "next".into(),
        )
        .unwrap();

    let dec = in_loop
        .build_iarithmetic(
            IArithmeticOp::Sub,
            OperandTy::Inferred,
            &i,
            &one,
            "dec".into(),
        )
        .unwrap();

    let cond = in_loop
        .build_icmp(ICond::Sgt, OperandTy::Inferred, &dec, &zero, "cond".into())
        .unwrap();

    // Close the back edge now that the values exist.
    acc_phi
        .add_branch((loop_b, next.clone()), &mut in_loop)
        .unwrap();
    i_phi.add_branch((loop_b, dec), &mut in_loop).unwrap();

    in_loop.build_conditional_br(cond, loop_b, exit).unwrap();

    builder
        .cursor_at_block(exit)
        .build_ret(Some(&next), i32_ty.into())
        .unwrap();

    let ir = IREmitter::emit(builder.build()).unwrap();

    assert_eq!(
        ir,
        concat!(
            "target triple = \"arm64-apple-macosx\"\n",
            "\n",
            "define i32 @sum_to(i32 %n) {\n",
            "entry:\n",
            "    br label %loop\n",
            "loop:\n",
            "    %acc = phi i32 [ 0, %entry ], [ %next, %loop ]\n",
            "    %i = phi i32 [ %n, %entry ], [ %dec, %loop ]\n",
            "    %next = add i32 %acc, %i\n",
            "    %dec = sub i32 %i, 1\n",
            "    %cond = icmp sgt i32 %dec, 0\n",
            "    br i1 %cond, label %loop, label %exit\n",
            "exit:\n",
            "    ret i32 %next\n",
            "}\n",
            "\n",
        ),
        "\n--- emitted ---\n{ir}"
    );
}

/// The harder back edge: it comes from a block *inside* the loop, not from the header.
///
/// An `if`/`else` sits in the body, so the latch is reached from two places and the
/// header's phis name a block that is neither their own branch source nor a direct
/// successor. Three things have to hold at once:
///
/// - the header's phis name `%latch`, a block created but not yet written when they
///   are built;
/// - `%latch` has its own phi merging the two arms of the `if`;
/// - the values closing the header's back edge (`%merged`, `%dec`) are defined in
///   `%latch`, so they only exist after its cursor is gone.
///
/// ```text
/// define i32 @alternating(i32 %n) {
/// entry:  br label %loop
/// loop:   %acc    = phi i32 [ 0, %entry ], [ %merged, %latch ]
///         %i      = phi i32 [ %n, %entry ], [ %dec, %latch ]
///         %bit    = and i32 %i, 1
///         %even   = icmp eq i32 %bit, 0
///         br i1 %even, label %then, label %else
/// then:   %a      = add i32 %acc, %i
///         br label %latch
/// else:   %b      = add i32 %acc, 1
///         br label %latch
/// latch:  %merged = phi i32 [ %a, %then ], [ %b, %else ]
///         %dec    = sub i32 %i, 1
///         %cont   = icmp sgt i32 %dec, 0
///         br i1 %cont, label %loop, label %exit
/// exit:   ret i32 %merged
/// }
/// ```
#[test]
fn a_frontend_can_close_a_back_edge_from_an_inner_block() {
    let mut ctx = Context::new(
        Triple::new(
            "arm64".to_string(),
            "apple".to_string(),
            "macosx".to_string(),
            None,
        ),
        DataLayout::default(),
    );

    let i32_ty = ctx.i32_ty();
    let mut builder = ctx.builder();

    let f = builder
        .define_function(
            "alternating".to_string(),
            &[(i32_ty, Some("n".to_string()))],
            i32_ty,
        )
        .unwrap();

    // Every block up front. A frontend knows the CFG shape from the syntax tree,
    // before it has emitted a single instruction.
    let entry = f
        .add_basic_block("entry".to_string(), &mut builder)
        .unwrap();
    let loop_b = f.add_basic_block("loop".to_string(), &mut builder).unwrap();
    let then_b = f.add_basic_block("then".to_string(), &mut builder).unwrap();
    let else_b = f.add_basic_block("else".to_string(), &mut builder).unwrap();
    let latch = f
        .add_basic_block("latch".to_string(), &mut builder)
        .unwrap();
    let exit = f.add_basic_block("exit".to_string(), &mut builder).unwrap();

    let n = f.nth_param(0, &builder).unwrap().clone();

    // Constants made before any cursor exists, then reused across blocks.
    let zero = builder.const_value(0i32, OperandTy::Inferred).unwrap();
    let one = builder.const_value(1i32, OperandTy::Inferred).unwrap();

    builder
        .cursor_at_block(entry)
        .build_unconditional_br(loop_b)
        .unwrap();

    // --- header: phis know only the `entry` edge so far ---
    let mut in_loop = builder.cursor_at_block(loop_b);

    let (acc_phi, acc) = in_loop
        .build_phi(&[(entry, zero.clone())], "acc".into())
        .unwrap();
    let (i_phi, i) = in_loop.build_phi(&[(entry, n)], "i".into()).unwrap();

    let bit = in_loop
        .build_iarithmetic(
            IArithmeticOp::And,
            OperandTy::Inferred,
            &i,
            &one,
            "bit".into(),
        )
        .unwrap();
    let even = in_loop
        .build_icmp(ICond::Eq, OperandTy::Inferred, &bit, &zero, "even".into())
        .unwrap();

    in_loop.build_conditional_br(even, then_b, else_b).unwrap();

    // --- the two arms ---
    let mut in_then = builder.cursor_at_block(then_b);
    let a = in_then
        .build_iarithmetic(
            IArithmeticOp::Add,
            OperandTy::Inferred,
            &acc,
            &i,
            "a".into(),
        )
        .unwrap();
    in_then.build_unconditional_br(latch).unwrap();

    let mut in_else = builder.cursor_at_block(else_b);
    let b = in_else
        .build_iarithmetic(
            IArithmeticOp::Add,
            OperandTy::Inferred,
            &acc,
            &one,
            "b".into(),
        )
        .unwrap();
    in_else.build_unconditional_br(latch).unwrap();

    // --- latch: merges the arms, then jumps back to the header ---
    let mut in_latch = builder.cursor_at_block(latch);

    let (_, merged) = in_latch
        .build_phi(&[(then_b, a), (else_b, b)], "merged".into())
        .unwrap();
    let dec = in_latch
        .build_iarithmetic(
            IArithmeticOp::Sub,
            OperandTy::Inferred,
            &i,
            &one,
            "dec".into(),
        )
        .unwrap();
    let cont = in_latch
        .build_icmp(ICond::Sgt, OperandTy::Inferred, &dec, &zero, "cont".into())
        .unwrap();

    in_latch.build_conditional_br(cont, loop_b, exit).unwrap();

    // --- close the header's back edge, now that `latch` has been written ---
    // The handles outlived their cursor, and `add_branch` wants a context rather than
    // a position, so this works with the builder alone.
    acc_phi
        .add_branch((latch, merged.clone()), &mut builder)
        .unwrap();
    i_phi.add_branch((latch, dec), &mut builder).unwrap();

    builder
        .cursor_at_block(exit)
        .build_ret(Some(&merged), i32_ty.into())
        .unwrap();

    let ir = IREmitter::emit(builder.build()).unwrap();

    assert_eq!(
        ir,
        concat!(
            "target triple = \"arm64-apple-macosx\"\n",
            "\n",
            "define i32 @alternating(i32 %n) {\n",
            "entry:\n",
            "    br label %loop\n",
            "loop:\n",
            "    %acc = phi i32 [ 0, %entry ], [ %merged, %latch ]\n",
            "    %i = phi i32 [ %n, %entry ], [ %dec, %latch ]\n",
            "    %bit = and i32 %i, 1\n",
            "    %even = icmp eq i32 %bit, 0\n",
            "    br i1 %even, label %then, label %else\n",
            "then:\n",
            "    %a = add i32 %acc, %i\n",
            "    br label %latch\n",
            "else:\n",
            "    %b = add i32 %acc, 1\n",
            "    br label %latch\n",
            "latch:\n",
            "    %merged = phi i32 [ %a, %then ], [ %b, %else ]\n",
            "    %dec = sub i32 %i, 1\n",
            "    %cont = icmp sgt i32 %dec, 0\n",
            "    br i1 %cont, label %loop, label %exit\n",
            "exit:\n",
            "    ret i32 %merged\n",
            "}\n",
            "\n",
        ),
        "\n--- emitted ---\n{ir}"
    );
}
