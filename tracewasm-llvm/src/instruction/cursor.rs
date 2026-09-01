use crate::{
    cfg::{
        basic_block::BasicBlockId,
        context::{Context, RegisterDef},
    },
    error::{AllocaError, GepError, InstructionError, PhiError, StoreError},
    instruction::{
        AllocaOperands, ConditionalBrOperands, GetElementPtrOperands, Instruction, InstructionKind,
        LoadOperands, PhiInstrHandler, PhiInstruction, StoreOperands, UnconditionalBrOperands,
    },
    interner::TyId,
    value::{I1Value, Value, ValueKind},
};
use rustc_hash::FxHashSet;

pub struct Cursor {
    pub(crate) block: BasicBlockId,
}

impl Cursor {
    pub fn add_phi(
        &self,
        branches: &[(BasicBlockId, Value)],
        reg: Option<&str>,
        ctx: &mut Context,
    ) -> Result<(PhiInstrHandler, Value), PhiError> {
        if branches.is_empty() {
            return Err(PhiError::PhiInstructionWithNoBranches);
        }

        let ref_ty = branches[0].1.ty();
        let func_id = ctx.get_block(self.block).func_id;
        let reg_name = ctx.name_for_reg(reg, func_id)?;
        let val = Value::from_register(reg_name, ref_ty, ctx);

        let phi_id = self.block.add_phi(
            PhiInstruction {
                branches: vec![],
                blocks: FxHashSet::default(),
                ref_ty,
                value: val.clone(),
            },
            ctx,
        )?;

        for (branch, val) in branches {
            phi_id.add_branch((*branch, val.clone()), ctx)?;
        }

        Ok((phi_id, val))
    }

    pub fn add_unconditional_br(
        self,
        label: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<(), InstructionError> {
        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::UnconditionalBr(UnconditionalBrOperands { label }),
                value: None,
            },
            ctx,
        )?;

        self.block.set_locked(ctx);

        Ok(())
    }

    pub fn add_conditional_br(
        self,
        cond: I1Value,
        true_label: BasicBlockId,
        false_label: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<(), InstructionError> {
        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::ConditionalBr(ConditionalBrOperands {
                    cond,
                    true_label,
                    false_label,
                }),
                value: None,
            },
            ctx,
        )?;

        self.block.set_locked(ctx);

        Ok(())
    }

    pub fn add_load(
        &self,
        ty: TyId,
        ptr: Value,
        align: Option<u32>,
        reg: Option<&str>,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        if !ptr.is_ptr(ctx) {
            return Err(InstructionError::PointerOperandExpected(
                ptr.ty().display(ctx).to_string(),
            ));
        }

        if !ty.is_first_class(ctx) {
            return Err(InstructionError::TypeNotLoadable(
                ty.display(ctx).to_string(),
            ));
        }

        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(InstructionError::AlignmentNotPowerOfTwo(align));
        }

        add_instruction_to_block_and_get_value(
            InstructionKind::Load(LoadOperands { ty, ptr, align }),
            ty,
            self.block,
            reg,
            ctx,
        )
    }

    pub fn add_store(
        &self,
        value: Value,
        ptr: Value,
        align: Option<u32>,
        ty: Option<TyId>,
        ctx: &mut Context,
    ) -> Result<(), InstructionError> {
        if !ptr.is_ptr(ctx) {
            return Err(InstructionError::PointerOperandExpected(
                ptr.ty().display(ctx).to_string(),
            ));
        }

        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(InstructionError::AlignmentNotPowerOfTwo(align));
        }

        let final_val = if let Some(ty) = ty {
            let value_ty = ctx.display(value.ty()).to_string();

            let Some(casted_value) = value.try_cast(ty, ctx) else {
                return Err(StoreError::StoredValueTypeMismatch(
                    value_ty,
                    ty.display(ctx).to_string(),
                )
                .into());
            };

            casted_value
        } else {
            value
        };

        if let Some(pointee_ty) = ptr.try_inferring_pointee_ty(self.block, ctx)
            && pointee_ty.ty != final_val.ty()
        {
            return Err(StoreError::StoredValueDoesNotMatchPointee(
                ctx.display(final_val.ty()).to_string(),
                ctx.display(pointee_ty.ty).to_string(),
            )
            .into());
        }

        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::Store(StoreOperands {
                    value: final_val,
                    ptr,
                    align,
                }),
                value: None,
            },
            ctx,
        )?;

        Ok(())
    }

    pub fn add_alloca(
        &self,
        ty: TyId,
        count: Option<(Value, Option<TyId>)>,
        align: Option<u32>,
        reg: Option<&str>,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        if !ty.is_first_class(ctx) {
            return Err(AllocaError::TypeNotAllocatable(ty.display(ctx).to_string()).into());
        }

        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(InstructionError::AlignmentNotPowerOfTwo(align));
        }

        let mut final_count: Option<Value> = None;

        if let Some((count_val, count_expected_ty)) = count {
            if !count_val.is_integer(ctx) {
                return Err(AllocaError::AllocaCountNotAnInteger(
                    ctx.display(count_val.ty()).to_string(),
                )
                .into());
            }

            let final_count_val = if let Some(count_expected_ty) = count_expected_ty {
                if !count_expected_ty.is_integer(ctx) {
                    return Err(AllocaError::AllocaCountNotAnInteger(
                        count_expected_ty.display(ctx).to_string(),
                    )
                    .into());
                }

                let count_ty = ctx.display(count_val.ty()).to_string();

                let Some(casted_val) = count_val.try_cast(count_expected_ty, ctx) else {
                    return Err(AllocaError::AllocaCountTypeMismatch(
                        count_ty,
                        count_expected_ty.display(ctx).to_string(),
                    )
                    .into());
                };

                casted_val
            } else {
                count_val
            };

            final_count = Some(final_count_val);
        }

        let result_ty = ctx.ptr_ty();

        add_instruction_to_block_and_get_value(
            InstructionKind::Alloca(AllocaOperands {
                ty,
                count: final_count,
                align,
            }),
            result_ty,
            self.block,
            reg,
            ctx,
        )
    }

    pub fn add_get_element_ptr(
        &self,
        source_ty: Option<TyId>,
        ptr: Value,
        indices: Vec<Value>,
        inbounds: Option<bool>,
        reg: Option<&str>,
        ctx: &mut Context,
    ) -> Result<Value, InstructionError> {
        let inbounds = inbounds.unwrap_or(false);

        if !ptr.is_ptr(ctx) {
            return Err(InstructionError::PointerOperandExpected(
                ptr.ty().display(ctx).to_string(),
            ));
        }

        for index in &indices {
            if !index.is_integer(ctx) {
                return Err(
                    GepError::IndexNotAnInteger(ctx.display(index.ty()).to_string()).into(),
                );
            }
        }

        let pointee_ty = ptr.try_inferring_pointee_ty(self.block, ctx);

        let final_source_ty = if let Some(source_ty) = source_ty {
            if !source_ty.is_first_class(ctx) {
                return Err(
                    GepError::SourceTypeNotSized(source_ty.display(ctx).to_string()).into(),
                );
            }

            if let Some(pointee_ty) = pointee_ty
                && source_ty != pointee_ty.ty
            {
                return Err(GepError::SourceTypeDoesNotMatchPointee(
                    source_ty.display(ctx).to_string(),
                    ctx.display(pointee_ty.ty).to_string(),
                )
                .into());
            }

            source_ty
        } else if let Some(pointee_ty) = pointee_ty {
            pointee_ty.ty
        } else {
            return Err(GepError::SourceTypeUnknown.into());
        };

        // The first index steps over the source type as pointer arithmetic rather than
        // descending into it, so the walk starts at the second.
        if indices.len() > 1 {
            final_source_ty.walk_pointee_ty_in_gep(&indices[1..], ctx)?;
        }

        let result_ty = ctx.ptr_ty();

        add_instruction_to_block_and_get_value(
            InstructionKind::GetElementPtr(GetElementPtrOperands {
                // The resolved type, not the caller's `Option`: when it was inferred
                // from the pointer, that is the type the instruction has to be emitted
                // with, and it is the one the walk above validated.
                source_ty: final_source_ty,
                ptr,
                indices: indices.into_boxed_slice(),
                inbounds,
            }),
            result_ty,
            self.block,
            reg,
            ctx,
        )
    }
}

fn add_instruction_to_block_and_get_value(
    kind: InstructionKind,
    result_ty: TyId,
    block: BasicBlockId,
    reg: Option<&str>,
    ctx: &mut Context,
) -> Result<Value, InstructionError> {
    let func_id = ctx.get_block(block).func_id;
    let reg_name = ctx.name_for_reg(reg, func_id)?;
    let val = Value::from_register(reg_name, result_ty, ctx);

    let ValueKind::Reg(reg) = val.kind() else {
        unreachable!("value is made out of register name just above")
    };

    let reg_name_id = reg.name;

    let instr_index = block.add_instruction(
        Instruction {
            kind,
            value: Some(val.clone()),
        },
        ctx,
    )?;

    let register_def_index = ctx.register_def_instr_index.entry(func_id).or_default();

    register_def_index.insert(reg_name_id, RegisterDef { block, instr_index });

    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cfg::{builder::Builder, context::Context},
        test_support::{add_fn, fixture, value},
        value::{ConstExpr, ConstValue, NullPtr, Type},
    };

    /// Interns `ty` and hands back its id, for the shapes that have no `Context`
    /// shorthand.
    fn intern(ty: Type, ctx: &mut Context) -> TyId {
        ctx.ty_interner.intern(ty).into()
    }

    /// How a type spells itself against `ctx`'s pool — which is the form an error
    /// carries, since `BuildError` holds the rendering rather than the type.
    fn rendered(ty: TyId, ctx: &Context) -> String {
        ty.display(ctx).to_string()
    }

    /// Instructions land in the block the cursor names, in order.
    #[test]
    fn instructions_append_in_order_to_the_open_block() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let i32_ty = ctx.i32_ty();
        let cursor = builder.cursor_at_block(entry);

        // A non-terminator first, so the block is still open for the branch that
        // follows it — a branch consumes the cursor and closes the block.
        cursor
            .add_alloca(i32_ty, None, None, None, &mut ctx)
            .unwrap();
        cursor.add_unconditional_br(body, &mut ctx).unwrap();

        let instrs = &ctx.blocks.get(entry.raw()).unwrap().instructions;

        assert_eq!(instrs.len(), 2);

        assert!(
            matches!(instrs[0].kind, InstructionKind::Alloca(_)),
            "the first instruction is the one added first"
        );

        assert!(
            matches!(
                instrs[1].kind,
                InstructionKind::UnconditionalBr(UnconditionalBrOperands{ label }) if label == body
            ),
            "and the branch is second"
        );

        assert!(
            instrs[1].value.is_none(),
            "a branch produces no value, so it defines no register"
        );

        assert!(
            ctx.blocks.get(body.raw()).unwrap().instructions.is_empty(),
            "the other block is untouched"
        );
    }

    /// A phi selects between incoming values, so with none there is nothing to
    /// select — and no first branch to take the phi's type from either.
    ///
    /// This is the one phi path that does not reach `Value::ty`, so it is the only
    /// one runnable until that lands.
    #[test]
    fn a_phi_needs_at_least_one_branch() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let _entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(body);

        let err = cursor
            .add_phi(&[], Some("result"), &mut ctx)
            .expect_err("a phi with no branches has no value and no type");

        assert!(matches!(err, PhiError::PhiInstructionWithNoBranches));

        assert!(
            ctx.blocks.get(body.raw()).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    /// The entry block has no predecessors, so a phi there has nothing to choose
    /// between.
    #[test]
    fn a_phi_cannot_go_in_the_entry_block() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let cursor = builder.cursor_at_block(entry);

        let err = cursor
            .add_phi(&[(entry, v)], Some("result"), &mut ctx)
            .expect_err("no predecessors to choose between");

        assert!(matches!(
            err,
            PhiError::PhiInstructionCannotBeAddedToEntryBasicBlock
        ));
    }

    /// In a later block a phi is fine, and each one gets its own index — tagged
    /// with the block, since an index alone would mean different phis in different
    /// blocks.
    #[test]
    fn phis_in_a_later_block_are_indexed_within_that_block() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let tail = f.add_basic_block("tail".to_string(), &mut ctx).unwrap();
        let (v1, v2, v3) = (value(1, &mut ctx), value(2, &mut ctx), value(3, &mut ctx));
        let cursor = builder.cursor_at_block(body);

        let (first, _) = cursor.add_phi(&[(entry, v1)], Some("a"), &mut ctx).unwrap();
        let (second, _) = cursor.add_phi(&[(entry, v2)], Some("b"), &mut ctx).unwrap();

        assert_eq!((first.index, first.block), (0, body));
        assert_eq!((second.index, second.block), (1, body));

        // A different block starts its own numbering, which is why the id has to
        // carry the block to be unambiguous.
        let cursor = builder.cursor_at_block(tail);
        let (elsewhere, _) = cursor.add_phi(&[(entry, v3)], Some("c"), &mut ctx).unwrap();

        assert_eq!((elsewhere.index, elsewhere.block), (0, tail));
        assert_ne!(elsewhere.block, first.block);
    }

    /// The phi's own type comes from its first branch, and the value it hands back
    /// carries that type — that is what makes the result usable downstream.
    #[test]
    fn a_phi_yields_a_value_of_its_branches_type() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let cursor = builder.cursor_at_block(body);

        let (_, result) = cursor
            .add_phi(&[(entry, v)], Some("merged"), &mut ctx)
            .unwrap();

        assert_eq!(
            ctx.ty_interner.value(result.ty().raw()),
            &Type::I32,
            "an i32 branch makes an i32 phi"
        );
    }

    /// A phi produces one value, so every incoming value has to have its type.
    #[test]
    fn phi_branches_must_all_share_the_phis_type() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let other = f.add_basic_block("other".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let an_i32 = value(1, &mut ctx);
        let an_i64 = Value::from_const(1i64, None, &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(body);

        let err = cursor
            .add_phi(
                &[(entry, an_i32), (other, an_i64)],
                Some("merged"),
                &mut ctx,
            )
            .expect_err("the second branch is an i64");

        assert!(
            matches!(
                &err,
                PhiError::PhiInstructionBranchTypeMismatch(phi, branch)
                    if phi == "i32" && branch == "i64"
            ),
            "the error must name both types, got: {err}"
        );
    }

    /// The other side of the check above: two branches whose types were built
    /// *separately* must still be accepted, because structurally equal types are one
    /// pool entry and so one id.
    ///
    /// This is what a non-number array length would break. If `Type::Array` held a
    /// `Value`, the two `[4 x i32]`s below would carry different constant ids, intern
    /// as two entries, and this phi would be refused as a type mismatch — valid IR
    /// rejected, with an error naming the same type twice.
    #[test]
    fn a_phi_accepts_branches_whose_equal_types_were_built_separately() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let other = f.add_basic_block("other".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        let i32_ty = intern(Type::I32, &mut ctx);

        let array = |ctx: &mut Context| {
            intern(
                Type::Array {
                    size: 4,
                    element_ty: i32_ty,
                },
                ctx,
            )
        };

        // Built one at a time, as two unrelated parts of a module would build them.
        let first_ty = array(&mut ctx);
        let second_ty = array(&mut ctx);

        assert_eq!(first_ty, second_ty, "`[4 x i32]` is one type, hence one id");

        let a = Value::from_register("a".to_string(), first_ty, &mut ctx);
        let b = Value::from_register("b".to_string(), second_ty, &mut ctx);

        let cursor = builder.cursor_at_block(body);

        let (_, merged) = cursor
            .add_phi(&[(entry, a), (other, b)], Some("merged"), &mut ctx)
            .expect("both branches are `[4 x i32]`");

        assert_eq!(
            ctx.display(merged.ty()).to_string(),
            "[4 x i32]",
            "and the phi carries that type"
        );
    }

    /// Phis have to precede every other instruction in their block, so once one
    /// has been emitted the window has closed.
    #[test]
    fn a_phi_cannot_follow_an_instruction() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let i32_ty = ctx.i32_ty();
        let cursor = builder.cursor_at_block(body);

        // A plain instruction, not a terminator: this is about phis coming *first*,
        // not about the block being closed, which is a separate rule.
        cursor
            .add_alloca(i32_ty, None, None, None, &mut ctx)
            .unwrap();

        let err = cursor
            .add_phi(&[(entry, v)], Some("result"), &mut ctx)
            .expect_err("the block already has an instruction");

        assert!(matches!(err, PhiError::PhiInstructionAddError));

        assert!(
            ctx.blocks.get(body.raw()).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    /// A block stays open until a terminator: plain instructions keep appending, and
    /// nothing locks it.
    #[test]
    fn a_block_stays_open_until_a_terminator() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let i32_ty = ctx.i32_ty();

        let cursor = builder.cursor_at_block(entry);

        for _ in 0..3 {
            cursor
                .add_alloca(i32_ty, None, None, None, &mut ctx)
                .expect("a block that has not branched still accepts instructions");
        }

        assert!(
            !ctx.blocks.get(entry.raw()).unwrap().is_locked,
            "only a terminator locks a block"
        );

        assert_eq!(ctx.blocks.get(entry.raw()).unwrap().instructions.len(), 3);
    }

    /// Both terminators lock, and locking is per *block*: closing one leaves its
    /// siblings open.
    #[test]
    fn a_terminator_locks_its_own_block_only() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let exit = f.add_basic_block("exit".to_string(), &mut ctx).unwrap();

        builder
            .cursor_at_block(entry)
            .add_unconditional_br(body, &mut ctx)
            .unwrap();

        assert!(ctx.blocks.get(entry.raw()).unwrap().is_locked);
        assert!(
            !ctx.blocks.get(body.raw()).unwrap().is_locked,
            "a branch closes the block it was written into, not the one it names"
        );

        let cond = Value::from_const(true, None, &mut ctx)
            .unwrap()
            .into_i1(&ctx)
            .unwrap();

        builder
            .cursor_at_block(body)
            .add_conditional_br(cond, exit, exit, &mut ctx)
            .unwrap();

        assert!(
            ctx.blocks.get(body.raw()).unwrap().is_locked,
            "a conditional branch locks too"
        );

        assert!(!ctx.blocks.get(exit.raw()).unwrap().is_locked);
    }

    /// Consuming the cursor stops *that* cursor being reused, but a fresh one can
    /// still be opened at the same block — which is what `is_locked` is for, and the
    /// only thing that catches this: the consumed-`self` rule cannot see a second
    /// cursor.
    #[test]
    fn a_fresh_cursor_on_a_locked_block_is_refused() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let i32_ty = ctx.i32_ty();

        builder
            .cursor_at_block(entry)
            .add_unconditional_br(body, &mut ctx)
            .unwrap();

        // A second cursor at the same block: the consumed-`self` rule cannot see this.
        let reopened = builder.cursor_at_block(entry);

        let err = reopened
            .add_alloca(i32_ty, None, None, None, &mut ctx)
            .expect_err("`entry` already ends in a branch");

        assert!(
            matches!(&err, InstructionError::BasicBlockAlreadyTerminated(name) if name == "entry"),
            "the error must name the block it refused, got: {err}"
        );

        assert_eq!(
            ctx.blocks.get(entry.raw()).unwrap().instructions.len(),
            1,
            "the refused instruction must not have been added"
        );
    }

    /// The same on the phi path. The lock check runs *before* the
    /// `instructions.is_empty()` one, so a terminated block reports being terminated
    /// rather than the vaguer `PhiInstructionAddError` — which is the more useful of
    /// the two, since both are true of a block that has branched.
    #[test]
    fn a_phi_on_a_locked_block_reports_the_block_is_terminated() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);

        builder
            .cursor_at_block(body)
            .add_unconditional_br(body, &mut ctx)
            .unwrap();

        let reopened = builder.cursor_at_block(body);

        let err = reopened
            .add_phi(&[(entry, v)], Some("result"), &mut ctx)
            .expect_err("`body` already ends in a branch");

        assert!(
            matches!(&err, PhiError::BasicBlockAlreadyTerminated(name) if name == "body"),
            "the error must name the block it refused, got: {err}"
        );

        assert!(
            ctx.blocks.get(body.raw()).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    /// Every builder that writes an instruction goes through the same lock, not just
    /// the one that happened to be tested first.
    #[test]
    fn a_locked_block_refuses_every_kind_of_instruction() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        let i32_ty = ctx.i32_ty();
        let ptr = Value::from_const(NullPtr, None, &mut ctx).unwrap();
        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();
        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        builder
            .cursor_at_block(entry)
            .add_unconditional_br(body, &mut ctx)
            .unwrap();

        let terminated = |r: Result<(), InstructionError>, what: &str| {
            assert!(
                matches!(
                    &r,
                    Err(InstructionError::BasicBlockAlreadyTerminated(name)) if name == "entry"
                ),
                "`{what}` must be refused by the lock, got: {r:?}"
            );
        };

        terminated(
            builder
                .cursor_at_block(entry)
                .add_alloca(i32_ty, None, None, None, &mut ctx)
                .map(|_| ()),
            "alloca",
        );

        terminated(
            builder
                .cursor_at_block(entry)
                .add_load(i32_ty, ptr.clone(), None, None, &mut ctx)
                .map(|_| ()),
            "load",
        );

        terminated(
            builder
                .cursor_at_block(entry)
                .add_store(seven, ptr.clone(), None, None, &mut ctx),
            "store",
        );

        terminated(
            builder
                .cursor_at_block(entry)
                .add_get_element_ptr(Some(i32_ty), ptr, vec![zero], None, None, &mut ctx)
                .map(|_| ()),
            "getelementptr",
        );

        assert_eq!(
            ctx.blocks.get(entry.raw()).unwrap().instructions.len(),
            1,
            "nothing was appended past the terminator"
        );
    }

    /// A second terminator is refused too, so a block ends in exactly one branch —
    /// which is what LLVM requires of a well-formed block.
    #[test]
    fn a_locked_block_refuses_a_second_terminator() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        builder
            .cursor_at_block(entry)
            .add_unconditional_br(body, &mut ctx)
            .unwrap();

        let err = builder
            .cursor_at_block(entry)
            .add_unconditional_br(body, &mut ctx)
            .expect_err("a block ends once");

        assert!(
            matches!(&err, InstructionError::BasicBlockAlreadyTerminated(name) if name == "entry"),
            "got: {err}"
        );

        assert_eq!(ctx.blocks.get(entry.raw()).unwrap().instructions.len(), 1);
    }

    /// The lock survives reopening: asking for a fresh cursor does not clear it, so a
    /// block cannot be reopened by going back to the builder.
    #[test]
    fn reopening_a_block_does_not_clear_its_lock() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let i32_ty = ctx.i32_ty();

        builder
            .cursor_at_block(entry)
            .add_unconditional_br(body, &mut ctx)
            .unwrap();

        for attempt in 0..3 {
            let reopened = builder.cursor_at_block(entry);

            assert!(
                reopened
                    .add_alloca(i32_ty, None, None, None, &mut ctx)
                    .is_err(),
                "attempt {attempt} must still be refused"
            );

            assert!(ctx.blocks.get(entry.raw()).unwrap().is_locked);
        }
    }

    /// A phi still goes in before the block is closed — locking is about the
    /// terminator, and an open block accepts phis and instructions as before.
    #[test]
    fn locking_does_not_disturb_a_well_formed_block() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let i32_ty = ctx.i32_ty();

        let cursor = builder.cursor_at_block(body);

        cursor
            .add_phi(&[(entry, v)], Some("m"), &mut ctx)
            .expect("a phi opens the block");

        cursor
            .add_alloca(i32_ty, None, None, None, &mut ctx)
            .expect("instructions follow the phi");

        cursor
            .add_unconditional_br(body, &mut ctx)
            .expect("and the terminator closes it");

        let block = ctx.blocks.get(body.raw()).unwrap();

        assert_eq!(block.phis.len(), 1);
        assert_eq!(block.instructions.len(), 2);
        assert!(block.is_locked, "closed only at the end");
    }

    /// A phi names one value per *predecessor*, so the same predecessor twice is a
    /// bug in the caller — and it is a different bug from an entry-block phi.
    #[test]
    fn a_phi_takes_each_predecessor_once() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let other = f.add_basic_block("other".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let (v1, v2, v3) = (value(1, &mut ctx), value(2, &mut ctx), value(3, &mut ctx));
        let cursor = builder.cursor_at_block(body);

        let (phi, _) = cursor
            .add_phi(&[(entry, v1), (other, v2)], Some("merged"), &mut ctx)
            .unwrap();

        let err = phi
            .add_branch((entry, v3), &mut ctx)
            .expect_err("`entry` is already a predecessor of this phi");

        assert!(
            matches!(err, PhiError::BasicBlockBranchAlreadyInPhiInstruction),
            "a repeated predecessor is not an entry-block error, got: {err}"
        );

        // One incoming value per predecessor, and the two branches it was built
        // with are the two it has — not four.
        let stored = &ctx.blocks.get(body.raw()).unwrap().phis[phi.index];

        assert_eq!(stored.branches.len(), 2);
        assert_eq!(stored.blocks.len(), 2);
    }

    /// A block with a pointer-typed value in hand, which every load test needs.
    fn block_with_ptr(ctx: &mut Context, builder: &mut Builder) -> (Cursor, Value) {
        let f = add_fn("f", builder, ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();
        let ptr = Value::from_const(NullPtr, None, ctx).unwrap();

        (builder.cursor_at_block(entry), ptr)
    }

    /// A `{ i32, double }` slot on the stack, plus the cursor to build against — the
    /// shape every struct-indexing test below needs.
    fn block_with_struct_slot(ctx: &mut Context, builder: &mut Builder) -> (Cursor, Value, TyId) {
        let f = add_fn("f", builder, ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();

        let struct_ty = intern(
            Type::Struct {
                fields: Box::new([i32_ty, f64_ty]),
                packed: false,
            },
            ctx,
        );

        let slot = cursor
            .add_alloca(struct_ty, None, None, Some("s"), ctx)
            .expect("a struct is allocatable");

        (cursor, slot, struct_ty)
    }

    /// A `{ i32, [4 x double] }` slot, for the tests that chain a `getelementptr`
    /// through a nested aggregate. Returns the cursor, the slot, and the two types.
    fn block_with_nested_slot(
        ctx: &mut Context,
        builder: &mut Builder,
    ) -> (Cursor, Value, TyId, TyId) {
        let f = add_fn("f", builder, ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();

        let array_ty = intern(
            Type::Array {
                size: 4,
                element_ty: f64_ty,
            },
            ctx,
        );

        let struct_ty = intern(
            Type::Struct {
                fields: Box::new([i32_ty, array_ty]),
                packed: false,
            },
            ctx,
        );

        let slot = cursor
            .add_alloca(struct_ty, None, None, Some("s"), ctx)
            .expect("a struct is allocatable");

        (cursor, slot, array_ty, f64_ty)
    }

    /// A `getelementptr` produces a pointer, so the *next* one has to be able to trace
    /// its pointee back through it — which is what makes a chain like
    /// `gep` → `gep` → `store` work without the caller restating the type each time.
    ///
    /// Before `GetElementPtr` was handled in the inference, this panicked rather than
    /// resolving: only `alloca` was matched.
    #[test]
    fn a_gep_through_a_gep_infers_its_pointee() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, array_ty, _) = block_with_nested_slot(&mut ctx, &mut builder);

        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();
        let one = Value::from_const(1i32, None, &mut ctx).unwrap();
        let two = Value::from_const(2i32, None, &mut ctx).unwrap();

        // `%f = gep { i32, [4 x double] }, ptr %s, i32 0, i32 1` — the array field.
        let field = cursor
            .add_get_element_ptr(
                None,
                slot,
                vec![zero.clone(), one],
                None,
                Some("f"),
                &mut ctx,
            )
            .expect("the alloca says what it points to");

        // `%e = gep [4 x double], ptr %f, i32 0, i32 2` — with no source type given,
        // so it has to come from the gep above.
        let elem = cursor
            .add_get_element_ptr(None, field, vec![zero, two], None, Some("e"), &mut ctx)
            .expect("the first gep says what it points to");

        let block = ctx.blocks.get(cursor.block.raw()).unwrap();

        let InstructionKind::GetElementPtr(second) = &block.instructions[2].kind else {
            panic!("expected a gep")
        };

        assert_eq!(
            second.source_ty, array_ty,
            "the second gep's source type is the first one's result pointee"
        );

        // And the chain lands on a `double`, which is what a store through it must be.
        let a_double = Value::from_const(1.0f64, None, &mut ctx).unwrap();
        let an_i32 = Value::from_const(1i32, None, &mut ctx).unwrap();

        assert!(
            cursor
                .add_store(a_double, elem.clone(), None, None, &mut ctx)
                .is_ok(),
            "the element is a double"
        );

        let err = cursor
            .add_store(an_i32, elem, None, None, &mut ctx)
            .expect_err("an i32 is not what this points to");

        assert!(
            matches!(
                &err,
                InstructionError::Store(StoreError::StoredValueDoesNotMatchPointee(value, pointee))
                    if value == "i32" && pointee == "double"
            ),
            "the error must name both types, got: {err}"
        );
    }

    /// The same inference from a `getelementptr` *constant expression* rather than an
    /// instruction. Nothing builds one yet, so this pins the arm before a builder
    /// exists to reach it.
    #[test]
    fn a_gep_constant_expression_reports_its_pointee() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _, array_ty, f64_ty) = block_with_nested_slot(&mut ctx, &mut builder);

        let ptr_ty = ctx.ptr_ty();
        let null = Value::from_const(NullPtr, None, &mut ctx).unwrap();
        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();
        let two = Value::from_const(2i32, None, &mut ctx).unwrap();

        // `getelementptr ([4 x double], ptr null, i32 0, i32 2)` — points at a double.
        let const_gep = Value::new(
            ptr_ty,
            ValueKind::ConstExpr(ConstExpr::GetElementPtr(Box::new(GetElementPtrOperands {
                source_ty: array_ty,
                ptr: null,
                indices: Box::new([zero, two]),
                inbounds: false,
            }))),
        );

        let pointee = const_gep
            .try_inferring_pointee_ty(cursor.block, &mut ctx)
            .expect("a gep constant expression says what it points to");

        assert_eq!(pointee.ty, f64_ty, "index 2 of `[4 x double]` is a double");

        assert_eq!(
            ctx.display(pointee.ty).to_string(),
            "double",
            "and it renders as one"
        );
    }

    /// A `getelementptr` with one index or none does not descend: the first index
    /// steps over the source type as pointer arithmetic, so the pointee is unchanged.
    #[test]
    fn a_gep_with_a_single_index_still_points_at_its_source_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _, _) = block_with_nested_slot(&mut ctx, &mut builder);

        let one = Value::from_const(1i32, None, &mut ctx).unwrap();

        // `%n = gep { i32, [4 x double] }, ptr %s, i32 1` — the *next* struct along.
        let next = cursor
            .add_get_element_ptr(None, slot, vec![one], None, Some("n"), &mut ctx)
            .expect("a single index is pointer arithmetic over the source type");

        let pointee = next
            .try_inferring_pointee_ty(cursor.block, &mut ctx)
            .expect("still traceable");

        assert_eq!(
            ctx.display(pointee.ty).to_string(),
            "{ i32, [4 x double] }",
            "one index steps over the type rather than into it"
        );
    }

    /// The types of the nested fixture below, so an assertion can name a level by
    /// what it means rather than by an id.
    struct DeepTys {
        outer: TyId,
        inner_array: TyId,
        inner: TyId,
        i64_array: TyId,
        i64: TyId,
        f64: TyId,
        ptr: TyId,
        i32: TyId,
    }

    /// A slot holding
    ///
    /// ```text
    /// %Inner = { i8, [3 x i64] }
    /// %Outer = { i32, [2 x %Inner], ptr, double }
    /// ```
    ///
    /// which is deep enough that an off-by-one in the walk lands on a real type
    /// rather than falling off the end — the failure mode a two-level fixture cannot
    /// catch. The index sequences the tests below use were checked against `llvm-as`:
    /// `0,1,1,1,2` assembles and one index further is refused, which is what pins the
    /// last level to a scalar.
    fn block_with_deep_slot(ctx: &mut Context, builder: &mut Builder) -> (Cursor, Value, DeepTys) {
        let f = add_fn("f", builder, ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let i8_ty = ctx.i8_ty();
        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f64_ty = ctx.f64_ty();
        let ptr_ty = ctx.ptr_ty();

        let i64_array = intern(
            Type::Array {
                size: 3,
                element_ty: i64_ty,
            },
            ctx,
        );

        let inner = intern(
            Type::Struct {
                fields: Box::new([i8_ty, i64_array]),
                packed: false,
            },
            ctx,
        );

        let inner_array = intern(
            Type::Array {
                size: 2,
                element_ty: inner,
            },
            ctx,
        );

        let outer = intern(
            Type::Struct {
                fields: Box::new([i32_ty, inner_array, ptr_ty, f64_ty]),
                packed: false,
            },
            ctx,
        );

        let slot = cursor
            .add_alloca(outer, None, None, Some("s"), ctx)
            .expect("a struct is allocatable");

        (
            cursor,
            slot,
            DeepTys {
                outer,
                inner_array,
                inner,
                i64_array,
                i64: i64_ty,
                f64: f64_ty,
                ptr: ptr_ty,
                i32: i32_ty,
            },
        )
    }

    /// A constant `i32` index, which is what a struct index has to be.
    fn idx(n: i32, ctx: &mut Context) -> Value {
        Value::from_const(n, None, ctx).expect("an i32 constant")
    }

    /// Every level of the nested type, asserted by both id and spelling.
    ///
    /// One index per level after the first, so an off-by-one anywhere in the walk
    /// moves *every* row below it — which is why each row names the exact type rather
    /// than just checking the walk succeeded.
    #[test]
    fn a_gep_walks_every_level_of_a_deeply_nested_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, tys) = block_with_deep_slot(&mut ctx, &mut builder);

        // The slot itself, before any walking, is the whole `%Outer`.
        assert_eq!(
            slot.try_inferring_pointee_ty(cursor.block, &mut ctx)
                .expect("an alloca says what it points to")
                .ty,
            tys.outer
        );

        // (indices after the leading 0, expected id, expected spelling)
        let cases: Vec<(Vec<i32>, TyId, &str)> = vec![
            (vec![0], tys.i32, "i32"),
            (vec![1], tys.inner_array, "[2 x { i8, [3 x i64] }]"),
            (vec![1, 1], tys.inner, "{ i8, [3 x i64] }"),
            (vec![1, 1, 1], tys.i64_array, "[3 x i64]"),
            (vec![1, 1, 1, 2], tys.i64, "i64"),
            (vec![2], tys.ptr, "ptr"),
            (vec![3], tys.f64, "double"),
        ];

        for (tail, expected_id, expected) in cases {
            let mut indices = vec![idx(0, &mut ctx)];

            for n in &tail {
                indices.push(idx(*n, &mut ctx));
            }

            let elem = cursor
                .add_get_element_ptr(None, slot.clone(), indices, None, None, &mut ctx)
                .unwrap_or_else(|e| panic!("gep 0,{tail:?} should walk: {e}"));

            let pointee = elem
                .try_inferring_pointee_ty(cursor.block, &mut ctx)
                .unwrap_or_else(|| panic!("gep 0,{tail:?} should report a pointee"));

            assert_eq!(
                ctx.display(pointee.ty).to_string(),
                expected,
                "gep 0,{tail:?} lands on the wrong type"
            );

            assert_eq!(
                pointee.ty, expected_id,
                "gep 0,{tail:?} renders as `{expected}` but is not that pool entry"
            );

            assert_eq!(
                ctx.display(elem.ty()).to_string(),
                "ptr",
                "a gep always yields a pointer"
            );
        }
    }

    /// One index past the last level names nothing. `llvm-as` agrees: appending an
    /// index to either sequence below is refused with "invalid getelementptr indices".
    #[test]
    fn a_gep_cannot_index_past_the_last_level() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _) = block_with_deep_slot(&mut ctx, &mut builder);

        // `0,1,1,1,2` ends on the `i64`; a sixth index has nothing to descend into.
        let too_deep = vec![
            idx(0, &mut ctx),
            idx(1, &mut ctx),
            idx(1, &mut ctx),
            idx(1, &mut ctx),
            idx(2, &mut ctx),
            idx(0, &mut ctx),
        ];

        let err = cursor
            .add_get_element_ptr(None, slot.clone(), too_deep, None, None, &mut ctx)
            .expect_err("an i64 has no elements");

        assert!(
            matches!(&err, InstructionError::Gep(GepError::TypeNotIndexable(t)) if t == "i64"),
            "the error must name the scalar it stopped at, got: {err}"
        );

        // And the same one level up, on the `double` field.
        let past_double = vec![idx(0, &mut ctx), idx(3, &mut ctx), idx(0, &mut ctx)];

        assert!(
            matches!(
                cursor.add_get_element_ptr(None, slot, past_double, None, None, &mut ctx),
                Err(InstructionError::Gep(GepError::TypeNotIndexable(t))) if t == "double"
            ),
            "a double has no elements either"
        );
    }

    /// A `load` in the middle of a chain: the scalar a deep `getelementptr` reaches is
    /// loadable at exactly the type the walk landed on.
    #[test]
    fn a_load_through_a_deep_gep_yields_the_walked_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, tys) = block_with_deep_slot(&mut ctx, &mut builder);

        let deep = vec![
            idx(0, &mut ctx),
            idx(1, &mut ctx),
            idx(1, &mut ctx),
            idx(1, &mut ctx),
            idx(2, &mut ctx),
        ];

        let elem = cursor
            .add_get_element_ptr(None, slot, deep, None, Some("e"), &mut ctx)
            .expect("the walk reaches the i64");

        let loaded = cursor
            .add_load(tys.i64, elem.clone(), None, Some("v"), &mut ctx)
            .expect("an i64 is loadable");

        assert_eq!(ctx.display(loaded.ty()).to_string(), "i64");
        assert_eq!(loaded.ty(), tys.i64);

        // Storing the loaded value straight back is the round trip, and the pointee
        // check has to accept it.
        assert!(
            cursor.add_store(loaded, elem, None, None, &mut ctx).is_ok(),
            "what was loaded from a slot must store back into it"
        );
    }

    /// A `load` of a *pointer* breaks the chain, which is the honest answer: the
    /// pointee of a pointer read out of memory is not knowable from the instruction
    /// that produced it, so the caller has to say what it is.
    #[test]
    fn a_gep_through_a_loaded_pointer_needs_an_explicit_source_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, tys) = block_with_deep_slot(&mut ctx, &mut builder);

        // `%p = gep %Outer, ptr %s, i32 0, i32 2` — the `ptr` field.
        let ptr_field = vec![idx(0, &mut ctx), idx(2, &mut ctx)];

        let ptr_field = cursor
            .add_get_element_ptr(None, slot, ptr_field, None, Some("p"), &mut ctx)
            .expect("field 2 is the pointer");

        // `%q = load ptr, ptr %p` — a pointer whose pointee nothing records.
        let loaded_ptr = cursor
            .add_load(tys.ptr, ptr_field, None, Some("q"), &mut ctx)
            .expect("a ptr is loadable");

        assert!(
            loaded_ptr
                .try_inferring_pointee_ty(cursor.block, &mut ctx)
                .is_none(),
            "a `load` says nothing about what its result points to"
        );

        let indices = vec![idx(0, &mut ctx), idx(1, &mut ctx)];

        let err = cursor
            .add_get_element_ptr(None, loaded_ptr.clone(), indices, None, None, &mut ctx)
            .expect_err("nothing can be inferred through a load");

        assert!(
            matches!(&err, InstructionError::Gep(GepError::SourceTypeUnknown)),
            "expected an unknown-source-type error, got: {err}"
        );

        // Given the type explicitly, the same walk goes through and lands where the
        // nested fixture says it should.
        let indices = vec![idx(0, &mut ctx), idx(1, &mut ctx)];

        let elem = cursor
            .add_get_element_ptr(
                Some(tys.inner),
                loaded_ptr,
                indices,
                None,
                Some("r"),
                &mut ctx,
            )
            .expect("with the source type given there is nothing to infer");

        let pointee = elem
            .try_inferring_pointee_ty(cursor.block, &mut ctx)
            .expect("the gep itself records its source type");

        assert_eq!(
            ctx.display(pointee.ty).to_string(),
            "[3 x i64]",
            "field 1 of `{{ i8, [3 x i64] }}` is the array"
        );

        assert_eq!(pointee.ty, tys.i64_array);
    }

    /// Memory is reached through a pointer, so `getelementptr` refuses anything else
    /// for the same reason `load` does.
    #[test]
    fn a_gep_needs_a_pointer_operand() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let not_a_ptr = value(7, &mut ctx);
        let zero = value(0, &mut ctx);

        let err = cursor
            .add_get_element_ptr(Some(i32_ty), not_a_ptr, vec![zero], None, None, &mut ctx)
            .expect_err("an i32 is not an address");

        assert!(
            matches!(&err, InstructionError::PointerOperandExpected(t) if t == "i32"),
            "the error must name the offending type, got: {err}"
        );
    }

    /// Every index scales an offset, so every one of them has to be an integer.
    #[test]
    fn a_gep_index_must_be_an_integer() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let a_float = Value::from_const(1.0f32, None, &mut ctx).unwrap();

        let err = cursor
            .add_get_element_ptr(Some(i32_ty), ptr, vec![a_float], None, None, &mut ctx)
            .expect_err("a float is not an index");

        assert!(
            matches!(&err, InstructionError::Gep(GepError::IndexNotAnInteger(t)) if t == "float"),
            "the error must name the offending type, got: {err}"
        );
    }

    /// With no source type given and a pointer nothing can be traced back to — a
    /// `null` constant — there is no type to emit the instruction with.
    #[test]
    fn a_gep_without_a_source_type_needs_an_inferable_pointer() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let zero = value(0, &mut ctx);

        let err = cursor
            .add_get_element_ptr(None, ptr, vec![zero], None, None, &mut ctx)
            .expect_err("null says nothing about its pointee");

        assert!(
            matches!(&err, InstructionError::Gep(GepError::SourceTypeUnknown)),
            "expected an unknown-source-type error, got: {err}"
        );
    }

    /// When the pointer *can* be traced back, a source type that disagrees with it is
    /// refused rather than silently preferred.
    #[test]
    fn a_gep_source_type_must_match_the_inferred_pointee() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _) = block_with_struct_slot(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let zero = value(0, &mut ctx);

        let err = cursor
            .add_get_element_ptr(Some(i32_ty), slot, vec![zero], None, None, &mut ctx)
            .expect_err("the slot holds a struct, not an i32");

        assert!(
            matches!(
                &err,
                InstructionError::Gep(GepError::SourceTypeDoesNotMatchPointee(given, inferred))
                    if given == "i32" && inferred == "{ i32, double }"
            ),
            "the error must name both types, got: {err}"
        );
    }

    /// Omitting the source type is allowed when the pointer says what it points to —
    /// and the *inferred* type is what the instruction is emitted with, not `None`.
    #[test]
    fn a_gep_emits_the_inferred_source_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, struct_ty) = block_with_struct_slot(&mut ctx, &mut builder);

        let zero = value(0, &mut ctx);
        let field = Value::from_const(1i32, None, &mut ctx).unwrap();

        let elem = cursor
            .add_get_element_ptr(None, slot, vec![zero, field], None, Some("f"), &mut ctx)
            .expect("the pointee is inferable from the alloca");

        assert_eq!(
            ctx.display(elem.ty()).to_string(),
            "ptr",
            "a gep yields a pointer"
        );

        let block = ctx.blocks.get(cursor.block.raw()).unwrap();

        let InstructionKind::GetElementPtr(GetElementPtrOperands {
            source_ty,
            ptr: _,
            indices: _,
            inbounds: _,
        }) = &block.instructions[1].kind
        else {
            panic!("expected a gep")
        };

        assert_eq!(
            *source_ty, struct_ty,
            "the resolved type is emitted, not the caller's `None`"
        );
    }

    /// LLVM requires a struct index to be a constant `i32` specifically — `llvm-as`
    /// refuses an `i64` one with "invalid getelementptr indices" — even though array
    /// indices may be any width, because the index names a field rather than scaling
    /// an offset.
    #[test]
    fn a_struct_index_must_be_a_constant_i32() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _) = block_with_struct_slot(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let zero = value(0, &mut ctx);
        let wide = Value::from_const(1i64, None, &mut ctx).unwrap();

        let err = cursor
            .add_get_element_ptr(
                None,
                slot.clone(),
                vec![zero.clone(), wide],
                None,
                None,
                &mut ctx,
            )
            .expect_err("an i64 struct index is not valid LLVM");

        assert!(
            matches!(
                &err,
                InstructionError::Gep(GepError::StructIndexNotAConstantI32(t)) if t == "i64"
            ),
            "the error must name the offending type, got: {err}"
        );

        // A register is refused for the same reason: the field has to be known now.
        let reg = Value::from_register("n".to_string(), i32_ty, &mut ctx);

        assert!(
            matches!(
                cursor.add_get_element_ptr(None, slot, vec![zero, reg], None, None, &mut ctx),
                Err(InstructionError::Gep(GepError::StructIndexNotAConstantI32(
                    _
                )))
            ),
            "a non-constant struct index names no field"
        );
    }

    /// A field index past the end names nothing, which `llvm-as` also refuses.
    /// Negative is the same error and not a huge positive one.
    #[test]
    fn a_struct_index_must_be_in_range() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, slot, _) = block_with_struct_slot(&mut ctx, &mut builder);

        let zero = value(0, &mut ctx);
        let past_end = Value::from_const(5i32, None, &mut ctx).unwrap();
        let negative = Value::from_const(-1i32, None, &mut ctx).unwrap();

        let err = cursor
            .add_get_element_ptr(
                None,
                slot.clone(),
                vec![zero.clone(), past_end],
                None,
                None,
                &mut ctx,
            )
            .expect_err("a two-field struct has no field 5");

        assert!(
            matches!(
                &err,
                InstructionError::Gep(GepError::StructIndexOutOfRange { index, fields })
                    if *index == 5 && *fields == 2
            ),
            "the error must name the index and the arity, got: {err}"
        );

        assert!(
            matches!(
                cursor.add_get_element_ptr(None, slot, vec![zero, negative], None, None, &mut ctx),
                Err(InstructionError::Gep(GepError::StructIndexOutOfRange {
                    index: -1,
                    ..
                }))
            ),
            "a negative index is out of range, not a large one"
        );
    }

    /// Only aggregates have anything to descend into: a second index against a scalar
    /// names nothing, and `llvm-as` refuses it too.
    #[test]
    fn only_an_aggregate_can_be_indexed_into() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        let slot = cursor
            .add_alloca(i32_ty, None, None, Some("p"), &mut ctx)
            .unwrap();

        let zero = value(0, &mut ctx);
        let one = Value::from_const(1i32, None, &mut ctx).unwrap();

        let err = cursor
            .add_get_element_ptr(None, slot, vec![zero, one], None, None, &mut ctx)
            .expect_err("an i32 has no elements");

        assert!(
            matches!(
                &err,
                InstructionError::Gep(GepError::TypeNotIndexable(t)) if t == "i32"
            ),
            "the error must name the type, got: {err}"
        );
    }

    /// `store i64 %v` where `%v` is an `i32` is not a store LLVM will assemble, and
    /// widening it would need a `zext`/`sext` the caller never asked for — so the
    /// declared type has to match what the register actually holds.
    #[test]
    fn a_store_refuses_a_register_that_is_not_the_declared_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let reg = Value::from_register("v".to_string(), i32_ty, &mut ctx);

        let err = cursor
            .add_store(reg, ptr, None, Some(i64_ty), &mut ctx)
            .expect_err("an i32 register is not an i64");

        assert!(
            matches!(&err, InstructionError::Store(StoreError::StoredValueTypeMismatch(got, declared))
                if got == "i32" && declared == "i64"),
            "the error must name both types, got: {err}"
        );
    }

    /// A constant, unlike a register, does fold into the declared type — the store
    /// is of the widened constant, so no instruction is skipped.
    #[test]
    fn a_store_widens_a_constant_to_the_declared_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i64_ty = ctx.i64_ty();
        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        cursor
            .add_store(seven, ptr, None, Some(i64_ty), &mut ctx)
            .expect("an i32 constant stores as an i64");

        let block = ctx.blocks.get(cursor.block.raw()).unwrap();

        let InstructionKind::Store(StoreOperands { value, .. }) = &block.instructions[0].kind
        else {
            panic!("expected a store")
        };

        // The *stored* value is the widened one — a store of the original `i32`
        // would be a different instruction than the caller asked for.
        assert_eq!(ctx.display(value.ty()).to_string(), "i64");

        let ValueKind::Const(id) = value.kind() else {
            panic!("expected a constant")
        };

        assert_eq!(*ctx.const_interner.value(id.raw()), ConstValue::I64(7));
    }

    /// `alloca` reserves room for a value, so the type has to have a size. `llvm-as`
    /// refuses `alloca void` ("void type only allowed for function results") and
    /// `alloca` of a function type ("invalid type for alloca") — the same pair
    /// `is_first_class` names, so this gates on the predicate that
    /// `only_void_and_function_types_are_unsized` pins in `value.rs`.
    #[test]
    fn an_alloca_refuses_an_unsized_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let void_ty = ctx.void_ty();

        let err = cursor
            .add_alloca(void_ty, None, None, None, &mut ctx)
            .expect_err("`void` has no size");

        assert!(
            matches!(&err, InstructionError::Alloca(AllocaError::TypeNotAllocatable(t)) if t == "void"),
            "expected a not-allocatable error, got: {err}"
        );
    }

    /// An aggregate is sized, so it allocates — the predicate is "has a size", not
    /// LLVM's narrower "first class".
    #[test]
    fn an_alloca_accepts_an_aggregate_and_yields_a_pointer() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        let array_ty = intern(
            Type::Array {
                size: 4,
                element_ty: i32_ty,
            },
            &mut ctx,
        );

        let slot = cursor
            .add_alloca(array_ty, None, None, Some("buf"), &mut ctx)
            .expect("`[4 x i32]` is sized");

        assert_eq!(
            ctx.display(slot.ty()).to_string(),
            "ptr",
            "an alloca yields a pointer, not the allocated type"
        );
    }

    /// `llvm-as` rejects a non-integer element count with "element count must have
    /// integer type", whether it is the value's own type or the one declared for it.
    #[test]
    fn an_alloca_count_must_be_an_integer() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();
        let a_float = Value::from_const(1.0f32, None, &mut ctx).unwrap();

        let err = cursor
            .add_alloca(i32_ty, Some((a_float, None)), None, None, &mut ctx)
            .expect_err("a float is not a count");

        assert!(
            matches!(&err, InstructionError::Alloca(AllocaError::AllocaCountNotAnInteger(t)) if t == "float"),
            "the error must name the offending type, got: {err}"
        );

        // The declared type is checked too, so a count that *is* an integer cannot be
        // relabelled into something that is not.
        let an_int = Value::from_const(4i32, None, &mut ctx).unwrap();

        assert!(
            matches!(
                cursor.add_alloca(i32_ty, Some((an_int, Some(f64_ty))), None, None, &mut ctx),
                Err(InstructionError::Alloca(
                    AllocaError::AllocaCountNotAnInteger(_)
                ))
            ),
            "`double` is not an integer, whoever supplied it"
        );
    }

    /// `i1` counts as an integer here because it does in LLVM: `alloca i32, i1 %c`
    /// assembles.
    #[test]
    fn an_alloca_count_may_be_an_i1() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let one = Value::from_const(true, None, &mut ctx).unwrap();

        assert!(
            cursor
                .add_alloca(i32_ty, Some((one, None)), None, None, &mut ctx)
                .is_ok(),
            "`alloca i32, i1 %c` is valid LLVM"
        );
    }

    /// A register count is checked, not converted — the same rule as a stored value,
    /// since widening one would need an instruction of its own.
    #[test]
    fn an_alloca_refuses_a_register_count_of_the_wrong_width() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, _) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let n = Value::from_register("n".to_string(), i32_ty, &mut ctx);

        let err = cursor
            .add_alloca(i32_ty, Some((n, Some(i64_ty))), None, None, &mut ctx)
            .expect_err("an i32 register is not an i64 count");

        assert!(
            matches!(&err, InstructionError::Alloca(AllocaError::AllocaCountTypeMismatch(got, declared))
                if got == "i32" && declared == "i64"),
            "the error must name both types, got: {err}"
        );
    }

    /// A register's definition is recorded with the block it was defined in, not just
    /// the instruction index.
    ///
    /// The index is a position in *that block's* instruction list, so on its own it
    /// says nothing — reading it against another block lands on an unrelated
    /// instruction, or past the end when that block is shorter. This is the shape
    /// `try_inferring_pointee_ty` depends on to walk back to an `alloca` in the entry
    /// block from a `store` in a later one.
    #[test]
    fn a_register_definition_records_the_block_it_was_defined_in() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let ptr = Value::from_const(NullPtr, None, &mut ctx).unwrap();

        // Two instructions in `entry`, so the register in `body` gets an index that
        // is only valid against `body` — index 0 of a one-instruction list.
        let in_entry = builder.cursor_at_block(entry);

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f64_ty = ctx.f64_ty();

        in_entry
            .add_load(i32_ty, ptr.clone(), None, Some("a"), &mut ctx)
            .unwrap();

        let second = in_entry
            .add_load(i64_ty, ptr.clone(), None, Some("b"), &mut ctx)
            .unwrap();

        let in_body = builder.cursor_at_block(body);

        let third = in_body
            .add_load(f64_ty, ptr, None, Some("c"), &mut ctx)
            .unwrap();

        let func_id = ctx.get_block(entry).func_id;

        let def_of = |val: &Value, ctx: &Context| {
            let ValueKind::Reg(reg) = val.kind() else {
                panic!("a load defines a register")
            };

            *ctx.register_defs(func_id)
                .get(&reg.name)
                .expect("the definition was recorded")
        };

        let second_def = def_of(&second, &ctx);
        let third_def = def_of(&third, &ctx);

        assert_eq!(second_def.block, entry);
        assert_eq!(second_def.instr_index, 1);
        assert_eq!(third_def.block, body, "`c` belongs to the block it was in");
        assert_eq!(third_def.instr_index, 0, "and `body` numbers from 0 again");

        // The indices collide across blocks, which is exactly why the block has to be
        // stored: `c`'s index resolved against `entry` reads `a`, a different
        // instruction of a different type.
        let loaded_ty_at = |b, index: usize, ctx: &Context| {
            let InstructionKind::Load(LoadOperands { ty, .. }) =
                &ctx.get_block(b).instructions[index].kind
            else {
                panic!("expected a load")
            };

            ctx.display(*ty).to_string()
        };

        assert_eq!(
            loaded_ty_at(third_def.block, third_def.instr_index, &ctx),
            "double"
        );

        assert_eq!(
            loaded_ty_at(entry, third_def.instr_index, &ctx),
            "i32",
            "the same index against the wrong block reads an unrelated instruction"
        );
    }

    /// A load yields a value of the type it *loaded*, not of the pointer it read
    /// through — `%x = load i32, ptr %p` defines an `i32`.
    #[test]
    fn a_load_yields_a_value_of_the_loaded_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        let loaded = cursor
            .add_load(i32_ty, ptr, None, Some("x"), &mut ctx)
            .expect("loading an i32 through a ptr is fine");

        assert_eq!(ctx.ty_interner.value(loaded.ty().raw()), &Type::I32);
    }

    /// The instruction records the register it defines, which is what an emitter
    /// writes in front of it.
    #[test]
    fn a_load_records_the_register_it_defines() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i64_ty = ctx.i64_ty();

        cursor
            .add_load(i64_ty, ptr, None, Some("x"), &mut ctx)
            .unwrap();

        let block = ctx.blocks.get(cursor.block.raw()).unwrap();

        assert_eq!(block.instructions.len(), 1);

        assert!(
            block.instructions[0].value.is_some(),
            "a load produces a value, so it defines a register"
        );
    }

    /// Memory is addressed through a pointer. Reaching it from an integer needs an
    /// `inttoptr` first, so a non-pointer operand is refused rather than folded.
    #[test]
    fn a_load_needs_a_pointer_operand() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);
        let not_a_ptr = value(7, &mut ctx);

        let i32_ty = ctx.i32_ty();

        let err = cursor
            .add_load(i32_ty, not_a_ptr, None, Some("x"), &mut ctx)
            .expect_err("an i32 is not an address");

        assert!(
            matches!(&err, InstructionError::PointerOperandExpected(t) if t == "i32"),
            "the error must name the offending type, got: {err}"
        );

        assert!(
            ctx.blocks.get(entry.raw()).unwrap().instructions.is_empty(),
            "the refused load must not have been added"
        );
    }

    /// `void` has no size, so there is nothing to load. `llvm-as` rejects it with
    /// "void type only allowed for function results".
    #[test]
    fn only_sized_types_can_be_loaded() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        // `Type::Func` is the other unsized case; it is covered in `value.rs`,
        // where a `FuncSignature` can be built.
        let void_ty = ctx.void_ty();

        let err = cursor
            .add_load(void_ty, ptr, None, None, &mut ctx)
            .expect_err("`void` has no size");

        assert!(
            matches!(&err, InstructionError::TypeNotLoadable(t) if t == "void"),
            "expected a not-loadable error, got: {err}"
        );
    }

    /// Aggregates are loadable even though LLVM does not call them "first class" —
    /// `load {i32, i32}` and `load [4 x i32]` both assemble.
    #[test]
    fn aggregates_can_be_loaded() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = intern(Type::I32, &mut ctx);

        for ty in [
            Type::Struct {
                fields: Box::new([i32_ty, i32_ty]),
                packed: false,
            },
            Type::Array {
                size: 4,
                element_ty: i32_ty,
            },
            Type::Ptr,
            Type::Double,
        ] {
            let id = intern(ty, &mut ctx);
            let spelled = rendered(id, &ctx);

            assert!(
                cursor
                    .add_load(id, ptr.clone(), None, None, &mut ctx)
                    .is_ok(),
                "`{spelled}` is loadable"
            );
        }
    }

    /// An explicit alignment must be a power of two. The cases below are the ones
    /// `llvm-as` accepts and rejects — note `1` is valid (it is `2^0`) and `0` is
    /// not, which an even-number test gets backwards in both directions.
    #[test]
    fn an_explicit_alignment_must_be_a_power_of_two() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        for align in [1, 2, 4, 8, 16, 4096] {
            assert!(
                cursor
                    .add_load(i32_ty, ptr.clone(), Some(align), None, &mut ctx)
                    .is_ok(),
                "align {align} is a power of two"
            );
        }

        for align in [0, 3, 6, 10, 12] {
            let err = cursor
                .add_load(i32_ty, ptr.clone(), Some(align), None, &mut ctx)
                .expect_err("not a power of two");

            assert!(
                matches!(&err, InstructionError::AlignmentNotPowerOfTwo(a) if *a == align),
                "expected an alignment error for {align}, got: {err}"
            );
        }
    }

    /// Omitting the alignment is how the ABI default is asked for, and is always
    /// allowed.
    #[test]
    fn an_omitted_alignment_is_allowed() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let i32_ty = ctx.i32_ty();

        assert!(cursor.add_load(i32_ty, ptr, None, None, &mut ctx).is_ok());
    }
}
