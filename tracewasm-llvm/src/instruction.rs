use crate::{
    cfg::{Cursor, basic_block::BasicBlockId, context::Context},
    error::BuildError,
    interner::TyId,
    value::{I1Value, Type, Value, ValueKind},
};
use rustc_hash::FxHashSet;

pub struct PhiInstruction {
    pub(crate) branches: Vec<(BasicBlockId, Value)>,
    pub(crate) blocks: FxHashSet<BasicBlockId>,
    ref_ty: TyId,
    value: Value,
}

#[derive(Debug, Clone, Copy)]
pub struct PhiInstrHandler {
    pub(crate) index: usize,
    pub(crate) block: BasicBlockId,
}

impl PhiInstrHandler {
    pub fn basic_block(&self) -> BasicBlockId {
        self.block
    }

    pub fn add_branch(
        &self,
        branch: (BasicBlockId, Value),
        ctx: &mut Context,
    ) -> Result<(), BuildError> {
        let index = self.index;

        // Read the phi's type through a shared borrow first: rendering a mismatch
        // needs the type pool, which cannot be reached while the block is borrowed
        // mutably. An id is `Copy`, so nothing is held across the switch.
        let ref_ty = ctx.get_block(self.block).phis[index].ref_ty;
        let branch_ty = branch.1.ty();

        if branch_ty != ref_ty {
            return Err(BuildError::PhiInstructionBranchTypeMismatch(
                ctx.ty_interner.display(ref_ty).to_string(),
                ctx.ty_interner.display(branch_ty).to_string(),
            ));
        }

        let block_id = branch.0;
        let instr = &mut ctx.get_block_mut(self.block).phis[index];

        if instr.blocks.contains(&block_id) {
            return Err(BuildError::BasicBlockBranchAlreadyInPhiInstruction);
        }

        instr.blocks.insert(block_id);
        instr.branches.push(branch);

        Ok(())
    }
}

pub enum InstructionKind {
    UnconditionalBr {
        label: BasicBlockId,
    },
    ConditionalBr {
        cond: I1Value,
        true_label: BasicBlockId,
        false_label: BasicBlockId,
    },
    Load {
        ty: TyId,
        ptr: Value,
        align: Option<u32>,
    },
    Store {
        value: Value,
        ptr: Value,
        align: Option<u32>,
    },
    Alloca {
        ty: TyId,
        count: Option<Value>,
        align: Option<u32>,
    },
}

pub struct Instruction {
    pub(crate) kind: InstructionKind,
    /// The register this instruction defines, for the `%x =` an emitter writes in
    /// front of it. `None` for the instructions that produce no value.
    pub(crate) value: Option<Value>,
}

impl Cursor {
    pub fn add_phi(
        &self,
        branches: &[(BasicBlockId, Value)],
        reg: Option<&str>,
        ctx: &mut Context,
    ) -> Result<(PhiInstrHandler, Value), BuildError> {
        if branches.is_empty() {
            return Err(BuildError::PhiInstructionWithNoBranches);
        }

        let ref_ty = branches[0].1.ty();
        let func_id = ctx.get_block(self.block).func_id;
        let reg_name = ctx.name_for_reg(reg, func_id)?;
        let val = Value::from_register(reg_name, ref_ty, ctx)?;

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

    pub fn add_unconditional_br(&self, label: BasicBlockId, ctx: &mut Context) {
        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::UnconditionalBr { label },
                value: None,
            },
            ctx,
        );
    }

    pub fn add_conditional_br(
        &self,
        cond: I1Value,
        true_label: BasicBlockId,
        false_label: BasicBlockId,
        ctx: &mut Context,
    ) {
        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::ConditionalBr {
                    cond,
                    true_label,
                    false_label,
                },
                value: None,
            },
            ctx,
        );
    }

    pub fn add_load(
        &self,
        ty: Type,
        ptr: Value,
        align: Option<u32>,
        reg: Option<&str>,
        ctx: &mut Context,
    ) -> Result<Value, BuildError> {
        let ptr_ty = ctx.ty_interner.value(ptr.ty().raw());

        if !ptr_ty.is_ptr() {
            return Err(BuildError::PointerOperandExpected(
                ptr_ty.display(&ctx.ty_interner).to_string(),
            ));
        }

        if !ty.is_first_class() {
            return Err(BuildError::TypeNotLoadable(
                ty.display(&ctx.ty_interner).to_string(),
            ));
        }

        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(BuildError::AlignmentNotPowerOfTwo(align));
        }

        let ty_id = ctx.ty_interner.intern(ty)?.into();

        Ok(add_instruction_to_block_and_get_value(
            InstructionKind::Load {
                ty: ty_id,
                ptr,
                align,
            },
            ty_id,
            self.block,
            reg,
            ctx,
        )?)
    }

    pub fn add_store(
        &self,
        value: Value,
        ptr: Value,
        align: Option<u32>,
        ty: Option<Type>,
        ctx: &mut Context,
    ) -> Result<(), BuildError> {
        let ptr_ty = ctx.ty_interner.value(ptr.ty().raw());

        if !ptr_ty.is_ptr() {
            return Err(BuildError::PointerOperandExpected(
                ptr_ty.display(&ctx.ty_interner).to_string(),
            ));
        }

        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(BuildError::AlignmentNotPowerOfTwo(align));
        }

        let final_val = if let Some(ty) = ty {
            let Some(casted_value) = value.try_cast(ty, ctx)? else {
                todo!() // RAISE ERROR: unable to cast the value in provided type
            };

            casted_value
        } else {
            value
        };

        if let Some(pointee_ty) = ptr.try_inferring_pointee_ty(self.block, ctx)
            && pointee_ty != final_val.ty()
        {
            todo!() // RAISE ERROR: the value type does not match with the type ptr points to!
        }

        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::Store {
                    value: final_val,
                    ptr,
                    align,
                },
                value: None,
            },
            ctx,
        );

        Ok(())
    }

    pub fn add_alloca(
        &self,
        ty: Type,
        count: Option<(Value, Option<Type>)>,
        align: Option<u32>,
        reg: Option<&str>,
        ctx: &mut Context,
    ) -> Result<Value, BuildError> {
        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(BuildError::AlignmentNotPowerOfTwo(align));
        }

        let mut final_count: Option<Value> = None;

        if let Some((count_val, count_expected_ty)) = count {
            if !count_val.is_integer(ctx) {
                todo!() // count is not of type integer
            }

            let final_count_val = if let Some(count_expected_ty) = count_expected_ty {
                if !count_expected_ty.is_integer() {
                    todo!() // RAISE ERROR: type passed for count is not integer
                }

                let Some(casted_val) = count_val.try_cast(count_expected_ty, ctx)? else {
                    todo!() // RAISE ERROR: value casting to type failed!
                };

                casted_val
            } else {
                count_val
            };

            final_count = Some(final_count_val);
        }

        Ok(add_instruction_to_block_and_get_value(
            InstructionKind::Alloca {
                ty: ctx.ty_interner.intern(ty)?.into(),
                count: final_count,
                align,
            },
            ctx.ty_interner.intern(Type::Ptr)?.into(),
            self.block,
            reg,
            ctx,
        )?)
    }
}

fn add_instruction_to_block_and_get_value(
    kind: InstructionKind,
    result_ty: TyId,
    block: BasicBlockId,
    reg: Option<&str>,
    ctx: &mut Context,
) -> Result<Value, BuildError> {
    let func_id = ctx.get_block(block).func_id;
    let reg_name = ctx.name_for_reg(reg, func_id)?;
    let val = Value::from_register(reg_name, result_ty, ctx)?;

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
    );

    let register_def_index = ctx.register_def_instr_index.entry(func_id).or_default();

    register_def_index.insert(reg_name_id, instr_index);

    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cfg::{Builder, context::Context},
        test_support::{fixture, value},
        value::NullPtr,
    };

    /// Interns `ty` and hands back its id, so a test can spell the shape it means
    /// instead of the pool bookkeeping that shape now costs.
    fn intern(ty: Type, ctx: &mut Context) -> TyId {
        ctx.ty_interner.intern(ty).expect("the type interns").into()
    }

    /// How a type spells itself against `ctx`'s pool — which is the form an error
    /// carries, since `BuildError` holds the rendering rather than the type.
    fn rendered(ty: &Type, ctx: &Context) -> String {
        ty.display(&ctx.ty_interner).to_string()
    }

    /// Instructions land in the block the cursor names, in order.
    #[test]
    fn instructions_append_in_order_to_the_open_block() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        cursor.add_unconditional_br(body, &mut ctx);
        cursor.add_unconditional_br(entry, &mut ctx);

        let instrs = &ctx.blocks.get(entry.raw()).unwrap().instructions;

        assert_eq!(instrs.len(), 2);

        assert!(
            matches!(
                instrs[0].kind,
                InstructionKind::UnconditionalBr { label } if label == body
            ),
            "the first instruction is the one added first"
        );

        assert!(
            instrs.iter().all(|i| i.value.is_none()),
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
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let _entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(body);

        let err = cursor
            .add_phi(&[], Some("result"), &mut ctx)
            .expect_err("a phi with no branches has no value and no type");

        assert!(matches!(err, BuildError::PhiInstructionWithNoBranches));

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
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let cursor = builder.cursor_at_block(entry);

        let err = cursor
            .add_phi(&[(entry, v)], Some("result"), &mut ctx)
            .expect_err("no predecessors to choose between");

        assert!(matches!(
            err,
            BuildError::PhiInstructionCannotBeAddedToEntryBasicBlock
        ));
    }

    /// In a later block a phi is fine, and each one gets its own index — tagged
    /// with the block, since an index alone would mean different phis in different
    /// blocks.
    #[test]
    fn phis_in_a_later_block_are_indexed_within_that_block() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
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
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
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
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
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
                BuildError::PhiInstructionBranchTypeMismatch(phi, branch)
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
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
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

        let a = Value::from_register("a".to_string(), first_ty, &mut ctx).unwrap();
        let b = Value::from_register("b".to_string(), second_ty, &mut ctx).unwrap();

        let cursor = builder.cursor_at_block(body);

        let (_, merged) = cursor
            .add_phi(&[(entry, a), (other, b)], Some("merged"), &mut ctx)
            .expect("both branches are `[4 x i32]`");

        assert_eq!(
            ctx.ty_interner.display(merged.ty()).to_string(),
            "[4 x i32]",
            "and the phi carries that type"
        );
    }

    /// Phis have to precede every other instruction in their block, so once one
    /// has been emitted the window has closed.
    #[test]
    fn a_phi_cannot_follow_an_instruction() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let cursor = builder.cursor_at_block(body);

        cursor.add_unconditional_br(body, &mut ctx);

        let err = cursor
            .add_phi(&[(entry, v)], Some("result"), &mut ctx)
            .expect_err("the block already has an instruction");

        assert!(matches!(err, BuildError::PhiInstructionAddError));

        assert!(
            ctx.blocks.get(body.raw()).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    /// A phi names one value per *predecessor*, so the same predecessor twice is a
    /// bug in the caller — and it is a different bug from an entry-block phi.
    #[test]
    fn a_phi_takes_each_predecessor_once() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
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
            matches!(err, BuildError::BasicBlockBranchAlreadyInPhiInstruction),
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
        let f = builder.add_function("f".to_string(), ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), ctx).unwrap();
        let ptr = Value::from_const(NullPtr, None, ctx).unwrap();

        (builder.cursor_at_block(entry), ptr)
    }

    /// A load yields a value of the type it *loaded*, not of the pointer it read
    /// through — `%x = load i32, ptr %p` defines an `i32`.
    #[test]
    fn a_load_yields_a_value_of_the_loaded_type() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        let loaded = cursor
            .add_load(Type::I32, ptr, None, Some("x"), &mut ctx)
            .expect("loading an i32 through a ptr is fine");

        assert_eq!(ctx.ty_interner.value(loaded.ty().raw()), &Type::I32);
    }

    /// The instruction records the register it defines, which is what an emitter
    /// writes in front of it.
    #[test]
    fn a_load_records_the_register_it_defines() {
        let (mut ctx, mut builder) = fixture();
        let (cursor, ptr) = block_with_ptr(&mut ctx, &mut builder);

        cursor
            .add_load(Type::I64, ptr, None, Some("x"), &mut ctx)
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
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);
        let not_a_ptr = value(7, &mut ctx);

        let err = cursor
            .add_load(Type::I32, not_a_ptr, None, Some("x"), &mut ctx)
            .expect_err("an i32 is not an address");

        assert!(
            matches!(&err, BuildError::PointerOperandExpected(t) if t == "i32"),
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
        let err = cursor
            .add_load(Type::Void, ptr, None, None, &mut ctx)
            .expect_err("`void` has no size");

        assert!(
            matches!(&err, BuildError::TypeNotLoadable(t) if t == "void"),
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
            let spelled = rendered(&ty, &ctx);

            assert!(
                cursor
                    .add_load(ty, ptr.clone(), None, None, &mut ctx)
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

        for align in [1, 2, 4, 8, 16, 4096] {
            assert!(
                cursor
                    .add_load(Type::I32, ptr.clone(), Some(align), None, &mut ctx)
                    .is_ok(),
                "align {align} is a power of two"
            );
        }

        for align in [0, 3, 6, 10, 12] {
            let err = cursor
                .add_load(Type::I32, ptr.clone(), Some(align), None, &mut ctx)
                .expect_err("not a power of two");

            assert!(
                matches!(&err, BuildError::AlignmentNotPowerOfTwo(a) if *a == align),
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

        assert!(
            cursor
                .add_load(Type::I32, ptr, None, None, &mut ctx)
                .is_ok()
        );
    }
}
