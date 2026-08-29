use crate::{
    cfg::{Cursor, basic_block::BasicBlockId, context::Context},
    constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID,
    error::BuildError,
    value::{I1Value, Type, Value},
};
use rustc_hash::FxHashSet;

pub struct PhiInstruction {
    pub(crate) branches: Vec<(BasicBlockId, Value)>,
    pub(crate) blocks: FxHashSet<BasicBlockId>,
    ref_ty: Type,
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
        let block = ctx
            .blocks
            .get_mut(self.block.raw())
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        let index = self.index;
        let instr = &mut block.phis[index];

        let branch_ty = branch.1.ty();

        if branch_ty != &instr.ref_ty {
            return Err(BuildError::PhiInstructionBranchTypeMismatch(
                instr.ref_ty.clone(),
                branch_ty.clone(),
            ));
        }

        let block_id = branch.0;

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
        ty: Type,
        ptr: Value,
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

        let func_id = ctx
            .blocks
            .get(self.block.raw())
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
            .func_id;

        let reg_name = ctx.name_for_reg(reg, func_id)?;
        let val = Value::from_register(reg_name, ref_ty.clone(), &mut ctx.str_interner)?;

        let phi_id = self.block.add_phi(
            PhiInstruction {
                branches: vec![],
                blocks: FxHashSet::default(),
                ref_ty: ref_ty.clone(),
                value: val.clone(),
            },
            ctx,
        )?;

        for (branch, val) in branches {
            phi_id.add_branch((*branch, val.clone()), ctx)?;
        }

        Ok((phi_id, val))
    }

    pub fn add_load(
        &self,
        ty: Type,
        ptr: Value,
        align: Option<u32>,
        reg: Option<&str>,
        ctx: &mut Context,
    ) -> Result<Value, BuildError> {
        let ptr_ty = ptr.ty();

        if !ptr_ty.is_ptr() {
            return Err(BuildError::PointerOperandExpected(ptr_ty.clone()));
        }

        if !ty.is_first_class() {
            return Err(BuildError::TypeNotLoadable(ty));
        }

        if let Some(align) = align
            && !align.is_power_of_two()
        {
            return Err(BuildError::AlignmentNotPowerOfTwo(align));
        }

        let func_id = ctx
            .blocks
            .get(self.block.raw())
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
            .func_id;

        let reg_name = ctx.name_for_reg(reg, func_id)?;
        let val = Value::from_register(reg_name, ty.clone(), &mut ctx.str_interner)?;

        self.block.add_instruction(
            Instruction {
                kind: InstructionKind::Load { ty, ptr, align },
                value: Some(val.clone()),
            },
            ctx,
        );

        // let val = add_instr!(InstructionKind::Load { ty, ptr, align }, ty)

        Ok(val)
    }
}
