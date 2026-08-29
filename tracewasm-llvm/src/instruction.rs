use crate::{
    cfg::{BasicBlockId, Context, Cursor},
    constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID,
    error::BuildError,
    value::{I1Value, Type, Value},
};
use rustc_hash::FxHashSet;

pub struct PhiInstruction {
    pub(crate) branches: Vec<(BasicBlockId, Value)>,
    pub(crate) blocks: FxHashSet<BasicBlockId>,
    ref_ty: Type,
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

        if branch_ty != instr.ref_ty {
            return Err(BuildError::PhiInstructionBranchTypeMismatch(
                instr.ref_ty.clone(),
                branch_ty,
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

pub enum Instruction {
    UnconditionalBr {
        label: BasicBlockId,
    },
    ConditionalBr {
        cond: I1Value,
        true_label: BasicBlockId,
        false_label: BasicBlockId,
    },
    Alloca,
}

impl Cursor {
    pub fn add_unconditional_br(&mut self, label: BasicBlockId, ctx: &mut Context) {
        self.block
            .add_instruction(Instruction::UnconditionalBr { label }, ctx);
    }

    pub fn add_conditional_br(
        &mut self,
        cond: I1Value,
        true_label: BasicBlockId,
        false_label: BasicBlockId,
        ctx: &mut Context,
    ) {
        self.block.add_instruction(
            Instruction::ConditionalBr {
                cond,
                true_label,
                false_label,
            },
            ctx,
        );
    }

    pub fn add_alloca(&mut self, ctx: &mut Context) {
        self.block.add_instruction(Instruction::Alloca, ctx);
    }

    pub fn add_phi(
        &mut self,
        branches: &[(BasicBlockId, Value)],
        reg: &str,
        ctx: &mut Context,
    ) -> Result<(PhiInstrHandler, Value), BuildError> {
        if branches.is_empty() {
            return Err(BuildError::PhiInstructionWithNoBranches);
        }

        let ref_ty = branches[0].1.ty();

        let phi_id = self.block.add_phi(
            PhiInstruction {
                branches: vec![],
                blocks: FxHashSet::default(),
                ref_ty: ref_ty.clone(),
            },
            ctx,
        )?;

        for (branch, val) in branches {
            phi_id.add_branch((*branch, val.clone()), ctx)?;
        }

        let func_id = ctx
            .blocks
            .get(self.block.raw())
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
            .func_id;

        let reg_name = ctx.name_for_reg(Some(reg), func_id);
        let val = Value::from_register(reg_name, ref_ty, &mut ctx.str_interner)?;

        Ok((phi_id, val))
    }
}
