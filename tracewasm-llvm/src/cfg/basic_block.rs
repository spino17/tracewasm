use crate::{
    cfg::{context::Context, function::FuncId},
    constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID,
    error::BuildError,
    instruction::{Instruction, PhiInstrHandler, PhiInstruction},
    interner::StrId,
};
use id_arena::Id;

pub struct BasicBlock {
    pub(crate) name: StrId,
    pub(crate) is_first: bool,
    pub(crate) func_id: FuncId,
    pub(crate) phis: Vec<PhiInstruction>,
    pub(crate) instructions: Vec<Instruction>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BasicBlockId(Id<BasicBlock>);

impl BasicBlockId {
    pub(crate) fn new(id: Id<BasicBlock>) -> Self {
        BasicBlockId(id)
    }

    pub(crate) fn raw(&self) -> Id<BasicBlock> {
        self.0
    }
}

impl Clone for BasicBlockId {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for BasicBlockId {}

impl BasicBlockId {
    pub(crate) fn add_instruction(&self, instr: Instruction, ctx: &mut Context) {
        let block = ctx
            .blocks
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        block.instructions.push(instr);
    }

    pub(crate) fn add_phi(
        &self,
        instr: PhiInstruction,
        ctx: &mut Context,
    ) -> Result<PhiInstrHandler, BuildError> {
        let block = ctx
            .blocks
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        if block.is_first {
            return Err(BuildError::PhiInstructionCannotBeAddedToEntryBasicBlock);
        }

        if !block.instructions.is_empty() {
            return Err(BuildError::PhiInstructionAddError);
        }

        let id = block.phis.len();

        block.phis.push(instr);

        Ok(PhiInstrHandler {
            index: id,
            block: *self,
        })
    }
}
