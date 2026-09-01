//! Basic blocks: a label, its phis, and a straight run of instructions.

use crate::{
    cfg::{context::Context, function::FuncId},
    error::{InstructionError, PhiError},
    instruction::{Instruction, PhiInstrHandler, PhiInstruction},
    interner::StrId,
};
use id_arena::Id;

/// One basic block.
///
/// Phis are kept apart from the other instructions rather than at the head of the
/// same list, because LLVM requires them to come first and separating them makes that
/// unrepresentable rather than merely checked.
pub struct BasicBlock {
    pub(crate) name: StrId,
    /// Whether this is the function's entry block. Entry blocks have no
    /// predecessors, so a phi cannot go in one.
    pub(crate) is_first: bool,
    pub(crate) func_id: FuncId,
    pub(crate) phis: Vec<PhiInstruction>,
    pub(crate) instructions: Vec<Instruction>,
    /// Set once a terminator has been added. See
    /// [`set_locked`](BasicBlockId::set_locked).
    pub(crate) is_locked: bool,
}

/// A handle to a [`BasicBlock`] in a [`Context`]'s arena.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BasicBlockId(Id<BasicBlock>);

impl BasicBlockId {
    /// Wraps an arena id. Only [`FuncId::add_basic_block`](crate::cfg::function::FuncId::add_basic_block)
    /// calls this.
    pub(crate) fn new(id: Id<BasicBlock>) -> Self {
        BasicBlockId(id)
    }

    /// The underlying arena id.
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
    /// Appends an instruction, returning its index in the block.
    ///
    /// # Errors
    ///
    /// [`InstructionError::BasicBlockAlreadyTerminated`] if the block already ends in
    /// a terminator.
    pub(crate) fn add_instruction(
        &self,
        instr: Instruction,
        ctx: &mut Context,
    ) -> Result<usize, InstructionError> {
        if ctx.get_block(*self).is_locked {
            return Err(InstructionError::BasicBlockAlreadyTerminated(
                self.name(ctx),
            ));
        }

        let block = ctx.get_block_mut(*self);
        let index = block.instructions.len();

        block.instructions.push(instr);

        Ok(index)
    }

    /// Appends a phi, returning a handle for adding incoming branches to it.
    ///
    /// # Errors
    ///
    /// The three checks run in this order, and the order is deliberate: a terminated
    /// block satisfies both the first and the last, and being terminated is the more
    /// useful thing to report.
    ///
    /// - [`PhiError::BasicBlockAlreadyTerminated`] — the block already ends in a
    ///   terminator.
    /// - [`PhiError::PhiInstructionCannotBeAddedToEntryBasicBlock`] — the entry block
    ///   has no predecessors, so a phi there selects on nothing.
    /// - [`PhiError::PhiInstructionAddError`] — an instruction is already present, and
    ///   phis come first.
    pub(crate) fn add_phi(
        &self,
        instr: PhiInstruction,
        ctx: &mut Context,
    ) -> Result<PhiInstrHandler, PhiError> {
        if ctx.get_block(*self).is_locked {
            return Err(PhiError::BasicBlockAlreadyTerminated(self.name(ctx)));
        }

        let block = ctx.get_block_mut(*self);

        if block.is_first {
            return Err(PhiError::PhiInstructionCannotBeAddedToEntryBasicBlock);
        }

        if !block.instructions.is_empty() {
            return Err(PhiError::PhiInstructionAddError);
        }

        let id = block.phis.len();

        block.phis.push(instr);

        Ok(PhiInstrHandler {
            index: id,
            block: *self,
        })
    }

    /// The block's label, for an error message.
    pub(crate) fn name(&self, ctx: &Context) -> String {
        let block = ctx.get_block(*self);

        ctx.str_interner.value(block.name.0).to_string()
    }

    /// Marks the block as ended, so nothing more can be added to it.
    ///
    /// Called by the terminator builders after the terminator itself is in. This is
    /// the *runtime* half of the guarantee that a block ends exactly once: the
    /// compile-time half is that those builders consume the
    /// [`Cursor`](crate::instruction::cursor::Cursor). Consuming the cursor cannot
    /// see a second cursor opened at the same block, which is what this catches.
    pub(crate) fn set_locked(&self, ctx: &mut Context) {
        let block = ctx.get_block_mut(*self);

        block.is_locked = true;
    }
}
