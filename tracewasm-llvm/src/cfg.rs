use crate::{
    constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID,
    error::BuildError,
    instruction::Instruction,
    interner::{ConstInterner, StrId, StrInterner},
    value::Value,
};
use id_arena::{Arena, Id};
use rustc_hash::FxHashSet;

#[derive(Default)]
pub struct PhiInstruction {
    branches: Vec<(BasicBlockId, Value)>,
    blocks: FxHashSet<BasicBlockId>,
}

impl PhiInstruction {
    pub fn add_branch(&mut self, branch: (BasicBlockId, Value)) -> Result<(), BuildError> {
        let block_id = branch.0;

        if self.blocks.contains(&block_id) {
            return Err(BuildError::BasicBlockBranchAlreadyInPhiInstruction);
        }

        self.blocks.insert(block_id);
        self.branches.push(branch);

        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct PhiInstrId {
    index: usize,
    block: BasicBlockId,
}

pub struct BasicBlock {
    name: StrId,
    is_first: bool,
    func_id: FuncId,
    phis: Vec<PhiInstruction>,
    instructions: Vec<Instruction>,
}

#[derive(PartialEq, Eq, Hash)]
pub struct BasicBlockId(Id<BasicBlock>);

impl Clone for BasicBlockId {
    fn clone(&self) -> Self {
        BasicBlockId(self.0)
    }
}

impl Copy for BasicBlockId {}

impl BasicBlockId {
    fn add_instruction(&self, instr: Instruction, ctx: &mut Context) {
        let block = ctx
            .blocks
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        block.instructions.push(instr);
    }

    fn add_phi(&self, instr: PhiInstruction, ctx: &mut Context) -> Result<PhiInstrId, BuildError> {
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

        Ok(PhiInstrId {
            index: id,
            block: *self,
        })
    }
}

pub struct Function {
    name: StrId,
    blocks: Vec<BasicBlockId>,
}

pub struct FuncId(Id<Function>);

impl Clone for FuncId {
    fn clone(&self) -> Self {
        FuncId(self.0)
    }
}

impl Copy for FuncId {}

impl FuncId {
    pub fn add_basic_block(
        &self,
        name: String,
        ctx: &mut Context,
    ) -> Result<BasicBlockId, BuildError> {
        let name_id = ctx.str_interner.intern(name)?;

        let is_first = ctx
            .funcs
            .get(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
            .blocks
            .is_empty();

        let id = BasicBlockId(ctx.blocks.alloc(BasicBlock {
            name: name_id.into(),
            is_first,
            func_id: *self,
            phis: vec![],
            instructions: vec![],
        }));

        ctx.funcs
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
            .blocks
            .push(id);

        Ok(id)
    }
}

pub struct Global {}

struct Module {
    triple: String,
    data_layout: String,
    globals: Vec<Global>,
    functions: Vec<FuncId>,
}

pub struct Context {
    blocks: Arena<BasicBlock>,
    funcs: Arena<Function>,
    str_interner: StrInterner,
    const_interner: ConstInterner,
}

impl Default for Context {
    fn default() -> Self {
        Context {
            blocks: Arena::default(),
            funcs: Arena::default(),
            str_interner: StrInterner::default(),
            const_interner: ConstInterner::default(),
        }
    }
}

pub struct Cursor {
    block: BasicBlockId,
}

impl Cursor {
    pub fn add_instruction(&mut self, instr: Instruction, ctx: &mut Context) {
        self.block.add_instruction(instr, ctx);
    }

    pub fn add_phi(
        &mut self,
        instr: PhiInstruction,
        ctx: &mut Context,
    ) -> Result<PhiInstrId, BuildError> {
        self.block.add_phi(instr, ctx)
    }
}

pub struct Builder {
    module: Module,
}

impl Builder {
    pub fn new(triple: String, data_layout: String) -> Self {
        Builder {
            module: Module {
                triple,
                data_layout,
                globals: vec![],
                functions: vec![],
            },
        }
    }

    pub fn cursor_at_block(&mut self, id: BasicBlockId) -> Cursor {
        Cursor { block: id }
    }

    pub fn add_function(&mut self, name: String, ctx: &mut Context) -> Result<FuncId, BuildError> {
        // TODO: check if func already exist with this name!
        let name_id = ctx.str_interner.intern(name)?;

        let id = FuncId(ctx.funcs.alloc(Function {
            name: name_id.into(),
            blocks: vec![],
        }));

        self.module.functions.push(id);

        Ok(id)
    }

    pub fn build(self) -> ControlFlowGraph {
        ControlFlowGraph {
            module: self.module,
        }
    }
}

pub struct ControlFlowGraph {
    module: Module,
}

impl ControlFlowGraph {
    pub fn emit_ll(&self, ctx: &Context) -> String {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_api_usage() {
        let mut ctx = Context::default();
        let mut builder = Builder::new("".to_string(), "".to_string());

        let func = builder.add_function("sum".to_string(), &mut ctx).unwrap();
        let entry = func.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        let mut cursor = builder.cursor_at_block(entry);

        let _cfg = builder.build();
    }
}
