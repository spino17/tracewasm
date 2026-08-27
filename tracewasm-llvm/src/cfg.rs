use crate::{constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID, error::BuildError};
use id_arena::{Arena, Id};

pub struct PhiInstruction {}

pub struct PhiInstrId(usize);

impl PhiInstrId {
    pub(crate) fn new(id: usize) -> Self {
        PhiInstrId(id)
    }
}

pub struct BasicBlock<I> {
    name: String,
    is_first: bool,
    func_id: FuncId<I>,
    phis: Vec<PhiInstruction>,
    instructions: Vec<I>,
}

pub struct BasicBlockId<I>(Id<BasicBlock<I>>);

impl<I> Clone for BasicBlockId<I> {
    fn clone(&self) -> Self {
        BasicBlockId(self.0)
    }
}

impl<I> Copy for BasicBlockId<I> {}

impl<I> BasicBlockId<I> {
    fn add_instruction(&self, instr: I, ctx: &mut Context<I>) {
        let block = ctx
            .blocks
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        block.instructions.push(instr);
    }

    fn add_phi(
        &self,
        instr: PhiInstruction,
        ctx: &mut Context<I>,
    ) -> Result<PhiInstrId, BuildError> {
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

        Ok(PhiInstrId::new(id))
    }
}

pub struct Function<I> {
    name: String,
    blocks: Vec<BasicBlockId<I>>,
}

pub struct FuncId<I>(Id<Function<I>>);

impl<I> Clone for FuncId<I> {
    fn clone(&self) -> Self {
        FuncId(self.0)
    }
}

impl<I> Copy for FuncId<I> {}

impl<I> FuncId<I> {
    pub fn add_basic_block(&self, name: String, ctx: &mut Context<I>) -> BasicBlockId<I> {
        let is_first = ctx
            .funcs
            .get(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
            .blocks
            .is_empty();

        let id = BasicBlockId(ctx.blocks.alloc(BasicBlock {
            name,
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

        id
    }
}

pub struct Global {}

struct Module<I> {
    triple: String,
    data_layout: String,
    globals: Vec<Global>,
    functions: Vec<FuncId<I>>,
}

pub struct Context<I> {
    blocks: Arena<BasicBlock<I>>,
    funcs: Arena<Function<I>>,
}

impl<I> Default for Context<I> {
    fn default() -> Self {
        Context {
            blocks: Arena::default(),
            funcs: Arena::default(),
        }
    }
}

pub struct Cursor<I> {
    block: BasicBlockId<I>,
}

impl<I> Cursor<I> {
    pub fn add_instruction(&mut self, instr: I, ctx: &mut Context<I>) {
        self.block.add_instruction(instr, ctx);
    }

    pub fn add_phi(
        &mut self,
        instr: PhiInstruction,
        ctx: &mut Context<I>,
    ) -> Result<PhiInstrId, BuildError> {
        self.block.add_phi(instr, ctx)
    }
}

pub struct Builder<I> {
    module: Module<I>,
}

impl<I> Builder<I> {
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

    pub fn cursor_at_block(&mut self, id: BasicBlockId<I>) -> Cursor<I> {
        Cursor { block: id }
    }

    pub fn add_function(&mut self, name: String, ctx: &mut Context<I>) -> FuncId<I> {
        let id = FuncId(ctx.funcs.alloc(Function {
            name,
            blocks: vec![],
        }));

        self.module.functions.push(id);

        id
    }

    pub fn build(self) -> ControlFlowGraph<I> {
        ControlFlowGraph {
            module: self.module,
        }
    }
}

pub struct ControlFlowGraph<I> {
    module: Module<I>,
}

impl<I> ControlFlowGraph<I> {
    pub fn emit_ll(&self, ctx: &Context<I>) -> String {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::Instruction;

    #[test]
    fn simple_api_usage() {
        let mut ctx = Context::<Instruction>::default();
        let mut builder = Builder::new("".to_string(), "".to_string());

        let func = builder.add_function("sum".to_string(), &mut ctx);
        let entry = func.add_basic_block("entry".to_string(), &mut ctx);

        let mut cursor = builder.cursor_at_block(entry);
        let _ = cursor.add_phi(PhiInstruction {}, &mut ctx).unwrap();

        let _cfg = builder.build();
    }
}
