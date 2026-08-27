use crate::constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID;
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
    fn is_phi_ongoing(&self, ctx: &Context<I>) -> bool {
        let block = ctx
            .blocks
            .get(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        block.instructions.is_empty()
    }

    fn add_instruction(&self, instr: I, ctx: &mut Context<I>) {
        let block = ctx
            .blocks
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        block.instructions.push(instr);
    }

    fn add_phi(&self, instr: PhiInstruction, ctx: &mut Context<I>) -> PhiInstrId {
        let block = ctx
            .blocks
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        let id = block.phis.len();

        block.phis.push(instr);

        PhiInstrId::new(id)
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
        let id = BasicBlockId(ctx.blocks.alloc(BasicBlock {
            name,
            func_id: *self,
            phis: vec![],
            instructions: vec![],
        }));

        let func = ctx
            .funcs
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        func.blocks.push(id);

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
    is_phi_ongoing: bool,
}

impl<I> Cursor<I> {
    pub fn add_instruction(&mut self, instr: I, ctx: &mut Context<I>) {
        self.is_phi_ongoing = false;
        self.block.add_instruction(instr, ctx);
    }

    pub fn add_phi(&mut self, instr: PhiInstruction, ctx: &mut Context<I>) -> PhiInstrId {
        if !self.is_phi_ongoing {
            panic!("phi instructions should be added at the start of the basic block")
        }

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

    pub fn cursor_at_block(&mut self, id: BasicBlockId<I>, ctx: &Context<I>) -> Cursor<I> {
        let is_phi_ongoing = id.is_phi_ongoing(ctx);

        Cursor {
            block: id,
            is_phi_ongoing,
        }
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

        let mut cursor = builder.cursor_at_block(entry, &ctx);
        cursor.add_phi(PhiInstruction {}, &mut ctx);

        let _cfg = builder.build();
    }
}
