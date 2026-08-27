use crate::{
    constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID,
    instruction::{Instruction, PhiInstrId, PhiInstruction},
};
use id_arena::{Arena, Id};

#[derive(Clone, Copy)]
pub struct BasicBlockId(Id<BasicBlock>);

pub struct BasicBlock {
    name: String,
    func_id: FuncId,
    phis: Vec<PhiInstruction>,
    instructions: Vec<Instruction>,
}

impl BasicBlock {
    fn is_phi_ongoing(&self) -> bool {
        self.instructions.is_empty()
    }
}

#[derive(Clone, Copy)]
pub struct FuncId(Id<Function>);

impl FuncId {
    pub fn add_basic_block(&self, name: String, ctx: &mut Context) -> BasicBlockId {
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

pub struct Function {
    name: String,
    blocks: Vec<BasicBlockId>,
}

pub struct Global {}

struct Module {
    triple: String,
    data_layout: String,
    globals: Vec<Global>,
    functions: Vec<FuncId>,
}

#[derive(Default)]
pub struct Context {
    blocks: Arena<BasicBlock>,
    funcs: Arena<Function>,
}

pub struct Cursor {
    block: BasicBlockId,
    is_phi_ongoing: bool,
}

impl Cursor {
    pub fn add_instruction(&mut self, instr: Instruction, ctx: &mut Context) {
        let block_id = self.block;

        let block = ctx
            .blocks
            .get_mut(block_id.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        if let Instruction::Phi(instr) = instr {
            if !self.is_phi_ongoing {
                panic!("phi instructions should be added at the start of the basic block")
            }

            block.phis.push(instr);
        } else {
            self.is_phi_ongoing = false;

            block.instructions.push(instr);
        }
    }

    pub fn add_phi(&mut self, instr: PhiInstruction, ctx: &mut Context) -> PhiInstrId {
        let block_id = self.block;

        let block = ctx
            .blocks
            .get_mut(block_id.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        if !self.is_phi_ongoing {
            panic!("phi instructions should be added at the start of the basic block")
        }

        let id = block.phis.len();
        block.phis.push(instr);

        PhiInstrId::new(id)
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

    pub fn cursor_at_block(&mut self, id: BasicBlockId, ctx: &Context) -> Cursor {
        let is_phi_ongoing = ctx
            .blocks
            .get(id.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
            .is_phi_ongoing();

        Cursor {
            block: id,
            is_phi_ongoing,
        }
    }

    pub fn add_function(&mut self, name: String, ctx: &mut Context) -> FuncId {
        let id = FuncId(ctx.funcs.alloc(Function {
            name,
            blocks: vec![],
        }));

        self.module.functions.push(id);

        id
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

        let func = builder.add_function("sum".to_string(), &mut ctx);
        let entry = func.add_basic_block("entry".to_string(), &mut ctx);

        let mut cursor = builder.cursor_at_block(entry, &ctx);
        cursor.add_instruction(Instruction::Phi(PhiInstruction {}), &mut ctx);

        let _cfg = builder.build();
    }
}
