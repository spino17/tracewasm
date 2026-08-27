#![allow(warnings)] // TODO: remove this, just use it while developing

use crate::{
    constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID,
    instruction::{Instruction, PhiInstruction},
};
use id_arena::{Arena, Id};

#[derive(Clone, Copy)]
pub struct BasicBlockId(Id<BasicBlock>);

pub struct BasicBlock {
    name: String,
    func_id: FuncId,
    instructions: Vec<Instruction>,
}

#[derive(Clone, Copy)]
pub struct FuncId(Id<Function>);

impl FuncId {
    pub fn add_basic_block(&self, name: String, ctx: &mut Context) -> BasicBlockId {
        BasicBlockId(ctx.blocks.alloc(BasicBlock {
            name,
            func_id: *self,
            instructions: vec![],
        }))
    }
}

pub struct Function {
    name: String,
    blocks: Vec<BasicBlockId>,
}

pub struct Global {}

#[derive(Default)]
pub struct Context {
    blocks: Arena<BasicBlock>,
    funcs: Arena<Function>,
}

struct Module {
    triple: String,
    data_layout: String,
    globals: Vec<Global>,
    functions: Vec<FuncId>,
}

struct Cursor {
    block: BasicBlockId,
}

pub struct CfgBuilderCursor {
    block: BasicBlockId,
}

impl CfgBuilderCursor {
    pub fn add_instruction(&self, instr: Instruction, ctx: &mut Context) {
        let block_id = self.block;

        let block = ctx
            .blocks
            .get_mut(block_id.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        block.instructions.push(instr);
    }
}

pub struct CfgBuilder {
    module: Module,
}

impl CfgBuilder {
    pub fn new(triple: String, data_layout: String) -> Self {
        CfgBuilder {
            module: Module {
                triple,
                data_layout,
                globals: vec![],
                functions: vec![],
            },
        }
    }

    pub fn at_block_end(&mut self, id: BasicBlockId) -> CfgBuilderCursor {
        CfgBuilderCursor { block: id }
    }

    pub fn add_function(&mut self, name: String, ctx: &mut Context) -> FuncId {
        let id = FuncId(ctx.funcs.alloc(Function {
            name,
            blocks: vec![],
        }));

        self.module.functions.push(id);

        id
    }

    pub fn get_basic_block<'a>(&self, id: BasicBlockId, ctx: &'a Context) -> &'a BasicBlock {
        ctx.blocks
            .get(id.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
    }

    pub fn get_basic_block_mut<'a>(
        &self,
        id: BasicBlockId,
        ctx: &'a mut Context,
    ) -> &'a mut BasicBlock {
        ctx.blocks
            .get_mut(id.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID)
    }

    pub fn build(self) -> Cfg {
        Cfg {
            module: self.module,
        }
    }
}

pub struct Cfg {
    module: Module,
}

impl Cfg {
    pub fn emit_ll(&self) -> String {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_api_usage() {
        let mut ctx = Context::default();
        let mut builder = CfgBuilder::new("".to_string(), "".to_string());

        let func = builder.add_function("sum".to_string(), &mut ctx);
        let entry = func.add_basic_block("entry".to_string(), &mut ctx);

        let cursor = builder.at_block_end(entry);
        cursor.add_instruction(Instruction::Phi(PhiInstruction {}), &mut ctx);

        let cfg = builder.build();
    }
}
