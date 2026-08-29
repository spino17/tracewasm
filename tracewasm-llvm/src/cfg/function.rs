use crate::{
    cfg::{
        basic_block::{BasicBlock, BasicBlockId},
        context::Context,
    },
    constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID,
    error::BuildError,
    interner::StrId,
};
use id_arena::Id;
use rustc_hash::FxHashSet;

pub struct Function {
    pub(crate) name: StrId,
    pub(crate) blocks: Vec<BasicBlockId>,
    pub(crate) block_names: FxHashSet<StrId>,
}

#[derive(Debug)]
pub struct FuncId(Id<Function>);

impl Clone for FuncId {
    fn clone(&self) -> Self {
        FuncId(self.0)
    }
}

impl Copy for FuncId {}

impl FuncId {
    pub(crate) fn new(id: Id<Function>) -> Self {
        FuncId(id)
    }

    pub(crate) fn raw(&self) -> Id<Function> {
        self.0
    }

    pub fn add_basic_block(
        &self,
        name: String,
        ctx: &mut Context,
    ) -> Result<BasicBlockId, BuildError> {
        let name_id: StrId = ctx.str_interner.intern(name)?.into();

        let func = ctx
            .funcs
            .get(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        if func.block_names.contains(&name_id) {
            return Err(BuildError::DuplicateBasicBlockName(
                ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let is_first = func.blocks.is_empty();

        let id = BasicBlockId::new(ctx.blocks.alloc(BasicBlock {
            name: name_id,
            is_first,
            func_id: *self,
            phis: vec![],
            instructions: vec![],
        }));

        let func = ctx
            .funcs
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        func.blocks.push(id);
        func.block_names.insert(name_id);

        Ok(id)
    }
}
