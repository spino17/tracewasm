use crate::{
    cfg::{
        ControlFlowGraph,
        basic_block::BasicBlockId,
        context::Context,
        function::{FuncId, Function},
        module::Module,
    },
    error::ContextError,
    instruction::cursor::Cursor,
    interner::StrId,
};
use rustc_hash::FxHashSet;

pub struct Builder {
    pub(crate) module: Module,
}

impl Builder {
    pub fn new(triple: String, data_layout: String) -> Self {
        Builder {
            module: Module {
                triple,
                data_layout,
                globals: vec![],
                functions: vec![],
                func_names: FxHashSet::default(),
            },
        }
    }

    pub fn cursor_at_block(&mut self, id: BasicBlockId) -> Cursor {
        Cursor { block: id }
    }

    pub fn add_function(
        &mut self,
        name: String,
        ctx: &mut Context,
    ) -> Result<FuncId, ContextError> {
        let name_id: StrId = ctx.str_interner.intern(name).into();

        if self.module.func_names.contains(&name_id) {
            return Err(ContextError::DuplicateFunctionName(
                ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let id = FuncId::new(ctx.funcs.alloc(Function {
            name: name_id,
            blocks: vec![],
            block_names: FxHashSet::default(),
        }));

        self.module.func_names.insert(name_id);
        self.module.functions.push(id);

        Ok(id)
    }

    pub fn build(self) -> ControlFlowGraph {
        ControlFlowGraph {
            module: self.module,
        }
    }
}
