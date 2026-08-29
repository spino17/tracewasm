use crate::{
    cfg::{
        basic_block::BasicBlockId,
        context::Context,
        function::{FuncId, Function},
    },
    error::BuildError,
    interner::StrId,
};
use rustc_hash::FxHashSet;

pub mod basic_block;
pub mod context;
pub mod function;

pub struct Global {}

struct Module {
    triple: String,
    data_layout: String,
    globals: Vec<Global>,
    functions: Vec<FuncId>,
    func_names: FxHashSet<StrId>,
}

pub struct Cursor {
    pub(crate) block: BasicBlockId,
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
                func_names: FxHashSet::default(),
            },
        }
    }

    pub fn cursor_at_block(&mut self, id: BasicBlockId) -> Cursor {
        Cursor { block: id }
    }

    pub fn add_function(&mut self, name: String, ctx: &mut Context) -> Result<FuncId, BuildError> {
        let name_id: StrId = ctx.str_interner.intern(name)?.into();

        if self.module.func_names.contains(&name_id) {
            return Err(BuildError::DuplicateFunctionName(
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

pub struct ControlFlowGraph {
    module: Module,
}

impl ControlFlowGraph {
    pub fn emit_ll(&self, _ctx: &Context) -> String {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::fixture;

    #[test]
    fn simple_api_usage() {
        let (mut ctx, mut builder) = fixture();
        let func = builder.add_function("sum".to_string(), &mut ctx).unwrap();
        let entry = func.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        cursor.add_unconditional_br(entry, &mut ctx);

        assert_eq!(
            ctx.blocks.get(entry.raw()).unwrap().instructions.len(),
            1,
            "the cursor writes into the block it was opened at"
        );

        let cfg = builder.build();

        assert_eq!(cfg.module.functions.len(), 1);
    }

    /// Every function is its own entry, and the module records them in the order
    /// they were added.
    #[test]
    fn functions_get_distinct_ids_in_order() {
        let (mut ctx, mut builder) = fixture();
        let a = builder.add_function("a".to_string(), &mut ctx).unwrap();
        let b = builder.add_function("b".to_string(), &mut ctx).unwrap();

        assert_ne!(a.raw(), b.raw());
        assert_eq!(builder.module.functions.len(), 2);
        assert_eq!(ctx.funcs.len(), 2);
    }

    /// LLVM identifies a definition by its name, so two `@sum`s in one module is
    /// not something to discover at emit time.
    #[test]
    fn a_duplicate_function_name_is_refused() {
        let (mut ctx, mut builder) = fixture();

        builder.add_function("sum".to_string(), &mut ctx).unwrap();

        let err = builder
            .add_function("sum".to_string(), &mut ctx)
            .expect_err("the name is taken");

        assert!(
            matches!(&err, BuildError::DuplicateFunctionName(name) if name == "sum"),
            "the error must name the collision, got: {err}"
        );

        assert_eq!(
            builder.module.functions.len(),
            1,
            "the refused function must not have been added"
        );
    }
}
