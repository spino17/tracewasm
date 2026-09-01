use crate::cfg::{basic_block::BasicBlockId, module::Module};
pub mod basic_block;
pub mod builder;
pub mod context;
pub mod emit;
pub mod function;
pub mod global;
pub mod module;

pub struct ControlFlowGraph {
    pub(crate) module: Module,
}

#[cfg(test)]
mod tests {
    use crate::{error::ContextError, test_support::fixture};

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
            matches!(&err, ContextError::DuplicateFunctionName(name) if name == "sum"),
            "the error must name the collision, got: {err}"
        );

        assert_eq!(
            builder.module.functions.len(),
            1,
            "the refused function must not have been added"
        );
    }
}
