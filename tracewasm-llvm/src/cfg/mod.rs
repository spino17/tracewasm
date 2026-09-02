//! The module being built: functions, blocks, and the context they live in.
//!
//! The pieces divide up like this:
//!
//! - [`context::Context`] owns the storage — arenas for blocks and functions, the
//!   interner pools, and the register bookkeeping.
//! - [`builder::Builder`] owns the module's own contents and hands out cursors.
//! - [`function::FuncId`] and [`basic_block::BasicBlockId`] are handles into the
//!   arenas; both carry their own methods, so `f.add_basic_block(..)` reads like a
//!   method on the function even though the storage lives in the context.
//! - [`walk::CfgVisitor`] traverses a finished graph, and [`emit::IREmitter`] is the
//!   implementation that renders it as text.

use crate::cfg::context::Context;

pub mod basic_block;
pub mod builder;
pub mod context;
pub mod emit;
pub mod function;
pub mod global;
pub mod module;
pub mod walk;

/// A finished module, ready to be walked or emitted.
///
/// Produced by [`Builder::build`](builder::Builder::build), which consumes the
/// builder — the graph is done being constructed, so nothing can be added to it. The
/// [`Context`](context::Context) it was built against is still needed to read it,
/// since everything inside is an id.
pub struct ControlFlowGraph {
    pub(crate) context: Context,
}

#[cfg(test)]
mod tests {
    use crate::{
        error::ContextError,
        test_support::{add_fn, fixture},
    };

    #[test]
    fn simple_api_usage() {
        let (mut ctx, mut builder) = fixture();
        let func = add_fn("sum", &mut builder, &mut ctx).unwrap();
        let entry = func.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        cursor.build_unconditional_br(entry, &mut ctx).unwrap();

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
        let a = add_fn("a", &mut builder, &mut ctx).unwrap();
        let b = add_fn("b", &mut builder, &mut ctx).unwrap();

        assert_ne!(a.raw(), b.raw());
        assert_eq!(builder.module.functions.len(), 2);
        assert_eq!(ctx.funcs.len(), 2);
    }

    /// LLVM identifies a definition by its name, so two `@sum`s in one module is
    /// not something to discover at emit time.
    #[test]
    fn a_duplicate_function_name_is_refused() {
        let (mut ctx, mut builder) = fixture();

        add_fn("sum", &mut builder, &mut ctx).unwrap();

        let err = add_fn("sum", &mut builder, &mut ctx).expect_err("the name is taken");

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
