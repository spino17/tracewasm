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

/// Basic blocks and the handle that addresses them.
pub mod basic_block;
/// The builder that extends a module.
pub mod builder;
/// Storage for everything a module's ids point into.
pub mod context;
/// Rendering a finished graph as textual LLVM IR.
pub mod emit;
/// Function definitions and their parameters.
pub mod function;
/// Module-level symbols: variables, definitions and declarations.
pub mod global;
/// The module and its target settings.
pub mod module;
/// Traversing a finished graph.
pub mod walk;

/// A finished module, ready to be walked or emitted.
///
/// Produced by [`Builder::build`](builder::Builder::build), which takes the
/// [`Context`] by value — so the graph owns everything its ids point into, and
/// walking or emitting it needs nothing else. Construction is over: the context has
/// been handed over, so no builder call can reach it any more.
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
        let mut builder = fixture();
        let func = add_fn("sum", &mut builder).unwrap();
        let entry = func.add_basic_block("entry".to_string(), &mut builder).unwrap();
        let cursor = builder.cursor_at_block(entry);

        cursor.build_unconditional_br(entry).unwrap();

        assert_eq!(
            builder.blocks.get(entry.raw()).unwrap().instructions.len(),
            1,
            "the cursor writes into the block it was opened at"
        );

        let cfg = builder.build();

        assert_eq!(cfg.context.module.functions.len(), 1);
    }

    /// Every function is its own entry, and the module records them in the order
    /// they were added.
    #[test]
    fn functions_get_distinct_ids_in_order() {
        let mut builder = fixture();
        let a = add_fn("a", &mut builder).unwrap();
        let b = add_fn("b", &mut builder).unwrap();

        assert_ne!(a.raw(), b.raw());
        assert_eq!(builder.module.functions.len(), 2);
        assert_eq!(builder.funcs.len(), 2);
    }

    /// LLVM identifies a definition by its name, so two `@sum`s in one module is
    /// not something to discover at emit time.
    #[test]
    fn a_duplicate_function_name_is_refused() {
        let mut builder = fixture();

        add_fn("sum", &mut builder).unwrap();

        let err = add_fn("sum", &mut builder).expect_err("the name is taken");

        assert!(
            matches!(&err, ContextError::DuplicateGlobalName(name) if name == "sum"),
            "the error must name the collision, got: {err}"
        );

        assert_eq!(
            builder.module.functions.len(),
            1,
            "the refused function must not have been added"
        );
    }
}
