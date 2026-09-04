//! The entry point a front end implements to lower its own instruction stream.

use crate::cfg::ControlFlowGraph;

/// Lowers a front end's instructions into a [`ControlFlowGraph`].
///
/// This crate does not know what the source language is: a front end implements this
/// trait over its own instruction type, builds the graph with
/// [`Builder`](crate::cfg::builder::Builder) and
/// [`Cursor`](crate::instruction::cursor::Cursor), and hands the result back. The
/// graph can then be rendered by [`IREmitter`](crate::cfg::emit::IREmitter).
pub trait Emitter {
    /// One instruction of the source language.
    type SourceInstr;

    /// Whatever the lowering needs besides the instructions themselves — a frame
    /// layout, a function name, a symbol table.
    type SourceInstrCtx;

    /// Lowers `stream` into a graph, consuming the emitter.
    fn emit_cfg(
        self,
        stream: &[Self::SourceInstr],
        ctx: &Self::SourceInstrCtx,
    ) -> Result<ControlFlowGraph, anyhow::Error>;
}
