//! Builds LLVM IR in memory and renders it as textual `.ll`.
//!
//! This is a construction layer for a compiler backend: you build a
//! [`ControlFlowGraph`](cfg::ControlFlowGraph) out of functions, basic blocks and
//! instructions, and [`IREmitter`](cfg::emit::IREmitter) turns it into text that
//! `llvm-as` accepts. Nothing here parses IR or links against libLLVM.
//!
//! # The shape of the API
//!
//! Three things are threaded through almost every call:
//!
//! - A [`Context`](cfg::context::Context) owns the arenas and the interner pools.
//!   Everything is addressed by id — [`TyId`](interner::TyId),
//!   [`StrId`](interner::StrId), [`FuncId`](cfg::function::FuncId),
//!   [`BasicBlockId`](cfg::basic_block::BasicBlockId) — and **an id only means
//!   anything against the context that issued it**.
//! - A [`Builder`](cfg::builder::Builder) owns the module: it adds functions and
//!   hands out cursors.
//! - A [`Cursor`](instruction::cursor::Cursor) points at one basic block and writes
//!   instructions into it.
//!
//! ```no_run
//! # use tracewasm_llvm::cfg::{builder::Builder, context::Context, emit::IREmitter};
//! let mut ctx = Context::default();
//! let mut builder = Builder::new("arm64-apple-macosx".to_string(), String::new());
//!
//! let i32_ty = ctx.i32_ty();
//! let f = builder.add_function("main".to_string(), &[], i32_ty, &mut ctx)?;
//! let entry = f.add_basic_block("entry".to_string(), &mut ctx)?;
//!
//! let zero = tracewasm_llvm::value::Value::from_const(0i32, None, &mut ctx)?;
//! builder.cursor_at_block(entry).build_ret(Some(zero), Some(i32_ty), &mut ctx)?;
//!
//! let ir = IREmitter::emit(builder.build(), &ctx)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! # Stricter than LLVM, on purpose
//!
//! The builders reject some IR that `llvm-as` would accept, so that a bug in the
//! compiler driving them surfaces at construction rather than as a miscompile. A
//! `getelementptr` with an out-of-range constant array index and a `load` whose type
//! disagrees with the pointer's inferred pointee are both legal LLVM and both refused
//! here. Where this crate is *looser* than LLVM that is a bug; where it is stricter it
//! is deliberate.
//!
//! # Types are interned
//!
//! A [`Type`](value::Type) names its children by [`TyId`](interner::TyId) rather than
//! holding them, so structurally equal types are one pool entry and comparing two
//! types is comparing two integers. The cost is that a type cannot print itself —
//! rendering needs the pool, via [`Context::display`](cfg::context::Context::display).

pub mod cfg;
pub mod constants;
pub mod emit;
pub mod error;
pub mod instruction;
pub mod interner;
pub mod value;

#[cfg(test)]
mod test_support;
