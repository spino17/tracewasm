//! Builds LLVM IR in memory and renders it as textual `.ll`.
//!
//! This is a construction layer for a compiler backend: you build a
//! [`ControlFlowGraph`](cfg::ControlFlowGraph) out of functions, basic blocks and
//! instructions, and [`IREmitter`](cfg::emit::IREmitter) turns it into text that
//! `llvm-as` accepts. Nothing here parses IR or links against libLLVM, except the
//! `jit` module, which is only built with the `jit` feature.
//!
//! # The shape of the API
//!
//! Three things are threaded through almost every call:
//!
//! - A [`Context`](cfg::context::Context) owns the arenas and the interner pools,
//!   and is created for a [`Target`](cfg::module::Target).
//!   Everything is addressed by id — [`TyId`](interner::TyId),
//!   [`StrId`](interner::StrId), [`FuncId`](cfg::function::FuncId),
//!   [`BasicBlockId`](cfg::basic_block::BasicBlockId) — and **an id only means
//!   anything against the context that issued it**.
//! - A [`Builder`](cfg::builder::Builder) owns the context (and so the module): it
//!   adds functions and hands out cursors.
//! - A [`Cursor`](instruction::cursor::Cursor) points at one basic block and writes
//!   instructions into it.
//!
//! ```
//! # use tracewasm_llvm::cfg::{context::Context, emit::IREmitter, module::Target};
//! // The target is fixed up front; `Unspecified` leaves it to whatever consumes the IR.
//! let ctx = Context::new(Target::Unspecified);
//! let mut builder = ctx.builder();
//!
//! let i32_ty = builder.i32_ty();
//! let f = builder.define_function("main", &[], i32_ty)?;
//! let entry = f.add_basic_block("entry", &mut builder)?;
//!
//! let zero = builder.const_value(0i32, tracewasm_llvm::instruction::cursor::OperandTy::Inferred)?;
//! builder.cursor_at_block(entry).build_ret(Some(zero), i32_ty.into())?;
//!
//! let ir = IREmitter::emit(builder.build())?;
//!
//! assert!(ir.contains("define i32 @main() {"));
//! assert!(ir.contains("ret i32 0"));
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
//!
//! # Running the IR
//!
//! With the `jit` feature, the `jit` module parses emitted (or any other) textual IR
//! into LLVM's ORC JIT, links host functions into it, optimizes and compiles it, and
//! hands back type-checked functions to call. It is the one part of the crate that
//! links libLLVM, so it's opt-in; see the module's own docs.

pub mod cfg;
pub mod constants;
pub mod error;
pub mod instruction;
pub mod interner;
#[cfg(feature = "jit")]
pub mod jit;
pub mod value;

#[cfg(test)]
mod test_support;

// Compiles the README's examples as doctests, so they can't go stale. They use the
// JIT, so only with the `jit` feature.
#[cfg(all(doctest, feature = "jit"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
