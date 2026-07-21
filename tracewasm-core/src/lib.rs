//! # tracewasm-core
//!
//! Core parsing and lowering for TraceWasm: turns a raw WebAssembly binary into
//! an owned, validated in-memory module ready for interpretation/tracing.
//!
//! ## Pipeline
//!
//! [`parser::TraceWasmParser::parse`] is the entry point. It first runs
//! `wasmparser`'s full validator over the bytes, then walks the module a second
//! time to build an owned [`ast::Module`]. Function bodies and constant
//! expressions are lowered by [`instruction`] into a flat instruction list where
//! structured control flow is resolved to absolute indices and operand-stack
//! heights are precomputed.
//!
//! ## Scope
//!
//! The parser targets core WebAssembly. It rejects the component model, GC
//! types, and non-function imports as [`error::TraceWasmError::Unsupported`];
//! anything the second pass cannot represent surfaces as the same error rather
//! than a panic.
//!
//! ## Modules
//!
//! - [`parser`] — binary → [`ast::Module`] (validate-then-build).
//! - [`ast`] — the owned module representation and its typed entity indices.
//! - [`instruction`] — control-flow lowering and stack-height precomputation.
//! - [`error`] — the crate's error type.

pub mod ast;
pub mod error;
pub mod instruction;
pub mod parser;
pub mod vm;
