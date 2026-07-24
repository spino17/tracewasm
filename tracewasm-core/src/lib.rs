//! # tracewasm-core
//!
//! Core parsing and lowering for TraceWasm: turns a raw WebAssembly binary into
//! an owned, validated in-memory module ready for interpretation/tracing.
//!
//! ## Pipeline
//!
//! [`module::Module::compile`] is the entry point. It first runs `wasmparser`'s
//! full validator over the bytes, then walks the module a second time to build
//! an owned [`module::Module`]. Function bodies and constant expressions are
//! lowered by [`instruction`] into a flat instruction list where structured
//! control flow is resolved to absolute indices and operand-stack heights are
//! precomputed. An [`instance::Instance`] then pairs a compiled module with a
//! [`memory::Memory`] and runs its functions via [`instance::TypedFunc`].
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
//! - [`module`] — binary → owned module representation and typed entity indices.
//! - [`instruction`] — control-flow lowering and stack-height precomputation.
//! - [`instance`] — the runtime instance and typed-function calling API.
//! - [`memory`] — the [`memory::Memory`] trait implemented by embedders.
//! - [`error`] — the crate's error type.
//!
//! The interpreter itself lives in a crate-internal `vm` module and is not part
//! of the public API.

pub mod error;
pub mod instance;
pub mod instruction;
pub mod memory;
pub mod module;
pub(crate) mod vm;
