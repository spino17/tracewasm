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
//! The parser targets core WebAssembly. It rejects the component model, imports
//! other than functions and globals, and 64-bit memory as
//! [`error::TraceWasmError::Unsupported`]; anything the second pass cannot
//! represent surfaces as the same error rather than a panic.
//!
//! GC types are refused a step earlier, by `wasmparser` while reading the type
//! section, and so arrive as [`error::TraceWasmError::Parsing`].
//!
//! ## Modules
//!
//! - [`module`] — binary → owned module representation and typed entity indices.
//! - [`instruction`] — control-flow lowering and stack-height precomputation.
//! - [`instance`] — the runtime instance and typed-function calling API.
//! - [`memory`] — the [`memory::Memory`] trait implemented by embedders.
//! - [`error`] — the crate's error type.
//! - [`tracewasm_unreachable`] — the crate's divergence helper for broken
//!   internal invariants.
//!
//! The interpreter itself lives in a crate-internal `vm` module and is not part
//! of the public API.

pub mod error;
pub mod instance;
pub mod instruction;
pub mod memory;
pub mod module;
pub(crate) mod vm;

/// The single place the crate diverges on a broken internal invariant.
pub mod tracewasm_unreachable {
    /// Panics. Called where a case is impossible unless one of this crate's own
    /// invariants has been broken — a `V128` reaching the value stack, say, when
    /// `Module::compile` is supposed to have rejected the module.
    ///
    /// A plain `unreachable!()` at each such site would inline its panic
    /// machinery — the formatting call, the location record — into whatever
    /// contains it. Several of those sites sit inside `#[inline(always)]`
    /// accessors that end up in the interpreter's dispatch, where that code would
    /// occupy registers and frame space on paths that never execute. Outlining it
    /// behind one `#[inline(never)]` function leaves a single `bl` at each site.
    ///
    /// Returns `!`, so the compiler knows control does not come back and needs no
    /// value from the caller's branch.
    #[inline(never)]
    pub fn unreachable() -> ! {
        unreachable!()
    }
}
