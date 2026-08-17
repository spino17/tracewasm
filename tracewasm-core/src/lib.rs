//! # tracewasm-core
//!
//! Core parsing and lowering for TraceWasm: turns a raw WebAssembly binary into
//! an owned, validated in-memory module ready for interpretation/tracing.
//!
//! ## Pipeline
//!
//! [`module::Module::compile`] is the entry point. It first runs `wasmparser`'s
//! full validator over the bytes, then walks the module a second time to build
//! an owned [`module::Module`], lowering each function body into a flat
//! instruction list with structured control flow resolved to absolute indices. An
//! [`instance::Instance`] then pairs a compiled module with a [`memory::Memory`]
//! and runs its functions via [`instance::TypedFunc`].
//!
//! ## Two machines
//!
//! A module is compiled for one of two machines, named by [`VirtualMachine`]:
//! [`Stack`] keeps wasm's own operand stack and precomputes the operand heights a
//! branch unwinds to — the reference for tracing fidelity — while [`Register`]
//! lowers the same operators into a register machine that moves values only when
//! it has to.
//!
//! ```ignore
//! let module = Module::<Stack>::compile(&wasm)?;
//! ```
//!
//! [`module::Module`] and [`instance::Instance`] are generic over that choice.
//! Everything it selects between — the instruction set, the frame layout, the
//! calling convention — is internal, which is why `VirtualMachine` is sealed and
//! carries no members: there is no third machine an embedder could add.
//!
//! [`error::TraceWasmError`] is deliberately *not* generic: nothing a trap carries
//! names an instruction, so parameterising it would spread the machine across
//! every `Result` in the crate to describe something the errors do not hold.
//!
//! Two things stay concrete on purpose. **Constant expressions are always lowered
//! for the stack machine** whatever the module's own, since they run once at
//! instantiation and never on a hot path; a lowered one is opaque as
//! [`module::ConstExpr`]. And execution currently accepts only a [`Stack`]
//! instance, because the register machine's execution is still being written.
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
//! - [`instance`] — the runtime instance and typed-function calling API.
//! - [`memory`] — the [`memory::Memory`] trait implemented by embedders.
//! - [`error`] — the crate's error type.
//! - [`tracewasm_unreachable`] — the crate's divergence helper for broken
//!   internal invariants.
//!
//! The interpreter itself lives in a crate-internal `runtime` module and is not
//! part of the public API.

/// Re-exported so implementors of
/// [`ImportRegistry`](instance::traits::ImportRegistry) — and the code
/// `#[imports]` generates for them — can name the error type its methods return
/// without taking a direct dependency on `anyhow`. A public trait that requires
/// naming a foreign type has to hand that type out.
pub use anyhow;

pub mod error;
pub mod instance;
pub mod memory;
pub mod module;

/// Lowering, and the machines it lowers for.
///
/// Crate-private: an instruction set, the frame it runs in and the calling
/// convention between them are implementation detail, and nothing in here is
/// usable from outside. [`VirtualMachine`] is the public face of the choice
/// between them.
pub(crate) mod instruction;
pub(crate) mod runtime;

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

/// Which machine a module is compiled for.
///
/// [`Stack`] keeps WebAssembly's own operand stack and is the reference for tracing
/// fidelity; [`Register`] lowers the same operators into a register machine that
/// moves values only when it has to. [`Module`](module::Module) and
/// [`Instance`](instance::Instance) are generic over this, so the choice is made
/// once — `Module::<Stack>::compile(..)` — and nothing downstream of it has to care.
///
/// **Sealed, and empty on purpose.** Everything that differs between the two
/// machines is internal: the instruction set, what lowering hands execution, where
/// live values are kept, how a callee's frame is found inside its caller's. There is
/// nothing for an embedder to supply and no third machine to add, so this is a bound
/// to name, not a trait to implement.
#[allow(
    private_bounds,
    reason = "the supertrait is crate-private on purpose — that is what seals this \
              trait, and what keeps the instruction types out of the public API"
)]
pub trait VirtualMachine: sealed::Internals {}

/// The machine details, kept out of the public API.
///
/// This is what lets [`VirtualMachine`] be a public name while everything behind it
/// stays private. The associated type is bounded by the crate-private `Instruction`
/// trait, so putting it on the public trait would drag that trait — and in turn the
/// frame traits, the untagged `Value`, the operand arenas — into the public
/// surface, none of which an embedder can use.
mod sealed {
    /// A machine's instruction set, and through its bound everything else the
    /// machine is made of.
    ///
    /// Crate-private, which is the seal: an embedder cannot name it, so it cannot
    /// implement [`VirtualMachine`](crate::VirtualMachine) either. It also has to be
    /// crate-private for its associated type to *be* one of the crate-private
    /// instruction enums — a `pub` trait's associated type is a public interface, and
    /// a private type may not satisfy one.
    pub(crate) trait Internals {
        /// What one lowered instruction of this machine is.
        type Instr: crate::instruction::Instruction;
    }
}

/// The instruction set behind a machine, spelled once so the crate's internals can
/// say `InstrOf<V>` instead of `<V as sealed::Internals>::Instr`.
pub(crate) type InstrOf<V> = <V as sealed::Internals>::Instr;

/// The stack machine: WebAssembly's own operand stack, with the operand heights a
/// branch unwinds to precomputed by lowering.
///
/// The reference implementation for tracing fidelity.
pub struct Stack;

impl sealed::Internals for Stack {
    type Instr = crate::instruction::stack::StackInstruction;
}

impl VirtualMachine for Stack {}

/// The register machine: the same operators lowered into named registers, so a value
/// moves only when it has to.
pub struct Register;

impl sealed::Internals for Register {
    type Instr = crate::instruction::register::RegInstruction;
}

impl VirtualMachine for Register {}
