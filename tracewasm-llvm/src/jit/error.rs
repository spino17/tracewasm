//! The errors the JIT reports, from parsing a module through to looking up a
//! compiled function.

use crate::error::TargetParseError;
use thiserror::Error;

/// Everything that can go wrong between [`JITHandler::new`](super::JITHandler::new)
/// and [`JITCompiledInstance::get_func`](super::JITCompiledInstance::get_func).
#[derive(Error, Debug)]
pub enum JITError {
    /// LLVM couldn't register the host machine's target. Returned by
    /// [`JITHandler::new`](super::JITHandler::new).
    #[error("initialization failed for native target")]
    InitializationError,
    /// [`JITHandler::target`](super::JITHandler::target) couldn't express the host's
    /// triple or data layout as a [`Target`](crate::cfg::module::Target): its layout
    /// has a specification the IR builder doesn't model.
    #[error("the host's target can't be expressed: {0}")]
    HostTarget(#[from] TargetParseError),
    /// The name given to [`parse_module`](super::JITHandler::parse_module)
    /// contains a NUL byte, which LLVM's C API can't take.
    #[error("module name contains a NUL byte")]
    InvalidModuleName,
    /// The IR parsed but failed LLVM's verifier (a use before its definition, a
    /// block without a terminator, …). Holds the verifier's report.
    #[error("module failed verification: {0}")]
    InvalidModule(String),
    /// The module's `target triple` names a different architecture or OS from the
    /// JIT's. The IR bakes in that platform's calling convention.
    #[error("module targets `{module}`, a different architecture or OS from the JIT's `{jit}`")]
    TripleMismatch {
        /// The module's triple, as written.
        module: String,
        /// The JIT's triple.
        jit: String,
    },
    /// The module's `target datalayout` isn't exactly the JIT's. Fields a layout
    /// leaves out take LLVM's defaults, which needn't be the host's.
    #[error("module's data layout `{module}` differs from the JIT's `{jit}`")]
    DataLayoutMismatch {
        /// The module's data layout, as written.
        module: String,
        /// The JIT's data layout.
        jit: String,
    },
    /// [`compile`](super::JITModule::compile) found a declared function or global
    /// that nothing defines: not a linked host function, and not an LLVM intrinsic.
    #[error("module declares `{0}`, but no host function is linked to it")]
    UnresolvedSymbol(String),
    /// A name given to [`link_host_func`](super::JITModule::link_host_func)
    /// contains a NUL byte.
    #[error("function name contains a NUL byte")]
    InvalidFuncName,
    /// A host function was linked to a name the module defines with a body, which
    /// would give the symbol two definitions.
    #[error("module defines `{0}` itself, so a host function can't be linked to it")]
    HostFuncDefinedInModule(String),
    /// [`link_host_func`](super::JITModule::link_host_func) was called after
    /// [`optimize`](super::JITModule::optimize), when the optimizer may already
    /// have treated the declaration as a C library function.
    #[error("can't link host function `{0}` after the module has been optimized")]
    HostFuncLinkedAfterOptimize(String),
    /// The module has a non-empty `llvm.global_ctors` or `llvm.global_dtors`, and
    /// ORC's C API has no way to run them.
    #[error("module has static constructors or destructors, which the JIT can't run")]
    StaticInitializers,
    /// A host function's Rust signature doesn't match the module's declaration of it.
    #[error("host function `{name}` is declared as `{declared}` but the host provides `{host}`")]
    HostFuncSignatureMismatch {
        /// The symbol being linked.
        name: String,
        /// The declaration's IR type, e.g. `i32 (i32)`.
        declared: String,
        /// The IR type of the Rust function, e.g. `i64 (i64)`.
        host: String,
    },
    /// [`get_func`](super::JITCompiledInstance::get_func) asked for a function the
    /// module doesn't export: undefined, `internal`/`private`, or `hidden`.
    #[error("module exports no function `{0}`")]
    FuncNotExported(String),
    /// [`get_func`](super::JITCompiledInstance::get_func) asked for a signature that
    /// doesn't match the function's definition.
    #[error("function `{name}` is defined as `{declared}` but was requested as `{requested}`")]
    FuncSignatureMismatch {
        /// The function looked up.
        name: String,
        /// The definition's IR type.
        declared: String,
        /// The IR type of the requested `P` and `R`.
        requested: String,
    },
    /// A different function is already linked under this name in the same module,
    /// by hand or by [`#[imported]`](super::imported).
    #[error("a different host function is already linked as `{0}`")]
    HostFuncAlreadyLinked(String),
    /// Any other failure LLVM reports: IR that doesn't parse, a pass pipeline or
    /// code generation error, or a failed link. Holds LLVM's message.
    #[error("error from LLVM Backend: {0}")]
    LLVMError(String),
}
