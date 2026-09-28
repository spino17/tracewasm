use thiserror::Error;

#[derive(Error, Debug)]
pub enum JITError {
    #[error("initialization failed for native target")]
    InitializationError,
    #[error("module name contains a NUL byte")]
    InvalidModuleName,
    #[error("module failed verification: {0}")]
    InvalidModule(String),
    #[error("module targets `{module}`, a different architecture or OS from the JIT's `{jit}`")]
    TripleMismatch { module: String, jit: String },
    #[error("module's data layout `{module}` differs from the JIT's `{jit}`")]
    DataLayoutMismatch { module: String, jit: String },
    #[error("module declares `{0}`, but no host function is linked to it")]
    UnresolvedSymbol(String),
    #[error("function name contains a NUL byte")]
    InvalidFuncName,
    #[error("module defines `{0}` itself, so a host function can't be linked to it")]
    HostFuncDefinedInModule(String),
    #[error("can't link host function `{0}` after the module has been optimized")]
    HostFuncLinkedAfterOptimize(String),
    #[error("module has static constructors or destructors, which the JIT can't run")]
    StaticInitializers,
    #[error("host function `{name}` is declared as `{declared}` but the host provides `{host}`")]
    HostFuncSignatureMismatch {
        name: String,
        declared: String,
        host: String,
    },
    #[error("module exports no function `{0}`")]
    FuncNotExported(String),
    #[error("function `{name}` is defined as `{declared}` but was requested as `{requested}`")]
    FuncSignatureMismatch {
        name: String,
        declared: String,
        requested: String,
    },
    #[error("a different host function is already linked as `{0}`")]
    HostFuncAlreadyLinked(String),
    #[error("error from LLVM Backend: {0}")]
    LLVMError(String),
}
