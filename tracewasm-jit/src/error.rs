use thiserror::Error;

#[derive(Error, Debug)]
pub enum JITError {
    #[error("initialization failed for native target")]
    InitializationError,
    #[error("module name contains a NUL byte")]
    InvalidModuleName,
    #[error("function name contains a NUL byte")]
    InvalidFuncName,
    #[error("module has no declaration for host function `{0}`")]
    HostFuncNotDeclared(String),
    #[error("host function `{name}` is declared as `{declared}` but the host provides `{host}`")]
    HostFuncSignatureMismatch {
        name: String,
        declared: String,
        host: String,
    },
    #[error("a different host function is already linked as `{0}`")]
    HostFuncAlreadyLinked(String),
    #[error("error from LLVM Backend: {0}")]
    LLVMError(String),
}
