use thiserror::Error;

#[derive(Error, Debug)]
pub enum JITError {
    #[error("initialization failed for native target")]
    InitializationError,
    #[error("module name contains a NUL byte")]
    InvalidModuleName,
    #[error("error from LLVM Backend: {0}")]
    LLVMError(String),
}
