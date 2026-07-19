use thiserror::Error;

#[derive(Error, Debug)]
pub enum TraceWasmError {
    #[error("not supported in TraceWasm: {0}")]
    Unsupported(String),
    #[error("error occured while parsing: {0}")]
    Parsing(wasmparser::Error),
    #[error("{0}")]
    Generic(anyhow::Error),
}

impl From<wasmparser::Error> for TraceWasmError {
    fn from(value: wasmparser::Error) -> Self {
        TraceWasmError::Parsing(value)
    }
}

impl From<anyhow::Error> for TraceWasmError {
    fn from(value: anyhow::Error) -> Self {
        TraceWasmError::Generic(value)
    }
}
