//! The crate-wide error type for parsing and lowering.
use thiserror::Error;

/// Any failure while validating, parsing, or lowering a WebAssembly module.
///
/// The `From` impls let both `wasmparser` decode/validation failures and
/// generic [`anyhow::Error`]s propagate through `?` in the parser and lowering
/// code.
#[derive(Error, Debug)]
pub enum TraceWasmError {
    /// A execution error reported while executing a WASM module function
    #[error("error occured while executing the WASM module func({0}): {1}")]
    Execution(u32, String),
    /// A well-formed construct that TraceWasm deliberately does not handle
    /// (e.g. the component model, GC types, or non-function imports). The string
    /// describes the specific unsupported feature.
    #[error("not supported in TraceWasm: {0}")]
    Unsupported(String),
    /// A structural/decode error reported by `wasmparser` while reading the
    /// binary (also produced by the up-front full validation pass).
    #[error("error occured while parsing: {0}")]
    Parsing(wasmparser::Error),
    /// A catch-all for errors carried as [`anyhow::Error`].
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
