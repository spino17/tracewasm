//! The crate-wide error type for parsing and lowering.
use thiserror::Error;

/// Any failure while validating, parsing, or lowering a WebAssembly module.
///
/// The `From<wasmparser::Error>` impl lets decode/validation failures propagate
/// through `?` in the parser and lowering code.
#[derive(Error, Debug)]
pub enum TraceWasmError {
    /// A execution error reported while executing a WASM module function
    #[error("error occured while executing the WASM module func({0}): {1}")]
    Execution(u32, String),
    /// A linear-memory access that ran past the memory's bounds (a wasm trap).
    /// Fields: a description of the access, the byte offset attempted, and the
    /// current memory length in bytes.
    #[error("out of bound memory access: {0} at offset `{1}` on memory with len `{2}`")]
    OutOfBoundMemoryAccess(String, usize, usize),
    /// A well-formed construct that TraceWasm deliberately does not handle
    /// (e.g. the component model, GC types, or non-function imports). The string
    /// describes the specific unsupported feature.
    #[error("not supported in TraceWasm: {0}")]
    Unsupported(String),
    /// A structural/decode error reported by `wasmparser` while reading the
    /// binary (also produced by the up-front full validation pass). The
    /// underlying error is flattened to its message so the `wasmparser` type does
    /// not appear in this crate's public API.
    #[error("error occured while parsing: {0}")]
    Parsing(String),
}

impl From<wasmparser::Error> for TraceWasmError {
    fn from(value: wasmparser::Error) -> Self {
        TraceWasmError::Parsing(value.to_string())
    }
}
