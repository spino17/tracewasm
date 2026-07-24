//! The crate-wide error type for parsing and lowering.
use thiserror::Error;

/// Any failure while validating, parsing, or lowering a WebAssembly module.
///
/// The `From<wasmparser::Error>` impl lets decode/validation failures propagate
/// through `?` in the parser and lowering code.
#[derive(Error, Debug)]
pub enum TraceWasmError {
    /// An execution error reported while executing a WASM module function
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
    /// The params or results supplied to / produced by a typed call don't match
    /// the function's signature. Fields: which side (`"params"`/`"results"`), the
    /// function index, the expected type list, and the actual values seen.
    #[error("incorrect {0} structure provided to func `{1}`: expected `{2}`, got `{3}`")]
    IncorrectParamsResultsStructure(String, u32, String, String),
    /// An imported function declared by the module (module name, function name)
    /// has no matching entry in the supplied import registry.
    #[error("import `{0}::{1}` not found in the registry")]
    ImportNotFound(String, String),
    /// The import registry declares a different number of functions than the
    /// module imports. Fields: the module's import count, the registry's count.
    #[error("import count mismatch: module imports `{0}` functions, registry provides `{1}`")]
    ImportCountMismatch(u32, u32),
    /// An imported function's registry signature does not match the module's
    /// declared import type. Fields: module name, function name, which side
    /// (`"params"`/`"results"`), the module's expected type list, and the
    /// registry's provided type list.
    #[error(
        "import `{0}::{1}` signature mismatch in {2}: module expects `{3}`, registry provides `{4}`"
    )]
    ImportSignatureMismatch(String, String, String, String, String),
    /// An imported global declared by the module (module name, global name) has a
    /// registry value whose type differs from the module's declared global type.
    /// Fields: module name, global name, the expected value type, and the value
    /// the registry provided.
    #[error(
        "import global `{0}::{1}` type mismatch: module expects `{2}`, registry provides `{3}`"
    )]
    ImportGlobalTypeMismatch(String, String, String, String),
    /// The import registry declares a different number of globals than the module
    /// imports. Fields: the module's imported-global count, the registry's count.
    #[error("import global count mismatch: module imports `{0}` globals, registry provides `{1}`")]
    ImportGlobalCountMismatch(u32, u32),
    /// A named export was requested but the module declares no export with that
    /// name. The string is the requested export name.
    #[error("export `{0}` not found in the module")]
    ExportNotFound(String),
    /// An export was requested as a particular kind but is something else; the
    /// string names the expected kind (e.g. `"function"`).
    #[error("export is not a {0}")]
    ExportNotA(String),
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
