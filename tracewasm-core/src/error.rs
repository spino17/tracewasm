//! The crate-wide error type for parsing and lowering.
use thiserror::Error;

use crate::module::{FuncIndex, TableIndex};

/// Any failure while validating, parsing, or lowering a WebAssembly module.
///
/// The `From<wasmparser::Error>` impl lets decode/validation failures propagate
/// through `?` in the parser and lowering code.
#[derive(Error, Debug)]
pub enum TraceWasmError {
    /// A trap or error raised while executing a single instruction, tagged with
    /// where it happened. The interpreter's driver loop attaches these coordinates
    /// to the [`InstructionExecutionError`] the instruction produced. Fields: the
    /// enclosing function index, the instruction's index in that function's
    /// lowered instruction list, and the underlying cause.
    #[error("error occured while executing instruction `{1}` in func({0:?}): {2}")]
    InstructionExecution(FuncIndex, usize, InstructionExecutionError),
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
    /// A table's initial element count exceeds the maximum the instance is
    /// willing to materialize (the declared maximum, capped by the instance
    /// [`Config`](crate::instance::config::Config)). Fields: the requested
    /// initial element count and the allowed maximum.
    #[error("table too large: initial `{0}` elements exceeds the allowed maximum `{1}`")]
    TableTooLarge(u64, u64),
    /// An active element segment writes past the end of its target table at
    /// instantiation. Fields: the write offset, the number of elements written,
    /// and the target table's length.
    #[error(
        "element segment out of bounds: writing `{1}` elements at offset `{0}` exceeds table length `{2}`"
    )]
    ElementSegmentOutOfBounds(usize, usize, usize),
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

/// The cause of a failure while executing one instruction, one variant per
/// instruction kind that can fail. The interpreter's driver loop tags this with
/// the enclosing function and instruction index via
/// [`Self::into_tracewasm_err`], producing a [`TraceWasmError::InstructionExecution`].
#[derive(Error, Debug)]
pub enum InstructionExecutionError {
    /// Reached an `unreachable` instruction (a wasm trap).
    #[error("reached an `unreachable` instruction")]
    Unreachable,
    /// A `call` failed; the string carries the underlying cause. Field: the
    /// callee's function index.
    #[error("call to func({0:?}): {1}")]
    Call(FuncIndex, String),
    /// A `call_indirect` failed — an out-of-bounds index, a null element, or a
    /// signature mismatch; the string carries which. Field: the table index.
    #[error("call_indirect via table({0:?}): {1}")]
    CallIndirect(TableIndex, String),
}

impl InstructionExecutionError {
    /// Tags this cause with where it happened, producing the crate-wide error.
    pub fn into_tracewasm_err(
        self,
        instr_index: usize,
        enclosing_func_index: FuncIndex,
    ) -> TraceWasmError {
        TraceWasmError::InstructionExecution(enclosing_func_index, instr_index, self)
    }
}
