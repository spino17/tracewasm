//! The crate-wide error type for parsing, lowering, instantiation, and execution.
use crate::{
    instruction::Instruction,
    module::{CustomSection, FuncIndex, TableIndex},
};
use rustc_demangle::demangle;
use std::fmt::Display;
use thiserror::Error;

/// Any failure while validating, parsing, or lowering a WebAssembly module.
///
/// The `From<wasmparser::Error>` impl lets decode/validation failures propagate
/// through `?` in the parser and lowering code.
#[derive(Error, Debug)]
pub enum TraceWasmError {
    /// A trap or error raised while executing a single instruction, tagged with
    /// where it happened. The interpreter's driver loop attaches these coordinates
    /// to the [`InstructionExecutionError`] the instruction produced.
    ///
    /// Fields: the enclosing function index, the instruction's index in that
    /// function's lowered instruction list, the instruction itself, the underlying
    /// cause, and the instruction's byte offset in the module binary. The offset
    /// is carried for DWARF lookup rather than display, so it is deliberately
    /// absent from the message.
    #[error("error occured while executing instruction `{1}`({2:?}) in func({0:?}): {3}")]
    InstructionExecution(
        FuncIndex,
        usize,
        Instruction,
        InstructionExecutionError,
        u32,
    ),
    /// A linear-memory access that ran past the memory's bounds (a wasm trap).
    /// Fields: a description of the access, the byte offset attempted, and the
    /// current memory length in bytes.
    #[error("{0:?}")]
    MemoryError(MemoryError),
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

/// One frame of a captured interpreter backtrace: the instruction (and its
/// enclosing function) that either trapped or called into the next-inner frame.
pub struct TraceRecord {
    /// The enclosing function this frame belongs to.
    pub func_index: FuncIndex,
    /// The instruction's index in that function's lowered instruction list.
    pub instr_index: usize,
    /// The instruction at that index.
    pub instr: Instruction,
    /// Whether this frame is a call into a deeper frame or the trapping leaf.
    pub kind: TraceRecordKind,
    /// The instruction's byte offset in the module binary, for resolving a source
    /// location against the module's DWARF (see [`Module::dwarf`](crate::module::Module::dwarf)).
    pub instr_offset: u32,
}

/// Distinguishes a caller frame (a `call`/`call_indirect` into a deeper frame)
/// from the innermost frame where execution actually trapped.
pub enum TraceRecordKind {
    /// A call frame leading into the next-inner frame. `callee_index` is the
    /// function called; `is_indirect` is `Some(table)` for a `call_indirect` and
    /// `None` for a direct `call`.
    Call {
        callee_index: FuncIndex,
        is_indirect: Option<TableIndex>,
    },
    /// The innermost frame: the instruction that trapped, carrying its message.
    NonCall(String),
}

/// A captured interpreter backtrace, innermost-first: frame `0` is where
/// execution trapped and each later frame is the caller that led to it.
pub struct StackTrace(Vec<TraceRecord>);

impl StackTrace {
    /// Renders the trace as a human-readable, innermost-first backtrace: frame
    /// `#0` is where execution trapped, and each subsequent frame is the caller
    /// that led to it.
    ///
    /// `top_enclosing_func_name` labels the header with the entry function (when
    /// known); `module` supplies the `name`-section names used for functions and
    /// tables, falling back to `func #N` / `table #N` when a name is absent.
    pub fn render(
        &self,
        top_enclosing_func_name: Option<&str>,
        custom_section: &CustomSection,
    ) -> String {
        let name_of_func = |f: FuncIndex| {
            custom_section
                .func_name(f)
                .map(|name| {
                    // Some toolchains emit the WAT-style `$`-prefixed symbol; strip
                    // it so the demangler sees the bare symbol. `{:#}` drops the
                    // trailing `::h<hash>` disambiguator, and a non-Rust symbol
                    // passes through unchanged.
                    let symbol = name.strip_prefix('$').unwrap_or(name);

                    format!("{:#}", demangle(symbol))
                })
                .unwrap_or_else(|| format!("func #{}", f.0))
        };

        let name_of_table = |t: TableIndex| {
            custom_section
                .table_name(t)
                .map(str::to_string)
                .unwrap_or_else(|| format!("table #{}", t.0))
        };

        let mut out = match top_enclosing_func_name {
            Some(name) => format!("Stack trace of `{name}` (most recent call first):\n\n"),
            None => String::from("Stack trace (most recent call first):\n\n"),
        };

        // Pre-resolve each frame's function name so the function column aligns.
        let frame_names: Vec<String> = self.0.iter().map(|r| name_of_func(r.func_index)).collect();
        let width = frame_names.iter().map(String::len).max().unwrap_or(0);

        for (i, (record, frame_name)) in self.0.iter().zip(&frame_names).enumerate() {
            let detail = match &record.kind {
                // The innermost frame: the instruction that actually trapped.
                TraceRecordKind::NonCall(err) => {
                    format!(
                        "at instr {} ({:?}) — trap: {err}",
                        record.instr_index, record.instr
                    )
                }
                // A caller frame: the call that led into the next-inner frame.
                TraceRecordKind::Call {
                    callee_index,
                    is_indirect,
                } => {
                    let via = match is_indirect {
                        Some(table) => format!(" (indirect, via {})", name_of_table(*table)),
                        None => String::new(),
                    };

                    format!(
                        "at instr {} — calls `{}`{via}",
                        record.instr_index,
                        name_of_func(*callee_index)
                    )
                }
            };

            let column = format!("{frame_name:<width$}", width = width);

            out.push_str(&format!("  #{i:<2} {column}  {detail}\n"));
        }

        out
    }
}

impl TraceWasmError {
    fn _extract_stack_trace(&self, trace: &mut Vec<TraceRecord>) -> Option<()> {
        let TraceWasmError::InstructionExecution(func_index, instr_index, instr, err, instr_offset) =
            self
        else {
            return None;
        };

        let callee_index: Option<(FuncIndex, Option<TableIndex>)> = match err {
            InstructionExecutionError::Call(callee_func_index, err) => {
                err._extract_stack_trace(trace);

                Some((*callee_func_index, None))
            }
            InstructionExecutionError::CallIndirect(table_index, err) => match err {
                CallIndirectError::FunctionCall(callee_func_index, err) => {
                    err._extract_stack_trace(trace);

                    Some((*callee_func_index, Some(*table_index)))
                }
                _ => None,
            },
            _ => None,
        };

        trace.push(TraceRecord {
            func_index: *func_index,
            kind: if let Some((callee_index, table_index)) = callee_index {
                TraceRecordKind::Call {
                    callee_index,
                    is_indirect: table_index,
                }
            } else {
                TraceRecordKind::NonCall(err.to_string())
            },
            instr: instr.clone(),
            instr_index: *instr_index,
            instr_offset: *instr_offset,
        });

        Some(())
    }

    pub(crate) fn extract_stack_trace(&self) -> Option<StackTrace> {
        let mut trace = vec![];

        self._extract_stack_trace(&mut trace)?;

        Some(StackTrace(trace))
    }
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
    /// A `call` failed; the boxed error carries the underlying cause (typically a
    /// nested [`TraceWasmError::InstructionExecution`] from the callee, or a host
    /// error for an imported callee). Field: the callee's function index.
    #[error("call to func({0:?}): {1}")]
    Call(FuncIndex, Box<TraceWasmError>),
    /// A `call_indirect` failed — an out-of-bounds index, a null element, a
    /// signature mismatch, or an error inside the callee. Fields: the table index
    /// and the specific [`CallIndirectError`].
    #[error("call_indirect via table({0:?}): {1}")]
    CallIndirect(TableIndex, CallIndirectError),
    /// A memory access failed (out of bounds, offset too large, or effective-
    /// address overflow). Field: the specific [`MemoryError`].
    #[error("{0}")]
    Memory(MemoryError),
}

impl InstructionExecutionError {
    /// Tags this cause with where it happened, producing the crate-wide error.
    pub fn into_tracewasm_err(
        self,
        instr_index: usize,
        enclosing_func_index: FuncIndex,
        instr: &Instruction,
        offset: u32,
    ) -> TraceWasmError {
        TraceWasmError::InstructionExecution(
            enclosing_func_index,
            instr_index,
            instr.clone(),
            self,
            offset,
        )
    }
}

/// Why a `call_indirect` failed: a table-access trap, a signature mismatch, or an
/// error raised inside the resolved callee.
#[derive(Error, Debug)]
pub enum CallIndirectError {
    /// The table index operand was outside the table's bounds (a wasm trap).
    #[error("table slot out of bounds")]
    TableSlotOutOfBounds,
    /// The referenced table slot held a null element (a wasm trap).
    #[error("null element in the table slot")]
    NullElementInTable,
    /// The callee's signature differs from the type the instruction expects (a
    /// wasm trap). Fields: the expected signature and the callee's actual signature.
    #[error("function signature mismatch: expected {0}, got {1}")]
    FunctionSignatureMismatch(String, String),
    /// The resolved callee itself failed; the boxed error carries the cause.
    /// Field: the callee's function index.
    #[error("call to func({0:?}): {1}")]
    FunctionCall(FuncIndex, Box<TraceWasmError>),
}

/// Which direction a failed memory access was going, for error reporting.
#[derive(Debug)]
pub enum MemoryAccessKind {
    /// A load.
    Read,
    /// A store.
    Write,
}

impl Display for MemoryAccessKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A linear-memory access failure (all are wasm traps).
#[derive(Error, Debug)]
pub enum MemoryError {
    /// The access ran past the memory's end. Fields: whether it was a read or a
    /// write, the byte offset attempted, and the current memory length.
    #[error("out of bounds access: {0} at {1} on memory with length {2}")]
    OutOfBoundsAccess(MemoryAccessKind, usize, usize), // (kind, offset, mem_len)
    /// A static `memarg` offset did not fit in a 32-bit memory's address space.
    #[error("offset too large for 32-bit memory")]
    OffsetTooLarge,
    /// The effective address (popped address + static offset) overflowed the
    /// 32-bit address space, so it is necessarily out of bounds. Fields: the
    /// popped address and the static offset.
    #[error(
        "effective address overflow: address `{0}` + offset `{1}` exceeds the 32-bit address space"
    )]
    EffectiveAddressOverflow(u32, u32),
    /// A `memory.grow` could not be satisfied — the requested delta would exceed
    /// the allowed maximum, or the page count overflowed. Fields: the maximum size
    /// in pages, the requested delta in pages, and the memory's current pages.
    ///
    /// Note this is *not* a trap: per the spec `memory.grow` reports failure by
    /// pushing `-1`, so the interpreter converts this into that value rather than
    /// propagating it. See [`Memory::grow`](crate::memory::Memory::grow).
    #[error(
        "memory grow failed: maximum cap on memory size in pages is `{0}`, request received for increasing `{1}` pages on a memory with `{2}` pages"
    )]
    GrowFailed(u64, u64, u64),
}

impl From<MemoryError> for InstructionExecutionError {
    fn from(value: MemoryError) -> Self {
        InstructionExecutionError::Memory(value)
    }
}

impl From<MemoryError> for TraceWasmError {
    fn from(value: MemoryError) -> Self {
        TraceWasmError::MemoryError(value)
    }
}

#[cfg(test)]
mod stack_trace_tests {
    use super::*;

    /// Build an `InstructionExecution` error: func `func`, instruction index
    /// `instr`, with the given per-instruction cause. The recorded `Instruction`
    /// is a fixed placeholder — the trace-extraction logic only clones it through,
    /// so its value is irrelevant to these tests.
    fn ie(func: u32, instr: usize, cause: InstructionExecutionError) -> TraceWasmError {
        TraceWasmError::InstructionExecution(FuncIndex(func), instr, Instruction::Nop, cause, 0)
    }

    /// Assert a record is a `NonCall` at `(func, instr)`.
    fn assert_noncall(rec: &TraceRecord, func: u32, instr: usize) {
        assert_eq!(rec.func_index, FuncIndex(func));
        assert_eq!(rec.instr_index, instr);
        assert!(
            matches!(rec.kind, TraceRecordKind::NonCall(_)),
            "expected a NonCall record"
        );
    }

    /// Assert a record is a `Call` at `(func, instr)` to `callee`, with the given
    /// `is_indirect` table (None for a direct call, Some(table) for indirect).
    fn assert_call(rec: &TraceRecord, func: u32, instr: usize, callee: u32, table: Option<u32>) {
        assert_eq!(rec.func_index, FuncIndex(func));
        assert_eq!(rec.instr_index, instr);
        match &rec.kind {
            TraceRecordKind::Call {
                callee_index,
                is_indirect,
            } => {
                assert_eq!(*callee_index, FuncIndex(callee));
                assert_eq!(*is_indirect, table.map(TableIndex));
            }
            TraceRecordKind::NonCall(_) => panic!("expected a Call record, got NonCall"),
        }
    }

    // top-level guard: a non-`InstructionExecution` error has no trace.
    #[test]
    fn non_instruction_error_yields_no_trace() {
        assert!(
            TraceWasmError::Unsupported("x".to_string())
                .extract_stack_trace()
                .is_none()
        );
        assert!(
            TraceWasmError::ExportNotA("function".to_string())
                .extract_stack_trace()
                .is_none()
        );
    }

    // leaf `Unreachable` (the catch-all IE arm) → single NonCall record.
    #[test]
    fn single_unreachable_frame() {
        let trace = ie(0, 7, InstructionExecutionError::Unreachable)
            .extract_stack_trace()
            .expect("top-level is an InstructionExecution");

        assert_eq!(trace.0.len(), 1);
        assert_noncall(&trace.0[0], 0, 7);
    }

    // every non-`FunctionCall` `CallIndirect` cause hits the inner `_ => None`
    // arm and becomes a NonCall leaf.
    #[test]
    fn call_indirect_leaf_traps_are_noncall() {
        let leaves = [
            CallIndirectError::TableSlotOutOfBounds,
            CallIndirectError::NullElementInTable,
            CallIndirectError::FunctionSignatureMismatch("(I32)".to_string(), "()".to_string()),
        ];

        for (i, leaf) in leaves.into_iter().enumerate() {
            let trace = ie(
                1,
                i,
                InstructionExecutionError::CallIndirect(TableIndex(3), leaf),
            )
            .extract_stack_trace()
            .unwrap();

            assert_eq!(trace.0.len(), 1);
            assert_noncall(&trace.0[0], 1, i);
        }
    }

    // direct `Call` recursion: caller frame recorded as a Call, callee's trap as
    // a NonCall, innermost first.
    #[test]
    fn direct_call_chain() {
        let err = ie(
            2,
            5,
            InstructionExecutionError::Call(
                FuncIndex(3),
                Box::new(ie(3, 1, InstructionExecutionError::Unreachable)),
            ),
        );

        let trace = err.extract_stack_trace().unwrap();

        assert_eq!(trace.0.len(), 2);
        assert_noncall(&trace.0[0], 3, 1); // innermost first
        assert_call(&trace.0[1], 2, 5, 3, None); // direct call → is_indirect None
    }

    // `CallIndirect::FunctionCall` recursion: Call record carries the table index.
    #[test]
    fn indirect_call_chain() {
        let err = ie(
            1,
            9,
            InstructionExecutionError::CallIndirect(
                TableIndex(4),
                CallIndirectError::FunctionCall(
                    FuncIndex(6),
                    Box::new(ie(6, 0, InstructionExecutionError::Unreachable)),
                ),
            ),
        );

        let trace = err.extract_stack_trace().unwrap();

        assert_eq!(trace.0.len(), 2);
        assert_noncall(&trace.0[0], 6, 0);
        assert_call(&trace.0[1], 1, 9, 6, Some(4)); // indirect → is_indirect Some(table)
    }

    // the `?`-free recursion: a nested non-`InstructionExecution` error (e.g. a
    // host/import failure) must NOT discard the trace — the calling frame is still
    // recorded, the non-instruction leaf is simply omitted.
    #[test]
    fn nested_non_instruction_error_keeps_caller_frame() {
        // direct call whose callee is an import that failed
        let via_call = ie(
            0,
            3,
            InstructionExecutionError::Call(
                FuncIndex(5),
                Box::new(TraceWasmError::ImportNotFound(
                    "env".to_string(),
                    "log".to_string(),
                )),
            ),
        );
        let trace = via_call.extract_stack_trace().unwrap();
        assert_eq!(trace.0.len(), 1, "caller frame must survive a non-IE leaf");
        assert_call(&trace.0[0], 0, 3, 5, None);

        // same for the indirect path
        let via_indirect = ie(
            0,
            2,
            InstructionExecutionError::CallIndirect(
                TableIndex(1),
                CallIndirectError::FunctionCall(
                    FuncIndex(9),
                    Box::new(TraceWasmError::ImportNotFound(
                        "env".to_string(),
                        "log".to_string(),
                    )),
                ),
            ),
        );
        let trace = via_indirect.extract_stack_trace().unwrap();
        assert_eq!(trace.0.len(), 1);
        assert_call(&trace.0[0], 0, 2, 9, Some(1));
    }

    // deep mixed chain A --call--> B --call_indirect--> C(unreachable): exercises
    // both recursion arms together and pins the innermost-first ordering.
    #[test]
    fn deep_mixed_chain_ordering() {
        let err = ie(
            0,
            10,
            InstructionExecutionError::Call(
                FuncIndex(1),
                Box::new(ie(
                    1,
                    4,
                    InstructionExecutionError::CallIndirect(
                        TableIndex(2),
                        CallIndirectError::FunctionCall(
                            FuncIndex(2),
                            Box::new(ie(2, 0, InstructionExecutionError::Unreachable)),
                        ),
                    ),
                )),
            ),
        );

        let trace = err.extract_stack_trace().unwrap();

        assert_eq!(trace.0.len(), 3);
        assert_noncall(&trace.0[0], 2, 0); // innermost: the trap
        assert_call(&trace.0[1], 1, 4, 2, Some(2)); // B called C indirectly via table 2
        assert_call(&trace.0[2], 0, 10, 1, None); // A called B directly
    }
}
