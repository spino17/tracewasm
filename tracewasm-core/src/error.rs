//! The crate-wide error type for parsing, lowering, instantiation, and execution.
use crate::{
    instruction::Instruction,
    module::{FuncIndex, Module, TableIndex},
};
use addr2line::LookupResult;
use rustc_demangle::demangle;
use std::{
    fmt::{self, Debug, Display},
    sync::Arc,
};
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
    /// A call would have nested deeper than
    /// [`Config::max_call_stack_depth`](crate::instance::config::Config::get_max_call_stack_depth).
    /// Field: the limit that was reached.
    ///
    /// Each active wasm frame costs a native Rust frame, so an unbounded guest
    /// recursion would overflow the *host* stack — which aborts the process
    /// instead of unwinding, leaving the embedder no way to recover. Trapping
    /// first keeps the failure an ordinary error.
    ///
    /// Raised at the call site, so it reaches the caller wrapped in an
    /// [`InstructionExecutionError::Call`] / [`CallIndirectError::FunctionCall`]
    /// and carries the usual backtrace.
    ///
    /// The message deliberately contains "call stack exhausted", the string the
    /// WebAssembly spec testsuite's `assert_exhaustion` directive matches on.
    #[error("call stack exhausted: exceeded the maximum call depth of {0}")]
    CallStackExhausted(u32),
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
    /// The module's initial memory size exceeds the maximum the instance is
    /// willing to materialize (its declared maximum, capped by the instance
    /// [`Config`](crate::instance::config::Config)). Fields: the requested initial
    /// page count and the allowed maximum.
    #[error("memory too large: initial `{0}` pages exceeds the allowed maximum `{1}`")]
    MemoryTooLarge(u64, u64),
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
pub struct StackTrace<'a>(Vec<TraceRecord>, &'a str, &'a Arc<Module>);

impl<'a> StackTrace<'a> {
    /// Resolves each frame of this trace against the module's `dwarf`, expanding
    /// it into its source-level frames (including any the compiler inlined).
    ///
    /// The instruction offsets recorded in the trace are byte offsets into the
    /// module binary, which is exactly how WebAssembly DWARF encodes code
    /// addresses, so they can be used as lookup probes directly.
    ///
    /// Every frame yields exactly one [`SourceTraceRecord`], even when DWARF has
    /// no coverage for it (an empty `inline_trace`), so the source trace never
    /// drops a frame relative to this one.
    ///
    /// # Errors
    ///
    /// Returns [`SourceStackTraceError::NoDebugInfo`] if the module carries no
    /// DWARF — i.e. the wasm was **not built with debug info**, which is the
    /// default for a release build. Resolving source locations requires the
    /// `.debug_*` custom sections, so enable debug info when producing the module
    /// (for a Cargo build, `debug = true` on the profile at the *workspace root*;
    /// a profile in a member manifest is ignored).
    ///
    /// Also fails if the DWARF cannot be indexed or a lookup errors; see
    /// [`SourceStackTraceError`] for the full set.
    pub fn to_source_trace(&self) -> Result<SourceStackTrace<'_>, SourceStackTraceError> {
        // A module built without debug info is the common case, not a bug — report
        // it rather than panicking on a diagnostics path.
        let Some(dwarf) = self.2.dwarf() else {
            return Err(SourceStackTraceError::NoDebugInfo);
        };

        let mut source_trace = vec![];
        // Built once: indexing the DWARF is proportional to its size, so doing it
        // per frame would re-parse the whole thing for every frame.
        let ctx = addr2line::Context::from_arc_dwarf(dwarf)
            .map_err(|err| SourceStackTraceError::ContextLoadFailed(err.to_string()))?;

        for (i, frame) in self.0.iter().enumerate() {
            let instruction_offset = frame.instr_offset;

            let mut frames = match ctx.find_frames(instruction_offset as u64) {
                // Only returned when the DWARF points at a split/supplementary
                // object, which a self-contained wasm module has no way to supply.
                // Report it rather than panicking — this is a diagnostics path.
                LookupResult::Load { .. } => {
                    return Err(SourceStackTraceError::SplitDwarfUnsupported);
                }
                LookupResult::Output(output) => output,
            }
            .map_err(|err| SourceStackTraceError::FindFramesFailed(err.to_string()))?;

            let mut inline_trace = vec![];

            while let Some(frame) = frames
                .next()
                .map_err(|err| SourceStackTraceError::NextFrameFetchFailed(err.to_string()))?
            {
                // Fall back through demangled → raw → `<unknown>`; a name we cannot
                // decode should degrade the trace, not panic while reporting it.
                let func_name = frame
                    .function
                    .as_ref()
                    .and_then(|f| {
                        f.demangle()
                            .or_else(|_| f.raw_name())
                            .ok()
                            .map(|name| name.into_owned())
                    })
                    .unwrap_or_else(|| "<unknown>".into());

                let (file, line) = match &frame.location {
                    Some(loc) => (loc.file.unwrap_or("<unknown>"), loc.line.unwrap_or(0)),
                    None => ("<unknown>", 0),
                };

                inline_trace.push(InlineFrameRecord {
                    file: file.to_string(),
                    line,
                    func_name,
                });
            }

            source_trace.push(SourceTraceRecord {
                trace_record_index: i as u32,
                inline_trace,
            });
        }

        Ok(SourceStackTrace(source_trace, self.1))
    }

    /// Renders the trace as a human-readable, innermost-first backtrace: frame
    /// `#0` is where execution trapped, and each subsequent frame is the caller
    /// that led to it.
    ///
    /// The header names the entry function the trace was captured for. Function
    /// and table names come from the module's `name` section, falling back to
    /// `func #N` / `table #N` when a name is absent.
    pub fn render(&self) -> String {
        let custom_section = &self.2.custom_section;

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

        let mut out = format!("Stack trace of `{}` (most recent call first):\n\n", self.1);

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

/// One source-level frame resolved from DWARF. A single wasm instruction can map
/// to several of these when the compiler inlined calls into it.
pub struct InlineFrameRecord {
    /// Source file the frame came from, or `<unknown>` if DWARF omits it.
    pub file: String,
    /// The function's demangled name, or `<unknown>` if DWARF omits it.
    pub func_name: String,
    /// Line number within [`Self::file`], or `0` if DWARF omits it.
    pub line: u32,
}

/// The source-level expansion of one interpreter frame.
pub struct SourceTraceRecord {
    /// Index of the [`TraceRecord`] in the originating [`StackTrace`] that this
    /// expands, so the two traces can be shown side by side.
    pub trace_record_index: u32,
    /// The source frames for that instruction, outermost call site last. Empty
    /// when the instruction's offset has no DWARF coverage — the record is still
    /// emitted so no interpreter frame goes missing from the trace.
    pub inline_trace: Vec<InlineFrameRecord>,
}

impl InlineFrameRecord {
    /// `file:line`, or just the file when DWARF recorded no line number.
    fn location(&self) -> String {
        if self.line == 0 {
            self.file.clone()
        } else {
            format!("{}:{}", self.file, self.line)
        }
    }
}

impl SourceTraceRecord {
    /// Renders this interpreter frame as a numbered block: the function the
    /// instruction resolved to and its source location, then each function it was
    /// inlined into, outward to the one that physically contains the code.
    ///
    /// The block is `\n`-terminated so records stack directly. A frame the DWARF
    /// does not cover still renders a line, so the numbering stays aligned with
    /// the interpreter trace.
    pub fn render(&self) -> String {
        let index = self.trace_record_index;

        // `inline_trace` is innermost-first: entry 0 is the deepest inlined
        // callee, and everything after it is a caller the compiler inlined it
        // into, ending at the real function.
        let Some((innermost, inlined_into)) = self.inline_trace.split_first() else {
            return format!("  #{index:<3} <no source information>\n");
        };

        let mut out = format!(
            "  #{index:<3} {}\n           at {}\n",
            innermost.func_name,
            innermost.location()
        );

        for caller in inlined_into {
            out.push_str(&format!(
                "        inlined into {}\n           at {}\n",
                caller.func_name,
                caller.location()
            ));
        }

        out
    }
}

/// A [`StackTrace`] resolved against the module's DWARF: one entry per
/// interpreter frame, each expanded into its source-level (and inlined) frames.
pub struct SourceStackTrace<'a>(Vec<SourceTraceRecord>, &'a str);

impl<'a> SourceStackTrace<'a> {
    /// The per-frame records, innermost frame first.
    pub fn records(&self) -> &[SourceTraceRecord] {
        &self.0
    }

    /// Renders the whole source trace, innermost frame first — the source-level
    /// counterpart of [`StackTrace::render`].
    ///
    /// The header names the entry function, carried over from the [`StackTrace`]
    /// this was resolved from. Frame numbers are the indices of the corresponding
    /// [`TraceRecord`]s in that trace, so the two renderings line up.
    pub fn render(&self) -> String {
        let mut out = format!("Source trace of `{}` (most recent call first):\n\n", self.1);

        for record in &self.0 {
            out.push_str(&record.render());
        }

        out
    }
}

/// A failure while resolving a [`StackTrace`] against DWARF debug info.
#[derive(Error, Debug)]
pub enum SourceStackTraceError {
    /// The DWARF sections could not be indexed into an `addr2line` context.
    #[error("context load from dwarf failed: {0}")]
    ContextLoadFailed(String),
    /// Looking up the frames covering an instruction's offset failed.
    #[error("find_frames failed: {0}")]
    FindFramesFailed(String),
    /// Advancing to the next inlined frame failed.
    #[error("failed to fetch the next frame: {0}")]
    NextFrameFetchFailed(String),
    /// The DWARF refers to a split unit (`.dwo`/`.dwp`) or a supplementary file,
    /// which is not loadable from a self-contained wasm module.
    #[error("split DWARF is not supported")]
    SplitDwarfUnsupported,
    /// The module carries no DWARF, so there is nothing to resolve against. This
    /// is the normal case for a release build — debug info has to be enabled for
    /// the wasm to contain `.debug_*` sections.
    #[error("the module has no debug info; rebuild the wasm with debug info enabled")]
    NoDebugInfo,
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
            InstructionExecutionError::CallIndirect(
                table_index,
                CallIndirectError::FunctionCall(callee_func_index, err),
            ) => {
                err._extract_stack_trace(trace);

                Some((*callee_func_index, Some(*table_index)))
            }
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
}

/// A failed [`TypedFunc::call`](crate::instance::TypedFunc::call), carrying the
/// context needed to explain it.
///
/// The underlying [`TraceWasmError`] is raised deep inside the interpreter, where
/// the entry function and the [`Module`] are not in reach. This pairs the cause
/// with both, so a caller can go straight from the error to a rendered backtrace
/// via [`Self::stack_trace`] without threading that context in themselves.
pub struct FuncCallError {
    func_name: String,
    err: TraceWasmError,
    module: Arc<Module>,
}

impl FuncCallError {
    /// Pairs a failure with the entry function and module it happened in.
    ///
    /// `err` must be a [`TraceWasmError::InstructionExecution`]: every error path
    /// out of the interpreter is tagged with the instruction that raised it, which
    /// is what lets [`Self::stack_trace`] always produce at least one frame. The
    /// `debug_assert` below pins that invariant at construction, so a violation
    /// surfaces where the error is built rather than where it is read.
    pub(crate) fn new(func_name: String, err: TraceWasmError, module: Arc<Module>) -> Self {
        debug_assert!(matches!(
            err,
            TraceWasmError::InstructionExecution(_, _, _, _, _)
        ));

        FuncCallError {
            func_name,
            err,
            module,
        }
    }

    /// The underlying cause, without the call context.
    pub fn cause(&self) -> &TraceWasmError {
        &self.err
    }

    /// The interpreter backtrace for this failure, innermost frame first.
    ///
    /// Always has at least one frame: this type is only constructed from a
    /// [`TraceWasmError::InstructionExecution`] cause.
    pub fn stack_trace(&self) -> StackTrace<'_> {
        let mut trace = vec![];

        self.err._extract_stack_trace(&mut trace);

        StackTrace(trace, &self.func_name, &self.module)
    }
}

// `Module` is large and deliberately not `Debug`, so this is written by hand to
// show the parts that identify the failure.
impl Debug for FuncCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FuncCallError")
            .field("func_name", &self.func_name)
            .field("err", &self.err)
            .finish_non_exhaustive()
    }
}

impl Display for FuncCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately a one-liner: the backtrace can be long, so it is opt-in
        // through `stack_trace().render()` rather than part of the message.
        write!(f, "call to `{}` failed: {}", self.func_name, self.err)
    }
}

impl std::error::Error for FuncCallError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.err)
    }
}

impl From<FuncCallError> for TraceWasmError {
    fn from(value: FuncCallError) -> Self {
        value.err
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
    #[error("division failed: {num}/{deno}")]
    Division { num: String, deno: String },
    #[error("remainder failed: {left} % {right}")]
    Remainder { left: String, right: String },
    /// A `trunc` conversion could not represent its operand in the target integer
    /// type — the operand was NaN, an infinity, or truncated to a value outside
    /// the target's range. Fields: the operand, and the target type's name.
    ///
    /// The saturating `trunc_sat` family clamps instead of failing, so it never
    /// produces this.
    #[error("float truncation of `{0}` to {1} failed")]
    FloatToIntTruncation(String, String),
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
mod source_trace_render_tests {
    use super::*;

    fn frame(func_name: &str, file: &str, line: u32) -> InlineFrameRecord {
        InlineFrameRecord {
            file: file.to_string(),
            func_name: func_name.to_string(),
            line,
        }
    }

    fn record(index: u32, inline_trace: Vec<InlineFrameRecord>) -> SourceTraceRecord {
        SourceTraceRecord {
            trace_record_index: index,
            inline_trace,
        }
    }

    #[test]
    fn single_frame_renders_function_and_location() {
        let out = record(0, vec![frame("my_crate::run", "src/lib.rs", 42)]).render();

        assert_eq!(out, "  #0   my_crate::run\n           at src/lib.rs:42\n");
    }

    // `inline_trace` is innermost-first, so entry 0 is the frame and the rest are
    // the callers it was inlined into.
    #[test]
    fn inlined_frames_are_attributed_to_their_callers() {
        let out = record(
            1,
            vec![
                frame("core::num::checked_mul", "core/src/num.rs", 1102),
                frame("my_crate::bench_bits", "src/lib.rs", 517),
            ],
        )
        .render();

        assert_eq!(
            out,
            "  #1   core::num::checked_mul\n           at core/src/num.rs:1102\n\
             \x20       inlined into my_crate::bench_bits\n           at src/lib.rs:517\n"
        );
    }

    #[test]
    fn frame_without_dwarf_coverage_still_renders() {
        // Must not vanish: the numbering has to stay aligned with the interpreter
        // trace even when DWARF covers nothing at that offset.
        assert_eq!(
            record(2, vec![]).render(),
            "  #2   <no source information>\n"
        );
    }

    #[test]
    fn missing_line_number_omits_the_colon() {
        let out = record(0, vec![frame("f", "<unknown>", 0)]).render();

        assert!(out.ends_with("at <unknown>\n"), "got: {out:?}");
    }

    #[test]
    fn whole_trace_renders_header_and_every_frame_in_order() {
        let trace = SourceStackTrace(
            vec![
                record(0, vec![frame("inner", "a.rs", 1)]),
                record(1, vec![]),
                record(2, vec![frame("outer", "b.rs", 9)]),
            ],
            "entry",
        );

        let out = trace.render();

        assert!(out.starts_with("Source trace of `entry` (most recent call first):\n\n"));
        assert_eq!(trace.records().len(), 3);

        // innermost first, one block per frame, none dropped
        let inner = out.find("inner").unwrap();
        let none = out.find("<no source information>").unwrap();
        let outer = out.find("outer").unwrap();
        assert!(inner < none && none < outer, "frames out of order: {out}");
    }

    // The entry-function name is carried by the trace itself, so the header always
    // names it — there is no anonymous form.
    #[test]
    fn empty_trace_still_renders_its_header() {
        let out = SourceStackTrace(vec![], "entry").render();

        assert_eq!(out, "Source trace of `entry` (most recent call first):\n\n");
    }
}

#[cfg(test)]
impl TraceWasmError {
    /// Test-only counterpart of [`FuncCallError::stack_trace`], returning just the
    /// frames.
    ///
    /// The public path borrows the entry-function name and the [`Module`] to build
    /// a [`StackTrace`], neither of which the extraction logic itself uses — this
    /// exposes the frames so that logic can be tested without standing up a whole
    /// module.
    fn stack_trace(&self) -> Option<Vec<TraceRecord>> {
        let mut trace = vec![];

        self._extract_stack_trace(&mut trace)?;

        Some(trace)
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
                .stack_trace()
                .is_none()
        );
        assert!(
            TraceWasmError::ExportNotA("function".to_string())
                .stack_trace()
                .is_none()
        );
    }

    // leaf `Unreachable` (the catch-all IE arm) → single NonCall record.
    #[test]
    fn single_unreachable_frame() {
        let trace = ie(0, 7, InstructionExecutionError::Unreachable)
            .stack_trace()
            .expect("top-level is an InstructionExecution");

        assert_eq!(trace.len(), 1);
        assert_noncall(&trace[0], 0, 7);
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
            .stack_trace()
            .unwrap();

            assert_eq!(trace.len(), 1);
            assert_noncall(&trace[0], 1, i);
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

        let trace = err.stack_trace().unwrap();

        assert_eq!(trace.len(), 2);
        assert_noncall(&trace[0], 3, 1); // innermost first
        assert_call(&trace[1], 2, 5, 3, None); // direct call → is_indirect None
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

        let trace = err.stack_trace().unwrap();

        assert_eq!(trace.len(), 2);
        assert_noncall(&trace[0], 6, 0);
        assert_call(&trace[1], 1, 9, 6, Some(4)); // indirect → is_indirect Some(table)
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
        let trace = via_call.stack_trace().unwrap();
        assert_eq!(trace.len(), 1, "caller frame must survive a non-IE leaf");
        assert_call(&trace[0], 0, 3, 5, None);

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
        let trace = via_indirect.stack_trace().unwrap();
        assert_eq!(trace.len(), 1);
        assert_call(&trace[0], 0, 2, 9, Some(1));
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

        let trace = err.stack_trace().unwrap();

        assert_eq!(trace.len(), 3);
        assert_noncall(&trace[0], 2, 0); // innermost: the trap
        assert_call(&trace[1], 1, 4, 2, Some(2)); // B called C indirectly via table 2
        assert_call(&trace[2], 0, 10, 1, None); // A called B directly
    }
}

#[cfg(test)]
mod func_call_error_tests {
    use super::*;

    fn err() -> TraceWasmError {
        TraceWasmError::InstructionExecution(
            FuncIndex(3),
            7,
            Instruction::Nop,
            InstructionExecutionError::Unreachable,
            0x2a,
        )
    }

    // The whole point of the type: it must work as an ordinary error.
    #[test]
    fn implements_std_error_with_display_and_source() {
        fn assert_is_error<E: std::error::Error + 'static>(_: &E) {}

        let module = crate::module::Module::compile(&wat_min()).unwrap();
        let e = FuncCallError::new("entry".to_string(), err(), module);

        assert_is_error(&e);
        assert!(
            e.to_string().starts_with("call to `entry` failed:"),
            "got: {e}"
        );
        assert!(std::error::Error::source(&e).is_some(), "cause must chain");
        assert!(
            format!("{e:?}").contains("func_name"),
            "Debug must not panic"
        );
    }

    // A release module has no `.debug_*` sections; that must be an error, not a panic.
    #[test]
    fn source_trace_without_debug_info_errors_instead_of_panicking() {
        let module = crate::module::Module::compile(&wat_min()).unwrap();
        let e = FuncCallError::new("entry".to_string(), err(), module);

        assert!(matches!(
            e.stack_trace().to_source_trace(),
            Err(SourceStackTraceError::NoDebugInfo)
        ));
    }

    /// The smallest valid module: `(module)`.
    fn wat_min() -> Vec<u8> {
        vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]
    }
}
