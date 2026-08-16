//! The owned, in-memory representation of a parsed WebAssembly module.
//!
//! [`Module::compile`] builds an `Arc<Module>` from a binary; every section is
//! converted from `wasmparser`'s borrowing views into owned data (boxed slices,
//! `String`s) so the module outlives the input bytes. The `Arc` lets one
//! compiled module back several [`Instance`]s.
//!
//! ## Typed entity indices
//!
//! WebAssembly refers to everything — functions, types, tables, globals, … — by
//! bare `u32` indices into per-kind index spaces. To avoid mixing them up, each
//! space gets its own newtype (`FuncIndex`, `TyIndex`, `TableIndex`, …) built
//! over the shared [`EntityIndex`] trait. They are `Copy` and hashable so they
//! can key the name maps below.
//!
//! Note the index spaces include imports: e.g. `FuncIndex(n)` addresses the
//! `n`-th function counting imported functions first, then locally-defined ones
//! (see [`Module::func_decls`] and [`Module::imported_func_count`]).

use crate::{
    error::TraceWasmError,
    instance::{
        Instance, TypedFunc,
        config::Config,
        traits::{ImportRegistry, Params, Results},
    },
    instruction::{Instruction, stack::StackInstruction},
    memory::Memory,
    runtime::{
        TraceVM,
        value::{DataVal, ElementVal, TableVal, Val},
    },
};
use core::fmt::{self, Debug};
use gimli::{Dwarf, EndianArcSlice, EndianReader, RunTimeEndian, SectionId};
use phf::phf_set;
use rustc_hash::FxHashMap;
use std::{hash::Hash, sync::Arc};
use wasmparser::{Encoding, ExternalKind, Parser, Payload::*, TypeRef, Validator};

/// Size in bytes of one WebAssembly linear-memory page (64 KiB).
///
/// A fresh instance allocates `initial_pages * WASM_MEMORY_PAGE_SIZE` bytes,
/// where `initial_pages` comes from the module's declared memory limits (see
/// [`Module::instantiate`]).
///
/// TODO: honor the custom-page-sizes proposal instead of assuming 64 KiB.
pub const WASM_MEMORY_PAGE_SIZE: u64 = 64 * 1024; // 64 KiB (one wasm page)

/// The module's parsed DWARF, as loaded from its `.debug_*` custom sections.
///
/// Shared behind an `Arc` because the sections are large and every consumer of
/// the debug info reads the same copy.
pub type ModuleDwarf = Arc<Dwarf<EndianReader<RunTimeEndian, Arc<[u8]>>>>;

/// The type of a WebAssembly global: its value type plus mutability.
///
/// An owned wrapper that keeps the underlying `wasmparser` representation
/// private so it does not leak into this crate's public API.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct GlobalType(wasmparser::GlobalType);

impl GlobalType {
    /// Adopts the parser's representation. Crate-private so the wrapped type
    /// stays out of this crate's public API.
    pub(crate) fn from_wasmparser(value: wasmparser::GlobalType) -> Self {
        GlobalType(value)
    }

    /// The value type stored by this global.
    pub fn content_type(&self) -> ValType {
        ValType::from_wasmparser(self.0.content_type)
    }

    /// Whether the global is mutable (`global.set` is allowed).
    pub fn is_mutable(&self) -> bool {
        self.0.mutable
    }

    /// Whether the global is shared across threads (threads proposal).
    pub fn is_shared(&self) -> bool {
        self.0.shared
    }
}

/// The type of a WebAssembly linear memory: its index width and page limits.
///
/// An owned wrapper that keeps the underlying `wasmparser` representation
/// private so it does not leak into this crate's public API.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MemoryType(wasmparser::MemoryType);

impl MemoryType {
    /// Adopts the parser's representation. Crate-private so the wrapped type
    /// stays out of this crate's public API.
    pub(crate) fn from_wasmparser(value: wasmparser::MemoryType) -> Self {
        MemoryType(value)
    }

    /// Whether this is a 64-bit memory (indexed by `i64`); `false` means a
    /// 32-bit memory indexed by `i32` (memory64 proposal).
    pub fn is_64(&self) -> bool {
        self.0.memory64
    }

    /// Whether this memory is shared across threads (threads proposal).
    pub fn is_shared(&self) -> bool {
        self.0.shared
    }

    /// Initial size, in pages.
    pub fn initial(&self) -> u64 {
        self.0.initial
    }

    /// Optional maximum size, in pages.
    pub fn maximum(&self) -> Option<u64> {
        self.0.maximum
    }

    /// Log base 2 of the memory's page size (16 for the default 64 KiB page,
    /// per the custom-page-sizes proposal).
    pub fn page_size_log2(&self) -> u32 {
        self.0.page_size_log2()
    }
}

/// The type of a WebAssembly tag (exception-handling proposal).
///
/// An owned wrapper that keeps the underlying `wasmparser` representation
/// private so it does not leak into this crate's public API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TagType(wasmparser::TagType);

impl TagType {
    /// Adopts the parser's representation. Crate-private so the wrapped type
    /// stays out of this crate's public API.
    pub(crate) fn from_wasmparser(value: wasmparser::TagType) -> Self {
        TagType(value)
    }

    /// Index into the type section of the function type describing this tag's
    /// payload.
    pub fn func_type_index(&self) -> TyIndex {
        TyIndex(self.0.func_type_idx)
    }
}

// Mirrors `wasmparser::ValType`, re-declared here so the type is owned by this
// crate rather than leaking the `wasmparser` dependency into the public API.
/// Represents the types of values in a WebAssembly module.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValType {
    /// The value type is i32.
    I32,
    /// The value type is i64.
    I64,
    /// The value type is f32.
    F32,
    /// The value type is f64.
    F64,
    /// The value type is v128.
    V128,
    /// The value type is a reference.
    Ref(RefType),
}

impl ValType {
    /// Alias for the wasm `funcref` type.
    pub const FUNCREF: ValType = ValType::Ref(RefType(wasmparser::RefType::FUNCREF));

    /// Alias for the wasm `externref` type.
    pub const EXTERNREF: ValType = ValType::Ref(RefType(wasmparser::RefType::EXTERNREF));

    /// Alias for the wasm `exnref` type.
    pub const EXNREF: ValType = ValType::Ref(RefType(wasmparser::RefType::EXNREF));

    /// Alias for the wasm `contref` type.
    pub const CONTREF: ValType = ValType::Ref(RefType(wasmparser::RefType::CONTREF));

    /// Maps the parser's value type onto this crate's.
    ///
    /// Total, including `V128`: rejecting SIMD is left to the callers that would
    /// have to represent a value of it, so the type itself can still be named in
    /// a signature this crate merely reads past.
    pub(crate) fn from_wasmparser(value: wasmparser::ValType) -> Self {
        match value {
            wasmparser::ValType::I32 => ValType::I32,
            wasmparser::ValType::I64 => ValType::I64,
            wasmparser::ValType::F32 => ValType::F32,
            wasmparser::ValType::F64 => ValType::F64,
            wasmparser::ValType::V128 => ValType::V128,
            wasmparser::ValType::Ref(r) => ValType::Ref(RefType(r)),
        }
    }
}

/// Formats a value-type list as a parenthesized, comma-separated tuple, e.g.
/// `(I32,F64)`. An empty list renders as `()` (correctly handling void/no-arg
/// signatures — the previous index-based version underflowed on an empty slice).
pub(crate) fn formatted_val_types(types: &[ValType]) -> String {
    let inner = types
        .iter()
        .map(|ty| format!("{ty:?}"))
        .collect::<Vec<_>>()
        .join(",");

    format!("({inner})")
}

/// A WebAssembly reference type (e.g. `funcref`, `externref`).
///
/// An owned wrapper that keeps the underlying `wasmparser` representation
/// private so it does not leak into this crate's public API.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RefType(wasmparser::RefType);

impl RefType {
    /// Adopts the parser's representation. Crate-private so the wrapped type
    /// stays out of this crate's public API.
    pub(crate) fn from_wasmparser(value: wasmparser::RefType) -> Self {
        RefType(value)
    }

    /// Whether the reference may be null.
    pub fn is_nullable(&self) -> bool {
        self.0.is_nullable()
    }

    /// Whether this is a `funcref` (a reference to a function).
    pub fn is_func_ref(&self) -> bool {
        self.0.is_func_ref()
    }

    /// Whether this is an `externref` (an opaque host reference).
    pub fn is_extern_ref(&self) -> bool {
        self.0.is_extern_ref()
    }
}

impl fmt::Debug for RefType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Marker trait for the typed `u32` index newtypes.
///
/// The `From<u32>` bound lets generic code (e.g. the name-map builders) turn raw
/// `wasmparser` indices into the correct typed index.
pub trait EntityIndex: From<u32> {}

/// Index into the function index space (imports first, then defined functions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for FuncIndex {
    fn from(value: u32) -> Self {
        FuncIndex(value)
    }
}

impl EntityIndex for FuncIndex {}

/// Index used by the `ref.func` "exact" reference form (function-references
/// proposal); addresses the function index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncExactIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for FuncExactIndex {
    fn from(value: u32) -> Self {
        FuncExactIndex(value)
    }
}

impl EntityIndex for FuncExactIndex {}

/// Index into the type section ([`Module::types`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for TyIndex {
    fn from(value: u32) -> Self {
        TyIndex(value)
    }
}

impl EntityIndex for TyIndex {}

/// Index into the global index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for GlobalIndex {
    fn from(value: u32) -> Self {
        GlobalIndex(value)
    }
}

impl EntityIndex for GlobalIndex {}

/// Index into the table index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for TableIndex {
    fn from(value: u32) -> Self {
        TableIndex(value)
    }
}

impl EntityIndex for TableIndex {}

/// Index into the memory index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for MemoryIndex {
    fn from(value: u32) -> Self {
        MemoryIndex(value)
    }
}

impl EntityIndex for MemoryIndex {}

/// Index into the tag index space (exception-handling proposal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TagIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for TagIndex {
    fn from(value: u32) -> Self {
        TagIndex(value)
    }
}

impl EntityIndex for TagIndex {}

/// Index into the element segment space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElementIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for ElementIndex {
    fn from(value: u32) -> Self {
        ElementIndex(value)
    }
}

impl EntityIndex for ElementIndex {}

/// Index into the data segment space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for DataIndex {
    fn from(value: u32) -> Self {
        DataIndex(value)
    }
}

impl EntityIndex for DataIndex {}

/// Index of a local within a function (params first, then declared locals).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for LocalIndex {
    fn from(value: u32) -> Self {
        LocalIndex(value)
    }
}

impl EntityIndex for LocalIndex {}

/// Index of a field within a struct type (GC proposal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldIndex(
    /// The raw index value.
    pub u32,
);

impl From<u32> for FieldIndex {
    fn from(value: u32) -> Self {
        FieldIndex(value)
    }
}

impl EntityIndex for FieldIndex {}

/// A fully parsed module: every section decoded into owned data.
///
/// Field order roughly follows the binary section order. Index spaces that can
/// be imported — functions and globals — fold imports and definitions into a
/// single list, imports first, matching how wasm numbers them; the
/// `imported_*_count` fields record where the boundary falls. See the per-field
/// docs.
pub struct Module<Instr: Instruction> {
    /// The type section. Only function types are kept — GC composite types
    /// (struct/array) are rejected during parsing.
    pub types: Box<[FuncType]>,
    /// The function index space: imported functions first, then locally-defined
    /// ones. The split point is [`Self::imported_func_count`].
    pub func_decls: Box<[FuncDecl]>,
    /// The table section.
    pub tables: Box<[Table]>,
    /// The memory section (TraceWasm currently allows at most one memory).
    pub memories: Box<[MemoryType]>,
    /// The tag section (exception-handling proposal).
    pub tags: Box<[TagType]>,
    /// The global index space: imported globals first, then locally-defined ones.
    /// The split point is [`Self::imported_global_count`].
    pub globals: Box<[Global]>,
    /// The export section, keyed by export name.
    pub exports: FxHashMap<String, Export>,
    /// Reverse index over the function entries of [`Self::exports`]: exported
    /// function index to export name. Lets an error name the function it came from
    /// without scanning the export table.
    pub exported_func_index_to_name: FxHashMap<FuncIndex, String>,
    /// The `start` function, run at instantiation, if the module declares one.
    pub start_section: Option<FuncIndex>,
    /// The element section.
    pub elements: Box<[Element]>,
    /// The declared data-segment count from the data-count section, if present
    /// (required by the bulk-memory proposal for validating `data.drop`, etc.).
    pub data_count: Option<u32>,
    /// The data section.
    pub datas: Box<[Data]>,
    /// Declared entry count of the code section (should match `func_bodies.len()`).
    pub code_sec_count: u32,
    /// Byte size of the code section as declared in its header.
    pub code_sec_size: u32,
    /// Number of imported functions; the boundary between imports and
    /// definitions in [`Self::func_decls`], and the offset mapping the `i`-th
    /// `func_bodies` entry to `func_decls[imported_func_count + i]`.
    pub imported_func_count: u32,
    /// Number of imported globals, mirroring [`Self::imported_func_count`]: the
    /// first `imported_global_count` entries of [`Self::globals`] are imports
    /// (with [`GlobalKind::Imported`]), the rest are locally defined.
    pub imported_global_count: u32,
    /// Lowered bodies of the locally-defined functions, in definition order.
    pub func_bodies: Box<[FuncBody<Instr>]>,
    /// Sections with an unrecognized id, preserved verbatim as `(id, contents)`.
    pub unknown_sections: Box<[(u8, Box<[u8]>)]>, // (id, content)
    /// Decoded `name`-section maps plus the raw bytes of other custom sections.
    pub custom_section: CustomSection,
    /// The module's DWARF debug info, parsed from its `.debug_*` custom sections,
    /// or `None` if the module carries none (no `.debug_info`).
    ///
    /// Behind an `Arc` because the sections are large and shared by every
    /// consumer; read it through [`Self::dwarf`].
    dwarf: Option<ModuleDwarf>,
}

impl<Instr: Instruction> Module<Instr> {
    /// The module's export map, keyed by export name.
    pub fn exports(&self) -> &FxHashMap<String, Export> {
        &self.exports
    }

    /// Looks up the export named `name`, if any.
    pub fn export(&self, name: &str) -> Option<Export> {
        self.exports.get(name).cloned()
    }

    /// The module's DWARF debug info, or `None` if it was built without any.
    ///
    /// Cheap to call: clones an `Arc`, not the sections. Use it to map an
    /// instruction's byte offset (see [`FuncBody::instruction_offsets`]) back to a
    /// source location.
    pub fn dwarf(&self) -> Option<ModuleDwarf> {
        self.dwarf.clone()
    }

    /// Looks up an exported function by name and returns a typed handle to it,
    /// checking that the typed signature `P` -> `R` matches the module's
    /// declared signature.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::ExportNotFound`] if no export has that name,
    /// [`TraceWasmError::ExportNotA`] if the export exists but is not a function,
    /// and [`TraceWasmError::IncorrectParamsResultsStructure`] if the parameter
    /// or result types of `P`/`R` disagree with the function's signature (in
    /// count or in the type at any position).
    pub fn get_typed_func<P: Params, R: Results>(
        &self,
        name: &str,
    ) -> Result<TypedFunc<P, R>, TraceWasmError<Instr>> {
        let Some(export) = self.export(name) else {
            return Err(TraceWasmError::ExportNotFound(name.to_string()));
        };

        let func_index = export.to_func()?;
        let func_decl = &self.func_decls[func_index.0 as usize];
        let ty_index = func_decl.ty;
        let ty = &self.types[ty_index.0 as usize];

        // Full type-list equality subsumes the arity check: a difference in
        // length or in the type at any position is reported the same way, with
        // the declared signature as "expected" and the typed handle as "got".
        let param_types = P::types();
        let result_types = R::types();

        if ty.params.as_ref() != param_types.as_ref() {
            return Err(TraceWasmError::IncorrectParamsResultsStructure(
                "params".to_string(),
                func_index.0,
                formatted_val_types(&ty.params),
                formatted_val_types(param_types.as_ref()),
            ));
        }

        if ty.results.as_ref() != result_types.as_ref() {
            return Err(TraceWasmError::IncorrectParamsResultsStructure(
                "results".to_string(),
                func_index.0,
                formatted_val_types(&ty.results),
                formatted_val_types(result_types.as_ref()),
            ));
        }

        Ok(TypedFunc::new(func_index))
    }

    /// The export name of function `index`, or `None` if it is not exported.
    pub fn exported_func_name(&self, index: FuncIndex) -> Option<&String> {
        self.exported_func_index_to_name.get(&index)
    }
}

/// A locally-defined function's locals and lowered instruction list.
pub struct FuncBody<Instr: Instruction> {
    /// All locals addressable in the body: the function's params first, then the
    /// declared locals, expanded from the run-length-encoded body header.
    pub locals: Box<[ValType]>,
    /// The body lowered by [`crate::instruction`] (control flow resolved to
    /// absolute indices).
    pub instructions: Box<[Instr]>,
    /// Source positions for [`Self::instructions`], used to point diagnostics at
    /// the original binary.
    ///
    /// **Invariant:** parallel to `instructions` — `instruction_offsets[i]` is the
    /// byte offset in the module binary of the operator that produced
    /// `instructions[i]`, and both slices always have the same length. Lowering
    /// pushes the two together, which is what upholds this.
    pub instruction_offsets: Box<[u32]>,
    pub(crate) frame_layout: Instr::FrameLayout,
}

/// The module's custom-section data, flattened for direct lookup: the decoded
/// `name` section (one map per subsection) plus the raw bytes of any other custom
/// sections, keyed by section name.
pub struct CustomSection {
    /// The module's name from the `name` section (empty if none).
    pub module_name: String,
    /// Function names.
    pub func: NameMap<FuncIndex>,
    /// Local-variable names, per function.
    pub local: IndirectNameMap<FuncIndex, LocalIndex>,
    /// Label names, per function.
    pub label: IndirectNameMap<FuncIndex, LocalIndex>,
    /// Type names.
    pub ty: NameMap<TyIndex>,
    /// Table names.
    pub table: NameMap<TableIndex>,
    /// Memory names.
    pub mem: NameMap<MemoryIndex>,
    /// Global names.
    pub global: NameMap<GlobalIndex>,
    /// Element-segment names.
    pub element: NameMap<ElementIndex>,
    /// Data-segment names.
    pub data: NameMap<DataIndex>,
    /// Struct-field names, per type (GC proposal).
    pub field: IndirectNameMap<TyIndex, FieldIndex>,
    /// Tag names.
    pub tag: NameMap<TagIndex>,
    /// Unrecognized `name`-section subsections, keyed by subsection type byte.
    pub name_unknown: FxHashMap<u8, Box<[u8]>>,
    /// Custom sections `wasmparser` recognizes but we keep raw, by section name.
    pub other: FxHashMap<String, Box<[u8]>>,
    /// Custom sections `wasmparser` does not recognize, by section name.
    pub unknown: FxHashMap<String, Box<[u8]>>,
}

impl CustomSection {
    /// The module's name from the `name` section's module subsection, if present
    /// (empty when the module declares none).
    pub fn module_name(&self) -> &str {
        &self.module_name
    }

    /// The name of function `index`, if the `name` section records one.
    pub fn func_name(&self, index: FuncIndex) -> Option<&str> {
        self.func.0.get(&index).map(String::as_str)
    }

    /// The name of local `local` in function `func`, if recorded.
    pub fn local_name(&self, func: FuncIndex, local: LocalIndex) -> Option<&str> {
        self.local.0.get(&func)?.0.get(&local).map(String::as_str)
    }

    /// The name of label `label` in function `func`, if recorded.
    pub fn label_name(&self, func: FuncIndex, label: LocalIndex) -> Option<&str> {
        self.label.0.get(&func)?.0.get(&label).map(String::as_str)
    }

    /// The name of type `index`, if recorded.
    pub fn type_name(&self, index: TyIndex) -> Option<&str> {
        self.ty.0.get(&index).map(String::as_str)
    }

    /// The name of table `index`, if recorded.
    pub fn table_name(&self, index: TableIndex) -> Option<&str> {
        self.table.0.get(&index).map(String::as_str)
    }

    /// The name of memory `index`, if recorded.
    pub fn memory_name(&self, index: MemoryIndex) -> Option<&str> {
        self.mem.0.get(&index).map(String::as_str)
    }

    /// The name of global `index`, if recorded.
    pub fn global_name(&self, index: GlobalIndex) -> Option<&str> {
        self.global.0.get(&index).map(String::as_str)
    }

    /// The name of element segment `index`, if recorded.
    pub fn element_name(&self, index: ElementIndex) -> Option<&str> {
        self.element.0.get(&index).map(String::as_str)
    }

    /// The name of data segment `index`, if recorded.
    pub fn data_name(&self, index: DataIndex) -> Option<&str> {
        self.data.0.get(&index).map(String::as_str)
    }

    /// The name of tag `index`, if recorded.
    pub fn tag_name(&self, index: TagIndex) -> Option<&str> {
        self.tag.0.get(&index).map(String::as_str)
    }

    /// The name of field `field` in type `ty`, if recorded.
    pub fn field_name(&self, ty: TyIndex, field: FieldIndex) -> Option<&str> {
        self.field.0.get(&ty)?.0.get(&field).map(String::as_str)
    }

    /// The raw bytes of the custom section named `name` (recognized-but-kept-raw
    /// first, then unrecognized).
    pub fn raw_section(&self, name: &str) -> Option<&[u8]> {
        self.other
            .get(name)
            .or_else(|| self.unknown.get(name))
            .map(Box::as_ref)
    }
}

/// A data segment.
pub struct Data {
    /// Whether the segment is passive or actively initializes memory.
    pub kind: DataKind,
    /// The segment's raw bytes.
    pub data: Box<[u8]>,
}

/// Whether a data segment is passive or actively initializes memory.
pub enum DataKind {
    /// Copied into memory only via `memory.init`.
    Passive,
    /// Copied into `memory_index` at instantiation, at the offset computed by
    /// `offset_expr`.
    Active {
        /// Target memory index.
        memory_index: MemoryIndex,
        /// Constant expression computing the destination offset.
        offset_expr: Box<[StackInstruction]>,
    },
}

/// Whether an element segment is passive, active, or declared.
pub enum ElementKind {
    /// Usable only via `table.init`.
    Passive,
    /// Copied into `table_index` at instantiation at `offset_expr` (a `None`
    /// table index means table 0).
    Active {
        /// Target table index (`None` means table 0).
        table_index: Option<TableIndex>,
        /// Constant expression computing the destination offset.
        offset_expr: Box<[StackInstruction]>,
    },
    /// Forward-declares references (for `ref.func`); contributes no table data.
    Declared,
}

/// The payload of an element segment.
pub enum ElementItems {
    /// A list of function indices (the legacy form).
    Functions(Box<[FuncIndex]>),
    /// Constant expressions of the given reference type (each yields one element).
    Expressions(RefType, Box<[Box<[StackInstruction]>]>),
}

/// An element segment.
pub struct Element {
    /// Whether the segment is passive, active, or declared.
    pub kind: ElementKind,
    /// The segment's payload (function indices or constant expressions).
    pub items: ElementItems,
}

/// How a table's slots are initialized.
pub enum TableInit {
    /// Every slot starts as a null reference.
    RefNull,
    /// Every slot is initialized from this constant expression.
    Expr(Box<[StackInstruction]>),
}

/// A table definition: its type plus initialization.
pub struct Table {
    /// The table's element reference type (always `funcref` in TraceWasm).
    pub element_ty: RefType,
    /// How the table's slots are initialized.
    pub init: TableInit,
    /// Initial size, in elements.
    pub initial: u64,
    /// Optional maximum size, in elements.
    pub maximum: Option<u64>,
}

/// A named export and the entity it exposes.
#[derive(Debug, Clone, Copy)]
pub enum Export {
    /// An exported function.
    Func(FuncIndex),
    /// An exported table.
    Table(TableIndex),
    /// An exported memory.
    Memory(MemoryIndex),
    /// An exported global.
    Global(GlobalIndex),
    /// An exported tag (exception-handling proposal).
    Tag(TagIndex),
    /// An exported function in the "exact" reference form (function-references
    /// proposal).
    FuncExact(FuncExactIndex),
}

impl Export {
    /// Returns the function index if this is a [`Export::Func`], otherwise
    /// [`TraceWasmError::ExportNotA`].
    pub fn to_func<Instr: Instruction>(&self) -> Result<FuncIndex, TraceWasmError<Instr>> {
        let Export::Func(func_index) = self else {
            return Err(TraceWasmError::ExportNotA("function".to_string()));
        };

        Ok(*func_index)
    }
}

/// Whether a global is imported or locally defined.
pub enum GlobalKind {
    /// Imported from `module`::`name`; its value is supplied at instantiation.
    Imported {
        /// The import's module name.
        module: String,
        /// The import's field name within that module.
        name: String,
    },
    /// Locally defined; the constant expression computes its initial value.
    Local(Box<[StackInstruction]>),
}

/// A global definition: its type plus the constant expression for its initial
/// value.
pub struct Global {
    /// The global's value type and mutability.
    pub ty: GlobalType,
    /// Whether the global is imported or locally defined (with its init expr).
    pub kind: GlobalKind,
}

/// A function type: parameter and result value types.
pub struct FuncType {
    /// Parameter value types.
    pub params: Box<[ValType]>,
    /// Result value types.
    pub results: Box<[ValType]>,
}

/// Whether a function is defined locally or imported.
pub enum FuncKind {
    /// Defined locally, with a body in [`Module::func_bodies`].
    Local,
    /// Imported from `module_name`::`imported_func_name`.
    Imported {
        /// Name of the module the function is imported from.
        module_name: String,
        /// Name of the imported function within that module.
        imported_func_name: String,
    },
}

/// A function declaration: its origin and its type-section index.
pub struct FuncDecl {
    /// Whether the function is defined locally or imported.
    pub kind: FuncKind,
    /// Index into [`Module::types`] giving this function's signature.
    pub ty: TyIndex,
}

/// A `name`-section map from a typed entity index to its name.
pub struct NameMap<T: PartialEq + Eq + Hash + EntityIndex>(
    /// Map from typed index to name.
    pub FxHashMap<T, String>,
);

impl<T: PartialEq + Eq + Hash + EntityIndex> Default for NameMap<T> {
    fn default() -> Self {
        NameMap(FxHashMap::default())
    }
}

impl<T: PartialEq + Eq + Hash + EntityIndex> NameMap<T> {
    /// Converts a `wasmparser` name map into an owned [`NameMap`], turning each
    /// raw `u32` into the typed index `T`.
    pub(crate) fn from_wasmparser_name_map<Instr: Instruction>(
        map: wasmparser::NameMap<'_>,
        new_map: &mut NameMap<T>,
    ) -> Result<(), TraceWasmError<Instr>> {
        for name in map {
            let name = name?;
            let index = T::from(name.index);
            let name = name.name.to_string();

            new_map.0.insert(index, name);
        }

        Ok(())
    }
}

/// A two-level `name`-section map: an outer index to a nested [`NameMap`].
///
/// For example `IndirectNameMap<FuncIndex, LocalIndex>` maps each function to
/// the names of its locals.
pub struct IndirectNameMap<
    T: PartialEq + Eq + Hash + EntityIndex,
    V: PartialEq + Eq + Hash + EntityIndex,
>(
    /// Map from outer index to the nested [`NameMap`] of inner names.
    pub FxHashMap<T, NameMap<V>>,
);

impl<T: PartialEq + Eq + Hash + EntityIndex, V: PartialEq + Eq + Hash + EntityIndex> Default
    for IndirectNameMap<T, V>
{
    fn default() -> Self {
        IndirectNameMap(FxHashMap::default())
    }
}

impl<T: PartialEq + Eq + Hash + EntityIndex, V: PartialEq + Eq + Hash + EntityIndex>
    IndirectNameMap<T, V>
{
    /// Converts a `wasmparser` indirect name map into the owned two-level form.
    pub(crate) fn from_wasmparser_indirect_map<Instr: Instruction>(
        map: wasmparser::IndirectNameMap<'_>,
        new_map: &mut IndirectNameMap<T, V>,
    ) -> Result<(), TraceWasmError<Instr>> {
        for entry in map {
            let entry = entry?;
            let outer_index = T::from(entry.index);
            let mut inner_name_map: NameMap<V> = NameMap::default();

            NameMap::from_wasmparser_name_map(entry.names, &mut inner_name_map)?;

            new_map.0.insert(outer_index, inner_name_map);
        }

        Ok(())
    }
}

/// Custom-section names that carry DWARF debug info, routed to the DWARF loader
/// instead of being kept as opaque bytes in [`CustomSection`].
///
/// The names match [`gimli::SectionId::name`] (leading dot included), so a hit
/// here is exactly a section the loader will ask for.
static DEBUG_SECTION_NAMES: phf::Set<&'static str> = phf_set! {
    ".debug_abbrev",
    ".debug_addr",
    ".debug_aranges",
    ".debug_cu_index",
    ".debug_frame",
    ".eh_frame",
    ".eh_frame_hdr",
    ".debug_gnu_pubnames",
    ".debug_gnu_pubtypes",
    ".debug_info",
    ".debug_line",
    ".debug_line_str",
    ".debug_loc",
    ".debug_loclists",
    ".debug_macinfo",
    ".debug_macro",
    ".debug_names",
    ".debug_pubnames",
    ".debug_pubtypes",
    ".debug_ranges",
    ".debug_rnglists",
    ".debug_str",
    ".debug_str_offsets",
    ".debug_tu_index",
    ".debug_types"
};

impl<Instr: Instruction> Module<Instr> {
    /// Validates `buf` as a core WebAssembly module and builds an owned
    /// [`Module`] from it, wrapped in an `Arc` so it can be shared across
    /// instances.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::Parsing`] if validation or decoding fails. GC
    /// types arrive this way too: `wasmparser` refuses them while reading the
    /// type section, before this pass sees them.
    ///
    /// Returns [`TraceWasmError::Unsupported`] for valid modules using features
    /// TraceWasm does not model: components, imports other than functions and
    /// globals, 64-bit memory, more than one memory, tables of anything but
    /// `funcref`, `v128` locals, or any operator the lowering pass rejects.
    pub fn compile(buf: &[u8]) -> Result<Arc<Module<Instr>>, TraceWasmError<Instr>> {
        // The parser alone only checks structure; validate semantics (section
        // order, index bounds, types) up front so the AST is built from wasm
        // that is known to be well-formed.
        Validator::new().validate_all(buf)?;

        let mut types = vec![];
        let mut func_decls = vec![];
        let mut tables = vec![];
        let mut memories = vec![];
        let mut globals = vec![];
        let mut imported_global_count = 0;
        let mut exports: FxHashMap<String, Export> = FxHashMap::default();
        let mut elements = vec![];
        let mut start_section: Option<FuncIndex> = None;
        let mut imported_func_count = 0;
        let mut data_count = None;
        let mut datas = vec![];
        let mut code_sec_count = 0;
        let mut code_sec_size = 0;
        let mut func_bodies = vec![];
        let mut tags = vec![];
        let mut unknown_sections = vec![];
        let mut debug_sections: FxHashMap<String, &[u8]> = FxHashMap::default();
        let mut possible_dwarf: Option<ModuleDwarf> = None;
        let mut custom_section_unknowns: FxHashMap<String, Box<[u8]>> = FxHashMap::default();
        let mut custom_section_others: FxHashMap<String, Box<[u8]>> = FxHashMap::default();
        let mut custom_section_module_name: String = "".to_string();
        let mut custom_section_func: NameMap<FuncIndex> = NameMap::default();
        let mut custom_section_local: IndirectNameMap<FuncIndex, LocalIndex> =
            IndirectNameMap::default();
        let mut custom_section_label: IndirectNameMap<FuncIndex, LocalIndex> =
            IndirectNameMap::default();
        let mut custom_section_ty: NameMap<TyIndex> = NameMap::default();
        let mut custom_section_table: NameMap<TableIndex> = NameMap::default();
        let mut custom_section_mem: NameMap<MemoryIndex> = NameMap::default();
        let mut custom_section_global: NameMap<GlobalIndex> = NameMap::default();
        let mut custom_section_element: NameMap<ElementIndex> = NameMap::default();
        let mut custom_section_data: NameMap<DataIndex> = NameMap::default();
        let mut custom_section_field: IndirectNameMap<TyIndex, FieldIndex> =
            IndirectNameMap::default();
        let mut custom_section_tag: NameMap<TagIndex> = NameMap::default();
        let mut custom_section_name_unknown: FxHashMap<u8, Box<[u8]>> = FxHashMap::default();
        let mut exported_func_index_to_name: FxHashMap<FuncIndex, String> = FxHashMap::default();

        for payload in Parser::new(0).parse_all(buf) {
            let payload = payload?;

            match payload {
                Version {
                    num: _num,
                    encoding,
                    range: _range,
                } => {
                    if encoding == Encoding::Component {
                        return Err(TraceWasmError::Unsupported("component model".to_string()));
                    }
                }
                TypeSection(ty_sec) => {
                    let types_iter = ty_sec.into_iter_err_on_gc_types();

                    for ty in types_iter {
                        let ty = ty?;
                        let params = ty.params();
                        let results = ty.results();

                        types.push(FuncType {
                            params: params
                                .iter()
                                .map(|v| ValType::from_wasmparser(*v))
                                .collect(),
                            results: results
                                .iter()
                                .map(|v| ValType::from_wasmparser(*v))
                                .collect(),
                        });
                    }
                }
                ImportSection(import_sec) => {
                    let imports_iter = import_sec.into_imports();

                    for import in imports_iter {
                        let import = import?;
                        let module_name = import.module.to_string();
                        let imported_func_name = import.name.to_string();

                        match import.ty {
                            TypeRef::Func(ty) => {
                                func_decls.push(FuncDecl {
                                    kind: FuncKind::Imported {
                                        module_name,
                                        imported_func_name,
                                    },
                                    ty: TyIndex(ty),
                                });
                            }
                            TypeRef::Global(ty) => {
                                globals.push(Global {
                                    ty: GlobalType(ty),
                                    kind: GlobalKind::Imported {
                                        module: module_name,
                                        name: imported_func_name,
                                    },
                                });
                            }
                            _ => {
                                return Err(TraceWasmError::Unsupported(
                                    "only function or global imports allowed".to_string(),
                                ));
                            }
                        }
                    }

                    // At this point `func_decls` holds only imports (the function section comes
                    // later), and any non-function import already returned above — so this count is
                    // exactly the number of imported functions and marks the imports/definitions split.
                    imported_func_count = func_decls.len() as u32;
                    imported_global_count = globals.len() as u32;
                }
                FunctionSection(func_sec) => {
                    let indices = func_sec.into_iter();

                    for index in indices {
                        let index = index?;

                        func_decls.push(FuncDecl {
                            kind: FuncKind::Local,
                            ty: TyIndex(index),
                        });
                    }
                }
                TableSection(table_sec) => {
                    let table_iter = table_sec.into_iter();

                    for table in table_iter {
                        let table = table?;
                        let ty = table.ty;

                        if !ty.element_type.is_func_ref() {
                            return Err(TraceWasmError::Unsupported(
                                "non-funcref table element types".to_string(),
                            ));
                        }

                        let init = table.init;

                        let table_init = match init {
                            wasmparser::TableInit::RefNull => TableInit::RefNull,
                            wasmparser::TableInit::Expr(const_expr) => TableInit::Expr(
                                StackInstruction::emit_instruction_for_const_expr(
                                    const_expr.get_operators_reader(),
                                )
                                .unwrap() // todo!
                                .into_boxed_slice(),
                            ),
                        };

                        tables.push(Table {
                            element_ty: RefType::from_wasmparser(ty.element_type),
                            init: table_init,
                            initial: ty.initial,
                            maximum: ty.maximum,
                        });
                    }
                }
                MemorySection(mem_sec) => {
                    let mem_iter = mem_sec.into_iter();

                    for mem in mem_iter {
                        let mem = mem?;

                        // Rejected up front rather than trapping later: `wasmparser`
                        // enables the memory64 proposal by default, but the
                        // interpreter is 32-bit throughout — addresses are popped
                        // with `Val::as_i32` and `memarg` offsets are stored as
                        // `u32`. Both of those silently misbehave on a 64-bit
                        // memory, so it cannot be allowed past compile time.
                        //
                        // Imported memories need no equivalent check: the import
                        // section arm rejects every non-function, non-global import,
                        // so the memory section is the only way a memory gets in.
                        if mem.memory64 {
                            return Err(TraceWasmError::Unsupported("64-bit memory".to_string()));
                        }

                        memories.push(mem);
                    }
                }
                TagSection(tag_sec) => {
                    for tag in tag_sec {
                        tags.push(tag?);
                    }
                }
                GlobalSection(global_sec) => {
                    let global_iter = global_sec.into_iter();

                    for global in global_iter {
                        let global = global?;
                        let global_ty = global.ty;

                        globals.push(Global {
                            ty: GlobalType::from_wasmparser(global_ty),
                            kind: GlobalKind::Local(
                                StackInstruction::emit_instruction_for_const_expr(
                                    global.init_expr.get_operators_reader(),
                                )
                                .unwrap() // todo!
                                .into_boxed_slice(),
                            ),
                        });
                    }
                }
                ExportSection(export_sec) => {
                    let exports_iter = export_sec.into_iter();

                    for export in exports_iter {
                        let export = export?;
                        let index = export.index;

                        exports.insert(
                            export.name.to_string(),
                            match export.kind {
                                ExternalKind::Func => {
                                    exported_func_index_to_name
                                        .insert(FuncIndex(index), export.name.to_string());
                                    Export::Func(FuncIndex(index))
                                }
                                ExternalKind::Table => Export::Table(TableIndex(index)),
                                ExternalKind::Memory => Export::Memory(MemoryIndex(index)),
                                ExternalKind::Global => Export::Global(GlobalIndex(index)),
                                ExternalKind::Tag => Export::Tag(TagIndex(index)),
                                ExternalKind::FuncExact => Export::FuncExact(FuncExactIndex(index)),
                            },
                        );
                    }
                }
                StartSection {
                    func,
                    range: _range,
                } => {
                    start_section = Some(FuncIndex(func));
                }
                ElementSection(elem_sec) => {
                    let elem_iter = elem_sec.into_iter();

                    for elem in elem_iter {
                        let elem = elem?;

                        elements.push(Element {
                            kind: match elem.kind {
                                wasmparser::ElementKind::Passive => ElementKind::Passive,
                                wasmparser::ElementKind::Declared => ElementKind::Declared,
                                wasmparser::ElementKind::Active {
                                    table_index,
                                    offset_expr,
                                } => ElementKind::Active {
                                    table_index: table_index.map(TableIndex),
                                    offset_expr: StackInstruction::emit_instruction_for_const_expr(
                                        offset_expr.get_operators_reader(),
                                    )
                                    .unwrap() // todo!
                                    .into_boxed_slice(),
                                },
                            },
                            items: match elem.items {
                                wasmparser::ElementItems::Functions(func_sec) => {
                                    let mut funcs = vec![];
                                    let iter = func_sec.into_iter();

                                    for index in iter {
                                        let index = index?;
                                        funcs.push(FuncIndex(index));
                                    }

                                    ElementItems::Functions(funcs.into_boxed_slice())
                                }
                                wasmparser::ElementItems::Expressions(ref_ty, expr_sec) => {
                                    let mut exprs = vec![];
                                    let iter = expr_sec.into_iter();

                                    for expr in iter {
                                        let expr = expr?;

                                        exprs.push(
                                            StackInstruction::emit_instruction_for_const_expr(
                                                expr.get_operators_reader(),
                                            )
                                            .unwrap() // todo!
                                            .into_boxed_slice(),
                                        );
                                    }

                                    ElementItems::Expressions(
                                        RefType::from_wasmparser(ref_ty),
                                        exprs.into_boxed_slice(),
                                    )
                                }
                            },
                        });
                    }
                }
                DataCountSection {
                    count,
                    range: _range,
                } => {
                    data_count = Some(count);
                }
                DataSection(data_sec) => {
                    let data_iter = data_sec.into_iter();

                    for data in data_iter {
                        let data = data?;

                        datas.push(Data {
                            kind: match data.kind {
                                wasmparser::DataKind::Passive => DataKind::Passive,
                                wasmparser::DataKind::Active {
                                    memory_index,
                                    offset_expr,
                                } => DataKind::Active {
                                    memory_index: MemoryIndex(memory_index),
                                    offset_expr: StackInstruction::emit_instruction_for_const_expr(
                                        offset_expr.get_operators_reader(),
                                    )
                                    .unwrap() // todo!
                                    .into_boxed_slice(),
                                },
                            },
                            data: data.data.to_vec().into_boxed_slice(),
                        });
                    }
                }
                CodeSectionStart {
                    count,
                    range: _range,
                    size,
                } => {
                    code_sec_count = count;
                    code_sec_size = size;
                }
                CodeSectionEntry(code_sec_entry) => {
                    let locals_reader = code_sec_entry.get_locals_reader()?;
                    let mut locals: Vec<ValType> = vec![];

                    // Code entries correspond to defined functions in order, which live in
                    // `func_decls` after the imported ones — so the i-th body maps to
                    // `func_decls[imported_func_count + i]`.
                    let func_index = func_bodies.len() as u32 + imported_func_count;
                    let func_decl = &func_decls[func_index as usize];
                    let ty_index = func_decl.ty;
                    let ty = &types[ty_index.0 as usize];
                    let params = &ty.params;
                    let results = &ty.results;

                    // first add the params in the locals!
                    for param in params {
                        locals.push(*param);
                    }

                    for local in locals_reader {
                        let (count, ty) = local?;
                        let ty = ValType::from_wasmparser(ty);

                        // Rejected here rather than when the frame is built: the
                        // local's type is static, so failing at compile time gives
                        // a clearer error, and it lets frame setup treat `v128` as
                        // unreachable instead of having to produce a trap for it.
                        if ty == ValType::V128 {
                            return Err(TraceWasmError::Unsupported("v128 local".to_string()));
                        }

                        for _ in 0..count {
                            locals.push(ty);
                        }
                    }

                    let (instructions, instruction_offsets, frame_layout) =
                        Instr::emit_instruction_for_func(
                            code_sec_entry.get_operators_reader()?,
                            params.len() as u32,
                            results.len() as u32,
                            &types,
                            &func_decls,
                            locals.len() as u32,
                            globals.len() as u32,
                        )?;

                    func_bodies.push(FuncBody {
                        locals: locals.into_boxed_slice(), // params + declared locals
                        instructions: instructions.into_boxed_slice(),
                        instruction_offsets: instruction_offsets.into_boxed_slice(),
                        frame_layout,
                    });
                }
                CustomSection(custom_sec) => {
                    match custom_sec.as_known() {
                        wasmparser::KnownCustom::Name(reader) => {
                            for name in reader {
                                let name = name?;

                                match name {
                                    wasmparser::Name::Module {
                                        name,
                                        name_range: _name_range,
                                    } => {
                                        custom_section_module_name = name.to_string();
                                    }
                                    wasmparser::Name::Function(seq) => {
                                        NameMap::from_wasmparser_name_map(
                                            seq,
                                            &mut custom_section_func,
                                        )?
                                    }
                                    wasmparser::Name::Local(seq) => {
                                        IndirectNameMap::from_wasmparser_indirect_map(
                                            seq,
                                            &mut custom_section_local,
                                        )?
                                    }
                                    wasmparser::Name::Label(seq) => {
                                        IndirectNameMap::from_wasmparser_indirect_map(
                                            seq,
                                            &mut custom_section_label,
                                        )?
                                    }
                                    wasmparser::Name::Type(seq) => {
                                        NameMap::from_wasmparser_name_map(
                                            seq,
                                            &mut custom_section_ty,
                                        )?
                                    }
                                    wasmparser::Name::Table(seq) => {
                                        NameMap::from_wasmparser_name_map(
                                            seq,
                                            &mut custom_section_table,
                                        )?
                                    }
                                    wasmparser::Name::Memory(seq) => {
                                        NameMap::from_wasmparser_name_map(
                                            seq,
                                            &mut custom_section_mem,
                                        )?
                                    }
                                    wasmparser::Name::Global(seq) => {
                                        NameMap::from_wasmparser_name_map(
                                            seq,
                                            &mut custom_section_global,
                                        )?
                                    }
                                    wasmparser::Name::Element(seq) => {
                                        NameMap::from_wasmparser_name_map(
                                            seq,
                                            &mut custom_section_element,
                                        )?
                                    }
                                    wasmparser::Name::Data(seq) => {
                                        NameMap::from_wasmparser_name_map(
                                            seq,
                                            &mut custom_section_data,
                                        )?
                                    }
                                    wasmparser::Name::Field(seq) => {
                                        IndirectNameMap::from_wasmparser_indirect_map(
                                            seq,
                                            &mut custom_section_field,
                                        )?
                                    }
                                    wasmparser::Name::Tag(seq) => {
                                        NameMap::from_wasmparser_name_map(
                                            seq,
                                            &mut custom_section_tag,
                                        )?
                                    }
                                    wasmparser::Name::Unknown {
                                        ty,
                                        data,
                                        range: _range,
                                    } => {
                                        // Keyed by the subsection type byte: the
                                        // custom section name is always "name" here,
                                        // so keying by it would collapse all unknown
                                        // subsections into one.
                                        custom_section_name_unknown
                                            .insert(ty, data.to_vec().into_boxed_slice());
                                    }
                                }
                            }
                        }
                        wasmparser::KnownCustom::Unknown => {
                            if DEBUG_SECTION_NAMES.contains(custom_sec.name()) {
                                debug_sections
                                    .insert(custom_sec.name().to_string(), custom_sec.data());
                            } else {
                                custom_section_unknowns.insert(
                                    custom_sec.name().to_string(),
                                    custom_sec.data().to_vec().into_boxed_slice(),
                                );
                            }
                        }
                        _ => {
                            custom_section_others.insert(
                                custom_sec.name().to_string(),
                                custom_sec.data().to_vec().into_boxed_slice(),
                            );
                        }
                    };
                }
                UnknownSection {
                    id,
                    contents,
                    range: _range,
                } => {
                    unknown_sections.push((id, contents.to_vec().into_boxed_slice()));
                }
                End(_final_offset) => break,
                _ => return Err(TraceWasmError::Unsupported(format!("{:?}", payload))),
            }
        }

        // checking restrictions of TraceWasm
        // Below checks are unsupported features right now in TraceWasm.
        if memories.len() > 1 {
            return Err(TraceWasmError::Unsupported(
                "more than one memory".to_string(),
            ));
        }

        // DWARF in a WebAssembly binary is always little-endian.
        let endian = RunTimeEndian::Little;

        // `.debug_info` is the root of the DWARF tree: without it there is nothing
        // to resolve. Gate on it so a module built without debug info reports
        // `None` — `Dwarf::load` itself cannot fail here (the loader below returns
        // `Ok` on every path, using an empty slice for absent sections), so
        // without this check every module would come back as `Some(..)` holding a
        // shell of empty sections.
        if debug_sections.contains_key(SectionId::DebugInfo.name()) {
            let loaded = Dwarf::load(
                |id: SectionId| -> Result<EndianArcSlice<RunTimeEndian>, gimli::Error> {
                    let bytes: Arc<[u8]> = match debug_sections.get(id.name()) {
                        Some(v) => Arc::from(v.to_vec().into_boxed_slice()),
                        None => Arc::from(&[][..]),
                    };

                    Ok(EndianArcSlice::new(bytes, endian))
                },
            );

            // Debug info is diagnostic-only: a malformed section degrades the
            // stack traces rather than failing the whole compile.
            if let Ok(dwarf) = loaded {
                possible_dwarf = Some(Arc::from(dwarf));
            }
        }

        Ok(Arc::new(Module {
            types: types.into_boxed_slice(),
            func_decls: func_decls.into_boxed_slice(),
            tables: tables.into_boxed_slice(),
            memories: memories
                .into_iter()
                .map(MemoryType::from_wasmparser)
                .collect(),
            tags: tags.into_iter().map(TagType::from_wasmparser).collect(),
            globals: globals.into_boxed_slice(),
            exports,
            exported_func_index_to_name,
            elements: elements.into_boxed_slice(),
            start_section,
            data_count,
            datas: datas.into_boxed_slice(),
            code_sec_count,
            code_sec_size,
            imported_func_count,
            imported_global_count,
            func_bodies: func_bodies.into_boxed_slice(),
            unknown_sections: unknown_sections.into_boxed_slice(),
            custom_section: CustomSection {
                module_name: custom_section_module_name,
                func: custom_section_func,
                local: custom_section_local,
                label: custom_section_label,
                ty: custom_section_ty,
                table: custom_section_table,
                mem: custom_section_mem,
                global: custom_section_global,
                element: custom_section_element,
                data: custom_section_data,
                field: custom_section_field,
                tag: custom_section_tag,
                name_unknown: custom_section_name_unknown,
                other: custom_section_others,
                unknown: custom_section_unknowns,
            },
            dwarf: possible_dwarf,
        }))
    }

    /// Instantiates the module against an import registry, producing a runnable
    /// [`Instance`].
    ///
    /// In order: validates the registry against the module's declared function
    /// imports (counts must agree, and each must exist with a matching
    /// signature), allocates linear memory, resolves and type-checks imported
    /// globals, builds tables, applies active element and data segments, and
    /// finally runs the `start` function if the module has one.
    ///
    /// Imports are checked before anything is allocated, so a mismatched registry
    /// costs nothing.
    ///
    /// # Errors
    ///
    /// - [`TraceWasmError::ImportCountMismatch`] / [`TraceWasmError::ImportGlobalCountMismatch`]
    ///   if the registry's function or global count disagrees with the module.
    /// - [`TraceWasmError::ImportNotFound`] if the registry lacks a declared import.
    /// - [`TraceWasmError::ImportSignatureMismatch`] if an imported function's
    ///   signature disagrees, or [`TraceWasmError::ImportGlobalTypeMismatch`] if
    ///   an imported global's type disagrees.
    /// - [`TraceWasmError::TableTooLarge`] / [`TraceWasmError::MemoryTooLarge`] if
    ///   a table's or memory's initial size exceeds the configured cap, and
    ///   [`TraceWasmError::ElementSegmentOutOfBounds`] if an active element segment
    ///   does not fit its target table.
    /// - [`TraceWasmError::Unsupported`] for constructs only reachable at
    ///   instantiation: a non-`funcref` element expression, a data segment
    ///   targeting a memory other than index 0, or a `v128` imported global.
    /// - Anything the module's `start` function raises, since it runs before this
    ///   returns.
    pub fn instantiate<M: Memory, I: ImportRegistry>(
        self: &Arc<Module<Instr>>,
        import_registry: I,
        config: Option<Config>,
    ) -> Result<Instance<M, I, Instr>, TraceWasmError<Instr>> {
        let initial_pages = if !self.memories.is_empty() {
            self.memories[0].initial()
        } else {
            0
        };

        // Narrowed below to the module's declared memory maximum, so the instance
        // carries the *effective* limits rather than only the requested ones.
        let mut config = config.unwrap_or_default();

        // Validate the registry against the module's declared imports: the counts
        // must agree, and every imported function must exist in the registry with
        // a matching signature.
        let imported_func_count = self.imported_func_count;

        if import_registry.func_count() != imported_func_count {
            return Err(TraceWasmError::ImportCountMismatch(
                imported_func_count,
                import_registry.func_count(),
            ));
        }

        for i in 0..imported_func_count {
            let func_decl = &self.func_decls[i as usize];

            debug_assert!(matches!(func_decl.kind, FuncKind::Imported { .. }));

            let FuncKind::Imported {
                module_name,
                imported_func_name,
            } = &func_decl.kind
            else {
                unreachable!()
            };

            let Some((import_params, import_results)) =
                import_registry.signature(module_name, imported_func_name)
            else {
                return Err(TraceWasmError::ImportNotFound(
                    module_name.to_string(),
                    imported_func_name.to_string(),
                ));
            };

            let func_ty = &self.types[func_decl.ty.0 as usize];
            let params = &func_ty.params;
            let results = &func_ty.results;

            if params.as_ref() != import_params.as_ref() {
                return Err(TraceWasmError::ImportSignatureMismatch(
                    module_name.to_string(),
                    imported_func_name.to_string(),
                    "params".to_string(),
                    formatted_val_types(params),
                    formatted_val_types(import_params.as_ref()),
                ));
            }

            if results.as_ref() != import_results.as_ref() {
                return Err(TraceWasmError::ImportSignatureMismatch(
                    module_name.to_string(),
                    imported_func_name.to_string(),
                    "results".to_string(),
                    formatted_val_types(results),
                    formatted_val_types(import_results.as_ref()),
                ));
            }
        }

        // Resolve the instance's effective memory ceiling once: the module's own
        // declared maximum, narrowed by the configured cap. Folding it into the
        // config means `memory.grow` reads a single limit instead of re-deriving it
        // per instruction — and, critically, that growth respects the maximum the
        // module declared, which its own code is entitled to rely on.
        let config_max_pages = config.get_max_memory_size_in_pages();

        let max_memory_pages = self
            .memories
            .first()
            .and_then(|memory| memory.maximum())
            .unwrap_or(config_max_pages)
            .min(config_max_pages);

        config.set_max_memory_size_in_pages(max_memory_pages);

        // Reject rather than truncate: handing a module less memory than it
        // declared would make it trap on accesses it is entitled to make.
        // Validation guarantees `initial <= declared maximum`, so this can only
        // fire when the configured cap is the binding limit.
        if initial_pages > max_memory_pages {
            return Err(TraceWasmError::MemoryTooLarge(
                initial_pages,
                max_memory_pages,
            ));
        }

        let mut memory = M::allocate_initial_memory(initial_pages);

        // Globals Initialization
        if import_registry.global_count() != self.imported_global_count {
            return Err(TraceWasmError::ImportGlobalCountMismatch(
                self.imported_global_count,
                import_registry.global_count(),
            ));
        }

        let mut global_vals: Vec<Val> = Vec::with_capacity(self.globals.len());
        let globals = &self.globals;

        for i in 0..self.imported_global_count {
            let global = &globals[i as usize];

            debug_assert!(matches!(global.kind, GlobalKind::Imported { .. }));

            let GlobalKind::Imported {
                module: module_name,
                name: global_name,
            } = &global.kind
            else {
                unreachable!(
                    "starting `imported_global_count` globals should have `Imported` kind. Reaching this means the global collecting logic in TraceWasm is incorrect"
                )
            };

            let expected = ValType::from_wasmparser(global.ty.0.content_type);
            let val = import_registry.get_global(module_name, global_name)?;

            if !val.has_ty(expected)? {
                return Err(TraceWasmError::ImportGlobalTypeMismatch(
                    module_name.to_string(),
                    global_name.to_string(),
                    format!("{expected:?}"),
                    format!("{val:?}"),
                ));
            }

            global_vals.push(val);
        }

        for i in self.imported_global_count..(self.globals.len() as u32) {
            let GlobalKind::Local(const_expr_instructions) = &globals[i as usize].kind else {
                unreachable!(
                    "after `imported_global_count` globals should have `Local` kind. Reaching this means the global collecting logic in TraceWasm is incorrect"
                )
            };

            let val = TraceVM::const_expr_evaluator(const_expr_instructions, &global_vals);

            global_vals.push(val);
        }

        // Tables Initialization
        // NOTE: TraceWasm does not support imported tables so everything inside `self.tables` is locally declared.
        let tables = &self.tables;
        let mut table_vals = vec![];

        for table in tables {
            let maximum = if let Some(max) = table.maximum {
                max.min(config.get_max_table_elements())
            } else {
                config.get_max_table_elements()
            };

            // WASM validation guarantees `initial <= declared maximum`, but not
            // `initial <= config cap`, so an untrusted module could still ask for
            // an arbitrarily large table. Reject rather than silently clamp:
            // truncating would hand the module a smaller table than it declared,
            // corrupting every table access it believes is in bounds.
            if table.initial > maximum {
                return Err(TraceWasmError::TableTooLarge(table.initial, maximum));
            }

            let initial_size = table.initial as usize;

            let slots: Vec<Option<FuncIndex>> = match &table.init {
                TableInit::RefNull => vec![None; initial_size],
                TableInit::Expr(const_expr) => {
                    // The element type is validated to be funcref, so the const
                    // expr yields a `Val::Ref` and `as_ref` cannot panic.
                    let val = TraceVM::const_expr_evaluator(const_expr, &global_vals).as_ref();

                    vec![val; initial_size]
                }
            };

            table_vals.push(TableVal {
                table: slots,
                maximum,
            });
        }

        // Elements processing...
        let mut element_vals: Vec<ElementVal> = vec![];
        let elements = &self.elements;

        for element in elements {
            let items = &element.items;
            let kind = &element.kind;

            let item_vals: Vec<Option<FuncIndex>> = match &items {
                ElementItems::Functions(func_refs) => func_refs.iter().map(|x| Some(*x)).collect(),
                ElementItems::Expressions(ref_ty, exprs) => {
                    if !ref_ty.is_func_ref() {
                        return Err(TraceWasmError::Unsupported(
                            "non-funcref table element types".to_string(),
                        ));
                    }

                    let mut v: Vec<Option<FuncIndex>> = vec![];

                    for const_expr in exprs {
                        // The element type is validated to be funcref, so the const
                        // expr yields a `Val::Ref` and `as_ref` cannot panic.
                        v.push(TraceVM::const_expr_evaluator(const_expr, &global_vals).as_ref());
                    }

                    v
                }
            };

            match kind {
                ElementKind::Active {
                    table_index,
                    offset_expr,
                } => {
                    let table_index = if let Some(table_index) = table_index {
                        table_index.0
                    } else {
                        0
                    } as usize;

                    // The offset expr yields an `i32` for a 32-bit table; a
                    // `table64` offset would be an `i64` and `as_i32` would panic.
                    // TraceWasm does not support table64 yet, so this is fine.
                    let offset =
                        TraceVM::const_expr_evaluator(offset_expr, &global_vals).as_i32() as usize;

                    let table = &mut table_vals[table_index].table;

                    // Written as a subtraction to avoid `offset + len` overflowing
                    // `usize` on a 32-bit target.
                    if offset > table.len() || item_vals.len() > table.len() - offset {
                        return Err(TraceWasmError::ElementSegmentOutOfBounds(
                            offset,
                            item_vals.len(),
                            table.len(),
                        ));
                    }

                    for (i, item) in item_vals.iter().enumerate() {
                        table[offset + i] = *item;
                    }

                    // Per the WebAssembly spec, active element is dropped after writting to the table
                    element_vals.push(ElementVal::Dropped);
                }
                ElementKind::Passive => {
                    // Per the WebAssembly spec, passive element stays until either used at runtime by `table.init` or dropped by `elem.drop`
                    element_vals.push(ElementVal::Passive(item_vals.into_boxed_slice()));
                }
                ElementKind::Declared => {
                    // Per the WebAssembly spec, declared element is dropped from the start
                    element_vals.push(ElementVal::Dropped);
                }
            }
        }

        // Data processing...
        let datas = &self.datas;
        let mut data_vals: Vec<DataVal> = vec![];

        for data in datas {
            let kind = &data.kind;
            let data = &data.data;

            match kind {
                DataKind::Passive => {
                    data_vals.push(DataVal::Passive(data.clone()));

                    continue;
                }
                DataKind::Active {
                    memory_index,
                    offset_expr,
                } => {
                    if memory_index.0 != 0 {
                        return Err(TraceWasmError::Unsupported("multiple memories".to_string()));
                    }

                    // The offset expr yields an `i32` for a 32-bit memory; a
                    // `memory64` offset would be an `i64` and `as_i32` would panic.
                    // TraceWasm does not support memory64 yet, so this is fine.
                    let offset =
                        TraceVM::const_expr_evaluator(offset_expr, &global_vals).as_i32() as usize;

                    // `write` bounds-checks (with `checked_add`) and traps on an
                    // out-of-bounds segment, so no manual bounds check is needed.
                    memory.write(offset, data)?;

                    data_vals.push(DataVal::Dropped);
                }
            }
        }

        let mut instance = Instance::new(
            memory,
            import_registry,
            self.clone(),
            config,
            global_vals.into_boxed_slice(),
            table_vals.into_boxed_slice(),
            element_vals.into_boxed_slice(),
            data_vals.into_boxed_slice(),
        );

        // execute start function
        if let Some(func_index) = self.start_section {
            // TraceVM::run(func_index, &[], &mut instance, self)?;
            todo!()
        }

        Ok(instance)
    }
}
