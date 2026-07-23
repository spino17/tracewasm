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
        traits::{ImportRegistry, Params, Results},
    },
    instruction::Instruction,
    memory::Memory,
    utils::formatted_val_types,
};
use core::fmt::{self, Debug};
use rustc_hash::FxHashMap;
use std::{hash::Hash, sync::Arc};
use wasmparser::{Encoding, ExternalKind, Parser, Payload::*, TypeRef, Validator};

/// Bytes of linear memory allocated for a fresh instance (one 64 KiB wasm page).
///
/// TODO: derive this from the module's declared memory limits instead of a fixed
/// default.
pub const WASM_MEMORY_INITIAL_ALLOCATION_SIZE: usize = 64 * 1024; // 64 KiB (one wasm page)

/// The type of a WebAssembly global: its value type plus mutability.
///
/// An owned wrapper that keeps the underlying `wasmparser` representation
/// private so it does not leak into this crate's public API.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct GlobalType(wasmparser::GlobalType);

impl GlobalType {
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

/// The type of a WebAssembly table: its element reference type and size limits.
///
/// An owned wrapper that keeps the underlying `wasmparser` representation
/// private so it does not leak into this crate's public API.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TableType(wasmparser::TableType);

impl TableType {
    pub(crate) fn from_wasmparser(value: wasmparser::TableType) -> Self {
        TableType(value)
    }

    /// The reference type of the table's elements.
    pub fn element_type(&self) -> RefType {
        RefType::from_wasmparser(self.0.element_type)
    }

    /// Whether this is a 64-bit table (indexed by `i64`); `false` means 32-bit
    /// (memory64 proposal).
    pub fn is_64(&self) -> bool {
        self.0.table64
    }

    /// Initial size, in elements.
    pub fn initial(&self) -> u64 {
        self.0.initial
    }

    /// Optional maximum size, in elements.
    pub fn maximum(&self) -> Option<u64> {
        self.0.maximum
    }
}

/// The type of a WebAssembly tag (exception-handling proposal).
///
/// An owned wrapper that keeps the underlying `wasmparser` representation
/// private so it does not leak into this crate's public API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TagType(wasmparser::TagType);

impl TagType {
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

/// A WebAssembly reference type (e.g. `funcref`, `externref`).
///
/// An owned wrapper that keeps the underlying `wasmparser` representation
/// private so it does not leak into this crate's public API.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RefType(wasmparser::RefType);

impl RefType {
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
pub struct FuncIndex(pub u32);

impl From<u32> for FuncIndex {
    fn from(value: u32) -> Self {
        FuncIndex(value)
    }
}

impl EntityIndex for FuncIndex {}

/// Index used by the `ref.func` "exact" reference form (function-references
/// proposal); addresses the function index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncExactIndex(pub u32);

impl From<u32> for FuncExactIndex {
    fn from(value: u32) -> Self {
        FuncExactIndex(value)
    }
}

impl EntityIndex for FuncExactIndex {}

/// Index into the type section ([`Module::types`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyIndex(pub u32);

impl From<u32> for TyIndex {
    fn from(value: u32) -> Self {
        TyIndex(value)
    }
}

impl EntityIndex for TyIndex {}

/// Index into the global index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalIndex(pub u32);

impl From<u32> for GlobalIndex {
    fn from(value: u32) -> Self {
        GlobalIndex(value)
    }
}

impl EntityIndex for GlobalIndex {}

/// Index into the table index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableIndex(pub u32);

impl From<u32> for TableIndex {
    fn from(value: u32) -> Self {
        TableIndex(value)
    }
}

impl EntityIndex for TableIndex {}

/// Index into the memory index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryIndex(pub u32);

impl From<u32> for MemoryIndex {
    fn from(value: u32) -> Self {
        MemoryIndex(value)
    }
}

impl EntityIndex for MemoryIndex {}

/// Index into the tag index space (exception-handling proposal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TagIndex(pub u32);

impl From<u32> for TagIndex {
    fn from(value: u32) -> Self {
        TagIndex(value)
    }
}

impl EntityIndex for TagIndex {}

/// Index into the element segment space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElementIndex(pub u32);

impl From<u32> for ElementIndex {
    fn from(value: u32) -> Self {
        ElementIndex(value)
    }
}

impl EntityIndex for ElementIndex {}

/// Index into the data segment space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataIndex(pub u32);

impl From<u32> for DataIndex {
    fn from(value: u32) -> Self {
        DataIndex(value)
    }
}

impl EntityIndex for DataIndex {}

/// Index of a local within a function (params first, then declared locals).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalIndex(pub u32);

impl From<u32> for LocalIndex {
    fn from(value: u32) -> Self {
        LocalIndex(value)
    }
}

impl EntityIndex for LocalIndex {}

/// Index of a field within a struct type (GC proposal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldIndex(pub u32);

impl From<u32> for FieldIndex {
    fn from(value: u32) -> Self {
        FieldIndex(value)
    }
}

impl EntityIndex for FieldIndex {}

/// A fully parsed module: every section decoded into owned data.
///
/// Field order roughly follows the binary section order. Index spaces that can
/// be imported (functions here) fold imports and definitions into a single
/// list; see the per-field docs.
pub struct Module {
    /// The type section. Only function types are kept — GC composite types
    /// (struct/array) are rejected during parsing.
    pub types: Box<[FuncType]>,
    /// The function index space: imported functions first, then locally-defined
    /// ones. The split point is [`Self::imported_func_count`].
    pub func_decls: Box<[FuncDecl]>,
    pub tables: Box<[Table]>,
    pub memories: Box<[MemoryType]>,
    pub tags: Box<[TagType]>,
    pub globals: Box<[Global]>,
    pub exports: FxHashMap<String, Export>,
    /// The `start` function, run at instantiation, if the module declares one.
    pub start_section: Option<FuncIndex>,
    pub elements: Box<[Element]>,
    /// The declared data-segment count from the data-count section, if present
    /// (required by the bulk-memory proposal for validating `data.drop`, etc.).
    pub data_count: Option<u32>,
    pub datas: Box<[Data]>,
    /// Declared entry count of the code section (should match `func_bodies.len()`).
    pub code_sec_count: u32,
    /// Byte size of the code section as declared in its header.
    pub code_sec_size: u32,
    /// Number of imported functions; the boundary between imports and
    /// definitions in [`Self::func_decls`], and the offset mapping the `i`-th
    /// `func_bodies` entry to `func_decls[imported_func_count + i]`.
    pub imported_func_count: u32,
    /// Lowered bodies of the locally-defined functions, in definition order.
    pub func_bodies: Box<[FuncBody]>,
    /// Sections with an unrecognized id, preserved verbatim as `(id, contents)`.
    pub unknown_sections: Box<[(u8, Box<[u8]>)]>, // (id, content)
    pub custom_sections: Box<[CustomSection]>,
}

impl Module {
    pub fn exports(&self) -> &FxHashMap<String, Export> {
        &self.exports
    }

    pub fn export(&self, name: &str) -> Option<Export> {
        self.exports.get(name).cloned()
    }
}

/// A locally-defined function's locals and lowered instruction list.
pub struct FuncBody {
    /// All locals addressable in the body: the function's params first, then the
    /// declared locals, expanded from the run-length-encoded body header.
    pub locals: Box<[ValType]>,
    /// The body lowered by [`crate::instruction`] (control flow resolved to
    /// absolute indices).
    pub instructions: Box<[Instruction]>,
}

/// One entry of the custom `name` section.
pub enum Name {
    Module(String),
    Function(NameMap<FuncIndex>),
    Local(IndirectNameMap<FuncIndex, LocalIndex>),
    Label(IndirectNameMap<FuncIndex, LocalIndex>),
    Type(NameMap<TyIndex>),
    Table(NameMap<TableIndex>),
    Memory(NameMap<MemoryIndex>),
    Global(NameMap<GlobalIndex>),
    Element(NameMap<ElementIndex>),
    Data(NameMap<DataIndex>),
    Field(IndirectNameMap<TyIndex, FieldIndex>),
    Tag(NameMap<TagIndex>),
    Unknown { ty: u8, data: Box<[u8]> },
}

/// The decoded payload of a custom section.
pub enum CustomSectionKind {
    /// The well-known `name` section, decoded into [`Name`] entries.
    Name(Box<[Name]>),
    /// A custom section `wasmparser` recognizes but that we keep as raw bytes.
    Others(Box<[u8]>),
    /// A custom section `wasmparser` does not recognize, kept as raw bytes.
    Unknown(Box<[u8]>),
}

/// A custom section: its name plus decoded/raw contents.
pub struct CustomSection {
    pub name: String,
    pub kind: CustomSectionKind,
}

/// A data segment.
pub struct Data {
    pub kind: DataKind,
    pub data: Box<[u8]>,
}

/// Whether a data segment is passive or actively initializes memory.
pub enum DataKind {
    /// Copied into memory only via `memory.init`.
    Passive,
    /// Copied into `memory_index` at instantiation, at the offset computed by
    /// `offset_expr`.
    Active {
        memory_index: MemoryIndex,
        offset_expr: Box<[Instruction]>,
    },
}

/// Whether an element segment is passive, active, or declared.
pub enum ElementKind {
    /// Usable only via `table.init`.
    Passive,
    /// Copied into `table_index` at instantiation at `offset_expr` (a `None`
    /// table index means table 0).
    Active {
        table_index: Option<TableIndex>,
        offset_expr: Box<[Instruction]>,
    },
    /// Forward-declares references (for `ref.func`); contributes no table data.
    Declared,
}

/// The payload of an element segment.
pub enum ElementItems {
    /// A list of function indices (the legacy form).
    Functions(Box<[FuncIndex]>),
    /// Constant expressions of the given reference type (each yields one element).
    Expressions(RefType, Box<[Box<[Instruction]>]>),
}

/// An element segment.
pub struct Element {
    pub kind: ElementKind,
    pub items: ElementItems,
}

/// How a table's slots are initialized.
pub enum TableInit {
    /// Every slot starts as a null reference.
    RefNull,
    /// Every slot is initialized from this constant expression.
    Expr(Box<[Instruction]>),
}

/// A table definition: its type plus initialization.
pub struct Table {
    pub ty: TableType,
    pub init: TableInit,
}

/// A named export and the entity it exposes.
#[derive(Debug, Clone, Copy)]
pub enum Export {
    Func(FuncIndex),
    Table(TableIndex),
    Memory(MemoryIndex),
    Global(GlobalIndex),
    Tag(TagIndex),
    FuncExact(FuncExactIndex),
}

impl Export {
    pub fn to_func(&self) -> Result<FuncIndex, TraceWasmError> {
        let Export::Func(func_index) = self else {
            todo!() // TODO - raise error
        };

        Ok(*func_index)
    }

    pub fn to_typed_func<P: Params, R: Results>(&self) -> Result<TypedFunc<P, R>, TraceWasmError> {
        Ok(TypedFunc::new(self.to_func()?))
    }
}

/// A global definition: its type plus the constant expression for its initial
/// value.
pub struct Global {
    pub ty: GlobalType,
    pub val: Box<[Instruction]>,
}

/// A function type: parameter and result value types.
pub struct FuncType {
    pub params: Box<[ValType]>,
    pub results: Box<[ValType]>,
}

/// Whether a function is defined locally or imported.
pub enum FuncKind {
    Local,
    Imported {
        module_name: String,
        imported_func_name: String,
    },
}

/// A function declaration: its origin and its type-section index.
pub struct FuncDecl {
    pub kind: FuncKind,
    /// Index into [`Module::types`] giving this function's signature.
    pub ty: TyIndex,
}

/// A `name`-section map from a typed entity index to its name.
pub struct NameMap<T: PartialEq + Eq + Hash + EntityIndex>(pub FxHashMap<T, String>);

impl<T: PartialEq + Eq + Hash + EntityIndex> NameMap<T> {
    /// Converts a `wasmparser` name map into an owned [`NameMap`], turning each
    /// raw `u32` into the typed index `T`.
    pub(crate) fn from_wasmparser_name_map(
        map: wasmparser::NameMap<'_>,
    ) -> Result<Self, TraceWasmError> {
        let mut new_map: FxHashMap<T, String> = FxHashMap::default();

        for name in map {
            let name = name?;
            let index = T::from(name.index);
            let name = name.name.to_string();

            new_map.insert(index, name);
        }

        Ok(NameMap(new_map))
    }
}

/// A two-level `name`-section map: an outer index to a nested [`NameMap`].
///
/// For example `IndirectNameMap<FuncIndex, LocalIndex>` maps each function to
/// the names of its locals.
pub struct IndirectNameMap<
    T: PartialEq + Eq + Hash + EntityIndex,
    V: PartialEq + Eq + Hash + EntityIndex,
>(pub FxHashMap<T, NameMap<V>>);

impl<T: PartialEq + Eq + Hash + EntityIndex, V: PartialEq + Eq + Hash + EntityIndex>
    IndirectNameMap<T, V>
{
    /// Converts a `wasmparser` indirect name map into the owned two-level form.
    pub(crate) fn from_wasmparser_indirect_map(
        map: wasmparser::IndirectNameMap<'_>,
    ) -> Result<Self, TraceWasmError> {
        let mut new_indirect_map: FxHashMap<T, NameMap<V>> = FxHashMap::default();

        for entry in map {
            let entry = entry?;
            let outer_index = T::from(entry.index);
            let names: NameMap<V> = NameMap::from_wasmparser_name_map(entry.names)?;

            new_indirect_map.insert(outer_index, names);
        }

        Ok(IndirectNameMap(new_indirect_map))
    }
}

impl Module {
    /// Validates `buf` as a core WebAssembly module and builds an owned
    /// [`Module`] from it, wrapped in an `Arc` so it can be shared across
    /// instances.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::Parsing`] if validation or decoding fails, and
    /// [`TraceWasmError::Unsupported`] for valid modules using features TraceWasm
    /// does not model (components, GC types, non-function imports, or any
    /// operator the lowering pass rejects).
    pub fn compile(buf: &[u8]) -> Result<Arc<Module>, TraceWasmError> {
        // The parser alone only checks structure; validate semantics (section
        // order, index bounds, types) up front so the AST is built from wasm
        // that is known to be well-formed.
        Validator::new().validate_all(buf)?;

        let mut types = vec![];
        let mut func_decls = vec![];
        let mut tables = vec![];
        let mut memories = vec![];
        let mut globals = vec![];
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
        let mut custom_sections = vec![];

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

                        let ty_index = if let TypeRef::Func(ty) = import.ty {
                            TyIndex(ty)
                        } else {
                            return Err(TraceWasmError::Unsupported(
                                "non-function imports".to_string(),
                            ));
                        };

                        func_decls.push(FuncDecl {
                            kind: FuncKind::Imported {
                                module_name,
                                imported_func_name,
                            },
                            ty: ty_index,
                        });
                    }

                    // At this point `func_decls` holds only imports (the function section comes
                    // later), and any non-function import already returned above — so this count is
                    // exactly the number of imported functions and marks the imports/definitions split.
                    imported_func_count = func_decls.len() as u32;
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
                        let init = table.init;

                        let table_init = match init {
                            wasmparser::TableInit::RefNull => TableInit::RefNull,
                            wasmparser::TableInit::Expr(const_expr) => TableInit::Expr(
                                Instruction::emit_instruction_from_operator_reader(
                                    const_expr.get_operators_reader(),
                                    None,
                                    &types,
                                    &func_decls,
                                )?
                                .into_boxed_slice(),
                            ),
                        };

                        tables.push(Table {
                            ty: TableType::from_wasmparser(ty),
                            init: table_init,
                        });
                    }
                }
                MemorySection(mem_sec) => {
                    let mem_iter = mem_sec.into_iter();

                    for mem in mem_iter {
                        let mem = mem?;

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
                            val: Instruction::emit_instruction_from_operator_reader(
                                global.init_expr.get_operators_reader(),
                                None,
                                &types,
                                &func_decls,
                            )?
                            .into_boxed_slice(),
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
                                ExternalKind::Func => Export::Func(FuncIndex(index)),
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
                                    offset_expr:
                                        Instruction::emit_instruction_from_operator_reader(
                                            offset_expr.get_operators_reader(),
                                            None,
                                            &types,
                                            &func_decls,
                                        )?
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
                                            Instruction::emit_instruction_from_operator_reader(
                                                expr.get_operators_reader(),
                                                None,
                                                &types,
                                                &func_decls,
                                            )?
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
                                    offset_expr:
                                        Instruction::emit_instruction_from_operator_reader(
                                            offset_expr.get_operators_reader(),
                                            None,
                                            &types,
                                            &func_decls,
                                        )?
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

                        for _ in 0..count {
                            locals.push(ValType::from_wasmparser(ty));
                        }
                    }

                    let instructions = Instruction::emit_instruction_from_operator_reader(
                        code_sec_entry.get_operators_reader()?,
                        Some((params.len() as u32, results.len() as u32)), // arity of this function
                        &types,
                        &func_decls,
                    )?
                    .into_boxed_slice();

                    func_bodies.push(FuncBody {
                        locals: locals.into_boxed_slice(), // params + declared locals
                        instructions,
                    });
                }
                CustomSection(custom_sec) => {
                    let name = custom_sec.name().to_string();

                    let kind = match custom_sec.as_known() {
                        wasmparser::KnownCustom::Name(reader) => {
                            let mut names: Vec<Name> = vec![];

                            for name in reader {
                                let name = name?;

                                let name = match name {
                                    wasmparser::Name::Module {
                                        name,
                                        name_range: _name_range,
                                    } => Name::Module(name.to_string()),
                                    wasmparser::Name::Function(seq) => {
                                        Name::Function(NameMap::from_wasmparser_name_map(seq)?)
                                    }
                                    wasmparser::Name::Local(seq) => Name::Local(
                                        IndirectNameMap::from_wasmparser_indirect_map(seq)?,
                                    ),
                                    wasmparser::Name::Label(seq) => Name::Label(
                                        IndirectNameMap::from_wasmparser_indirect_map(seq)?,
                                    ),
                                    wasmparser::Name::Type(seq) => {
                                        Name::Type(NameMap::from_wasmparser_name_map(seq)?)
                                    }
                                    wasmparser::Name::Table(seq) => {
                                        Name::Table(NameMap::from_wasmparser_name_map(seq)?)
                                    }
                                    wasmparser::Name::Memory(seq) => {
                                        Name::Memory(NameMap::from_wasmparser_name_map(seq)?)
                                    }
                                    wasmparser::Name::Global(seq) => {
                                        Name::Global(NameMap::from_wasmparser_name_map(seq)?)
                                    }
                                    wasmparser::Name::Element(seq) => {
                                        Name::Element(NameMap::from_wasmparser_name_map(seq)?)
                                    }
                                    wasmparser::Name::Data(seq) => {
                                        Name::Data(NameMap::from_wasmparser_name_map(seq)?)
                                    }
                                    wasmparser::Name::Field(seq) => Name::Field(
                                        IndirectNameMap::from_wasmparser_indirect_map(seq)?,
                                    ),
                                    wasmparser::Name::Tag(seq) => {
                                        Name::Tag(NameMap::from_wasmparser_name_map(seq)?)
                                    }
                                    wasmparser::Name::Unknown {
                                        ty,
                                        data,
                                        range: _range,
                                    } => Name::Unknown {
                                        ty,
                                        data: data.to_vec().into_boxed_slice(),
                                    },
                                };

                                names.push(name);
                            }

                            CustomSectionKind::Name(names.into_boxed_slice())
                        }
                        wasmparser::KnownCustom::Unknown => CustomSectionKind::Unknown(
                            custom_sec.data().to_vec().into_boxed_slice(),
                        ),
                        _ => {
                            CustomSectionKind::Others(custom_sec.data().to_vec().into_boxed_slice())
                        }
                    };

                    custom_sections.push(CustomSection { name, kind });
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
            elements: elements.into_boxed_slice(),
            start_section,
            data_count,
            datas: datas.into_boxed_slice(),
            code_sec_count,
            code_sec_size,
            imported_func_count,
            func_bodies: func_bodies.into_boxed_slice(),
            unknown_sections: unknown_sections.into_boxed_slice(),
            custom_sections: custom_sections.into_boxed_slice(),
        }))
    }

    /// Instantiates the module against an import registry, producing a runnable
    /// [`Instance`].
    ///
    /// Allocates the instance's linear memory and validates the registry against
    /// the module's declared imports: the function counts must agree, and every
    /// imported function must exist in the registry with a matching signature.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::ImportCountMismatch`] if the counts differ,
    /// [`TraceWasmError::ImportedFunctionNotFound`] if the registry lacks a
    /// declared import, and [`TraceWasmError::ImportSignatureMismatch`] if an
    /// import's signature disagrees with the module.
    pub fn instantiate<M: Memory, I: ImportRegistry>(
        self: Arc<Module>,
        import_registry: I,
    ) -> Result<Instance<M, I>, TraceWasmError> {
        // TODO: take this from the module itself if specified! and make it tunable
        let memory = M::allocate_initial_memory(WASM_MEMORY_INITIAL_ALLOCATION_SIZE);

        // Validate the registry against the module's declared imports: the counts
        // must agree, and every imported function must exist in the registry with
        // a matching signature.
        let imported_func_count = self.imported_func_count;

        if import_registry.size() != imported_func_count {
            return Err(TraceWasmError::ImportCountMismatch(
                imported_func_count,
                import_registry.size(),
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
                return Err(TraceWasmError::ImportedFunctionNotFound(
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
                    formatted_val_types(&import_params),
                ));
            }

            if results.as_ref() != import_results.as_ref() {
                return Err(TraceWasmError::ImportSignatureMismatch(
                    module_name.to_string(),
                    imported_func_name.to_string(),
                    "results".to_string(),
                    formatted_val_types(results),
                    formatted_val_types(&import_results),
                ));
            }
        }

        Ok(Instance::new(memory, import_registry, self.clone()))
    }
}
