use crate::{error::TraceWasmError, instruction::Instruction};
use rustc_hash::FxHashMap;
use std::hash::Hash;
use wasmparser::{GlobalType, MemoryType, RefType, TableType, TagType, ValType};

pub trait EntityIndex: From<u32> {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncIndex(pub u32);

impl From<u32> for FuncIndex {
    fn from(value: u32) -> Self {
        FuncIndex(value)
    }
}

impl EntityIndex for FuncIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncExactIndex(pub u32);

impl From<u32> for FuncExactIndex {
    fn from(value: u32) -> Self {
        FuncExactIndex(value)
    }
}

impl EntityIndex for FuncExactIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyIndex(pub u32);

impl From<u32> for TyIndex {
    fn from(value: u32) -> Self {
        TyIndex(value)
    }
}

impl EntityIndex for TyIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalIndex(pub u32);

impl From<u32> for GlobalIndex {
    fn from(value: u32) -> Self {
        GlobalIndex(value)
    }
}

impl EntityIndex for GlobalIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableIndex(pub u32);

impl From<u32> for TableIndex {
    fn from(value: u32) -> Self {
        TableIndex(value)
    }
}

impl EntityIndex for TableIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryIndex(pub u32);

impl From<u32> for MemoryIndex {
    fn from(value: u32) -> Self {
        MemoryIndex(value)
    }
}

impl EntityIndex for MemoryIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TagIndex(pub u32);

impl From<u32> for TagIndex {
    fn from(value: u32) -> Self {
        TagIndex(value)
    }
}

impl EntityIndex for TagIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElementIndex(pub u32);

impl From<u32> for ElementIndex {
    fn from(value: u32) -> Self {
        ElementIndex(value)
    }
}

impl EntityIndex for ElementIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataIndex(pub u32);

impl From<u32> for DataIndex {
    fn from(value: u32) -> Self {
        DataIndex(value)
    }
}

impl EntityIndex for DataIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalIndex(pub u32); // index of local inside a function

impl From<u32> for LocalIndex {
    fn from(value: u32) -> Self {
        LocalIndex(value)
    }
}

impl EntityIndex for LocalIndex {}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldIndex(pub u32); // field of a struct type

impl From<u32> for FieldIndex {
    fn from(value: u32) -> Self {
        FieldIndex(value)
    }
}

impl EntityIndex for FieldIndex {}

pub struct Module {
    pub types: Box<[FuncType]>, // we error for GC proposal types like struct, so it's only function types
    pub func_decls: Box<[FuncDecl]>, // imports + local functions. First comes the imports and then local functions
    pub tables: Box<[Table]>,
    pub memories: Box<[MemoryType]>,
    pub tags: Box<[TagType]>,
    pub globals: Box<[Global]>,
    pub exports: Box<[Export]>,
    pub start_section: Option<FuncIndex>,
    pub elements: Box<[Element]>,
    pub data_count: Option<u32>,
    pub datas: Box<[Data]>,
    pub code_sec_count: u32,
    pub code_sec_size: u32,
    pub imported_func_count: u32,
    pub func_bodies: Box<[FuncBody]>,
    pub unknown_sections: Box<[(u8, Box<[u8]>)]>, // (id, content)
    pub custom_sections: Box<[CustomSection]>,
}

pub struct FuncBody {
    pub locals: Box<[ValType]>, // params + declared locals inside the function body
    pub instructions: Box<[Instruction]>,
}

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

pub enum CustomSectionKind {
    Name(Box<[Name]>),
    Others(Box<[u8]>),
    Unknown(Box<[u8]>),
}

pub struct CustomSection {
    pub name: String,
    pub kind: CustomSectionKind,
}

pub struct Data {
    pub kind: DataKind,
    pub data: Box<[u8]>,
}

pub enum DataKind {
    Passive,
    Active {
        memory_index: MemoryIndex,
        offset_expr: Box<[Instruction]>,
    },
}

pub enum ElementKind {
    Passive,
    Active {
        table_index: Option<TableIndex>,
        offset_expr: Box<[Instruction]>,
    },
    Declared,
}

pub enum ElementItems {
    Functions(Box<[FuncIndex]>),
    Expressions(RefType, Box<[Box<[Instruction]>]>),
}

pub struct Element {
    pub kind: ElementKind,
    pub items: ElementItems,
}

pub enum TableInit {
    RefNull,
    Expr(Box<[Instruction]>),
}

pub struct Table {
    pub ty: TableType,
    pub init: TableInit,
}

pub enum ExportKind {
    Func(FuncIndex),
    Table(TableIndex),
    Memory(MemoryIndex),
    Global(GlobalIndex),
    Tag(TagIndex),
    FuncExact(FuncExactIndex),
}

pub struct Export {
    pub name: String,
    pub kind: ExportKind,
}

pub struct Global {
    pub ty: GlobalType,
    pub val: Box<[Instruction]>,
}

pub struct FuncType {
    pub params: Box<[ValType]>,
    pub results: Box<[ValType]>,
}

pub enum FuncKind {
    Local,
    Imported {
        module_name: String,
        imported_func_name: String,
    },
}

pub struct FuncDecl {
    pub kind: FuncKind,
    pub ty_index: TyIndex,
}

pub struct NameMap<T: PartialEq + Eq + Hash + EntityIndex>(pub FxHashMap<T, String>); // index -> name

impl<T: PartialEq + Eq + Hash + EntityIndex> NameMap<T> {
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

pub struct IndirectNameMap<
    T: PartialEq + Eq + Hash + EntityIndex,
    V: PartialEq + Eq + Hash + EntityIndex,
>(
    pub FxHashMap<T, NameMap<V>>, // e.g. func_index -> { local_index -> name } so IndirectNameMap<FuncIndex, LocalIndex>
);

impl<T: PartialEq + Eq + Hash + EntityIndex, V: PartialEq + Eq + Hash + EntityIndex>
    IndirectNameMap<T, V>
{
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
