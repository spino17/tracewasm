use crate::instruction::Instruction;
use anyhow::Result;
use wasmparser::{
    ExternalKind, GlobalType, MemoryType, Parser, Payload::*, RefType, TableType, TypeRef, ValType,
};
pub struct TraceWasmParser;

#[derive(Clone, Copy)]
pub struct FuncIndex(pub u32);
#[derive(Clone, Copy)]
pub struct FuncExactIndex(pub u32);
#[derive(Clone, Copy)]
pub struct FuncTyIndex(pub u32);
#[derive(Clone, Copy)]
pub struct GlobalIndex(pub u32);
#[derive(Clone, Copy)]
pub struct TableIndex(pub u32);
#[derive(Clone, Copy)]
pub struct MemoryIndex(pub u32);
#[derive(Clone, Copy)]
pub struct TagIndex(pub u32);

pub struct Module {
    pub func_types: Box<[FuncType]>,
    pub func_decls: Box<[FuncDecl]>,
    pub tables: Box<[Table]>,
    pub memories: Box<[MemoryType]>,
    pub globals: Box<[Global]>,
    pub exports: Box<[Export]>,
    pub start_section: FuncIndex,
    pub elements: Box<[Element]>,
    pub data_count: u32,
    pub code_sec_count: u32,
    pub code_sec_size: u32,
    pub imported_func_count: u32,
    pub func_bodies: Box<[Box<[Instruction]>]>,
}

pub struct FuncBody(pub Box<[Instruction]>);

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
    pub ty_index: FuncTyIndex,
}

impl TraceWasmParser {
    pub fn parse(buf: &[u8]) -> Result<Module, anyhow::Error> {
        let mut func_types = vec![];
        let mut func_decls = vec![];
        let mut tables = vec![];
        let mut memories = vec![];
        let mut globals = vec![];
        let mut exports = vec![];
        let mut elements = vec![];
        let mut datas = vec![];
        let mut start_section = FuncIndex(0);
        let mut imported_func_count = 0;
        let mut data_count = 0;
        let mut code_sec_count = 0;
        let mut code_sec_size = 0;
        let mut func_bodies = vec![];

        for payload in Parser::new(0).parse_all(buf) {
            let payload = payload?;

            match payload {
                TypeSection(ty_sec) => {
                    let func_types_iter = ty_sec.into_iter_err_on_gc_types();

                    for ty in func_types_iter {
                        let ty = ty?;
                        let params = ty.params();
                        let results = ty.results();

                        func_types.push(FuncType {
                            params: params.to_vec().into_boxed_slice(),
                            results: results.to_vec().into_boxed_slice(),
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
                            FuncTyIndex(ty)
                        } else {
                            return Err(anyhow::Error::msg(
                                "non-function imports are not allowed in TraceWasm",
                            ));
                        };

                        func_decls.push(FuncDecl {
                            kind: FuncKind::Imported {
                                module_name,
                                imported_func_name,
                            },
                            ty_index,
                        });
                    }

                    imported_func_count = func_decls.len() as u32;
                }
                FunctionSection(func_sec) => {
                    let indices = func_sec.into_iter();

                    for index in indices {
                        let index = index?;

                        func_decls.push(FuncDecl {
                            kind: FuncKind::Local,
                            ty_index: FuncTyIndex(index),
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
                                )?
                                .into_boxed_slice(),
                            ),
                        };

                        tables.push(Table {
                            ty,
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
                GlobalSection(global_sec) => {
                    let global_iter = global_sec.into_iter();

                    for global in global_iter {
                        let global = global?;
                        let global_ty = global.ty;

                        globals.push(Global {
                            ty: global_ty,
                            val: Instruction::emit_instruction_from_operator_reader(
                                global.init_expr.get_operators_reader(),
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

                        exports.push(Export {
                            name: export.name.to_string(),
                            kind: match export.kind {
                                ExternalKind::Func => ExportKind::Func(FuncIndex(index)),
                                ExternalKind::Table => ExportKind::Table(TableIndex(index)),
                                ExternalKind::Memory => ExportKind::Memory(MemoryIndex(index)),
                                ExternalKind::Global => ExportKind::Global(GlobalIndex(index)),
                                ExternalKind::Tag => ExportKind::Tag(TagIndex(index)),
                                ExternalKind::FuncExact => {
                                    ExportKind::FuncExact(FuncExactIndex(index))
                                }
                            },
                        });
                    }
                }
                StartSection {
                    func,
                    range: _range,
                } => {
                    start_section = FuncIndex(func);
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
                                    table_index: table_index.map(|i| TableIndex(i)),
                                    offset_expr:
                                        Instruction::emit_instruction_from_operator_reader(
                                            offset_expr.get_operators_reader(),
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
                                            )?
                                            .into_boxed_slice(),
                                        );
                                    }

                                    ElementItems::Expressions(ref_ty, exprs.into_boxed_slice())
                                }
                            },
                        });
                    }
                }
                DataCountSection {
                    count,
                    range: _range,
                } => {
                    data_count = count;
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
                    let instructions = Instruction::emit_instruction_from_operator_reader(
                        code_sec_entry.get_operators_reader()?,
                    )?
                    .into_boxed_slice();

                    func_bodies.push(instructions);
                }
                _ => continue,
            }
        }

        Ok(Module {
            func_types: func_types.into_boxed_slice(),
            func_decls: func_decls.into_boxed_slice(),
            tables: tables.into_boxed_slice(),
            memories: memories.into_boxed_slice(),
            globals: globals.into_boxed_slice(),
            exports: exports.into_boxed_slice(),
            elements: elements.into_boxed_slice(),
            start_section,
            data_count,
            code_sec_count,
            code_sec_size,
            imported_func_count,
            func_bodies: func_bodies.into_boxed_slice(),
        })
    }
}
