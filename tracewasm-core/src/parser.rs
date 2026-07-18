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
    pub func_types: Box<[FuncType]>, // ordering of type indices
    pub func_decls: Box<[FuncDecl]>, // ordering of func indices. Imports are included
    pub globals: Box<[Global]>,
    pub exports: Box<[Export]>,
    pub memories: Box<[MemoryType]>,
    pub start_section: FuncIndex,
}

pub enum ElementKind {
    Passive,
    Active {
        table_index: Option<TableIndex>,
        offset_expr: Instruction,
    },
    Declared,
}

pub enum ElementItems {
    Functions(Box<[FuncIndex]>),
    Expressions(RefType, Box<[Instruction]>),
}

pub struct Element {
    pub kind: ElementKind,
    pub items: ElementItems,
}

pub enum TableInit {
    RefNull,
    Expr(Instruction),
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
    pub val: Instruction,
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

// e.g. (import "env" "_call" (func $_ZN13async_weil_rs6future5_call17hfde84aebd9442f2fE (;0;) (type 5)))
pub struct FuncDecl {
    pub kind: FuncKind,
    pub ty_index: FuncTyIndex,
}

impl TraceWasmParser {
    pub fn parse(buf: &[u8]) -> Result<Module, anyhow::Error> {
        let mut func_types = vec![];
        let mut func_decls = vec![];
        let mut globals = vec![];
        let mut exports = vec![];
        let mut tables = vec![];
        let mut memories = vec![];
        let mut elements = vec![];
        let mut start_section = FuncIndex(0);

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
                            wasmparser::TableInit::Expr(const_expr) => {
                                TableInit::Expr(Instruction::from_operator(
                                    const_expr.get_operators_reader().read()?,
                                )?)
                            }
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
                        let mut operator_reader = global.init_expr.get_operators_reader();
                        let operator = operator_reader.read()?;

                        globals.push(Global {
                            ty: global_ty,
                            val: Instruction::from_operator(operator)?,
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
                                    offset_expr: Instruction::from_operator(
                                        offset_expr.get_operators_reader().read()?,
                                    )?,
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
                                        exprs.push(Instruction::from_operator(
                                            expr.get_operators_reader().read()?,
                                        )?);
                                    }

                                    ElementItems::Expressions(ref_ty, exprs.into_boxed_slice())
                                }
                            },
                        });
                    }
                }
                _ => continue,
            }
        }

        Ok(Module {
            func_types: func_types.into_boxed_slice(),
            func_decls: func_decls.into_boxed_slice(),
            globals: globals.into_boxed_slice(),
            exports: exports.into_boxed_slice(),
            memories: memories.into_boxed_slice(),
            start_section,
        })
    }
}
