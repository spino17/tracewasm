use crate::{
    ast::{
        CustomSection, CustomSectionKind, Data, DataKind, Element, ElementItems, ElementKind,
        Export, ExportKind, FuncDecl, FuncExactIndex, FuncIndex, FuncKind, FuncType, Global,
        GlobalIndex, IndirectNameMap, MemoryIndex, Module, Name, NameMap, Table, TableIndex,
        TableInit, TagIndex, TyIndex,
    },
    error::TraceWasmError,
    instruction::Instruction,
};
use anyhow::Result;
use wasmparser::{Encoding, ExternalKind, Parser, Payload::*, TypeRef, Validator};

pub struct TraceWasmParser;

impl TraceWasmParser {
    pub fn parse(buf: &[u8]) -> Result<Module, TraceWasmError> {
        // The parser alone only checks structure; validate semantics (section
        // order, index bounds, types) up front so the AST is built from wasm
        // that is known to be well-formed.
        Validator::new().validate_all(buf)?;

        let mut types = vec![];
        let mut func_decls = vec![];
        let mut tables = vec![];
        let mut memories = vec![];
        let mut globals = vec![];
        let mut exports = vec![];
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
                            ty_index: TyIndex(index),
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
                CustomSection(custom_sec) => {
                    let name = custom_sec.name().to_string();

                    let kind = match custom_sec.as_known() {
                        wasmparser::KnownCustom::Name(reader) => {
                            let mut names: Vec<Name> = vec![];

                            for name in reader {
                                let Ok(name) = name else {
                                    continue;
                                };

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

        Ok(Module {
            types: types.into_boxed_slice(),
            func_decls: func_decls.into_boxed_slice(),
            tables: tables.into_boxed_slice(),
            memories: memories.into_boxed_slice(),
            tags: tags.into_boxed_slice(),
            globals: globals.into_boxed_slice(),
            exports: exports.into_boxed_slice(),
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
        })
    }
}
