use anyhow::Result;
use wasmparser::{Parser, Payload::*, TypeRef, ValType};

pub struct TraceWasmParser;

pub struct FuncIndex(u32);
pub struct FuncTyIndex(u32);

pub struct Module {
    pub func_types: Box<[FuncType]>, // ordering of type indices
    pub func_decls: Box<[FuncDecl]>, // ordering of func indices. Imports are included
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
    pub fn parse(buf: &[u8]) -> Result<(), anyhow::Error> {
        let mut func_types = vec![];
        let mut func_decls: Vec<FuncDecl> = vec![];

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
                TableSection(_) => {
                    todo!()
                }
                MemorySection(mem_sec) => {
                    todo!()
                }
                GlobalSection(global_sec) => {
                    todo!()
                }
                _ => continue,
            }
            // match on payload
        }

        Ok(())
    }
}
