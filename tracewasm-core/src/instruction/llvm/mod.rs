use crate::{
    VirtualMachine,
    module::{FuncIndex, Module, ValType},
};
use rustc_hash::FxHashMap;
use std::sync::Arc;
use tracewasm_llvm::{
    cfg::{
        ControlFlowGraph,
        builder::Builder,
        context::Context,
        global::{DeclaredFunc, DefinedFunc, GlobalId},
        module::{DataLayout, DataLayoutSpec, Endianness, Mangling, Triple},
    },
    instruction::cursor::RegName,
    interner::TyId,
};

#[derive(Default)]
pub struct WasmInstrLLVMPassManager {
    declared_funcs: FxHashMap<FuncIndex, GlobalId<DeclaredFunc>>,
    defined_funcs: FxHashMap<FuncIndex, GlobalId<DefinedFunc>>,
}

impl WasmInstrLLVMPassManager {
    pub fn compile<V: VirtualMachine>(
        mut self,
        module: &Arc<Module<V>>,
    ) -> Result<ControlFlowGraph, anyhow::Error> {
        // TODO: get this from the system on which this function is called!
        let mut ctx = Context::new(
            Triple::new(
                "arm64".to_string(),
                "apple".to_string(),
                "macosx".to_string(),
                None,
            ),
            DataLayout::new(vec![
                DataLayoutSpec::Endianness(Endianness::Little),
                DataLayoutSpec::Mangling(Mangling::MachO),
                DataLayoutSpec::StackAlignment(128),
            ]),
        );

        let mut builder = ctx.builder();

        let ty_decls = &module.types;
        let func_decls = &module.func_decls;
        let imported_func_count = module.imported_func_count; // this many functions will be declared! rest are defined!

        for func_index in 0..func_decls.len() {
            let func_decl = &func_decls[func_index];
            let ty = func_decl.ty;
            let func_ty = &ty_decls[ty.0 as usize];
            let params = &func_ty.params;
            let results = &func_ty.results;

            let (llvm_params, llvm_result) =
                Self::llvm_signature_from_wasm(params, results, &mut builder);

            if func_index < imported_func_count as usize {
                let func = builder.declare_function(
                    format!("fn{}", func_index),
                    &llvm_params,
                    llvm_result,
                )?;

                self.declared_funcs
                    .insert(FuncIndex(func_index as u32), func);
            } else {
                let mut llvm_param_decls = vec![];
                let mut counter = 0;

                for param in llvm_params {
                    llvm_param_decls.push((param, RegName::Named(format!("param{}", counter))));

                    counter += 1;
                }

                let func = builder.define_function(
                    format!("fn{}", func_index),
                    &llvm_param_decls,
                    llvm_result,
                )?;

                self.compile_func(FuncIndex(func_index as u32), func, &mut builder, module)?;

                self.defined_funcs
                    .insert(FuncIndex(func_index as u32), func);
            }
        }

        Ok(builder.build())
    }

    fn compile_func<V: VirtualMachine>(
        &mut self,
        func_index: FuncIndex,
        func: GlobalId<DefinedFunc>,
        builder: &mut Builder,
        module: &Arc<Module<V>>,
    ) -> Result<(), anyhow::Error> {
        todo!()
    }

    fn llvm_ty_from_wasm(ty: &ValType, ctx: &mut Context) -> TyId {
        match ty {
            ValType::I32 => ctx.i32_ty(),
            ValType::I64 => ctx.i64_ty(),
            ValType::F32 => ctx.f32_ty(),
            ValType::F64 => ctx.f64_ty(),
            ValType::Ref(_) => ctx.ptr_ty(),
            ValType::V128 => unreachable!("v128 is rejected at Module check time"),
        }
    }

    fn llvm_signature_from_wasm(
        params: &[ValType],
        results: &[ValType],
        ctx: &mut Context,
    ) -> (Vec<TyId>, TyId) {
        let mut llvm_params = vec![];

        for param_ty in params {
            llvm_params.push(Self::llvm_ty_from_wasm(param_ty, ctx));
        }

        llvm_params.push(ctx.ptr_ty()); // pointer to runtime struct containing mmap memory pointers etc.

        let llvm_result = if results.len() > 1 {
            let mut fields = vec![];

            for result in results {
                fields.push(Self::llvm_ty_from_wasm(result, ctx));
            }

            ctx.struct_ty(&fields, false).unwrap()
        } else {
            Self::llvm_ty_from_wasm(&results[0], ctx)
        };

        (llvm_params, llvm_result)
    }
}
