//! The embedder-facing runtime: an instantiated [`Module`] plus the
//! type-safe [`TypedFunc`] handle for calling its functions.

use crate::{
    error::TraceWasmError,
    instance::traits::{ImportRegistry, Params, Results, assert_imported_func_trait},
    memory::Memory,
    module::{FuncIndex, Module, ValType},
    utils::formatted_val_types,
    vm::{TraceVM, stack::Val},
};
use std::{marker::PhantomData, sync::Arc};

pub mod traits;

/// A compiled module paired with the mutable state needed to run it: its linear
/// [`Memory`] and the [`ImportRegistry`] resolving its imported functions.
///
/// Generic over `M`/`I` so embedders choose their own memory backing store and
/// import implementation. The module is shared via `Arc`, so one compiled
/// module can back several instances.
pub struct Instance<M, I> {
    memory: M,
    import_registry: I,
    module: Arc<Module>,
}

impl<M: Memory, I: ImportRegistry> Instance<M, I> {
    pub fn new(memory: M, import_registry: I, module: Arc<Module>) -> Self {
        Instance {
            memory,
            import_registry,
            module,
        }
    }
}

/// A statically-typed handle to a module function: `P` is its parameter tuple
/// and `R` its result tuple.
///
/// The type parameters let [`Self::call`] accept native Rust values and return
/// native Rust values, converting to/from runtime `Val`s internally.
pub struct TypedFunc<P, R> {
    func_index: FuncIndex,
    phantom: PhantomData<(P, R)>,
}

impl<P: Params, R: Results> TypedFunc<P, R> {
    /// Calls the function on `instance` with `params`, returning its results.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::IncorrectParamsResultsStructure`] if the
    /// function's actual results do not match `R`, and propagates any
    /// [`TraceWasmError`] produced during execution (traps, etc.).
    pub fn call<M: Memory, I: ImportRegistry>(
        &self,
        params: P,
        instance: &mut Instance<M, I>,
    ) -> Result<R, TraceWasmError> {
        let params = params.to_vals();

        let results = TraceVM::run(
            self.func_index,
            &params,
            &instance.module,
            &mut instance.memory,
            &mut instance.import_registry,
        )?;

        let Some(res) = R::from_vals(&results) else {
            return Err(TraceWasmError::IncorrectParamsResultsStructure(
                "results".to_string(),
                self.func_index.0,
                formatted_val_types(&R::types()),
                format!("{:?}", results),
            ));
        };

        Ok(res)
    }
}

pub struct ImportedFunctions {}

impl ImportRegistry for ImportedFunctions {
    fn execute(
        &mut self,
        module_name: &str,
        func_name: &str,
        params: &[Val],
    ) -> Result<Box<[Val]>, TraceWasmError> {
        match (module_name, func_name) {
            ("env", "host1") => {
                // check the types of params first! whether they match the signature!
                // parse params here!
                let params: (i32, i32) = (params[0].as_i32(), params[1].as_i32()); // this should come from params
                let results = self.host1(params.0, params.1);
                let res: Vec<Val> = vec![Val::I32(results)];

                Ok(res.into_boxed_slice())
            }
            ("env", "host2") => {
                // parse params here!
                let params: (i32, i32, i64) =
                    (params[0].as_i32(), params[1].as_i32(), params[2].as_i64()); // this should come from params
                let results = self.host2(params.0, params.1, params.2);
                let res: Vec<Val> = vec![Val::I32(results)];

                Ok(res.into_boxed_slice())
            }
            _ => Err(TraceWasmError::ImportedFunctionNotFound(
                module_name.to_string(),
                func_name.to_string(),
            )),
        }
    }

    fn signature(
        &self,
        module_name: &str,
        func_name: &str,
    ) -> Option<(Box<[ValType]>, Box<[ValType]>)> {
        let sig = match (module_name, func_name) {
            ("env", "host1") => {
                let params = [ValType::I32, ValType::I32].to_vec();
                let results = [ValType::I32].to_vec();

                (params.into_boxed_slice(), results.into_boxed_slice())
            }
            ("env", "host2") => {
                let params = [ValType::I32, ValType::I32, ValType::I64].to_vec();
                let results = [ValType::I32].to_vec();

                (params.into_boxed_slice(), results.into_boxed_slice())
            }
            _ => return None,
        };

        Some(sig)
    }

    fn size(&self) -> u32 {
        2 // number of functions in imports impl block
    }
}

// #[imports] imports macro. identifies the module name of the function with what is specified in the `module` macro!
// This macro makes sure all the functions inside this impl has ctx of exactly same type!
impl ImportedFunctions {
    // #[module("env")] // specifies the module name
    pub fn host1(&mut self, _a: i32, _b: i32) -> i32 {
        // constructed from the signature
        let inner = |_ctx: &mut ImportedFunctions, _params: (i32, i32)| (0,);

        assert_imported_func_trait(inner); // type-check it!

        todo!() // user implemented logic
    }

    // #[module("env")] // specifies the module name
    pub fn host2(&mut self, _a: i32, _b: i32, _c: i64) -> i32 {
        // constructed from the signature
        let inner = |_ctx: &mut ImportedFunctions, _params: (i32, i32, i64)| (0,);

        assert_imported_func_trait(inner); // type-check it!

        todo!()
    }
}
