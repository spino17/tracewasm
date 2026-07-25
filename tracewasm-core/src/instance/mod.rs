//! The embedder-facing runtime: an instantiated [`Module`] plus the
//! type-safe [`TypedFunc`] handle for calling its functions.

use crate::{
    error::TraceWasmError,
    instance::{
        config::Config,
        traits::{ImportRegistry, Params, Results},
    },
    memory::Memory,
    module::{FuncIndex, Module, formatted_val_types},
    vm::{
        TraceVM,
        stack::{DataVal, ElementVal, TableVal, Val},
    },
};
use std::{marker::PhantomData, sync::Arc};

pub mod config;
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
    config: Config,
    global_vals: Box<[Val]>,
    table_vals: Vec<TableVal>,
    element_vals: Box<[ElementVal]>,
    data_vals: Box<[DataVal]>,
}

impl<M: Memory, I: ImportRegistry> Instance<M, I> {
    /// Internal constructor assembling an instance from its parts. Crate-private
    /// because it performs no validation; the public path is
    /// [`Module::instantiate`](crate::module::Module::instantiate), which checks
    /// the registry against the module's imports first.
    pub(crate) fn new(
        memory: M,
        import_registry: I,
        module: Arc<Module>,
        config: Config,
        global_vals: Box<[Val]>,
        table_vals: Vec<TableVal>,
        element_vals: Box<[ElementVal]>,
        data_vals: Box<[DataVal]>,
    ) -> Self {
        Instance {
            memory,
            import_registry,
            module,
            config,
            global_vals,
            table_vals,
            element_vals,
            data_vals,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn memory_view(&self) -> &M {
        &self.memory
    }

    pub fn memory_view_mut(&mut self) -> &mut M {
        &mut self.memory
    }
}

/// A statically-typed handle to a module function: `P` is its parameter tuple
/// and `R` its result tuple.
///
/// The type parameters let [`Self::call`] accept native Rust values and return
/// native Rust values, converting to/from runtime `Val`s internally.
#[derive(Clone, Copy)]
pub struct TypedFunc<P, R> {
    func_index: FuncIndex,
    phantom: PhantomData<(P, R)>,
}

impl<P: Params, R: Results> TypedFunc<P, R> {
    pub(crate) fn new(func_index: FuncIndex) -> Self {
        TypedFunc {
            func_index,
            phantom: PhantomData,
        }
    }

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
        // Marshalled into a stack-allocated `ParamVals` (no heap for <=5 params).
        let params = params.to_vals();

        let results = TraceVM::run(
            self.func_index,
            params.as_ref(),
            &instance.module,
            &mut instance.memory,
            &mut instance.import_registry,
            &mut instance.global_vals,
            &mut instance.table_vals,
        )?;

        let Some(res) = R::from_vals(results.as_ref()) else {
            return Err(TraceWasmError::IncorrectParamsResultsStructure(
                "results".to_string(),
                self.func_index.0,
                formatted_val_types(R::types().as_ref()),
                format!("{:?}", results),
            ));
        };

        Ok(res)
    }
}
