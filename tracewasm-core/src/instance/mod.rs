//! The embedder-facing runtime: an instantiated [`Module`] plus the
//! type-safe [`TypedFunc`] handle for calling its functions.

use crate::{
    error::FuncCallError,
    instance::{
        config::Config,
        traits::{ImportRegistry, Params, Results},
    },
    memory::Memory,
    module::{FuncIndex, Module},
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
    // Assembles every piece of instance state; grouping them into a struct would
    // just move the same fields behind another type.
    #[allow(clippy::too_many_arguments)]
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

    /// The resource limits [`Config`] this instance was created with.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Shared access to the instance's linear [`Memory`].
    pub fn memory_view(&self) -> &M {
        &self.memory
    }

    /// Mutable access to the instance's linear [`Memory`].
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
    /// Returns a [`FuncCallError`] if execution fails (a trap, or an error from an
    /// imported function). It wraps the underlying cause together with the entry
    /// function and module, so callers can get a rendered backtrace from
    /// [`FuncCallError::stack_trace`] without threading those in themselves.
    ///
    /// Params and results are not re-checked here: `TypedFunc<P, R>` is only
    /// handed out after [`Module::get_typed_func`](crate::module::Module::get_typed_func)
    /// has matched `P`/`R` against the module's declared signature.
    pub fn call<M: Memory, I: ImportRegistry>(
        &self,
        params: P,
        instance: &mut Instance<M, I>,
    ) -> Result<R, FuncCallError> {
        // Marshalled into a stack-allocated `ParamVals` (no heap for <=5 params).
        let params = params.to_vals();

        let results = match TraceVM::run(
            self.func_index,
            params.as_ref(),
            &instance.module,
            &mut instance.memory,
            &mut instance.import_registry,
            &mut instance.global_vals,
            &mut instance.table_vals,
        ) {
            Ok(res) => res,
            Err(err) => {
                // A `TypedFunc` is only handed out for an export, so the lookup
                // resolves; fall back rather than unwrapping, since panicking
                // while building an error would replace a diagnosable failure
                // with a crash.
                let func_name = instance
                    .module
                    .exported_func_name(self.func_index)
                    .map(String::as_str)
                    .unwrap_or("<unknown>");

                return Err(FuncCallError::new(
                    func_name.to_string(),
                    err,
                    instance.module.clone(),
                ));
            }
        };

        // `get_typed_func` already matched `R` against the module's declared
        // results, so this holds. It cannot be reported as a `FuncCallError`
        // either, since that type's invariant is an `InstructionExecution` cause.
        let res = R::from_vals(results.as_ref())
            .expect("panicking this means the logic of function signature validation in `get_typed_func` in module/mod.rs is incorrect");

        Ok(res)
    }
}
