//! The embedder-facing runtime: an instantiated [`Module`] plus the
//! type-safe [`TypedFunc`] handle for calling its functions.

use crate::{
    error::TraceWasmError,
    instance::traits::{Params, Results},
    memory::{Memory, linear::LinearMemory},
    module::{FuncIndex, Module},
    utils::formatted_val_types,
    vm::TraceVM,
};
use std::marker::PhantomData;

pub mod traits;

/// A module paired with the mutable state (linear memory) needed to run it.
///
/// Generic over the [`Memory`] backing store so embedders can choose their own
/// implementation.
pub struct Instance<M> {
    memory: M,
    module: Module,
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
    pub fn call<M: Memory>(
        &self,
        params: P,
        instance: &mut Instance<M>,
    ) -> Result<R, TraceWasmError> {
        let params = params.to_vals();

        let results = TraceVM::run(
            self.func_index,
            &params,
            &instance.module,
            &mut instance.memory,
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

fn _check_api_usage(module: Module) -> Result<(i32,), TraceWasmError> {
    let f: TypedFunc<(i32, f32), (i32,)> = TypedFunc {
        func_index: FuncIndex(10),
        phantom: PhantomData,
    };

    let mut instance = Instance {
        memory: LinearMemory::new(10),
        module,
    };

    let results = f.call((1, 2.0), &mut instance)?;

    Ok(results)
}
