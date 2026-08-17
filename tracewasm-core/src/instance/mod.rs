//! The embedder-facing runtime: an instantiated [`Module`] plus the
//! type-safe [`TypedFunc`] handle for calling its functions.

use crate::{
    InstrOf, VirtualMachine,
    error::FuncCallError,
    instance::{
        config::Config,
        traits::{ImportRegistry, Params, Results},
    },
    instruction::Instruction,
    memory::{Memory, MemoryView},
    module::{FuncIndex, Module},
    runtime::{
        TraceVM,
        value::{DataVal, ElementVal, TableVal, Val},
    },
};
use std::{marker::PhantomData, sync::Arc};

/// Per-instance resource limits.
pub mod config;
pub mod traits;

/// A compiled module paired with everything mutable that running it touches: the
/// frame store, linear memory, globals, tables, and the [`ImportRegistry`]
/// resolving its imported functions.
///
/// All interpreter state lives here rather than in the driver, so a frame is
/// described by a few indices into this rather than by owned buffers, and a
/// nested call costs no allocation.
///
/// Generic over `M`/`I` so embedders choose their own memory backing store and
/// import implementation, and over `V` so the frame store matches the
/// machine — see [`VirtualMachine`]. The module is shared via `Arc`, so one compiled module
/// can back several instances.
pub struct Instance<M, I, V: VirtualMachine> {
    /// The guest's linear memory.
    pub(crate) memory: M,
    /// Live values for every frame in the call chain, laid out however the
    /// lowering's [`RuntimeFrame`](crate::instruction::RuntimeFrame) chooses —
    /// an operand stack for the stack machine, a register file for the register
    /// machine. Either way one store is shared by the whole chain and each frame
    /// addresses its own region by base offset, so a nested call allocates
    /// nothing.
    ///
    /// Reset at the start of each [`TypedFunc::call`], which is what leaves an
    /// instance usable after a trap.
    pub(crate) frame: <InstrOf<V> as Instruction>::RuntimeFrame,
    /// Host functions backing the module's declared imports.
    pub(crate) import_registry: I,
    /// The compiled module this instance was created from.
    pub(crate) module: Arc<Module<V>>,
    /// The limits this instance was created under, already narrowed against what
    /// the module declares.
    pub(crate) config: Config,
    /// Global values, imported ones first, indexed by the module's global index
    /// space. Tagged [`Val`] rather than untagged
    /// [`Value`](crate::runtime::value::Value) because a global's
    /// type is read at runtime when the host asks for it.
    pub(crate) global_vals: Box<[Val]>,
    /// Materialized tables, indexed by table index.
    pub(crate) table_vals: Box<[TableVal]>,
    /// Element segments, retained so a passive segment can still be applied by a
    /// later `table.init`, and dropped by `elem.drop`.
    ///
    /// **Not read yet:** neither machine lowers a table bulk operator, so
    /// instantiation currently builds this and nothing consumes it — unlike
    /// [`Self::data_vals`], which `memory.init` and `data.drop` do read. Kept
    /// because it is what those operators will need, at the cost of the segment
    /// walk in `instantiate` on every instance.
    #[allow(
        dead_code,
        reason = "groundwork for table.init/elem.drop; an unsilenced warning here \
                  masks the ones that mean something"
    )]
    pub(crate) element_vals: Box<[ElementVal]>,
    /// Data segments, retained for `memory.init` and dropped by `data.drop`, for
    /// the same reason as [`Self::element_vals`].
    pub(crate) data_vals: Box<[DataVal]>,
}

impl<M: Memory, I: ImportRegistry, V: VirtualMachine> Instance<M, I, V> {
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
        module: Arc<Module<V>>,
        config: Config,
        global_vals: Box<[Val]>,
        table_vals: Box<[TableVal]>,
        element_vals: Box<[ElementVal]>,
        data_vals: Box<[DataVal]>,
    ) -> Self {
        Instance {
            memory,
            frame: <InstrOf<V> as Instruction>::RuntimeFrame::default(),
            import_registry,
            module,
            config,
            global_vals,
            table_vals,
            element_vals,
            data_vals,
        }
    }

    /// The resource limits in force for this instance.
    ///
    /// These are the *effective* limits: instantiation narrows the configured
    /// memory cap to the module's own declared maximum, so this can report a lower
    /// value than was supplied.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Read access to the instance's linear memory.
    ///
    /// Returned as an opaque [`MemoryView`] rather than as `&M`: that trait is the
    /// read/write half, so an embedder gets every bounds-checked accessor and none
    /// of [`Memory`]'s resizing.
    pub fn memory_view(&self) -> &impl MemoryView {
        &self.memory
    }

    /// Read/write access to the instance's linear memory.
    ///
    /// Also an opaque [`MemoryView`], for the reason [`MemoryView`] itself gives:
    /// growth is reserved to the interpreter, because the limits it must respect —
    /// the module's declared maximum, narrowed by the instance [`Config`] — live on
    /// this `Instance` and not on the memory. Handing out `&mut M` would let a host
    /// call [`Memory::grow`] straight past both.
    pub fn memory_view_mut(&mut self) -> &mut impl MemoryView {
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
    /// Wraps a function index in the `P`/`R` types the caller will use it with.
    ///
    /// Crate-private because it asserts the pairing rather than checking it; the
    /// public path is
    /// [`Module::get_typed_func`](crate::module::Module::get_typed_func), which
    /// matches `P` and `R` against the module's declared signature first.
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
        instance: &mut Instance<M, I, crate::Stack>,
    ) -> Result<R, FuncCallError> {
        // Marshalled into a stack-allocated `ParamVals` (no heap for <=5 params).
        let params = params.to_vals();
        let module = instance.module.clone();

        let results = TraceVM::run(self.func_index, params.as_ref(), instance, &module)?;

        // `get_typed_func` already matched `R` against the module's declared
        // results, so this holds. It could not be reported as a `FuncCallError`
        // anyway: that type describes a trap inside the guest and its trace must
        // start at the instruction that raised one, which a signature mismatch
        // here is not.
        let res = R::from_vals(results.as_ref())
            .expect("panicking this means the logic of function signature validation in `get_typed_func` in module/mod.rs is incorrect");

        Ok(res)
    }
}
