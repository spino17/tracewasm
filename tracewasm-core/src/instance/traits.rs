//! The typed-function API's value-conversion traits.
//!
//! [`WasmTy`] maps a Rust scalar to a single WebAssembly value type; the
//! [`Params`]/[`Results`] traits — implemented for the empty tuple `()` and for
//! tuples up to arity 5 (`Params`) and arity 3 (`Results`) — let a whole
//! function signature be expressed as Rust types, so
//! [`crate::instance::TypedFunc`] can convert arguments and results without the
//! caller touching raw `Val`s.

use crate::{memory::MemoryView, module::ValType};
use smallvec::{Array, SmallVec, smallvec};

// The runtime value type lives in the crate-internal `runtime` module; re-export it
// here so it has a public path (`instance::traits::Val`) for the `ImportRegistry`
// signatures and the `#[imports]`-generated code, without exposing the VM.
pub use crate::runtime::value::Val;

/// A Rust type corresponding to a single WebAssembly value type.
pub trait WasmTy: Sized {
    /// Converts this value into a runtime `Val`.
    fn to_val(&self) -> Val;

    /// Recovers a value from a runtime `Val`, returning `None` on a type mismatch.
    fn from_val(v: Val) -> Option<Self>;

    /// The WebAssembly value type this Rust type maps to.
    fn ty() -> ValType;
}

impl WasmTy for i32 {
    fn to_val(&self) -> Val {
        Val::I32(*self)
    }

    fn from_val(v: Val) -> Option<i32> {
        if let Val::I32(v) = v { Some(v) } else { None }
    }

    fn ty() -> ValType {
        ValType::I32
    }
}

impl WasmTy for i64 {
    fn to_val(&self) -> Val {
        Val::I64(*self)
    }

    fn from_val(v: Val) -> Option<Self> {
        if let Val::I64(v) = v { Some(v) } else { None }
    }

    fn ty() -> ValType {
        ValType::I64
    }
}

impl WasmTy for f32 {
    fn to_val(&self) -> Val {
        Val::F32(*self)
    }

    fn from_val(v: Val) -> Option<Self> {
        if let Val::F32(v) = v { Some(v) } else { None }
    }

    fn ty() -> ValType {
        ValType::F32
    }
}

impl WasmTy for f64 {
    fn to_val(&self) -> Val {
        Val::F64(*self)
    }

    fn from_val(v: Val) -> Option<Self> {
        if let Val::F64(v) = v { Some(v) } else { None }
    }

    fn ty() -> ValType {
        ValType::F64
    }
}

// These structs are wrappers over `SmallVec` to avoid heap allocations in the VM hot path.

/// A function's parameter value types, stack-allocated for arities up to 5.
#[derive(Debug)]
pub struct ParamValTypes(SmallVec<[ValType; 5]>);
/// A function's parameter values, stack-allocated for arities up to 5.
#[derive(Debug)]
pub struct ParamVals(SmallVec<[Val; 5]>);

/// A function's result value types, stack-allocated for arities up to 3.
#[derive(Debug)]
pub struct ResultValTypes(SmallVec<[ValType; 3]>);
/// A function's result values, stack-allocated for arities up to 3.
#[derive(Debug)]
pub struct ResultVals(SmallVec<[Val; 3]>);

impl ParamVals {
    /// Wraps a `SmallVec` of param values.
    pub fn new(s: SmallVec<[Val; 5]>) -> Self {
        ParamVals(s)
    }
}

impl ResultVals {
    /// Wraps a `SmallVec` of result values.
    pub fn new(s: SmallVec<[Val; 3]>) -> Self {
        ResultVals(s)
    }
}

/// Owning iterator over the [`Val`]s of a [`ResultVals`], in order.
pub struct ResultValsIter {
    results: ResultVals,
    index: usize,
}

impl Iterator for ResultValsIter {
    type Item = Val;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.results.0.len() {
            return None;
        }

        let item = self.results.0[self.index];

        self.index += 1;

        Some(item)
    }
}

impl IntoIterator for ResultVals {
    type Item = Val;
    type IntoIter = ResultValsIter;

    fn into_iter(self) -> Self::IntoIter {
        ResultValsIter {
            results: self,
            index: 0,
        }
    }
}

impl AsRef<[Val]> for ParamVals {
    fn as_ref(&self) -> &[Val] {
        &self.0
    }
}

impl AsRef<[ValType]> for ParamValTypes {
    fn as_ref(&self) -> &[ValType] {
        &self.0
    }
}

impl AsRef<[Val]> for ResultVals {
    fn as_ref(&self) -> &[Val] {
        &self.0
    }
}

impl AsRef<[ValType]> for ResultValTypes {
    fn as_ref(&self) -> &[ValType] {
        &self.0
    }
}

/// Internal helper for wrapping a `SmallVec` into one of the newtype wrappers
/// ([`ParamVals`], [`ResultVals`], and their type counterparts); used by the
/// tuple impls to build their marshalled values.
pub trait FromSmallVec<A: Array> {
    /// Wraps `s` into the newtype.
    fn from_small_vec(s: SmallVec<A>) -> Self;
}

/// Conversions and type info shared by a function's parameter and result tuples:
/// value ↔ `Val` marshalling, arity, and the tuple's value types.
pub trait FuncSignatureEntity<A: Array, B: Array, V: FromSmallVec<A>, T: FromSmallVec<B>>:
    Sized
{
    /// Converts the tuple into runtime values, in declaration order.
    fn to_vals(&self) -> V;
    /// The number of values in the tuple.
    fn len() -> usize;
    /// The value types, in order.
    fn types() -> T;
}

impl FromSmallVec<[Val; 5]> for ParamVals {
    fn from_small_vec(s: SmallVec<[Val; 5]>) -> Self {
        ParamVals(s)
    }
}

impl FromSmallVec<[ValType; 5]> for ParamValTypes {
    fn from_small_vec(s: SmallVec<[ValType; 5]>) -> Self {
        ParamValTypes(s)
    }
}

impl FromSmallVec<[Val; 3]> for ResultVals {
    fn from_small_vec(s: SmallVec<[Val; 3]>) -> Self {
        ResultVals(s)
    }
}

impl FromSmallVec<[ValType; 3]> for ResultValTypes {
    fn from_small_vec(s: SmallVec<[ValType; 3]>) -> Self {
        ResultValTypes(s)
    }
}

/// A function's parameter list, expressed as a tuple of [`WasmTy`] types.
pub trait Params: FuncSignatureEntity<[Val; 5], [ValType; 5], ParamVals, ParamValTypes> {}

/// A function's result list, expressed as a tuple of [`WasmTy`] types.
pub trait Results: FuncSignatureEntity<[Val; 3], [ValType; 3], ResultVals, ResultValTypes> {
    /// Rebuilds the tuple from runtime values, returning `None` if the count or
    /// types do not match.
    fn from_vals(vals: &[Val]) -> Option<Self>;
}

/// The empty signature: a function taking no parameters, or returning no results.
impl<A: Array, B: Array, V: FromSmallVec<A>, T: FromSmallVec<B>> FuncSignatureEntity<A, B, V, T>
    for ()
{
    fn to_vals(&self) -> V {
        V::from_small_vec(smallvec![])
    }

    fn len() -> usize {
        0
    }

    fn types() -> T {
        T::from_small_vec(smallvec![])
    }
}

impl Params for () {}

impl Results for () {
    fn from_vals(vals: &[Val]) -> Option<Self> {
        vals.is_empty().then_some(())
    }
}

/// Implements [`FuncSignatureEntity`] (at the param sizes `[Val; 5]`/
/// `[ValType; 5]`) and [`Params`] for a tuple, given just its element type params
/// (e.g. `impl_params!(A1, A2)`).
///
/// Mirrors the hand-written reference impl: the array/wrapper types are fixed to
/// the param sizes while `V`/`T` stay generic, so the impl satisfies the `Params`
/// supertrait bound. Field bindings and arity are derived — `to_vals`
/// destructures the tuple (binding each field to its type-param name), and `len`
/// counts the idents.
macro_rules! impl_params {
    ($($ty:ident),+ $(,)?) => {
        impl<V: FromSmallVec<[Val; 5]>, T: FromSmallVec<[ValType; 5]>, $($ty: WasmTy),+>
            FuncSignatureEntity<[Val; 5], [ValType; 5], V, T> for ($($ty,)+)
        {
            #[allow(non_snake_case)] // bindings reuse the (uppercase) type-param names
            fn to_vals(&self) -> V {
                let ($($ty,)+) = self;
                V::from_small_vec(smallvec![$($ty.to_val()),+])
            }

            fn len() -> usize {
                [$(stringify!($ty)),+].len()
            }

            fn types() -> T {
                T::from_small_vec(smallvec![$($ty::ty()),+])
            }
        }

        impl<$($ty: WasmTy),+> Params for ($($ty,)+) {}
    };
}

/// Implements [`FuncSignatureEntity`] (at the result sizes `[Val; 3]`/
/// `[ValType; 3]`) and [`Results`] for a tuple. Same shape as [`impl_params`] but
/// sized for results; `from_vals` rebuilds the tuple from a value slice,
/// consuming one value per element in order.
macro_rules! impl_results {
    ($($ty:ident),+ $(,)?) => {
        impl<V: FromSmallVec<[Val; 3]>, T: FromSmallVec<[ValType; 3]>, $($ty: WasmTy),+>
            FuncSignatureEntity<[Val; 3], [ValType; 3], V, T> for ($($ty,)+)
        {
            #[allow(non_snake_case)] // bindings reuse the (uppercase) type-param names
            fn to_vals(&self) -> V {
                let ($($ty,)+) = self;
                V::from_small_vec(smallvec![$($ty.to_val()),+])
            }

            fn len() -> usize {
                [$(stringify!($ty)),+].len()
            }

            fn types() -> T {
                T::from_small_vec(smallvec![$($ty::ty()),+])
            }
        }

        impl<$($ty: WasmTy),+> Results for ($($ty,)+) {
            fn from_vals(vals: &[Val]) -> Option<Self> {
                let mut iter = vals.iter();
                // Pull one value per element, in field order; too few → `None`.
                let tuple = ($($ty::from_val(*iter.next()?)?,)+);
                // Reject a value count larger than the tuple's arity.
                if iter.next().is_some() {
                    return None;
                }

                Some(tuple)
            }
        }
    };
}

// Params: tuples of arity 1..=5 (inline capacity 5).
impl_params!(A1);
impl_params!(A1, A2);
impl_params!(A1, A2, A3);
impl_params!(A1, A2, A3, A4);
impl_params!(A1, A2, A3, A4, A5);

// Results: tuples of arity 1..=3 (inline capacity 3).
impl_results!(A1);
impl_results!(A1, A2);
impl_results!(A1, A2, A3);

/// A host function usable as a wasm import: a callable taking `&mut Ctx` and a
/// [`Params`] tuple, returning a [`Results`] tuple.
///
/// Blanket-implemented for every matching closure/`fn`, so it is a bound to
/// write against rather than something to implement. [`assert_imported_func_trait`]
/// exists to turn it into a compile-time check.
pub trait ImportedFunc<Ctx, P, R>: Fn(&mut Ctx, P) -> R {}
impl<P: Params, R: Results, Ctx, F: Fn(&mut Ctx, P) -> R> ImportedFunc<Ctx, P, R> for F {}

/// An imported function's signature as `(params, results)` value-type lists.
pub type ImportSignature = (ParamValTypes, ResultValTypes);

/// Resolves a module's imported functions to host implementations.
///
/// Supplied by the embedder at [`Module::instantiate`](crate::module::Module::instantiate),
/// which cross-checks it against the module's declared imports before running.
pub trait ImportRegistry {
    /// Invokes the imported function `module_name::func_name` with `params`,
    /// returning its results.
    ///
    /// # Errors
    ///
    /// Whatever the host function chose to fail with. The interpreter treats it
    /// as a trap in the calling frame, wrapping it in an
    /// [`InstructionExecutionError::Call`](crate::error::InstructionExecutionError::Call)
    /// naming the callee.
    fn execute<V: MemoryView>(
        &mut self,
        module_name: &str,
        func_name: &str,
        params: &[Val],
        memory_view: &mut V,
    ) -> Result<ResultVals, anyhow::Error>;

    /// Returns the `(params, results)` signature of `module_name::func_name`, or
    /// `None` if the registry has no such function.
    fn signature(&self, module_name: &str, func_name: &str) -> Option<ImportSignature>;

    /// The number of functions the registry provides (checked against the
    /// module's import count at instantiation).
    fn func_count(&self) -> u32;

    /// The number of globals the registry provides (checked against the module's
    /// imported-global count at instantiation).
    fn global_count(&self) -> u32;

    /// Returns the value of the imported global `module_name::global_name`, or
    /// an error if the registry has no such global — the `#[imports]` macro
    /// generates `anyhow!("import not found: …")` for that case.
    /// The value's type is cross-checked against the module's declared global
    /// type at instantiation.
    ///
    /// Called only once per global, during [`Module::instantiate`](crate::module::Module::instantiate):
    /// the resolved value is copied into the [`Instance`](crate::instance::Instance)
    /// and read from there for the rest of the module's execution.
    fn get_global(&self, module_name: &str, global_name: &str) -> Result<Val, anyhow::Error>;
}

/// Compile-time assertion helper: calling it forces `F` to satisfy
/// [`ImportedFunc`], i.e. that a host function's params form a [`Params`] tuple
/// and its results a [`Results`] tuple. Generates no code.
///
/// **Nothing currently calls it.** The `#[imports]` macro checks the same
/// property with its own generated `__assert_params`/`__assert_results` helpers.
/// This is kept as the hand-written equivalent, for asserting the bound on a
/// function the macro does not see.
pub fn assert_imported_func_trait<P: Params, R: Results, Ctx, F: ImportedFunc<Ctx, P, R>>(_f: F) {}
