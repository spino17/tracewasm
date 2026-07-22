//! The typed-function API's value-conversion traits.
//!
//! [`WasmTy`] maps a Rust scalar to a single WebAssembly value type; the
//! [`Params`]/[`Results`] traits — implemented for tuples up to arity 5 and for
//! the empty tuple `()` — let a whole function signature be expressed as Rust
//! types, so [`crate::instance::TypedFunc`] can convert arguments and results
//! without the caller touching raw `Val`s.

use crate::{module::ValType, vm::stack::Val};

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

/// Shared conversion for a function-signature tuple: turns the Rust values into
/// the runtime `Val`s pushed as arguments.
pub trait FuncSignatureEntity: Sized {
    /// Converts the tuple into runtime values, in declaration order.
    fn to_vals(&self) -> Vec<Val>;
}

/// A function's parameter list, expressed as a tuple of [`WasmTy`] types.
pub trait Params: FuncSignatureEntity {}

/// A function's result list, expressed as a tuple of [`WasmTy`] types.
pub trait Results: FuncSignatureEntity {
    /// The number of result values.
    fn len() -> usize;

    /// Rebuilds the tuple from runtime values, returning `None` if the count or
    /// types do not match.
    fn from_vals(vals: &[Val]) -> Option<Self>;

    /// The result value types, in order.
    fn types() -> Vec<ValType>;
}

/// The empty signature: a function taking no parameters, or returning no results.
impl FuncSignatureEntity for () {
    fn to_vals(&self) -> Vec<Val> {
        Vec::new()
    }
}

impl Params for () {}

impl Results for () {
    fn len() -> usize {
        0
    }

    fn from_vals(vals: &[Val]) -> Option<Self> {
        vals.is_empty().then_some(())
    }

    fn types() -> Vec<ValType> {
        Vec::new()
    }
}

/// Implements [`FuncSignatureEntity`], [`Params`], and [`Results`] for a tuple,
/// given just its element type params (e.g. `impl_tuple!(A1, A2)`).
///
/// Field indices and arity are derived rather than passed in: `to_vals`
/// destructures the tuple (binding each field to its type-param name), and
/// `from_vals` consumes a value iterator in order — so no `self.0` / `vals[i]` /
/// length literal is needed.
macro_rules! impl_tuple {
    ($($ty:ident),+ $(,)?) => {
        impl<$($ty: WasmTy),+> FuncSignatureEntity for ($($ty,)+) {
            #[allow(non_snake_case)] // bindings reuse the (uppercase) type-param names
            fn to_vals(&self) -> Vec<Val> {
                let ($($ty,)+) = self;
                vec![$($ty.to_val()),+]
            }
        }

        impl<$($ty: WasmTy),+> Params for ($($ty,)+) {}

        impl<$($ty: WasmTy),+> Results for ($($ty,)+) {
            fn len() -> usize {
                Self::types().len()
            }

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

            fn types() -> Vec<ValType> {
                vec![$($ty::ty()),+]
            }
        }
    };
}

impl_tuple!(A1);
impl_tuple!(A1, A2);
impl_tuple!(A1, A2, A3);
impl_tuple!(A1, A2, A3, A4);
impl_tuple!(A1, A2, A3, A4, A5);
