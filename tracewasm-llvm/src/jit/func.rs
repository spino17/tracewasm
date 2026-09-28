use crate::jit::JITCompiledInstance;
use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMContextRef, LLVMTypeRef};
use std::marker::PhantomData;

// The traits below are sealed: an impl that named the wrong IR type, or a host
// function whose `addr` wasn't really an `extern "C"` function of that type,
// would let safe code call through a mismatched signature.
mod sealed {
    pub trait Sealed {}
    pub trait SealedParams {}
}

/// A Rust type that can be passed to a host function, with its IR type.
///
/// Only types whose C ABI needs no `zeroext`/`signext` attribute are included,
/// so `i8`, `u8`, `i16`, `u16` and `bool` are deliberately missing.
pub trait LLVMFuncParam: sealed::Sealed + Copy + 'static {
    /// # Safety
    /// `ctx` must be a valid, live LLVM context.
    #[doc(hidden)]
    unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef;
}

/// A Rust type that a host function can return, with its IR type. `()` is `void`.
pub trait LLVMFuncResult: sealed::Sealed + 'static {
    /// # Safety
    /// `ctx` must be a valid, live LLVM context.
    #[doc(hidden)]
    unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef;
}

/// A tuple of parameter types, e.g. `(i64, i64)`, with their IR types.
pub trait LLVMFuncParams: sealed::SealedParams + 'static {
    /// # Safety
    /// `ctx` must be a valid, live LLVM context.
    #[doc(hidden)]
    unsafe fn llvm_types(ctx: LLVMContextRef) -> Vec<LLVMTypeRef>;
}

/// The signature of a function crossing the JIT boundary.
pub trait LLVMFunc {
    /// The parameters as a tuple, e.g. `(i64, i64)`.
    type Params;
    type Results;
}

/// An `extern "C"` function pointer that can be linked into the JIT.
pub trait LLVMHostFunc: sealed::Sealed + Copy + 'static {
    #[doc(hidden)]
    fn addr(self) -> usize;

    /// # Safety
    /// `ctx` must be a valid, live LLVM context.
    #[doc(hidden)]
    unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef;
}

/// The IR function type taking `P` and returning `R`.
///
/// # Safety
/// `ctx` must be a valid, live LLVM context.
pub(crate) unsafe fn fn_type<P: LLVMFuncParams, R: LLVMFuncResult>(
    ctx: LLVMContextRef,
) -> LLVMTypeRef {
    unsafe {
        let mut params = P::llvm_types(ctx);

        LLVMFunctionType(
            R::llvm_type(ctx),
            params.as_mut_ptr(),
            params.len() as u32,
            0, // not variadic
        )
    }
}

/// A compiled function whose signature was checked against its IR definition
/// when it was looked up. `call` accepts exactly the parameters `P` and returns `R`.
///
/// It borrows the instance it came from, so it can't be called after the JIT
/// that owns its code is gone.
pub struct Func<'a, P, R> {
    addr: usize,
    _inst: PhantomData<&'a JITCompiledInstance<'a>>,
    _sig: PhantomData<fn(P) -> R>,
}

impl<P, R> LLVMFunc for Func<'_, P, R> {
    type Params = P;
    type Results = R;
}

impl<P, R> Clone for Func<'_, P, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P, R> Copy for Func<'_, P, R> {}

impl<'a, P, R> Func<'a, P, R> {
    /// # Safety
    /// `addr` must be the address of compiled code whose IR signature matches
    /// `P` and `R`, and that code must stay live for `'a`.
    pub(crate) unsafe fn from_addr(addr: usize) -> Self {
        Func {
            addr,
            _inst: PhantomData,
            _sig: PhantomData,
        }
    }
}

macro_rules! scalar {
    ($($t:ty => $llvm:ident),* $(,)?) => {$(
        impl sealed::Sealed for $t {}

        impl LLVMFuncParam for $t {
            unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef {
                unsafe { $llvm(ctx) }
            }
        }

        impl LLVMFuncResult for $t {
            unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef {
                unsafe { $llvm(ctx) }
            }
        }
    )*};
}

scalar!(
    i32 => LLVMInt32TypeInContext,
    u32 => LLVMInt32TypeInContext,
    i64 => LLVMInt64TypeInContext,
    u64 => LLVMInt64TypeInContext,
    f32 => LLVMFloatTypeInContext,
    f64 => LLVMDoubleTypeInContext,
);

// IR pointers are opaque, so the pointee doesn't matter. It must be `Sized`, though:
// `*const [u8]` or `*const str` is two words, not one `ptr`.
macro_rules! pointer {
    ($($p:ty),*) => {$(
        impl<T: 'static> sealed::Sealed for $p {}

        impl<T: 'static> LLVMFuncParam for $p {
            unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef {
                unsafe { LLVMPointerTypeInContext(ctx, 0) }
            }
        }

        impl<T: 'static> LLVMFuncResult for $p {
            unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef {
                unsafe { LLVMPointerTypeInContext(ctx, 0) }
            }
        }
    )*};
}

pointer!(*const T, *mut T);

impl sealed::Sealed for () {}

impl LLVMFuncResult for () {
    unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef {
        unsafe { LLVMVoidTypeInContext(ctx) }
    }
}

macro_rules! func {
    ($($a:ident),*) => {
        impl<$($a: LLVMFuncParam,)*> sealed::SealedParams for ($($a,)*) {}

        impl<$($a: LLVMFuncParam,)*> LLVMFuncParams for ($($a,)*) {
            #[allow(unused_variables, unused_unsafe)] // with no parameters, `ctx` goes unused
            unsafe fn llvm_types(ctx: LLVMContextRef) -> Vec<LLVMTypeRef> {
                unsafe { vec![$(<$a as LLVMFuncParam>::llvm_type(ctx)),*] }
            }
        }

        impl<$($a: LLVMFuncParam,)* R: LLVMFuncResult> sealed::Sealed for extern "C" fn($($a),*) -> R {}

        impl<$($a: LLVMFuncParam,)* R: LLVMFuncResult> LLVMHostFunc for extern "C" fn($($a),*) -> R {
            fn addr(self) -> usize {
                self as usize
            }

            unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef {
                unsafe { fn_type::<($($a,)*), R>(ctx) }
            }
        }

        impl<$($a: LLVMFuncParam,)* R: LLVMFuncResult> Func<'_, ($($a,)*), R> {
            #[allow(non_snake_case, clippy::too_many_arguments)]
            pub fn call(&self, $($a: $a),*) -> R {
                // SAFETY: `from_addr`'s contract: `addr` is live compiled code whose
                // IR signature is exactly this one.
                let f: extern "C" fn($($a),*) -> R = unsafe { std::mem::transmute(self.addr) };

                f($($a),*)
            }
        }
    };
}

func!();
func!(A1);
func!(A1, A2);
func!(A1, A2, A3);
func!(A1, A2, A3, A4);
func!(A1, A2, A3, A4, A5);
func!(A1, A2, A3, A4, A5, A6);
func!(A1, A2, A3, A4, A5, A6, A7);
func!(A1, A2, A3, A4, A5, A6, A7, A8);
