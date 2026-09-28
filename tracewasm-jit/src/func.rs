use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMContextRef, LLVMTypeRef};

// The traits below are sealed: an impl that named the wrong IR type, or a host
// function whose `addr` wasn't really an `extern "C"` function of that type,
// would let safe code call through a mismatched signature.
mod sealed {
    pub trait Sealed {}
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

/// An `extern "C"` function pointer that can be linked into the JIT.
pub trait LLVMHostFunc: sealed::Sealed + Copy + 'static {
    #[doc(hidden)]
    fn addr(self) -> usize;

    /// # Safety
    /// `ctx` must be a valid, live LLVM context.
    #[doc(hidden)]
    unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef;
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

macro_rules! host_func {
    ($($a:ident),*) => {
        impl<$($a: LLVMFuncParam,)* R: LLVMFuncResult> sealed::Sealed for extern "C" fn($($a),*) -> R {}

        impl<$($a: LLVMFuncParam,)* R: LLVMFuncResult> LLVMHostFunc for extern "C" fn($($a),*) -> R {
            fn addr(self) -> usize {
                self as usize
            }

            unsafe fn llvm_type(ctx: LLVMContextRef) -> LLVMTypeRef {
                unsafe {
                    let mut params: Vec<LLVMTypeRef> = vec![$(<$a as LLVMFuncParam>::llvm_type(ctx)),*];

                    LLVMFunctionType(
                        R::llvm_type(ctx),
                        params.as_mut_ptr(),
                        params.len() as u32,
                        0, // not variadic
                    )
                }
            }
        }
    };
}

host_func!();
host_func!(A1);
host_func!(A1, A2);
host_func!(A1, A2, A3);
host_func!(A1, A2, A3, A4);
host_func!(A1, A2, A3, A4, A5);
host_func!(A1, A2, A3, A4, A5, A6);
host_func!(A1, A2, A3, A4, A5, A6, A7);
host_func!(A1, A2, A3, A4, A5, A6, A7, A8);
