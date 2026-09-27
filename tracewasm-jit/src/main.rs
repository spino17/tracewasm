use llvm_sys::core::*;
use llvm_sys::error::*;
use llvm_sys::ir_reader::LLVMParseIRInContext2;
use llvm_sys::orc2::lljit::*;
use llvm_sys::orc2::*;
use llvm_sys::prelude::LLVMModuleRef;
use llvm_sys::target::{LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget};
use std::ffi::{CStr, c_char};
use std::ptr;

// The module: one imported function, one global string, one exported function.
const IR: &str = r#"
@msg = private constant [13 x i8] c"Hello, world!"

declare void @host_print(ptr, i64)

define i32 @hello(i32 %x) {
entry:
  call void @host_print(ptr @msg, i64 13)
  %r = add i32 %x, 1
  ret i32 %r
}
"#;

// The host implementation of the import. Its signature must match the declaration.
extern "C" fn host_print(ptr: *const u8, len: u64) {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    println!("{}", String::from_utf8_lossy(bytes));
}

// The Rust type of the export, matching `i32 @hello(i32)`.
type HelloFn = unsafe extern "C" fn(i32) -> i32;

fn check(err: LLVMErrorRef) -> Result<(), String> {
    if err.is_null() {
        return Ok(());
    }
    unsafe {
        let msg = LLVMGetErrorMessage(err); // consumes `err`
        let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
        LLVMDisposeErrorMessage(msg);
        Err(s)
    }
}

fn main() -> Result<(), String> {
    unsafe {
        // 1. Initialize code generation for the machine we're running on.
        if LLVM_InitializeNativeTarget() != 0 || LLVM_InitializeNativeAsmPrinter() != 0 {
            return Err("failed to initialize native target".into());
        }

        // 2. Create the JIT.
        let mut jit: LLVMOrcLLJITRef = ptr::null_mut();
        check(LLVMOrcCreateLLJIT(&mut jit, ptr::null_mut()))?;
        let dylib = LLVMOrcLLJITGetMainJITDylib(jit);

        // 3. Parse the IR text into a module, in a context we own for now.
        let ctx = LLVMContextCreate();

        let buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            IR.as_ptr() as *const c_char,
            IR.len(),
            c"hello_module".as_ptr(),
        );

        let mut module: LLVMModuleRef = ptr::null_mut();
        let mut msg: *mut c_char = ptr::null_mut();
        let failed = LLVMParseIRInContext2(ctx, buf, &mut module, &mut msg) != 0;

        LLVMDisposeMemoryBuffer(buf); // the "2" variant doesn't take ownership

        if failed {
            let err = CStr::from_ptr(msg).to_string_lossy().into_owned();

            LLVMDisposeMessage(msg);
            LLVMContextDispose(ctx);

            return Err(format!("IR parse error: {err}"));
        }

        LLVMSetTarget(module, LLVMOrcLLJITGetTripleString(jit));
        LLVMSetDataLayout(module, LLVMOrcLLJITGetDataLayoutStr(jit));

        // 4. Hand the context to ORC, then the module to the JIT.
        let tsc = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(ctx); // takes ownership of `ctx`
        let tsm = LLVMOrcCreateNewThreadSafeModule(module, tsc);

        LLVMOrcDisposeThreadSafeContext(tsc); // the module keeps its own reference
        check(LLVMOrcLLJITAddLLVMIRModule(jit, dylib, tsm))?;

        // 5. Define the import: `host_print` lives at our Rust function's address.
        let mut pair = LLVMOrcCSymbolMapPair {
            Name: LLVMOrcLLJITMangleAndIntern(jit, c"host_print".as_ptr()),
            Sym: LLVMJITEvaluatedSymbol {
                Address: host_print as usize as u64,
                Flags: LLVMJITSymbolFlags {
                    GenericFlags: LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported
                        as u8
                        | LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsCallable as u8,
                    TargetFlags: 0,
                },
            },
        };

        let mu = LLVMOrcAbsoluteSymbols(&mut pair, 1);

        check(LLVMOrcJITDylibDefine(dylib, mu))?;

        // 6. Look up the export. This is when the module is actually compiled and linked.
        let mut addr: LLVMOrcExecutorAddress = 0;

        check(LLVMOrcLLJITLookup(jit, &mut addr, c"hello".as_ptr()))?;

        let hello: HelloFn = std::mem::transmute(addr as usize);

        // 7. Call it.
        let result = hello(41);

        println!("hello(41) returned {result}");

        // 8. Tear down. `hello` must not be called after this.
        check(LLVMOrcDisposeLLJIT(jit))?;
    }

    Ok(())
}
