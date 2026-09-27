use llvm_sys::core::*;
use llvm_sys::error::*;
use llvm_sys::ir_reader::LLVMParseIRInContext2;
use llvm_sys::orc2::lljit::*;
use llvm_sys::orc2::*;
use llvm_sys::prelude::{LLVMContextRef, LLVMModuleRef};
use llvm_sys::target::{LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget};
use llvm_sys::target_machine::*;
use llvm_sys::transforms::pass_builder::*;
use std::ffi::{CStr, c_char};
use std::ptr;

// Two exports and one import.
//
// `sum_to` is written the way a simple front end emits code: every local lives in
// an `alloca`, and every use is a load or store. The optimizer should promote the
// locals to registers and then replace the whole loop with a closed-form formula.
const IR: &str = r#"
@msg = private constant [13 x i8] c"Hello, world!"

declare void @host_print(ptr, i64)

define i32 @hello(i32 %x) {
entry:
  call void @host_print(ptr @msg, i64 13)
  %r = add i32 %x, 1
  ret i32 %r
}

define i64 @sum_to(i64 %n) {
entry:
  %i = alloca i64
  %acc = alloca i64
  store i64 0, ptr %i
  store i64 0, ptr %acc
  br label %loop

loop:
  %iv = load i64, ptr %i
  %done = icmp sge i64 %iv, %n
  br i1 %done, label %exit, label %body

body:
  %a = load i64, ptr %acc
  %a2 = add i64 %a, %iv
  store i64 %a2, ptr %acc
  %iv2 = add i64 %iv, 1
  store i64 %iv2, ptr %i
  br label %loop

exit:
  %r = load i64, ptr %acc
  ret i64 %r
}
"#;

/// The IR optimization pipeline, in `opt -passes=...` syntax.
const PIPELINE: &CStr = c"default<O3>";

// The host implementation of the import. Its signature must match the declaration.
extern "C" fn host_print(ptr: *const u8, len: u64) {
    // SAFETY: generated code passes a pointer to `len` valid bytes.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    println!("{}", String::from_utf8_lossy(bytes));
}

// Rust types of the exports, matching their IR signatures.
type HelloFn = unsafe extern "C" fn(i32) -> i32;
type SumToFn = unsafe extern "C" fn(i64) -> i64;

/// Converts an `LLVMErrorRef` into a `Result`, consuming the error.
fn check(err: LLVMErrorRef) -> Result<(), String> {
    if err.is_null() {
        return Ok(());
    }
    // SAFETY: `err` is a non-null error from LLVM; getting its message consumes it,
    // and the message is a valid C string until we dispose of it.
    unsafe {
        let msg = LLVMGetErrorMessage(err);
        let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
        LLVMDisposeErrorMessage(msg);
        Err(s)
    }
}

/// Copies an LLVM-allocated message into a `String` and frees it.
///
/// # Safety
/// `msg` must be null or a message allocated by LLVM that hasn't been freed.
unsafe fn take_message(msg: *mut c_char) -> String {
    if msg.is_null() {
        return "unknown error".into();
    }
    unsafe {
        let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
        LLVMDisposeMessage(msg);
        s
    }
}

/// A target machine for the exact CPU we're running on.
///
/// # Safety
/// `triple` must be a valid C string, and the native target must be initialized.
unsafe fn host_target_machine(
    triple: *const c_char,
    level: LLVMCodeGenOptLevel,
) -> Result<LLVMTargetMachineRef, String> {
    unsafe {
        let mut target: LLVMTargetRef = ptr::null_mut();
        let mut err: *mut c_char = ptr::null_mut();
        if LLVMGetTargetFromTriple(triple, &mut target, &mut err) != 0 {
            return Err(take_message(err));
        }

        let cpu = LLVMGetHostCPUName();
        let features = LLVMGetHostCPUFeatures();
        let tm = LLVMCreateTargetMachine(
            target,
            triple,
            cpu,
            features,
            level,
            LLVMRelocMode::LLVMRelocDefault,
            LLVMCodeModel::LLVMCodeModelJITDefault,
        );
        LLVMDisposeMessage(cpu);
        LLVMDisposeMessage(features);

        if tm.is_null() {
            Err("failed to create target machine".into())
        } else {
            Ok(tm)
        }
    }
}

/// Parses textual IR into a module owned by `ctx`.
///
/// # Safety
/// `ctx` must be a valid, live LLVM context.
unsafe fn parse_ir(ctx: LLVMContextRef, ir: &str, name: &CStr) -> Result<LLVMModuleRef, String> {
    unsafe {
        let buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            ir.as_ptr() as *const c_char,
            ir.len(),
            name.as_ptr(),
        );

        let mut module: LLVMModuleRef = ptr::null_mut();
        let mut msg: *mut c_char = ptr::null_mut();
        let failed = LLVMParseIRInContext2(ctx, buf, &mut module, &mut msg) != 0;
        LLVMDisposeMemoryBuffer(buf); // the "2" variant doesn't take ownership

        if failed {
            Err(format!("IR parse error: {}", take_message(msg)))
        } else {
            Ok(module)
        }
    }
}

/// Runs an IR optimization pipeline over the whole module.
///
/// # Safety
/// `module` and `tm` must be valid, live LLVM handles.
unsafe fn optimize(
    module: LLVMModuleRef,
    tm: LLVMTargetMachineRef,
    pipeline: &CStr,
) -> Result<(), String> {
    unsafe {
        let opts = LLVMCreatePassBuilderOptions();
        let result = check(LLVMRunPasses(module, pipeline.as_ptr(), tm, opts));
        LLVMDisposePassBuilderOptions(opts);
        result
    }
}

/// Prints a module's IR to stdout.
///
/// # Safety
/// `module` must be a valid, live LLVM module.
unsafe fn print_module(title: &str, module: LLVMModuleRef) {
    unsafe {
        let text = LLVMPrintModuleToString(module);
        println!(
            "===== {title} =====\n{}",
            CStr::from_ptr(text).to_string_lossy()
        );
        LLVMDisposeMessage(text);
    }
}

/// Defines one host function as an absolute symbol in the JIT.
///
/// # Safety
/// `jit` must be a valid, live JIT, and `addr` must be the address of a function
/// whose signature matches the symbol's declaration in the IR.
unsafe fn define_host_fn(jit: LLVMOrcLLJITRef, name: &CStr, addr: usize) -> Result<(), String> {
    unsafe {
        let mut pair = LLVMOrcCSymbolMapPair {
            // Applies the platform's symbol mangling (a leading `_` on macOS).
            Name: LLVMOrcLLJITMangleAndIntern(jit, name.as_ptr()),
            Sym: LLVMJITEvaluatedSymbol {
                Address: addr as u64,
                Flags: LLVMJITSymbolFlags {
                    GenericFlags: LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported
                        as u8
                        | LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsCallable as u8,
                    TargetFlags: 0,
                },
            },
        };
        let mu = LLVMOrcAbsoluteSymbols(&mut pair, 1);
        check(LLVMOrcJITDylibDefine(LLVMOrcLLJITGetMainJITDylib(jit), mu))
    }
}

/// Looks up an export and returns it as the function-pointer type `F`.
///
/// # Safety
/// `jit` must be a valid, live JIT. `F` must be the exact `unsafe extern "C" fn`
/// type of the export, and the returned pointer must not be used after the JIT is
/// disposed.
unsafe fn lookup<F: Copy>(jit: LLVMOrcLLJITRef, name: &CStr) -> Result<F, String> {
    assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<usize>());
    let mut addr: LLVMOrcExecutorAddress = 0;
    unsafe {
        check(LLVMOrcLLJITLookup(jit, &mut addr, name.as_ptr()))?;
        Ok(std::mem::transmute_copy(&(addr as usize)))
    }
}

fn main() -> Result<(), String> {
    // SAFETY: every LLVM handle below is created here, used only while live, and
    // each ownership transfer is noted where it happens.
    unsafe {
        // 1. Initialize code generation for the machine we're running on.
        if LLVM_InitializeNativeTarget() != 0 || LLVM_InitializeNativeAsmPrinter() != 0 {
            return Err("failed to initialize native target".into());
        }

        // 2. Create the JIT, with code generation at the aggressive level.
        let default_triple = LLVMGetDefaultTargetTriple();
        let jit_tm = host_target_machine(
            default_triple,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        );
        LLVMDisposeMessage(default_triple);
        let jit_tm = jit_tm?;

        let jtmb = LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(jit_tm); // takes jit_tm
        let builder = LLVMOrcCreateLLJITBuilder();
        LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(builder, jtmb); // takes jtmb

        let mut jit: LLVMOrcLLJITRef = ptr::null_mut();
        check(LLVMOrcCreateLLJIT(&mut jit, builder))?; // takes builder

        // 3. Parse the IR, then set the JIT's target so layouts match the host.
        let ctx = LLVMContextCreate();
        let module = match parse_ir(ctx, IR, c"hello_module") {
            Ok(m) => m,
            Err(e) => {
                LLVMContextDispose(ctx);
                return Err(e);
            }
        };

        let triple = LLVMOrcLLJITGetTripleString(jit); // owned by the JIT
        LLVMSetTarget(module, triple);
        LLVMSetDataLayout(module, LLVMOrcLLJITGetDataLayoutStr(jit));

        print_module("before optimization", module);

        // 4. Run the IR optimizer, using a target machine for the host CPU.
        let opt_tm = host_target_machine(triple, LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)?;
        let opt_result = optimize(module, opt_tm, PIPELINE);
        LLVMDisposeTargetMachine(opt_tm);
        opt_result?;

        print_module("after optimization", module);

        // 5. Hand the context to ORC, then the module to the JIT.
        let tsc = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(ctx); // takes ctx
        let tsm = LLVMOrcCreateNewThreadSafeModule(module, tsc);
        LLVMOrcDisposeThreadSafeContext(tsc); // the module keeps its own reference
        check(LLVMOrcLLJITAddLLVMIRModule(
            jit,
            LLVMOrcLLJITGetMainJITDylib(jit),
            tsm,
        ))?;

        // 6. Define the import before the first lookup triggers linking.
        define_host_fn(jit, c"host_print", host_print as *const () as usize)?;

        // 7. Look up the exports. The first lookup compiles and links the module.
        let hello: HelloFn = lookup(jit, c"hello")?;
        let sum_to: SumToFn = lookup(jit, c"sum_to")?;

        // 8. Call them.
        println!("===== running =====");
        println!("hello(41) returned {}", hello(41));
        println!("sum_to(10) returned {}", sum_to(10));
        println!("sum_to(1_000_000) returned {}", sum_to(1_000_000));

        // 9. Tear down. No function pointer from the JIT may be used after this.
        check(LLVMOrcDisposeLLJIT(jit))?;
    }

    Ok(())
}
