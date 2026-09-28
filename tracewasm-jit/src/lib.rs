use crate::error::JITError;
use llvm_sys::core::*;
use llvm_sys::error::*;
use llvm_sys::ir_reader::LLVMParseIRInContext2;
use llvm_sys::orc2::lljit::*;
use llvm_sys::orc2::*;
use llvm_sys::prelude::LLVMContextRef;
use llvm_sys::prelude::LLVMModuleRef;
use llvm_sys::target::{LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget};
use llvm_sys::target_machine::*;
use llvm_sys::transforms::pass_builder::LLVMCreatePassBuilderOptions;
use llvm_sys::transforms::pass_builder::LLVMDisposePassBuilderOptions;
use llvm_sys::transforms::pass_builder::LLVMRunPasses;
use std::ffi::CString;
use std::ffi::{CStr, c_char};
use std::ptr;

pub mod error;

const PIPELINE: &CStr = c"default<O3>";

pub struct JITHandler {
    jit: LLVMOrcLLJITRef,
}

impl Drop for JITHandler {
    fn drop(&mut self) {
        // Nothing to report an error to from `drop`, but it must still be consumed.
        let _ = check(unsafe { LLVMOrcDisposeLLJIT(self.jit) });
    }
}

impl JITHandler {
    pub fn new(level: LLVMCodeGenOptLevel) -> Result<Self, JITError> {
        let jit = unsafe {
            if LLVM_InitializeNativeTarget() != 0 || LLVM_InitializeNativeAsmPrinter() != 0 {
                return Err(JITError::InitializationError);
            }

            let default_triple = LLVMGetDefaultTargetTriple();
            let jit_tm = host_target_machine(default_triple, level);

            LLVMDisposeMessage(default_triple);

            let jit_tm = jit_tm?;
            let jtmb = LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(jit_tm); // takes jit_tm
            let builder = LLVMOrcCreateLLJITBuilder();

            LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(builder, jtmb); // takes jtmb

            let mut jit: LLVMOrcLLJITRef = ptr::null_mut();

            check(LLVMOrcCreateLLJIT(&mut jit, builder))?; // takes builder

            jit
        };

        Ok(JITHandler { jit })
    }

    pub fn jit_module(&self, name: &str, module_str: &str) -> Result<JITModule<'_>, JITError> {
        let c_name = CString::new(name).map_err(|_| JITError::InvalidModuleName)?;

        let (ctx, module) = unsafe {
            let ctx = LLVMContextCreate();

            let module = match parse_ir(ctx, module_str, &c_name) {
                Ok(m) => m,
                Err(e) => {
                    LLVMContextDispose(ctx);

                    return Err(e);
                }
            };

            LLVMSetTarget(module, LLVMOrcLLJITGetTripleString(self.jit));
            LLVMSetDataLayout(module, LLVMOrcLLJITGetDataLayoutStr(self.jit));

            (ctx, module)
        };

        Ok(JITModule {
            ctx,
            module,
            jit: self,
        })
    }
}

/// A parsed module that owns its LLVM context. Borrows the `JITHandler` it came
/// from, so it can't outlive the JIT whose triple and data layout it uses.
///
/// ```compile_fail,E0597
/// # use tracewasm_jit::JITHandler;
/// # use llvm_sys::target_machine::LLVMCodeGenOptLevel;
/// let mut module = {
///     let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap();
///     jit.jit_module("m", "").unwrap()
/// }; // `jit` dropped here while `module` still borrows it
/// module.optimize().unwrap();
/// ```
pub struct JITModule<'jit> {
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
    jit: &'jit JITHandler,
}

impl Drop for JITModule<'_> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeModule(self.module);
            LLVMContextDispose(self.ctx);
        }
    }
}

impl JITModule<'_> {
    pub fn optimize(&mut self) -> Result<(), JITError> {
        unsafe {
            let triple = LLVMOrcLLJITGetTripleString(self.jit.jit); // owned by the JIT
            let opt_tm =
                host_target_machine(triple, LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)?;
            let opt_result = optimize(self.module, opt_tm, PIPELINE);

            LLVMDisposeTargetMachine(opt_tm);

            opt_result
        }
    }
}

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
) -> Result<LLVMTargetMachineRef, JITError> {
    unsafe {
        let mut target: LLVMTargetRef = ptr::null_mut();
        let mut err: *mut c_char = ptr::null_mut();

        if LLVMGetTargetFromTriple(triple, &mut target, &mut err) != 0 {
            return Err(JITError::LLVMError(take_message(err)));
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
            Err(JITError::LLVMError(
                "failed to create target machine".to_string(),
            ))
        } else {
            Ok(tm)
        }
    }
}

/// Converts an `LLVMErrorRef` into a `Result`, consuming the error.
fn check(err: LLVMErrorRef) -> Result<(), JITError> {
    if err.is_null() {
        return Ok(());
    }

    // SAFETY: `err` is a non-null error from LLVM; getting its message consumes it,
    // and the message is a valid C string until we dispose of it.
    unsafe {
        let msg = LLVMGetErrorMessage(err);
        let s = CStr::from_ptr(msg).to_string_lossy().into_owned();

        LLVMDisposeErrorMessage(msg);

        Err(JITError::LLVMError(s))
    }
}

/// Parses textual IR into a module owned by `ctx`.
///
/// # Safety
/// `ctx` must be a valid, live LLVM context.
unsafe fn parse_ir(ctx: LLVMContextRef, ir: &str, name: &CStr) -> Result<LLVMModuleRef, JITError> {
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
            Err(JITError::LLVMError(format!(
                "IR parse error: {}",
                take_message(msg)
            )))
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
) -> Result<(), JITError> {
    unsafe {
        let opts = LLVMCreatePassBuilderOptions();
        let result = check(LLVMRunPasses(module, pipeline.as_ptr(), tm, opts));

        LLVMDisposePassBuilderOptions(opts);

        result
    }
}
