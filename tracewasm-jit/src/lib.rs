use crate::error::JITError;
use crate::func::LLVMHostFunc;
use llvm_sys::core::*;
use llvm_sys::error::*;
use llvm_sys::ir_reader::LLVMParseIRInContext2;
use llvm_sys::orc2::lljit::*;
use llvm_sys::orc2::*;
use llvm_sys::prelude::LLVMContextRef;
use llvm_sys::prelude::LLVMModuleRef;
use llvm_sys::prelude::LLVMTypeRef;
use llvm_sys::target::{LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget};
use llvm_sys::target_machine::*;
use llvm_sys::transforms::pass_builder::LLVMCreatePassBuilderOptions;
use llvm_sys::transforms::pass_builder::LLVMDisposePassBuilderOptions;
use llvm_sys::transforms::pass_builder::LLVMRunPasses;
use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::{CStr, c_char};
use std::ptr;

pub mod error;
pub mod func;

const PIPELINE: &CStr = c"default<O3>";

pub struct JITHandler {
    jit: LLVMOrcLLJITRef,
    // Host symbols live in the JIT's main dylib, shared by every module, so the
    // same name can only ever be bound to one function: name -> (address, type).
    host_funcs: RefCell<HashMap<String, (usize, TypeId)>>,
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

        Ok(JITHandler {
            jit,
            host_funcs: RefCell::new(HashMap::new()),
        })
    }

    /// Defines `name` in the JIT, or does nothing if it's already bound to this
    /// exact function.
    fn define_host_func(
        &self,
        name: &str,
        c_name: &CStr,
        addr: usize,
        ty: TypeId,
    ) -> Result<(), JITError> {
        let mut host_funcs = self.host_funcs.borrow_mut();

        if let Some(&existing) = host_funcs.get(name) {
            return if existing == (addr, ty) {
                Ok(())
            } else {
                Err(JITError::HostFuncAlreadyLinked(name.into()))
            };
        }

        unsafe {
            define_host_fn(self.jit, c_name, addr)?;
        }

        host_funcs.insert(name.into(), (addr, ty));

        Ok(())
    }

    pub fn parse_module(&self, name: &str, module_str: &str) -> Result<JITModule<'_>, JITError> {
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
///     jit.parse_module("m", "").unwrap()
/// }; // `jit` dropped here while `module` still borrows it
/// module.optimize(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap();
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

impl<'jit> JITModule<'jit> {
    pub fn optimize(&mut self, level: LLVMCodeGenOptLevel) -> Result<(), JITError> {
        unsafe {
            let triple = LLVMOrcLLJITGetTripleString(self.jit.jit); // owned by the JIT
            let opt_tm = host_target_machine(triple, level)?;
            let opt_result = optimize(self.module, opt_tm, PIPELINE);

            LLVMDisposeTargetMachine(opt_tm);

            opt_result
        }
    }

    /// Links a host function to the module's `declare` of the same name. The IR
    /// signature is inferred from `F` and must match the declaration exactly.
    ///
    /// A function item has to be coerced to a pointer first; `_` lets the compiler
    /// fill in the types:
    ///
    /// ```
    /// # use tracewasm_jit::JITHandler;
    /// # use llvm_sys::target_machine::LLVMCodeGenOptLevel;
    /// extern "C" fn add(a: i64, b: i64) -> i64 { a + b }
    ///
    /// let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap();
    /// let module = jit.parse_module("m", "declare i64 @add(i64, i64)").unwrap();
    ///
    /// module.link_host_func("add", add as extern "C" fn(_, _) -> _).unwrap();
    /// ```
    pub fn link_host_func<F: LLVMHostFunc>(&self, name: &str, func: F) -> Result<(), JITError> {
        let c_name = CString::new(name).map_err(|_| JITError::InvalidFuncName)?;

        unsafe {
            let decl = LLVMGetNamedFunction(self.module, c_name.as_ptr());

            // A body in the module would collide with the host symbol.
            if decl.is_null() || LLVMIsDeclaration(decl) == 0 {
                return Err(JITError::HostFuncNotDeclared(name.into()));
            }

            // Types are uniqued per context, so pointer equality is type equality.
            let declared = LLVMGlobalGetValueType(decl);
            let host = F::llvm_type(self.ctx);

            if declared != host {
                return Err(JITError::HostFuncSignatureMismatch {
                    name: name.into(),
                    declared: type_to_string(declared),
                    host: type_to_string(host),
                });
            }
        }

        self.jit
            .define_host_func(name, &c_name, func.addr(), TypeId::of::<F>())
    }
}

pub struct JITCompiledInstance<'jit> {
    jit: &'jit JITHandler,
}

impl<'jit> JITCompiledInstance<'jit> {
    pub fn get_func() -> Func {
        todo!()
    }
}

pub struct Func {}

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

/// Defines one host function as an absolute symbol in the JIT.
///
/// # Safety
/// `jit` must be a valid, live JIT, and `addr` must be the address of a function
/// whose signature matches the symbol's declaration in the IR.
unsafe fn define_host_fn(jit: LLVMOrcLLJITRef, name: &CStr, addr: usize) -> Result<(), JITError> {
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
        let err = LLVMOrcJITDylibDefine(LLVMOrcLLJITGetMainJITDylib(jit), mu);

        // On failure the dylib doesn't take `mu`, so it's still ours to free.
        if !err.is_null() {
            LLVMOrcDisposeMaterializationUnit(mu);
        }

        check(err)
    }
}

/// Renders an IR type as text, e.g. `i64 (i64, i64)`.
///
/// # Safety
/// `ty` must be a valid, live LLVM type.
unsafe fn type_to_string(ty: LLVMTypeRef) -> String {
    unsafe { take_message(LLVMPrintTypeToString(ty)) }
}
