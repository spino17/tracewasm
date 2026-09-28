use crate::error::JITError;
use crate::func::{Func, LLVMFuncParams, LLVMFuncResult, LLVMHostFunc, fn_type};
use llvm_sys::LLVMLinkage;
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
use std::sync::OnceLock;

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
        // Registering targets mutates a global list without locking, so racing
        // `new` calls on different threads can corrupt it. Do it exactly once.
        static NATIVE_TARGET: OnceLock<bool> = OnceLock::new();

        let initialized = *NATIVE_TARGET.get_or_init(|| unsafe {
            LLVM_InitializeNativeTarget() == 0 && LLVM_InitializeNativeAsmPrinter() == 0
        });

        if !initialized {
            return Err(JITError::InitializationError);
        }

        let jit = unsafe {
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

    /// Hands the module to the JIT. Code is generated lazily, on the first
    /// `get_func`; every host function the module declares must be linked first.
    ///
    /// # Safety
    /// Calling a compiled function runs the module's code, which the type system
    /// can't see into: that code must not have undefined behaviour for any
    /// arguments it can be called with.
    pub unsafe fn compile(self) -> Result<JITCompiledInstance<'jit>, JITError> {
        // Recorded now, because ORC owns the module (and may free it) from here on.
        let exports = unsafe { exported_signatures(self.module) };

        // ORC takes the context and module, so our `Drop` must not run.
        let this = std::mem::ManuallyDrop::new(self);

        unsafe {
            let tsc = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(this.ctx); // takes ctx
            let tsm = LLVMOrcCreateNewThreadSafeModule(this.module, tsc); // takes module

            LLVMOrcDisposeThreadSafeContext(tsc); // the module keeps its own reference

            // Takes `tsm` even on failure.
            check(LLVMOrcLLJITAddLLVMIRModule(
                this.jit.jit,
                LLVMOrcLLJITGetMainJITDylib(this.jit.jit),
                tsm,
            ))?;
        }

        Ok(JITCompiledInstance {
            jit: this.jit,
            exports,
        })
    }
}

/// A module that has been handed to the JIT. Its compiled code lives as long as
/// the `JITHandler`.
pub struct JITCompiledInstance<'jit> {
    jit: &'jit JITHandler,
    // Exported function name -> its IR signature, as text.
    exports: HashMap<String, String>,
}

impl<'jit> JITCompiledInstance<'jit> {
    /// Looks up an exported function taking the parameter tuple `P` and returning
    /// `R`, which must match its IR definition exactly. The first lookup generates the module's code.
    ///
    /// ```
    /// # use tracewasm_jit::JITHandler;
    /// # use llvm_sys::target_machine::LLVMCodeGenOptLevel;
    /// let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap();
    /// let ir = "define i64 @add(i64 %a, i64 %b) {\n  %r = add i64 %a, %b\n  ret i64 %r\n}";
    /// let module = jit.parse_module("m", ir).unwrap();
    /// // SAFETY: `add` is defined for all inputs.
    /// let instance = unsafe { module.compile() }.unwrap();
    ///
    /// let add = instance.get_func::<(i64, i64), i64>("add").unwrap();
    ///
    /// assert_eq!(add.call(2, 3), 5);
    /// ```
    ///
    /// `call` only accepts the looked-up signature:
    ///
    /// ```compile_fail,E0308
    /// # use tracewasm_jit::JITHandler;
    /// # use llvm_sys::target_machine::LLVMCodeGenOptLevel;
    /// # let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap();
    /// # let ir = "define i64 @add(i64 %a, i64 %b) {\n  %r = add i64 %a, %b\n  ret i64 %r\n}";
    /// # let instance = unsafe { jit.parse_module("m", ir).unwrap().compile() }.unwrap();
    /// let add = instance.get_func::<(i64, i64), i64>("add").unwrap();
    ///
    /// add.call(2.0_f64, 3);
    /// ```
    ///
    /// and can't outlive the instance:
    ///
    /// ```compile_fail,E0597
    /// # use tracewasm_jit::JITHandler;
    /// # use llvm_sys::target_machine::LLVMCodeGenOptLevel;
    /// # let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap();
    /// # let ir = "define i64 @add(i64 %a, i64 %b) {\n  %r = add i64 %a, %b\n  ret i64 %r\n}";
    /// let add = {
    ///     let instance = unsafe { jit.parse_module("m", ir).unwrap().compile() }.unwrap();
    ///     instance.get_func::<(i64, i64), i64>("add").unwrap()
    /// }; // `instance` dropped here while `add` still borrows it
    ///
    /// add.call(2, 3);
    /// ```
    pub fn get_func<P: LLVMFuncParams, R: LLVMFuncResult>(
        &self,
        name: &str,
    ) -> Result<Func<'_, P, R>, JITError> {
        let declared = self
            .exports
            .get(name)
            .ok_or_else(|| JITError::FuncNotExported(name.into()))?;
        let requested = signature_of::<P, R>();

        if *declared != requested {
            return Err(JITError::FuncSignatureMismatch {
                name: name.into(),
                declared: declared.clone(),
                requested,
            });
        }

        let c_name = CString::new(name).map_err(|_| JITError::InvalidFuncName)?;
        let mut addr: LLVMOrcExecutorAddress = 0;

        unsafe {
            // Applies the platform's symbol mangling itself.
            check(LLVMOrcLLJITLookup(self.jit.jit, &mut addr, c_name.as_ptr()))?;

            // SAFETY: the signature matches the IR definition, and the code lives
            // in the JIT, which outlives `self`.
            Ok(Func::from_addr(addr as usize))
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

/// The IR signatures of the functions a module defines with external linkage,
/// which are the only ones the JIT can look up.
///
/// # Safety
/// `module` must be a valid, live LLVM module.
unsafe fn exported_signatures(module: LLVMModuleRef) -> HashMap<String, String> {
    let mut exports = HashMap::new();

    unsafe {
        let mut f = LLVMGetFirstFunction(module);

        while !f.is_null() {
            if LLVMIsDeclaration(f) == 0 && LLVMGetLinkage(f) == LLVMLinkage::LLVMExternalLinkage {
                let mut len = 0;
                let name = LLVMGetValueName2(f, &mut len);
                let name = std::slice::from_raw_parts(name as *const u8, len);

                exports.insert(
                    String::from_utf8_lossy(name).into_owned(),
                    type_to_string(LLVMGlobalGetValueType(f)),
                );
            }

            f = LLVMGetNextFunction(f);
        }
    }

    exports
}

/// The IR signature taking `P` and returning `R`, as text. Types from different contexts can't be compared
/// directly, so signatures are compared by how they print.
fn signature_of<P: LLVMFuncParams, R: LLVMFuncResult>() -> String {
    // SAFETY: the context is created, used and disposed of here.
    unsafe {
        let ctx = LLVMContextCreate();
        let s = type_to_string(fn_type::<P, R>(ctx));

        LLVMContextDispose(ctx);

        s
    }
}

/// Renders an IR type as text, e.g. `i64 (i64, i64)`.
///
/// # Safety
/// `ty` must be a valid, live LLVM type.
unsafe fn type_to_string(ty: LLVMTypeRef) -> String {
    unsafe { take_message(LLVMPrintTypeToString(ty)) }
}
