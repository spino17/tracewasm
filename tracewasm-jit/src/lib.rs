use crate::error::JITError;
use crate::func::{Func, LLVMFuncParams, LLVMFuncResult, LLVMHostFunc, fn_type};
use llvm_sys::analysis::{LLVMVerifierFailureAction, LLVMVerifyModule};
use llvm_sys::core::*;
use llvm_sys::error::*;
use llvm_sys::ir_reader::LLVMParseIRInContext2;
use llvm_sys::orc2::lljit::*;
use llvm_sys::orc2::*;
use llvm_sys::prelude::LLVMContextRef;
use llvm_sys::prelude::LLVMModuleRef;
use llvm_sys::prelude::LLVMTypeRef;
use llvm_sys::prelude::LLVMValueRef;
use llvm_sys::target::{LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget};
use llvm_sys::target_machine::*;
use llvm_sys::transforms::pass_builder::LLVMCreatePassBuilderOptions;
use llvm_sys::transforms::pass_builder::LLVMDisposePassBuilderOptions;
use llvm_sys::transforms::pass_builder::LLVMRunPasses;
use llvm_sys::{LLVMAttributeFunctionIndex, LLVMLinkage, LLVMVisibility};
use std::any::TypeId;
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::{CStr, c_char, c_void};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::OnceLock;
use std::sync::mpsc;

pub mod error;
pub mod func;

/// An IR optimization level, as in `opt -O<n>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptLevel {
    O0,
    O1,
    O2,
    O3,
    Os,
    Oz,
}

impl OptLevel {
    fn pipeline(self) -> &'static CStr {
        match self {
            OptLevel::O0 => c"default<O0>",
            OptLevel::O1 => c"default<O1>",
            OptLevel::O2 => c"default<O2>",
            OptLevel::O3 => c"default<O3>",
            OptLevel::Os => c"default<Os>",
            OptLevel::Oz => c"default<Oz>",
        }
    }

    // The level the pipeline's target machine reports to passes that query it.
    fn codegen_level(self) -> LLVMCodeGenOptLevel {
        match self {
            OptLevel::O0 => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            OptLevel::O1 => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
            OptLevel::O2 | OptLevel::Os | OptLevel::Oz => {
                LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault
            }
            OptLevel::O3 => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        }
    }
}

pub struct JITHandler {
    jit: LLVMOrcLLJITRef,
    // Every compiled module gets its own JITDylib, which needs a unique name.
    next_dylib: Cell<u64>,
}

impl Drop for JITHandler {
    fn drop(&mut self) {
        // Nothing to report an error to from `drop`, but it must still be consumed.
        let _ = check(unsafe { LLVMOrcDisposeLLJIT(self.jit) });
    }
}

impl JITHandler {
    /// Creates a JIT for the host machine. `level` is the backend's code
    /// generation level; IR optimization is chosen per module with
    /// [`JITModule::optimize`].
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
            next_dylib: Cell::new(0),
        })
    }

    /// Parses and verifies textual IR.
    ///
    /// A module that names no target gets the JIT's. One that does must name the
    /// JIT's architecture and exactly the JIT's data layout: code generated for
    /// another layout can't be retargeted by swapping the layout string.
    pub fn parse_module(&self, name: &str, module_str: &str) -> Result<JITModule<'_>, JITError> {
        let c_name = CString::new(name).map_err(|_| JITError::InvalidModuleName)?;

        let module = unsafe {
            let ctx = LLVMContextCreate();

            let module = match parse_ir(ctx, module_str, &c_name) {
                Ok(m) => m,
                Err(e) => {
                    LLVMContextDispose(ctx);

                    return Err(e);
                }
            };

            // From here on, `Drop` cleans up on every early return.
            JITModule {
                ctx,
                module,
                jit: self,
                host_funcs: HashMap::new(),
                optimized: false,
            }
        };

        unsafe {
            module.adopt_target()?;
            verify(module.module)?;
        }

        Ok(module)
    }

    fn fresh_dylib_name(&self) -> CString {
        let n = self.next_dylib.get();

        self.next_dylib.set(n + 1);

        CString::new(format!("module.{n}")).expect("no NUL in a formatted number")
    }
}

/// A parsed module that owns its LLVM context. Borrows the `JITHandler` it came
/// from, so it can't outlive the JIT whose triple and data layout it uses.
///
/// ```compile_fail,E0597
/// # use tracewasm_jit::{JITHandler, OptLevel};
/// # use llvm_sys::target_machine::LLVMCodeGenOptLevel;
/// let mut module = {
///     let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap();
///     jit.parse_module("m", "").unwrap()
/// }; // `jit` dropped here while `module` still borrows it
/// module.optimize(OptLevel::O2).unwrap();
/// ```
pub struct JITModule<'jit> {
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
    jit: &'jit JITHandler,
    // Linked host functions: name -> (address, Rust type).
    host_funcs: HashMap<String, (usize, TypeId)>,
    // Linking must come first; see `link_host_func`.
    optimized: bool,
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
    /// Runs the standard `default<O…>` IR pipeline, tuned for the host CPU. Link
    /// host functions first: [`link_host_func`](Self::link_host_func) refuses
    /// once the module has been optimized.
    pub fn optimize(&mut self, level: OptLevel) -> Result<(), JITError> {
        // Set even if the pipeline fails part-way: passes may already have run.
        self.optimized = true;

        unsafe {
            let triple = LLVMOrcLLJITGetTripleString(self.jit.jit); // owned by the JIT
            let opt_tm = host_target_machine(triple, level.codegen_level())?;
            let opt_result = optimize(self.module, opt_tm, level.pipeline());

            LLVMDisposeTargetMachine(opt_tm);

            opt_result
        }
    }

    /// Links a host function to the symbol `name`. If the module declares it, the
    /// IR signature is inferred from `F` and must match the declaration exactly;
    /// if it doesn't, nothing calls it and there is nothing to check.
    ///
    /// Linking must happen before [`optimize`](Self::optimize). The optimizer
    /// assumes a function named like a C library function (`abs`, `strlen`, …)
    /// behaves like it, and may fold or delete calls on that basis. Linking marks
    /// the declaration `nobuiltin` to stop that, which only helps if it's done
    /// before the optimizer runs.
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
    /// let mut module = jit.parse_module("m", "declare i64 @add(i64, i64)").unwrap();
    ///
    /// module.link_host_func("add", add as extern "C" fn(_, _) -> _).unwrap();
    /// ```
    pub fn link_host_func<F: LLVMHostFunc>(&mut self, name: &str, func: F) -> Result<(), JITError> {
        if self.optimized {
            return Err(JITError::HostFuncLinkedAfterOptimize(name.into()));
        }

        let c_name = CString::new(name).map_err(|_| JITError::InvalidFuncName)?;
        let linked = (func.addr(), TypeId::of::<F>());

        match self.host_funcs.get(name) {
            Some(&existing) if existing == linked => return Ok(()),
            Some(_) => return Err(JITError::HostFuncAlreadyLinked(name.into())),
            None => {}
        }

        unsafe {
            let decl = LLVMGetNamedFunction(self.module, c_name.as_ptr());

            if !decl.is_null() {
                // A body in the module would collide with the host symbol.
                if LLVMIsDeclaration(decl) == 0 {
                    return Err(JITError::HostFuncDefinedInModule(name.into()));
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

                let kind =
                    LLVMGetEnumAttributeKindForName(c"nobuiltin".as_ptr(), "nobuiltin".len());

                LLVMAddAttributeAtIndex(
                    decl,
                    LLVMAttributeFunctionIndex,
                    LLVMCreateEnumAttribute(self.ctx, kind, 0),
                );
            }
        }

        self.host_funcs.insert(name.into(), linked);

        Ok(())
    }

    /// Compiles the module into its own JITDylib, so the same IR can be compiled
    /// any number of times without its symbols colliding. Every exported function
    /// is compiled and linked here, so code generation and linking errors surface
    /// now rather than at `get_func`.
    ///
    /// Every symbol the module declares must be a linked host function (or an LLVM
    /// intrinsic). Library calls the backend introduces itself, such as `memcpy`
    /// for a large `llvm.memcpy`, resolve against the host process.
    ///
    /// Static constructors and destructors (`llvm.global_ctors`/`llvm.global_dtors`)
    /// are rejected: ORC's C API has no way to run them. The optimizer can often
    /// fold a constructor into the globals it initializes, which removes it.
    ///
    /// # Safety
    /// Calling a compiled function runs the module's code, which the type system
    /// can't see into: that code must not have undefined behaviour for any
    /// arguments it can be called with.
    pub unsafe fn compile(mut self) -> Result<JITCompiledInstance<'jit>, JITError> {
        unsafe {
            self.check_resolved()?;
        }

        // Taken out so nothing is left for `ManuallyDrop` below to leak.
        let host_funcs = std::mem::take(&mut self.host_funcs);

        // Recorded now, because ORC owns the module (and may free it) from here on.
        let exported = unsafe { exported_functions(self.module) };
        let jit = self.jit;
        let es = unsafe { LLVMOrcLLJITGetExecutionSession(jit.jit) };
        let mut jd: LLVMOrcJITDylibRef = ptr::null_mut();

        unsafe {
            check(LLVMOrcExecutionSessionCreateJITDylib(
                es,
                &mut jd,
                jit.fresh_dylib_name().as_ptr(),
            ))?;
        }

        // From here on, the instance's `Drop` frees whatever reached the dylib.
        let mut instance = JITCompiledInstance {
            jd,
            exports: HashMap::new(),
            _jit: PhantomData,
        };

        unsafe {
            let mut generator: LLVMOrcDefinitionGeneratorRef = ptr::null_mut();

            check(LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                &mut generator,
                LLVMOrcLLJITGetGlobalPrefix(jit.jit),
                None,
                ptr::null_mut(),
            ))?;

            LLVMOrcJITDylibAddGenerator(jd, generator); // takes generator

            for (name, &(addr, _)) in &host_funcs {
                let c_name = CString::new(name.as_str()).expect("checked by link_host_func");

                define_host_fn(jit.jit, jd, &c_name, addr)?;
            }
        }

        // ORC takes the context and module, so our `Drop` must not run.
        let this = ManuallyDrop::new(self);

        unsafe {
            let tsc = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(this.ctx); // takes ctx
            let tsm = LLVMOrcCreateNewThreadSafeModule(this.module, tsc); // takes module

            LLVMOrcDisposeThreadSafeContext(tsc); // the module keeps its own reference

            // Takes `tsm` even on failure.
            check(LLVMOrcLLJITAddLLVMIRModule(jit.jit, jd, tsm))?;
        }

        // One lookup for everything: it compiles and links the whole module.
        let names: Vec<&CStr> = exported
            .iter()
            .map(|(_, c_name, _)| c_name.as_c_str())
            .collect();
        let addrs = unsafe { lookup_all(jit.jit, jd, &names)? };

        for ((name, _, signature), addr) in exported.into_iter().zip(addrs) {
            instance.exports.insert(name, (signature, addr));
        }

        Ok(instance)
    }

    /// Sets the JIT's triple and data layout, or checks the module's against them.
    ///
    /// # Safety
    /// `self.module` must be live.
    unsafe fn adopt_target(&self) -> Result<(), JITError> {
        unsafe {
            let jit_triple = LLVMOrcLLJITGetTripleString(self.jit.jit);
            let jit_layout = LLVMOrcLLJITGetDataLayoutStr(self.jit.jit);
            let module_triple = CStr::from_ptr(LLVMGetTarget(self.module));
            let module_layout = CStr::from_ptr(LLVMGetDataLayoutStr(self.module));

            if !module_triple.is_empty()
                && platform(module_triple) != platform(CStr::from_ptr(jit_triple))
            {
                return Err(JITError::TripleMismatch {
                    module: module_triple.to_string_lossy().into_owned(),
                    jit: CStr::from_ptr(jit_triple).to_string_lossy().into_owned(),
                });
            }

            if !module_layout.is_empty() && module_layout != CStr::from_ptr(jit_layout) {
                return Err(JITError::DataLayoutMismatch {
                    module: module_layout.to_string_lossy().into_owned(),
                    jit: CStr::from_ptr(jit_layout).to_string_lossy().into_owned(),
                });
            }

            // Same architecture and OS; the spelling (vendor, OS version) is the JIT's.
            LLVMSetTarget(self.module, jit_triple);
            LLVMSetDataLayout(self.module, jit_layout);
        }

        Ok(())
    }

    /// Fails on the first declared symbol that nothing will define.
    ///
    /// # Safety
    /// `self.module` must be live.
    unsafe fn check_resolved(&self) -> Result<(), JITError> {
        unsafe {
            for name in [c"llvm.global_ctors", c"llvm.global_dtors"] {
                let list = LLVMGetNamedGlobal(self.module, name.as_ptr());

                // An empty list (what the optimizer can leave behind) runs nothing.
                if !list.is_null() && LLVMGetArrayLength2(LLVMGlobalGetValueType(list)) > 0 {
                    return Err(JITError::StaticInitializers);
                }
            }

            let mut f = LLVMGetFirstFunction(self.module);

            while !f.is_null() {
                if LLVMIsDeclaration(f) != 0 {
                    let name = value_name(f);

                    if !name.starts_with("llvm.") && !self.host_funcs.contains_key(&name) {
                        return Err(JITError::UnresolvedSymbol(name));
                    }
                }

                f = LLVMGetNextFunction(f);
            }

            let mut g = LLVMGetFirstGlobal(self.module);

            while !g.is_null() {
                if LLVMIsDeclaration(g) != 0 {
                    return Err(JITError::UnresolvedSymbol(value_name(g)));
                }

                g = LLVMGetNextGlobal(g);
            }
        }

        Ok(())
    }
}

/// A compiled module, in its own JITDylib. Dropping it frees its code, so every
/// [`Func`] borrows it.
pub struct JITCompiledInstance<'jit> {
    jd: LLVMOrcJITDylibRef,
    // Exported function name -> (its IR signature as text, its address).
    exports: HashMap<String, (String, LLVMOrcExecutorAddress)>,
    _jit: PhantomData<&'jit JITHandler>,
}

impl Drop for JITCompiledInstance<'_> {
    fn drop(&mut self) {
        // Frees the code and symbols. The C API can't remove the (now empty)
        // dylib itself; the session frees it along with the JIT.
        let _ = check(unsafe { LLVMOrcJITDylibClear(self.jd) });
    }
}

impl<'jit> JITCompiledInstance<'jit> {
    /// Looks up an exported function taking the parameter tuple `P` and returning
    /// `R`, which must match its IR definition exactly.
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
        let (declared, addr) = self
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

        // SAFETY: the signature matches the IR definition, and the code lives
        // until `self` is dropped, which the returned borrow prevents.
        Ok(unsafe { Func::from_addr(*addr as usize) })
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

/// Runs LLVM's verifier. The parser only checks syntax; passes and code
/// generation assume everything else (dominance, terminators, types) holds.
///
/// # Safety
/// `module` must be a valid, live LLVM module.
unsafe fn verify(module: LLVMModuleRef) -> Result<(), JITError> {
    unsafe {
        let mut msg: *mut c_char = ptr::null_mut();
        let broken = LLVMVerifyModule(
            module,
            LLVMVerifierFailureAction::LLVMReturnStatusAction,
            &mut msg,
        ) != 0;
        let msg = take_message(msg); // allocated even on success

        if broken {
            Err(JITError::InvalidModule(msg.trim_end().into()))
        } else {
            Ok(())
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

/// Defines one host function as an absolute symbol in `jd`.
///
/// # Safety
/// `jit` and `jd` must be live, and `addr` must be the address of a function
/// whose signature matches the symbol's declaration in the IR.
unsafe fn define_host_fn(
    jit: LLVMOrcLLJITRef,
    jd: LLVMOrcJITDylibRef,
    name: &CStr,
    addr: usize,
) -> Result<(), JITError> {
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
        let err = LLVMOrcJITDylibDefine(jd, mu);

        // On failure the dylib doesn't take `mu`, so it's still ours to free.
        if !err.is_null() {
            LLVMOrcDisposeMaterializationUnit(mu);
        }

        check(err)
    }
}

// Mangled name -> address, for every symbol a lookup found.
type LookupResult = Result<HashMap<String, LLVMOrcExecutorAddress>, JITError>;

/// Looks up `names` in `jd` in one go, compiling and linking whatever they depend
/// on. Returns their addresses in the same order.
///
/// # Safety
/// `jit` and `jd` must be live.
unsafe fn lookup_all(
    jit: LLVMOrcLLJITRef,
    jd: LLVMOrcJITDylibRef,
    names: &[&CStr],
) -> Result<Vec<LLVMOrcExecutorAddress>, JITError> {
    extern "C" fn on_result(
        err: LLVMErrorRef,
        pairs: LLVMOrcCSymbolMapPairs,
        len: usize,
        ctx: *mut c_void,
    ) {
        // SAFETY: `ctx` is the sender `lookup_all` boxed for this call, and ORC
        // calls this exactly once, so it's reclaimed exactly once.
        let tx = unsafe { Box::from_raw(ctx as *mut mpsc::Sender<LookupResult>) };

        let result = check(err).map(|()| {
            (0..len)
                .map(|i| {
                    // SAFETY: on success `pairs` holds `len` entries, live for this
                    // call, each naming a live pool entry.
                    unsafe {
                        let pair = &*pairs.add(i);
                        let name = CStr::from_ptr(LLVMOrcSymbolStringPoolEntryStr(pair.Name));

                        (name.to_string_lossy().into_owned(), pair.Sym.Address)
                    }
                })
                .collect()
        });

        let _ = tx.send(result);
    }

    if names.is_empty() {
        return Ok(Vec::new());
    }

    unsafe {
        let mut mangled = Vec::with_capacity(names.len());
        let mut symbols: Vec<LLVMOrcCLookupSetElement> = names
            .iter()
            .map(|name| {
                // Applies the platform's symbol mangling (a leading `_` on macOS).
                let entry = LLVMOrcLLJITMangleAndIntern(jit, name.as_ptr());

                mangled.push(
                    CStr::from_ptr(LLVMOrcSymbolStringPoolEntryStr(entry))
                        .to_string_lossy()
                        .into_owned(),
                );

                LLVMOrcCLookupSetElement {
                    // ORC takes this reference; we must not release it.
                    Name: entry,
                    LookupFlags: LLVMOrcSymbolLookupFlags::LLVMOrcSymbolLookupFlagsRequiredSymbol,
                }
            })
            .collect();
        let mut search = [LLVMOrcCJITDylibSearchOrderElement {
            JD: jd,
            JDLookupFlags:
                LLVMOrcJITDylibLookupFlags::LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly,
        }];
        let (tx, rx) = mpsc::channel::<LookupResult>();

        // The callback may run on another thread, possibly after this call returns,
        // so it owns the sender rather than borrowing it from this frame.
        LLVMOrcExecutionSessionLookup(
            LLVMOrcLLJITGetExecutionSession(jit),
            LLVMOrcLookupKind::LLVMOrcLookupKindStatic,
            search.as_mut_ptr(),
            search.len(),
            symbols.as_mut_ptr(),
            symbols.len(),
            on_result,
            Box::into_raw(Box::new(tx)) as *mut c_void,
        );

        let found = rx
            .recv()
            .unwrap_or_else(|_| Err(JITError::LLVMError("lookup never completed".into())))?;

        mangled
            .iter()
            .map(|name| {
                found
                    .get(name)
                    .copied()
                    .ok_or_else(|| JITError::LLVMError(format!("lookup didn't return `{name}`")))
            })
            .collect()
    }
}

/// The functions a module defines that the JIT can look up (externally visible
/// linkage, not `hidden`): (name, name for lookup, IR signature as text). Names that
/// aren't UTF-8 or contain NUL can't be requested by a `&str`, so they're skipped.
///
/// # Safety
/// `module` must be a valid, live LLVM module.
unsafe fn exported_functions(module: LLVMModuleRef) -> Vec<(String, CString, String)> {
    let mut exports = Vec::new();

    unsafe {
        let mut f = LLVMGetFirstFunction(module);

        while !f.is_null() {
            let linkage = LLVMGetLinkage(f);
            let visible = matches!(
                linkage,
                LLVMLinkage::LLVMExternalLinkage
                    | LLVMLinkage::LLVMWeakAnyLinkage
                    | LLVMLinkage::LLVMWeakODRLinkage
                    | LLVMLinkage::LLVMLinkOnceAnyLinkage
                    | LLVMLinkage::LLVMLinkOnceODRLinkage
            ) && LLVMGetVisibility(f) != LLVMVisibility::LLVMHiddenVisibility;

            // Hidden symbols aren't exported from their JITDylib, so a lookup
            // can't see them.
            if LLVMIsDeclaration(f) == 0 && visible {
                let mut len = 0;
                let bytes = LLVMGetValueName2(f, &mut len);
                let bytes = std::slice::from_raw_parts(bytes as *const u8, len);

                if let (Ok(name), Ok(c_name)) = (std::str::from_utf8(bytes), CString::new(bytes)) {
                    exports.push((
                        name.to_owned(),
                        c_name,
                        type_to_string(LLVMGlobalGetValueType(f)),
                    ));
                }
            }

            f = LLVMGetNextFunction(f);
        }
    }

    exports
}

/// A global value's name.
///
/// # Safety
/// `value` must be a valid, live LLVM value.
unsafe fn value_name(value: LLVMValueRef) -> String {
    unsafe {
        let mut len = 0;
        let bytes = LLVMGetValueName2(value, &mut len);

        String::from_utf8_lossy(std::slice::from_raw_parts(bytes as *const u8, len)).into_owned()
    }
}

/// A triple's (architecture, OS), which together fix the calling convention the
/// IR was written for. Aliases are folded together and OS versions dropped, so
/// `arm64-apple-macosx` and `aarch64-apple-darwin25.1.0` are the same platform.
fn platform(triple: &CStr) -> (String, String) {
    // SAFETY: `triple` is a valid C string; the result is ours to free.
    let normalized = unsafe { take_message(LLVMNormalizeTargetTriple(triple.as_ptr())) };
    let mut parts = normalized.split('-');

    let arch = match parts.next().unwrap_or("") {
        "arm64" => "aarch64",
        "amd64" => "x86_64",
        arch => arch,
    };
    let os = parts.nth(1).unwrap_or("");
    let os = match os.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.') {
        "macos" | "macosx" => "darwin",
        os => os,
    };

    (arch.to_owned(), os.to_owned())
}

/// The IR signature taking `P` and returning `R`, as text. Types from different
/// contexts can't be compared directly, so signatures are compared by how they print.
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
