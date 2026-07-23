//! The TraceWasm interpreter: a tree-walker over the flat, pre-lowered
//! instruction list produced by [`crate::instruction`].
//!
//! ## Execution model
//!
//! Each function body is a `Vec<Instruction>` with control flow already
//! resolved to absolute instruction indices and operand-stack heights
//! precomputed. Execution is a simple `pc` loop: `TraceVMState::execute` runs
//! one instruction and reports what the loop should do next via
//! `ExecutionResult` — advance (`Next`), jump (`JumpTo`), or (implicitly, by
//! advancing past the final `End`) return from the function.
//!
//! ## One shared operand stack across all frames
//!
//! Rather than giving every call its own operand stack, the whole call tree
//! shares a single `Stack`. A call does not allocate a new stack; the callee
//! simply builds its operands on top of the caller's. Recursion still uses the
//! native Rust call stack (one `TraceVM::execute` frame per active wasm call),
//! but the potentially-large value stack is allocated exactly once.
//!
//! ## Frame base height vs. recorded height
//!
//! Because the operand stack is shared, a frame's operands do not start at
//! absolute height 0 — they start at the height the stack had when the frame was
//! entered. That offset is the frame's **base height** (`caller_base_height`).
//!
//! The lowered instructions, however, store **frame-relative** heights
//! (`recorded_height`), computed as if the function ran on an empty stack. The
//! interpreter therefore converts relative → absolute at every height-sensitive
//! operation with a single rule:
//!
//! ```text
//! absolute_height = caller_base_height + recorded_height (+ arity)
//! ```
//!
//! A call establishes the callee's base as "current height minus the args"
//! (the args are popped off the shared stack and rebound as the callee's
//! locals), so the callee's results end up exactly where the caller's arguments
//! were. Instruction indices, by contrast, are per-function: each
//! `TraceVM::execute` invocation has its own `instructions` slice and `pc`.

use crate::{
    error::TraceWasmError,
    instance::traits::ImportRegistry,
    instruction::Instruction,
    memory::Memory,
    module::{FuncIndex, FuncKind, Module},
    utils::formatted_val_types,
    vm::stack::{Locals, Stack, Val},
};

pub(crate) mod stack;

/// What the driver loop should do after executing one instruction.
enum ExecutionResult {
    /// Set `pc` to this absolute (per-function) instruction index.
    JumpTo(usize),
    /// Advance to the next instruction (`pc + 1`); falling off the end of the
    /// function's instruction list ends the frame.
    Next,
}

/// The mutable state of a single in-flight function activation.
///
/// `stack`, `memory`, and `import_registry` are borrowed because they are shared
/// across the whole call tree (see the module docs); only `locals` is owned per
/// activation.
struct TraceVMState<'a, M, I> {
    /// The operand stack, shared with every other active frame.
    stack: &'a mut Stack<Val>,
    /// Linear memory, shared across the module.
    memory: &'a mut M,
    /// This activation's local slots (params followed by declared locals).
    locals: Locals,
    /// The registry resolving imported-function calls, shared across the call tree.
    import_registry: &'a mut I,
}

impl<'a, M: Memory, I: ImportRegistry> TraceVMState<'a, M, I> {
    /// Executes a single instruction against this activation's state and returns
    /// the control-flow decision for the driver loop.
    ///
    /// `func_index` identifies the running function (used only for error
    /// reporting). `caller_base_height` is this frame's base height on the shared
    /// stack; it is added to the instructions' frame-relative `recorded_height`
    /// to obtain absolute stack heights (see the module docs). `module` is needed
    /// to resolve callees on `Call`.
    fn execute(
        &mut self,
        instruction: &Instruction,
        func_index: FuncIndex,
        caller_base_height: u32,
        module: &Module,
    ) -> Result<ExecutionResult, TraceWasmError> {
        let res = match instruction {
            // TODO - give complete stack trace! - with range information of the
            // original WASM instruction
            Instruction::Unreachable => {
                return Err(TraceWasmError::Execution(
                    func_index.0,
                    "unreachable panic!".to_string(),
                ));
            }
            Instruction::Nop => ExecutionResult::Next,
            Instruction::Block {
                end_index: _end_index,
            } => ExecutionResult::Next,
            Instruction::Loop => ExecutionResult::Next,
            Instruction::If {
                else_index,
                end_index,
            } => {
                let cond = self.stack.pop().as_i32();

                if cond != 0 {
                    ExecutionResult::Next
                } else {
                    if let Some(else_index) = else_index {
                        ExecutionResult::JumpTo(*else_index + 1) // first instruction of the else branch
                    } else {
                        ExecutionResult::JumpTo(*end_index)
                    }
                }
            }
            // this instruction would be encountered only when control flow is coming after completing `if` branch
            // because if the condition was `false` and the control went to `else` branch, it jumps to the first
            // instruction of `else` branch and not the `else` instruction.
            Instruction::Else { if_end_index } => ExecutionResult::JumpTo(*if_end_index),
            Instruction::End {
                arity,
                recorded_height,
            } => {
                // Sanity check the height model: when a block closes, the stack must
                // hold exactly its `arity` results above the label's recorded height.
                // Both are frame-relative, so shift by this frame's base to compare
                // against the shared stack's absolute height.
                debug_assert!(
                    self.stack.height() == *recorded_height + *arity + caller_base_height
                );
                ExecutionResult::Next
            }
            Instruction::Br {
                target_index,
                arity,
                recorded_height,
            } => {
                // Unwind to the target label's absolute height (frame base + its
                // recorded height) while keeping the top `arity` values, then jump.
                self.stack
                    .truncate_by_preserving_arity(*recorded_height + caller_base_height, *arity);

                ExecutionResult::JumpTo(*target_index)
            }
            Instruction::BrIf {
                target_index,
                arity,
                recorded_height,
            } => {
                let cond = self.stack.pop().as_i32();

                if cond != 0 {
                    self.stack.truncate_by_preserving_arity(
                        *recorded_height + caller_base_height,
                        *arity,
                    );

                    ExecutionResult::JumpTo(*target_index)
                } else {
                    ExecutionResult::Next
                }
            }
            Instruction::BrTable { targets } => {
                // the branch index is an unsigned i32; go through u32 so a
                // high-bit-set value maps to a large index (→ default), not a
                // sign-extended one.
                let index = self.stack.pop().as_i32() as u32 as usize;
                let target_count = targets.len() - 1;

                let branch = if target_count <= index {
                    &targets[target_count] // always the last element of targets
                } else {
                    &targets[index]
                };

                self.stack.truncate_by_preserving_arity(
                    branch.recorded_height + caller_base_height,
                    branch.arity,
                );

                ExecutionResult::JumpTo(branch.target_index)
            }
            Instruction::Return {
                target_index,
                arity,
                recorded_height,
            } => {
                self.stack
                    .truncate_by_preserving_arity(*recorded_height + caller_base_height, *arity);

                ExecutionResult::JumpTo(*target_index)
            }
            Instruction::Call {
                func_index: callee_func_index,
                params_count,
            } => {
                // The callee's frame begins just below the arguments: everything up
                // to this height belongs to the caller and is left untouched. The
                // args are popped off the shared stack and rebound as the callee's
                // locals, so on return the callee's results occupy exactly the slots
                // the arguments did.
                let caller_base_height_for_callee = self.stack.height() - *params_count;
                let params = self.stack.pops_and_reverse(*params_count);
                let imported_func_count = module.imported_func_count;

                // Route on the *callee*: an imported callee is dispatched to the
                // registry; a local one is interpreted recursively.
                if callee_func_index.0 < imported_func_count {
                    let func_decl = &module.func_decls[callee_func_index.0 as usize];

                    debug_assert!(matches!(func_decl.kind, FuncKind::Imported { .. }));

                    let FuncKind::Imported {
                        module_name,
                        imported_func_name,
                    } = &func_decl.kind
                    else {
                        unreachable!()
                    };

                    let results =
                        self.import_registry
                            .execute(module_name, imported_func_name, &params)?;

                    // push results to the stack
                    for res in results {
                        self.stack.push(res);
                    }
                } else {
                    // local function execution
                    TraceVM::execute(
                        *callee_func_index,
                        &params,
                        module,
                        self.stack,
                        self.memory,
                        caller_base_height_for_callee,
                        self.import_registry,
                    )?;
                }

                ExecutionResult::Next
            }
        };

        Ok(res)
    }
}

pub(crate) struct TraceVM;

impl TraceVM {
    /// Runs one (locally-defined) function to completion on the shared stack.
    ///
    /// Called both as the top-level entry (with an empty stack and
    /// `caller_base_height == 0`) and recursively from the `Call` instruction.
    /// Arguments arrive in `params` (declaration order); results are left on
    /// `stack` above `caller_base_height` for the caller to consume — this
    /// function returns `()`, not the results.
    ///
    /// `caller_base_height` is the height the shared stack had on entry, i.e.
    /// this frame's base; see the module docs for how it maps the instructions'
    /// frame-relative heights onto the shared stack.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::IncorrectParamsResultsStructure`] if `params`
    /// don't match the function's signature, [`TraceWasmError::Execution`] on a
    /// trap (`unreachable`), and propagates errors from nested calls (including
    /// imported-function calls).
    pub fn execute<M: Memory, I: ImportRegistry>(
        func_index: FuncIndex,
        params: &[Val],
        module: &Module,
        stack: &mut Stack<Val>,
        memory: &mut M,
        caller_base_height: u32,
        import_registry: &mut I,
    ) -> Result<(), TraceWasmError> {
        // `func_bodies` holds only locally-defined functions, so shift the global
        // function index down by the number of imports to index into it.
        let imported_func_count = module.imported_func_count;

        debug_assert!(func_index.0 >= imported_func_count);

        let func_decl = &module.func_decls[func_index.0 as usize];
        let ty = &module.types[func_decl.ty.0 as usize];
        let params_ty = &ty.params;
        let func_body = &module.func_bodies[(func_index.0 - imported_func_count) as usize];
        let instructions = &func_body.instructions;
        // `locals` in the body is laid out params-first, then declared locals,
        // and `locals_ty[i]` is the declared type of local slot `i`.
        let locals_ty = &func_body.locals;

        // The caller must supply exactly one argument per declared parameter
        if params.len() != params_ty.len() {
            return Err(TraceWasmError::IncorrectParamsResultsStructure(
                "params".to_string(),
                func_index.0,
                formatted_val_types(params_ty),
                format!("{:?}", params),
            ));
        }

        // their types should also match
        for i in 0..params_ty.len() {
            let ty = params_ty[i];
            let param = params[i];

            if !param.has_ty(ty)? {
                return Err(TraceWasmError::IncorrectParamsResultsStructure(
                    "params".to_string(),
                    func_index.0,
                    formatted_val_types(params_ty),
                    format!("{:?}", params),
                ));
            }
        }

        // Build the activation's local slots. Per the WebAssembly spec, a
        // function's locals are the parameters (bound to the incoming arguments,
        // in order) followed by the declared locals.
        let mut locals: Vec<Val> = Vec::with_capacity(locals_ty.len());

        // Parameters occupy the first `params.len()` slots. Their count and types
        // were already validated above, so take the values as-is.
        locals.extend_from_slice(params);

        // The remaining declared locals are default-initialized: the spec requires
        // each to start at the zero value of its type (0 / 0.0 / null ref).
        for i in params.len()..locals_ty.len() {
            let ty = locals_ty[i];

            locals.push(Val::zero_of_ty(ty)?);
        }

        let mut state = TraceVMState {
            stack,
            memory,
            locals: Locals::new(locals),
            import_registry,
        };

        // Driver loop. `pc` indexes this function's instruction list only.
        let mut pc = 0;

        loop {
            let instr = &instructions[pc];

            match state.execute(instr, func_index, caller_base_height, module)? {
                ExecutionResult::JumpTo(next_pc) => {
                    pc = next_pc;

                    continue;
                }
                ExecutionResult::Next => {
                    pc += 1;

                    // Advancing past the last instruction means we just executed
                    // the function's terminating `End`: the frame is complete.
                    // (`return` and branches out of the outermost block also land
                    // here, since their target is that final `End`.)
                    if pc == instructions.len() {
                        break;
                    }

                    continue;
                }
            }
        }

        // The frame's results are now the top values on the shared stack, sitting
        // above `caller_base_height`; the caller reads them from there.
        Ok(())
    }

    /// Top-level entry point: runs a locally-defined function on a fresh operand
    /// stack and returns its result values (in declaration order).
    ///
    /// This is the wrapper around [`Self::execute`] that owns the shared stack
    /// and extracts the results, which `execute` itself leaves on the stack.
    ///
    /// # Errors
    ///
    /// Propagates any [`TraceWasmError`] from execution (traps, argument/result
    /// mismatches, errors returned by imported functions, …).
    pub(crate) fn run<M: Memory, I: ImportRegistry>(
        func_index: FuncIndex,
        params: &[Val],
        module: &Module,
        memory: &mut M,
        import_registry: &mut I,
    ) -> Result<Box<[Val]>, TraceWasmError> {
        let mut stack: Stack<Val> = Stack::default();

        // A fresh stack starts at height 0, so this frame's base is 0.
        Self::execute(
            func_index,
            params,
            module,
            &mut stack,
            memory,
            0,
            import_registry,
        )?;

        // How many result values the function leaves on the stack.
        let func_decl = &module.func_decls[func_index.0 as usize];
        let results_len = module.types[func_decl.ty.0 as usize].results.len() as u32;

        Ok(stack.pops_and_reverse(results_len).into_boxed_slice())
    }
}
