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
    error::{
        CallIndirectError::{self, FunctionCall},
        InstructionExecutionError, MemoryError, TraceWasmError,
    },
    instance::traits::{ImportRegistry, ResultVals},
    instruction::Instruction,
    memory::Memory,
    module::{FuncIndex, FuncKind, LocalIndex, Module, formatted_val_types},
    vm::stack::{Stack, TableVal, Val},
};
use smallvec::{SmallVec, smallvec};

pub(crate) mod stack;

/// A function activation's local slots: its parameters followed by its declared
/// locals, addressed by `local.get`/`local.set` index.
pub(crate) struct Locals {
    inner: SmallVec<[Val; 16]>, // size = params + declared locals
}

impl Locals {
    /// Wraps a fully-populated slot vector (params first, then zero-initialized
    /// declared locals). The caller owns getting the length and contents right.
    pub fn new(locals: SmallVec<[Val; 16]>) -> Self {
        Locals { inner: locals }
    }

    /// Writes `val` into the slot addressed by `index`.
    ///
    /// Panics if `index` is out of range; validation guarantees in-range indices
    /// for well-formed modules.
    pub fn set(&mut self, index: LocalIndex, val: Val) {
        self.inner[index.0 as usize] = val;
    }

    /// Reads the value in the slot addressed by `index` (values are `Copy`).
    ///
    /// Panics if `index` is out of range; validation guarantees in-range indices
    /// for well-formed modules.
    pub fn get(&self, index: LocalIndex) -> Val {
        self.inner[index.0 as usize]
    }
}

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
/// `stack`, `memory`, `import_registry`, `globals`, and `tables` are borrowed
/// because they are shared across the whole call tree (see the module docs);
/// only `locals` is owned per activation.
struct TraceVMState<'a, M, I> {
    /// The operand stack, shared with every other active frame.
    stack: &'a mut Stack<Val>,
    /// Linear memory, shared across the module.
    memory: &'a mut M,
    /// This activation's local slots (params followed by declared locals).
    locals: Locals,
    /// The registry resolving imported-function calls, shared across the call tree.
    import_registry: &'a mut I,
    /// The module's global values, shared across the call tree.
    globals: &'a mut [Val],
    /// The module's tables, shared across the call tree.
    tables: &'a mut Vec<TableVal>,
}

impl<'a, M: Memory, I: ImportRegistry> TraceVMState<'a, M, I> {
    fn call_func(
        &mut self,
        func_index: FuncIndex,
        params_count: u32,
        module: &Module,
    ) -> Result<(), TraceWasmError> {
        // The callee's frame begins just below the arguments: everything up
        // to this height belongs to the caller and is left untouched. The
        // args are popped off the shared stack and rebound as the callee's
        // locals, so on return the callee's results occupy exactly the slots
        // the arguments did.
        let caller_base_height_for_callee = self.stack.height() - params_count;
        let params = self.stack.pop_params(params_count);
        let imported_func_count = module.imported_func_count;

        // Route on the *callee*: an imported callee is dispatched to the
        // registry; a local one is interpreted recursively.
        if func_index.0 < imported_func_count {
            let func_decl = &module.func_decls[func_index.0 as usize];

            debug_assert!(matches!(func_decl.kind, FuncKind::Imported { .. }));

            let FuncKind::Imported {
                module_name,
                imported_func_name,
            } = &func_decl.kind
            else {
                unreachable!()
            };

            // `execute` returns a stack-allocated `ResultVals` (no heap for <=3 results).
            let results =
                self.import_registry
                    .execute(module_name, imported_func_name, params.as_ref())?;

            // push results to the stack
            for res in results {
                self.stack.push(res);
            }
        } else {
            // local function execution
            TraceVM::execute(
                func_index,
                params.as_ref(),
                module,
                self.stack,
                self.memory,
                caller_base_height_for_callee,
                self.import_registry,
                self.globals,
                self.tables,
            )?;
        }

        Ok(())
    }

    /// Pops the address operand of a memory access and resolves it to an
    /// effective address by adding the instruction's static `memarg` offset.
    ///
    /// Shared by every load/store arm, so they all inherit the same overflow and
    /// offset-range trapping.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::OffsetTooLarge`] if the static offset does not fit
    /// a 32-bit address space, and [`MemoryError::EffectiveAddressOverflow`] if
    /// the sum leaves it. The access itself is bounds-checked by [`Memory`].
    fn pop_effective_address(&mut self, memarg_offset: u64) -> Result<usize, MemoryError> {
        let addr = self.stack.pop().as_i32() as u32;
        let static_offset =
            u32::try_from(memarg_offset).map_err(|_| MemoryError::OffsetTooLarge)?;

        // Effective address = popped address + static offset, computed with
        // a checked add: a u32 overflow is past the 32-bit memory space, so
        // it traps rather than wrapping to a wrong (in-bounds) address.
        let effective_offset = addr
            .checked_add(static_offset)
            .ok_or(MemoryError::EffectiveAddressOverflow(addr, static_offset))?;

        Ok(effective_offset as usize)
    }

    /// Executes a single instruction against this activation's state and returns
    /// the control-flow decision for the driver loop.
    ///
    /// `caller_base_height` is this frame's base height on the shared stack; it is
    /// added to the instructions' frame-relative `recorded_height` to obtain
    /// absolute stack heights (see the module docs). `module` is needed to resolve
    /// callees on `Call`. Errors are returned bare as
    /// [`InstructionExecutionError`]; the driver loop tags them with the enclosing
    /// function and instruction index.
    fn execute(
        &mut self,
        instruction: &Instruction,
        caller_base_height: u32,
        module: &Module,
    ) -> Result<ExecutionResult, InstructionExecutionError> {
        let res = match instruction {
            // TODO - give complete stack trace! - with range information of the
            // original WASM instruction
            Instruction::Unreachable => {
                return Err(InstructionExecutionError::Unreachable);
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
                self.call_func(*callee_func_index, *params_count, module)
                    .map_err(|err| {
                        InstructionExecutionError::Call(*callee_func_index, Box::new(err))
                    })?;

                ExecutionResult::Next
            }
            Instruction::CallIndirect {
                params,
                results,
                table_index,
            } => {
                let table = &self.tables[table_index.0 as usize];
                // The index operand is an unsigned i32; a negative value becomes a
                // large `usize` and fails the bounds check below.
                let slot = self.stack.pop().as_i32() as u32 as usize;

                // Trap if the index is past the table's end (wasm: "undefined element").
                let Some(func_ref) = table.table.get(slot).copied() else {
                    return Err(InstructionExecutionError::CallIndirect(
                        *table_index,
                        CallIndirectError::TableSlotOutOfBounds,
                    ));
                };

                // Trap on a null element (wasm: "uninitialized element").
                let Some(callee_func_index) = func_ref else {
                    return Err(InstructionExecutionError::CallIndirect(
                        *table_index,
                        CallIndirectError::NullElementInTable,
                    ));
                };

                let func = &module.func_decls[callee_func_index.0 as usize];
                let ty = &module.types[func.ty.0 as usize];

                let declared_params = &ty.params;
                let declared_results = &ty.results;

                // Trap if the callee's signature differs from the type the
                // instruction expects (wasm: "indirect call type mismatch").
                if params.as_ref() != declared_params.as_ref()
                    || results.as_ref() != declared_results.as_ref()
                {
                    return Err(InstructionExecutionError::CallIndirect(
                        *table_index,
                        CallIndirectError::FunctionSignatureMismatch(
                            format!(
                                "{} -> {}",
                                formatted_val_types(declared_params),
                                formatted_val_types(declared_results)
                            ),
                            format!(
                                "{} -> {}",
                                formatted_val_types(params),
                                formatted_val_types(results)
                            ),
                        ),
                    ));
                }

                self.call_func(callee_func_index, declared_params.len() as u32, module)
                    .map_err(|err| {
                        InstructionExecutionError::CallIndirect(
                            *table_index,
                            FunctionCall(callee_func_index, Box::new(err)),
                        )
                    })?;

                ExecutionResult::Next
            }
            Instruction::I32Const { value } => {
                self.stack.push(Val::I32(*value));

                ExecutionResult::Next
            }
            Instruction::I64Const { value } => {
                self.stack.push(Val::I64(*value));

                ExecutionResult::Next
            }
            Instruction::F32Const { value } => {
                self.stack.push(Val::F32(*value));

                ExecutionResult::Next
            }
            Instruction::F64Const { value } => {
                self.stack.push(Val::F64(*value));

                ExecutionResult::Next
            }
            Instruction::RefNull => {
                self.stack.push(Val::Ref(None));

                ExecutionResult::Next
            }
            Instruction::RefFunc { function_index } => {
                self.stack.push(Val::Ref(Some(*function_index)));

                ExecutionResult::Next
            }
            Instruction::I32Add => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.wrapping_add(b)));

                ExecutionResult::Next
            }
            Instruction::I32Sub => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.wrapping_sub(b)));

                ExecutionResult::Next
            }
            Instruction::I32Mul => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.wrapping_mul(b)));

                ExecutionResult::Next
            }
            Instruction::I64Add => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.wrapping_add(b)));

                ExecutionResult::Next
            }
            Instruction::I64Sub => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.wrapping_sub(b)));

                ExecutionResult::Next
            }
            Instruction::I64Mul => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.wrapping_mul(b)));

                ExecutionResult::Next
            }
            Instruction::Drop => {
                let _ = self.stack.pop();

                ExecutionResult::Next
            }
            Instruction::Select => {
                let cond = self.stack.pop().as_i32();
                let b = self.stack.pop();
                let a = self.stack.pop();

                // true condition
                if cond != 0 {
                    self.stack.push(a);
                } else {
                    self.stack.push(b);
                }

                ExecutionResult::Next
            }
            Instruction::LocalGet { index } => {
                self.stack.push(self.locals.get(*index));

                ExecutionResult::Next
            }
            Instruction::LocalSet { index } => {
                let val = self.stack.pop();

                self.locals.set(*index, val);

                ExecutionResult::Next
            }
            Instruction::LocalTee { index } => {
                let val = self.stack.tee();

                self.locals.set(*index, val);

                ExecutionResult::Next
            }
            Instruction::GlobalGet { index } => {
                self.stack.push(self.globals[index.0 as usize]);

                ExecutionResult::Next
            }
            Instruction::GlobalSet { index } => {
                let val = self.stack.pop();

                self.globals[index.0 as usize] = val;

                ExecutionResult::Next
            }
            Instruction::I32Load { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_i32(effective_offset)?;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I32Load8U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_u8(effective_offset)? as i32;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I32Load8S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_i8(effective_offset)? as i32;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I32Load16U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_u16(effective_offset)? as i32;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I32Load16S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_i16(effective_offset)? as i32;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I64Load { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_i64(effective_offset)?;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load8U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_u8(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load8S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_i8(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load16U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_u16(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load16S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_i16(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load32U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_u32(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load32S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_i32(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::F32Load { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_f32(effective_offset)?;

                self.stack.push(Val::F32(val));

                ExecutionResult::Next
            }
            Instruction::F64Load { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.memory.read_f64(effective_offset)?;

                self.stack.push(Val::F64(val));

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
    /// Returns [`TraceWasmError::Unsupported`] if a declared local has an
    /// unsupported type (`V128`, via [`Val::zero_of_ty`]); wraps a failed
    /// instruction as [`TraceWasmError::InstructionExecution`] (tagged with the
    /// enclosing function and instruction index); and propagates errors from
    /// nested and imported-function calls.
    fn execute<M: Memory, I: ImportRegistry>(
        func_index: FuncIndex,
        params: &[Val],
        module: &Module,
        stack: &mut Stack<Val>,
        memory: &mut M,
        caller_base_height: u32,
        import_registry: &mut I,
        globals: &mut [Val],
        tables: &mut Vec<TableVal>,
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

        // No runtime params check: `Module::compile` runs `Validator::validate_all`,
        // so every call site is type-correct — the recursive `Call` arm (the common
        // path) and the start function included, not just the typed `TypedFunc<P, R>`
        // entry, which adds a second guard on the top-level call. Validation covers
        // the input module but not TraceWasm's own lowering, so a `debug_assert`
        // still guards against a lowering bug producing a wrong param count, at zero
        // release cost.
        debug_assert_eq!(
            params.len(),
            params_ty.len(),
            "lowering produced wrong param count for func {}",
            func_index.0,
        );

        // Build the activation's local slots. Per the WebAssembly spec, a
        // function's locals are the parameters (bound to the incoming arguments,
        // in order) followed by the declared locals.
        let mut locals: SmallVec<[Val; 16]> = smallvec![]; // stack-allocated upto 16 locals per function

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
            globals,
            tables,
        };

        // Driver loop. `pc` indexes this function's instruction list only.
        let mut pc = 0;

        loop {
            let instr = &instructions[pc];

            let res = state
                .execute(instr, caller_base_height, module)
                .map_err(|err| err.into_tracewasm_err(pc, func_index, instr))?;

            match res {
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
        func_name: Option<&str>,
        params: &[Val],
        module: &Module,
        memory: &mut M,
        import_registry: &mut I,
        global_vals: &mut [Val],
        table_vals: &mut Vec<TableVal>,
    ) -> Result<ResultVals, TraceWasmError> {
        let mut stack: Stack<Val> = Stack::default();

        // A fresh stack starts at height 0, so this frame's base is 0.
        if let Err(err) = Self::execute(
            func_index,
            params,
            module,
            &mut stack,
            memory,
            0,
            import_registry,
            global_vals,
            table_vals,
        ) {
            if let Some(trace) = err.extract_stack_trace() {
                // TODO - propogate trace also!
                let _trace = trace.render(func_name, &module.custom_section);
            }

            return Err(err);
        }

        // How many result values the function leaves on the stack.
        let func_decl = &module.func_decls[func_index.0 as usize];
        let results_len = module.types[func_decl.ty.0 as usize].results.len() as u32;

        Ok(stack.pop_results(results_len))
    }

    /// Evaluates a constant-expression instruction sequence to its single
    /// resulting [`Val`], on a small dedicated stack. Used to compute
    /// global/table/data/element initializers at instantiation.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::Unsupported`] if the sequence contains an
    /// instruction not permitted in a constant expression.
    pub(crate) fn const_expr_evaluator(
        instructions: &[Instruction],
        globals: &[Val],
    ) -> Result<Val, TraceWasmError> {
        let mut stack: Stack<Val> = Stack::for_const_expr_evaluation();

        for instr in instructions {
            match instr {
                Instruction::I32Const { value } => {
                    stack.push(Val::I32(*value));
                }
                Instruction::I64Const { value } => {
                    stack.push(Val::I64(*value));
                }
                Instruction::F32Const { value } => {
                    stack.push(Val::F32(*value));
                }
                Instruction::F64Const { value } => stack.push(Val::F64(*value)),
                Instruction::GlobalGet { index } => {
                    stack.push(globals[index.0 as usize]);
                }
                Instruction::RefNull => stack.push(Val::Ref(None)),
                Instruction::RefFunc { function_index } => {
                    stack.push(Val::Ref(Some(*function_index)));
                }
                Instruction::I32Add => {
                    let b = stack.pop().as_i32();
                    let a = stack.pop().as_i32();

                    stack.push(Val::I32(a.wrapping_add(b)));
                }
                Instruction::I32Sub => {
                    let b = stack.pop().as_i32();
                    let a = stack.pop().as_i32();

                    stack.push(Val::I32(a.wrapping_sub(b)));
                }
                Instruction::I32Mul => {
                    let b = stack.pop().as_i32();
                    let a = stack.pop().as_i32();

                    stack.push(Val::I32(a.wrapping_mul(b)));
                }
                Instruction::I64Add => {
                    let b = stack.pop().as_i64();
                    let a = stack.pop().as_i64();

                    stack.push(Val::I64(a.wrapping_add(b)));
                }
                Instruction::I64Sub => {
                    let b = stack.pop().as_i64();
                    let a = stack.pop().as_i64();

                    stack.push(Val::I64(a.wrapping_sub(b)));
                }
                Instruction::I64Mul => {
                    let b = stack.pop().as_i64();
                    let a = stack.pop().as_i64();

                    stack.push(Val::I64(a.wrapping_mul(b)));
                }
                Instruction::End { .. } => {}
                _ => {
                    return Err(TraceWasmError::Unsupported(format!(
                        "instruction `{:?}` in const expression evaluator",
                        instr
                    )));
                }
            }
        }

        let val = stack.pop();

        Ok(val)
    }
}
