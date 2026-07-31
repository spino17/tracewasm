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
//! ## One shared stack across all frames — locals included
//!
//! Rather than giving every call its own operand stack, the whole call tree
//! shares a single `Stack`. A call does not allocate a new stack; the callee
//! simply builds its operands on top of the caller's. Recursion still uses the
//! native Rust call stack (one `TraceVM::execute` frame per active wasm call),
//! but the potentially-large value stack is allocated exactly once.
//!
//! A frame's **locals live on that same shared stack**, not in a per-activation
//! vector. The layout of one frame, from its base upwards, is:
//!
//! ```text
//!   caller_base_height ─┐
//!                       ▼
//!   ... caller's stack | p0 p1 … | l0 l1 … | operands … →
//!                        └ params ┘└ decl. ┘▲
//!                        └──── locals ─────┘└─ frame_base_height
//! ```
//!
//! The arguments are *already* on the stack when the callee is entered, so they
//! are left in place and become local slots `0..params_len` rather than being
//! popped and copied. Frame setup then pushes zero values for the declared
//! locals (`params_len..locals_len`), so `local.get`/`local.set` for slot `i`
//! is a direct index at `caller_base_height + i`.
//!
//! ## The two base heights
//!
//! Because both locals and operands share the stack, a frame needs *two* bases,
//! and conflating them corrupts memory:
//!
//! - **`caller_base_height`** — the bottom of the locals region, i.e. the height
//!   the stack had on entry minus the arguments. Used only by
//!   `get_local`/`set_local`, and as the truncation target on frame exit.
//! - **`frame_base_height`** — the bottom of the *operand* region, i.e.
//!   `caller_base_height + locals_len`. Used by every height-sensitive control
//!   operation.
//!
//! The lowered instructions store **frame-relative** operand heights
//! (`recorded_height`), computed as if the function ran on an empty operand
//! stack with its locals held separately. The interpreter converts
//! relative → absolute with a single rule:
//!
//! ```text
//! absolute_height = frame_base_height + recorded_height (+ arity)
//! ```
//!
//! On frame exit the stack is truncated to `caller_base_height` preserving the
//! result arity, which drops the locals and leaves the results exactly where the
//! caller's arguments were — so the caller does nothing after the call returns.
//! Instruction indices, by contrast, are per-function: each `TraceVM::execute`
//! invocation has its own `instructions` slice and `pc`.

use crate::{
    error::{
        CallIndirectError::{self, FunctionCall},
        InstructionExecutionError, MemoryAccessKind, MemoryError, TraceWasmError,
    },
    instance::{
        Instance,
        traits::{ImportRegistry, ResultVals},
    },
    instruction::Instruction,
    memory::Memory,
    module::{FuncIndex, FuncKind, LocalIndex, Module, formatted_val_types},
    vm::stack::{DataVal, Stack, Val},
};
use std::ops::{BitAnd, BitOr, BitXor, Neg};

pub(crate) mod stack;

// Bounds for the `trunc` targets. Each `_HIGH` is the power of two just *past*
// the target's range and is treated as exclusive; see `trunc_float_to_int`.
//
// The casts land on those powers of two precisely *because* they round: no
// `_HIGH` maximum is representable in the type it is cast through, so each one
// rounds up to the next power of two, which is. The trailing comment on every
// line is the resulting value — keep them in step if these are ever rewritten.
//
// This makes the intermediate `as f32` on the 32-bit lines load-bearing despite
// looking redundant. Dropping it leaves `i32::MAX as f64 == 2147483647.0`, an
// *inclusive* maximum, and the exclusive test would then reject `i32::MAX`
// itself. The 64-bit lines need no such step because `i64::MAX`/`u64::MAX` are
// already unrepresentable in `f64`.
const I32_TRUNC_LOW: f64 = i32::MIN as f32 as f64; // -2^31 = -2147483648
const I32_TRUNC_HIGH: f64 = i32::MAX as f32 as f64; // 2^31 = 2147483648
const U32_TRUNC_HIGH: f64 = u32::MAX as f32 as f64; // 2^32 = 4294967296
const I64_TRUNC_LOW: f64 = i64::MIN as f64; // -2^63 = -9223372036854775808
const I64_TRUNC_HIGH: f64 = i64::MAX as f64; // 2^63 = 9223372036854775808
const U64_TRUNC_HIGH: f64 = u64::MAX as f64; // 2^64 = 18446744073709551616

pub(crate) struct TraceVM;

impl TraceVM {
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
        instance: &mut Instance<M, I>,
        module: &Module,
    ) -> Result<ResultVals, TraceWasmError> {
        let mut call_stack_depth: u32 = 0;
        let mut stack: Stack<Val> = Stack::default();

        for param in params {
            stack.push(*param);
        }

        // A fresh stack starts at height 0, so this frame's base is 0.
        Self::execute(
            func_index,
            &mut stack,
            instance,
            module,
            &mut call_stack_depth,
        )?;

        // How many result values the function leaves on the stack.
        let func_decl = &module.func_decls[func_index.0 as usize];
        let results_len = module.types[func_decl.ty.0 as usize].results.len() as u32;

        Ok(stack.pop_results(results_len))
    }

    /// Runs one (locally-defined) function to completion on the shared stack.
    ///
    /// Called both as the top-level entry (with an empty stack, so this frame's
    /// base is 0) and recursively from the `Call`/`CallIndirect` instructions.
    ///
    /// **Arguments are passed on the stack, not as a slice.** The caller leaves
    /// them as the topmost values and this function adopts them in place as the
    /// leading local slots; it derives its own base as
    /// `stack.height() - params_len`. Results are likewise left on `stack` for
    /// the caller to consume, which is why this returns `()` rather than the
    /// results. See the module docs for the full frame layout.
    ///
    /// # Errors
    ///
    /// Always a [`TraceWasmError::InstructionExecution`]: every failure — a trap,
    /// an error from a nested call, or one from an imported function — is tagged
    /// here with the instruction that raised it. Frame setup cannot fail, since
    /// `Module::compile` already rejected the local types the VM does not model.
    ///
    /// [`FuncCallError`](crate::error::FuncCallError) depends on this being the
    /// only error shape that escapes the interpreter.
    fn execute<M: Memory, I: ImportRegistry>(
        func_index: FuncIndex,
        stack: &mut Stack<Val>,
        instance: &mut Instance<M, I>,
        module: &Module,
        call_stack_depth: &mut u32,
    ) -> Result<(), TraceWasmError> {
        // `func_bodies` holds only locally-defined functions, so shift the global
        // function index down by the number of imports to index into it.
        let imported_func_count = module.imported_func_count;

        debug_assert!(func_index.0 >= imported_func_count);

        let func_decl = &module.func_decls[func_index.0 as usize];
        let ty = &module.types[func_decl.ty.0 as usize];
        let params_len = ty.params.len();
        let results_len = ty.results.len();
        let func_body = &module.func_bodies[(func_index.0 - imported_func_count) as usize];
        let instructions = &func_body.instructions;
        let instruction_offsets = &func_body.instruction_offsets;

        // `locals` in the body is laid out params-first, then declared locals,
        // and `locals_ty[i]` is the declared type of local slot `i`. So
        // `locals_len >= params_len` always, and the subtraction below cannot
        // underflow.
        let locals_ty = &func_body.locals;
        let locals_len = locals_ty.len();

        // Frame setup. The arguments are already the topmost values on the shared
        // stack, so the locals region starts just below them and the args become
        // slots `0..params_len` without being copied anywhere.
        let caller_base_height = stack.height() - params_len as u32;

        // The declared locals follow the params and must start at the zero value
        // of their type, per the spec. Pushing them here is what makes the locals
        // region contiguous, so `get_local` can index it directly.
        for i in 0..(locals_len - params_len) {
            let ty = locals_ty[i + params_len];

            stack.push(Val::zero_of_ty(ty));
        }

        // Entering a frame. The matching decrement is after the driver loop; the
        // error paths out of the loop deliberately skip it, because an error always
        // propagates to the top-level `run` (nothing in the interpreter catches
        // one) and the counter is a fresh local per `run`. The depth *limit* is
        // enforced at the call site in `call_func`, not here, so that the top-level
        // frame is always admitted — an error raised during frame setup would
        // otherwise escape without the enclosing-instruction tag that
        // `FuncCallError` requires.
        *call_stack_depth += 1;

        let mut state = TraceVMState {
            stack,
            caller_base_height,
            frame_base_height: caller_base_height + locals_len as u32,
            instance,
        };

        // Driver loop. `pc` indexes this function's instruction list only.
        //
        // `pc` is passed to `execute` by value and the next one comes back as the
        // return value, deliberately: lending it out as `&mut` would give it an
        // address and force it to a stack slot, reloaded on every iteration.
        // `instr_count` is hoisted for the same reason — reading
        // `instructions.len()` in the condition reloads the slice length each time
        // round.
        let mut pc = 0;
        let instr_count = instructions.len();

        loop {
            let instr = &instructions[pc];

            pc = state
                .execute(instr, module, pc, call_stack_depth)
                .map_err(|err| {
                    // `pc` is unchanged on the error path (the failing arm returns
                    // before the update), so it still names the faulting instruction.
                    err.into_tracewasm_err(pc, func_index, instr, instruction_offsets[pc])
                })?;

            // Advancing past the last instruction means we just executed the
            // function's terminating `End`: the frame is complete. (`return` and
            // branches out of the outermost block also land here, since their
            // target is that final `End`.) Branch targets are always real
            // instruction indices, so only a fall-through can reach `len`.
            if pc == instr_count {
                break;
            }
        }

        *call_stack_depth -= 1;

        // Tear the frame down: truncating to `caller_base_height` (the locals
        // base, *not* `frame_base_height`) discards this frame's locals while
        // preserving the results on top. They therefore land in exactly the slots
        // the caller's arguments occupied, which is what lets the caller do
        // nothing at all after a `Call` returns.
        stack.truncate_by_preserving_arity(caller_base_height, results_len as u32);

        Ok(())
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
/// Nothing here is owned per activation: both the stack (which holds this
/// frame's locals *and* operands) and the instance are borrowed, because they
/// are shared across the whole call tree. The two height fields are all that
/// distinguishes one activation from another — see the module docs for the frame
/// layout they describe.
struct TraceVMState<'a, M, I> {
    /// The shared stack: the caller's values, then this frame's locals, then
    /// this frame's operands.
    stack: &'a mut Stack<Val>,
    /// Bottom of this frame's **locals** region: the height the stack had on
    /// entry, minus the arguments (which are reused in place as the leading
    /// local slots). Also the truncation target on frame exit.
    caller_base_height: u32,
    /// Bottom of this frame's **operand** region:
    /// `caller_base_height + locals_len`. Every `recorded_height` is relative to
    /// this, *not* to `caller_base_height`.
    frame_base_height: u32,
    instance: &'a mut Instance<M, I>,
}

impl<'a, M: Memory, I: ImportRegistry> TraceVMState<'a, M, I> {
    /// Reads local slot `index` from this frame's locals region.
    ///
    /// Indexes the backing storage directly rather than going through the
    /// stack's push/pop API: locals sit *below* the stack pointer for the whole
    /// life of the frame, so they are always live storage that the operand
    /// discipline never touches. Validation guarantees `index < locals_len`.
    fn get_local(&self, index: LocalIndex) -> Val {
        self.stack.inner[(index.0 + self.caller_base_height) as usize]
    }

    /// Writes local slot `index` in this frame's locals region. See
    /// [`Self::get_local`] for why this indexes the backing storage directly.
    fn set_local(&mut self, index: LocalIndex, val: Val) {
        self.stack.inner[(index.0 + self.caller_base_height) as usize] = val;
    }

    /// Truncates `operand` toward zero, checking the result is representable in an
    /// integer type spanning the half-open range `[low, high)`.
    ///
    /// Shared by the eight `iNN.trunc_fNN_{s,u}` instructions, which differ only in
    /// those bounds and in how they reinterpret the result. Two details make it worth
    /// centralising rather than repeating per arm:
    ///
    /// * **Truncation happens before the range test.** `trunc_u` of `-0.9` is `-0.0`,
    ///   a valid zero — testing `operand >= 0.0` first would trap on it instead.
    /// * **The upper bound is exclusive.** An exclusive power of two is exactly
    ///   representable where the target's maximum often is not: `i32::MAX as f32`
    ///   rounds *up* to 2^31, so comparing against it would wrongly admit 2^31
    ///   itself, and `i64::MAX as f64` rounds up to 2^63 the same way. Callers
    ///   promote `f32` operands to `f64` so the 64-bit bounds stay precise too.
    ///
    /// # Errors
    ///
    /// Returns [`InstructionExecutionError::FloatToIntTruncation`] for a NaN or
    /// infinite operand, and for a truncated value outside `[low, high)`.
    ///
    /// Those are two checks but one condition: the range test alone would reject all
    /// three special values, since NaN compares false against everything and the
    /// infinities fall outside any finite range. The explicit guard is kept for
    /// legibility — it states the intent without asking the reader to work through
    /// IEEE comparison semantics — so note that no input can reach the range test
    /// only to be caught by the guard, and removing it would not change behaviour.
    fn trunc_float_to_int(
        operand: f64,
        low: f64,
        high: f64,
        target: &str,
    ) -> Result<f64, InstructionExecutionError> {
        if operand.is_nan() || operand.is_infinite() {
            return Err(InstructionExecutionError::FloatToIntTruncation(
                operand.to_string(),
                target.to_string(),
            ));
        }

        let truncated = operand.trunc();

        if !(low..high).contains(&truncated) {
            return Err(InstructionExecutionError::FloatToIntTruncation(
                operand.to_string(),
                target.to_string(),
            ));
        }

        Ok(truncated)
    }

    /// Pops the address operand of a memory access and resolves it to an
    /// effective address by adding the instruction's static `memarg` offset.
    ///
    /// Shared by every load/store arm, so they all inherit the same overflow and
    /// offset-range trapping.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::EffectiveAddressOverflow`] if the sum leaves the
    /// 32-bit address space. The access itself is bounds-checked by [`Memory`].
    ///
    /// The static offset needs no range check of its own: `wasmparser` rejects a
    /// `memarg` offset above `u32::MAX` on an `i32`-indexed memory, and 64-bit
    /// memories are refused at compile time (see `Module::compile`), so the only
    /// memories that reach here are 32-bit.
    fn pop_effective_address(&mut self, memarg_offset: u32) -> Result<usize, MemoryError> {
        let addr = self.stack.pop().as_i32() as u32;

        // Effective address = popped address + static offset, computed with
        // a checked add: a u32 overflow is past the 32-bit memory space, so
        // it traps rather than wrapping to a wrong (in-bounds) address.
        let effective_offset = addr
            .checked_add(memarg_offset)
            .ok_or(MemoryError::EffectiveAddressOverflow(addr, memarg_offset))?;

        Ok(effective_offset as usize)
    }

    fn call_func(
        &mut self,
        func_index: FuncIndex,
        params_count: u32,
        module: &Module,
        call_stack_depth: &mut u32,
    ) -> Result<(), TraceWasmError> {
        // The arguments are the topmost values on the shared stack. How they are
        // handed over depends on the callee, so neither branch below pops them
        // up front:
        //
        // - a local callee adopts them in place as its leading local slots, so
        //   they are never copied;
        // - an imported callee needs them as a `&[Val]` for the host boundary, so
        //   that branch pops them into a `ParamVals`.
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

            let params = self.stack.pop_params(params_count);

            // `execute` returns a stack-allocated `ResultVals` (no heap for <=3 results).
            let results = self.instance.import_registry.execute(
                module_name,
                imported_func_name,
                params.as_ref(),
                &mut self.instance.memory,
            )?;

            // push results to the stack
            for res in results {
                self.stack.push(res);
            }
        } else {
            // Only a local callee creates a new interpreter frame — and therefore a
            // new native frame — so the depth limit is checked here rather than
            // around the whole of `call_func`. An imported callee runs on the host's
            // own stack and consumes no interpreter depth.
            //
            // `call_stack_depth` is this (the caller's) depth, so the callee would
            // be at `+ 1`; `>=` is the test for "the callee would exceed".
            let max_depth = self.instance.config().get_max_call_stack_depth();

            if *call_stack_depth >= max_depth {
                return Err(TraceWasmError::CallStackExhausted(max_depth));
            }

            // local function execution
            TraceVM::execute(
                func_index,
                self.stack,
                self.instance,
                module,
                call_stack_depth,
            )?;
        }

        Ok(())
    }

    /// Executes a single instruction against this activation's state and returns
    /// the control-flow decision for the driver loop.
    ///
    /// Heights are resolved against `frame_base_height` (see the module docs).
    /// `module` is needed to resolve callees on `Call`.
    ///
    /// ## Why this signature, precisely
    ///
    /// This is the hottest boundary in the interpreter — crossed once per wasm
    /// instruction — so both halves of it are shaped around the aarch64/SysV rule
    /// that a return value over 16 bytes comes back through an `sret` pointer
    /// (i.e. memory) instead of registers:
    ///
    /// * **`pc` is taken by value and the next `pc` is returned.** Taking
    ///   `&mut usize` instead forces the caller's `pc` to have an address, so it
    ///   cannot stay in a register across this (non-inlined) call: the caller then
    ///   spills it and reloads it every iteration. Returning it keeps it in a
    ///   register.
    /// * **The error is boxed.** [`InstructionExecutionError`] is 56 bytes — its
    ///   variants carry `String`s and a `Box<TraceWasmError>` — so an unboxed
    ///   `Result` would be returned via `sret`, making *every successful
    ///   instruction* pay a 56-byte stack write plus a reload to report a failure
    ///   that essentially never happens. `Box` shrinks the error to 8 bytes, so
    ///   `Result<usize, _>` fits in two registers. The cost is one allocation on
    ///   the error path, which is already the slow path.
    ///
    /// The `ExecutionResult` match stays inside this function so the enum never
    /// crosses the boundary either; LLVM folds each arm's constant straight into
    /// the returned `pc`.
    fn execute(
        &mut self,
        instruction: &Instruction,
        module: &Module,
        pc: usize,
        call_stack_depth: &mut u32,
    ) -> Result<usize, InstructionExecutionError> {
        let res = match instruction {
            Instruction::Call {
                func_index: callee_func_index,
                params_count,
            } => {
                self.call_func(*callee_func_index, *params_count, module, call_stack_depth)
                    .map_err(|err| {
                        InstructionExecutionError::Call(*callee_func_index, Box::new(err))
                    })?;

                ExecutionResult::Next
            }
            Instruction::CallIndirect {
                ty_index,
                table_index,
            } => {
                let table = &self.instance.table_vals[table_index.0 as usize];
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

                let func_ty = &module.types[ty_index.0 as usize];
                let params = &func_ty.params;
                let results = &func_ty.results;

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

                self.call_func(
                    callee_func_index,
                    declared_params.len() as u32,
                    module,
                    call_stack_depth,
                )
                .map_err(|err| {
                    InstructionExecutionError::CallIndirect(
                        *table_index,
                        FunctionCall(callee_func_index, Box::new(err)),
                    )
                })?;

                ExecutionResult::Next
            }
            Instruction::Unreachable => {
                return Err(InstructionExecutionError::Unreachable);
            }
            Instruction::Nop => ExecutionResult::Next,
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
            Instruction::RefIsNull => {
                let func_ref = self.stack.pop().as_ref();

                if func_ref.is_none() {
                    self.stack.push(Val::I32(1));
                } else {
                    self.stack.push(Val::I32(0));
                }

                ExecutionResult::Next
            }
            Instruction::MemorySize => {
                self.stack
                    .push(Val::I32(self.instance.memory.size_in_pages() as i32));

                ExecutionResult::Next
            }
            Instruction::MemoryGrow => {
                // The delta is an unsigned i32; going through `u32` keeps a
                // high-bit-set value from sign-extending into a different number.
                let delta_in_pages = self.stack.pop().as_i32() as u32 as u64;
                let max_pages = self.instance.config.get_max_memory_size_in_pages();

                // `instantiate` already narrowed this to the module's declared
                // maximum, so the configured cap is the effective ceiling here.
                match self.instance.memory.grow(delta_in_pages, max_pages) {
                    Ok(old_page) => self.stack.push(Val::I32(old_page as i32)),
                    // Per the spec `memory.grow` does not trap: a request it cannot
                    // satisfy reports `-1` and execution continues.
                    Err(_) => self.stack.push(Val::I32(-1)),
                }

                ExecutionResult::Next
            }
            Instruction::MemoryCopy => {
                let len = self.stack.pop().as_i32() as usize;
                let src = self.stack.pop().as_i32() as usize;
                let dest = self.stack.pop().as_i32() as usize;

                self.instance.memory.copy_within(dest, src, len)?;

                ExecutionResult::Next
            }
            Instruction::MemoryFill => {
                let len = self.stack.pop().as_i32() as usize;
                let val = self.stack.pop().as_i32() as u32;
                let dest = self.stack.pop().as_i32() as usize;

                self.instance.memory.fill(dest, val, len)?;

                ExecutionResult::Next
            }
            Instruction::MemoryInit { data_index } => {
                let len = self.stack.pop().as_i32() as usize;
                let src = self.stack.pop().as_i32() as usize;
                let dest = self.stack.pop().as_i32() as usize;

                // A dropped segment reads as *empty*, not as an outright trap: the
                // spec replaces its bytes with the empty sequence, so a
                // zero-length `memory.init` after `data.drop` still succeeds while
                // any non-empty read fails the bounds check below.
                let segment: &[u8] = match &self.instance.data_vals[*data_index as usize] {
                    DataVal::Dropped => &[],
                    DataVal::Passive(segment) => segment,
                };

                // The source range is validated against the segment before
                // anything is written, and `checked_add` stops a huge `len` from
                // wrapping past the comparison.
                let end = src
                    .checked_add(len)
                    .filter(|end| *end <= segment.len())
                    .ok_or(MemoryError::OutOfBoundsAccess(
                        MemoryAccessKind::Read,
                        src,
                        segment.len(),
                    ))?;

                // `write` bounds-checks the destination, so a trap on either side
                // leaves memory untouched.
                self.instance.memory.write(dest, &segment[src..end])?;

                ExecutionResult::Next
            }
            Instruction::DataDrop { data_index } => {
                self.instance.data_vals[*data_index as usize] = DataVal::Dropped;

                ExecutionResult::Next
            }
            Instruction::I32Load { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_i32(effective_offset)?;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I32Load8U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_u8(effective_offset)? as i32;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I32Load8S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_i8(effective_offset)? as i32;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I32Load16U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_u16(effective_offset)? as i32;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I32Load16S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_i16(effective_offset)? as i32;

                self.stack.push(Val::I32(val));

                ExecutionResult::Next
            }
            Instruction::I64Load { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_i64(effective_offset)?;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load8U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_u8(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load8S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_i8(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load16U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_u16(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load16S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_i16(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load32U { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_u32(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::I64Load32S { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_i32(effective_offset)? as i64;

                self.stack.push(Val::I64(val));

                ExecutionResult::Next
            }
            Instruction::F32Load { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_f32(effective_offset)?;

                self.stack.push(Val::F32(val));

                ExecutionResult::Next
            }
            Instruction::F64Load { offset, align: _ } => {
                let effective_offset = self.pop_effective_address(*offset)?;
                let val = self.instance.memory.read_f64(effective_offset)?;

                self.stack.push(Val::F64(val));

                ExecutionResult::Next
            }
            Instruction::I32Store { offset, align: _ } => {
                let val = self.stack.pop().as_i32();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.instance
                    .memory
                    .write_u32(effective_offset, val as u32)?;

                ExecutionResult::Next
            }
            Instruction::I32Store8 { offset, align: _ } => {
                let val = self.stack.pop().as_i32();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.instance.memory.write_u8(effective_offset, val as u8)?;

                ExecutionResult::Next
            }
            Instruction::I32Store16 { offset, align: _ } => {
                let val = self.stack.pop().as_i32();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.instance
                    .memory
                    .write_u16(effective_offset, val as u16)?;

                ExecutionResult::Next
            }
            Instruction::I64Store { offset, align: _ } => {
                let val = self.stack.pop().as_i64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.instance
                    .memory
                    .write_u64(effective_offset, val as u64)?;

                ExecutionResult::Next
            }
            Instruction::I64Store8 { offset, align: _ } => {
                let val = self.stack.pop().as_i64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.instance.memory.write_u8(effective_offset, val as u8)?;

                ExecutionResult::Next
            }
            Instruction::I64Store16 { offset, align: _ } => {
                let val = self.stack.pop().as_i64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.instance
                    .memory
                    .write_u16(effective_offset, val as u16)?;

                ExecutionResult::Next
            }
            Instruction::I64Store32 { offset, align: _ } => {
                let val = self.stack.pop().as_i64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.instance
                    .memory
                    .write_u32(effective_offset, val as u32)?;

                ExecutionResult::Next
            }
            Instruction::F32Store { offset, align: _ } => {
                let val = self.stack.pop().as_f32();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.instance.memory.write_f32(effective_offset, val)?;

                ExecutionResult::Next
            }
            Instruction::F64Store { offset, align: _ } => {
                let val = self.stack.pop().as_f64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.instance.memory.write_f64(effective_offset, val)?;

                ExecutionResult::Next
            }
            Instruction::I32Clz => {
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.leading_zeros() as i32));

                ExecutionResult::Next
            }
            Instruction::I32Ctz => {
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.trailing_zeros() as i32));

                ExecutionResult::Next
            }
            Instruction::I32Popcnt => {
                let a = self.stack.pop().as_i32();

                // Counts set bits in the two's-complement representation, so a
                // negative operand counts its sign bits too — which is what the
                // spec's bit-level definition asks for.
                self.stack.push(Val::I32(a.count_ones() as i32));

                ExecutionResult::Next
            }
            Instruction::I32Eqz => {
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(if a == 0 { 1 } else { 0 }));

                ExecutionResult::Next
            }
            Instruction::I32Extend8S => {
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a as i8 as i32));

                ExecutionResult::Next
            }
            Instruction::I32Extend16S => {
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a as i16 as i32));

                ExecutionResult::Next
            }
            Instruction::I32WrapI64 => {
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I32(a as i32));

                ExecutionResult::Next
            }
            Instruction::I32TruncF32U => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = self.stack.pop().as_f32() as f64;
                let truncated = Self::trunc_float_to_int(a, 0.0, U32_TRUNC_HIGH, "u32")?;

                // The result is the `u32` bit pattern held in an `i32`, so values
                // above `i32::MAX` come back out negative.
                self.stack.push(Val::I32(truncated as u32 as i32));

                ExecutionResult::Next
            }
            Instruction::I32TruncF32S => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = self.stack.pop().as_f32() as f64;
                let truncated = Self::trunc_float_to_int(a, I32_TRUNC_LOW, I32_TRUNC_HIGH, "i32")?;

                self.stack.push(Val::I32(truncated as i32));

                ExecutionResult::Next
            }
            Instruction::I32TruncF64U => {
                let a = self.stack.pop().as_f64();
                let truncated = Self::trunc_float_to_int(a, 0.0, U32_TRUNC_HIGH, "u32")?;

                self.stack.push(Val::I32(truncated as u32 as i32));

                ExecutionResult::Next
            }
            Instruction::I32TruncF64S => {
                let a = self.stack.pop().as_f64();
                let truncated = Self::trunc_float_to_int(a, I32_TRUNC_LOW, I32_TRUNC_HIGH, "i32")?;

                self.stack.push(Val::I32(truncated as i32));

                ExecutionResult::Next
            }
            Instruction::I32TruncSatF32U => {
                let a = self.stack.pop().as_f32() as u32;

                self.stack.push(Val::I32(a as i32));

                ExecutionResult::Next
            }
            Instruction::I32TruncSatF32S => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::I32(a as i32));

                ExecutionResult::Next
            }
            Instruction::I32TruncSatF64U => {
                // Saturate to `u32`, the *target* width — going through `u64` here
                // would clamp at the wrong bound and then wrap on the way down.
                let a = self.stack.pop().as_f64() as u32;

                self.stack.push(Val::I32(a as i32));

                ExecutionResult::Next
            }
            Instruction::I32TruncSatF64S => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::I32(a as i32));

                ExecutionResult::Next
            }
            Instruction::I32ReinterpretF32 => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::I32(a.to_bits() as i32));

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
            Instruction::I32DivU => {
                let b = self.stack.pop().as_i32() as u32;
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::I32(a.checked_div(b).ok_or(
                    InstructionExecutionError::Division {
                        num: a.to_string(),
                        deno: b.to_string(),
                    },
                )? as i32));

                ExecutionResult::Next
            }
            Instruction::I32DivS => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.checked_div(b).ok_or(
                    InstructionExecutionError::Division {
                        num: a.to_string(),
                        deno: b.to_string(),
                    },
                )?));

                ExecutionResult::Next
            }
            Instruction::I32RemU => {
                let b = self.stack.pop().as_i32() as u32;
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::I32(a.checked_rem(b).ok_or(
                    InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    },
                )? as i32));

                ExecutionResult::Next
            }
            Instruction::I32RemS => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                // A zero divisor is the *only* trap here. Unlike `i32.div_s`,
                // `rem_s` does not trap on overflow: the spec defines
                // `i32::MIN % -1` as `0`, which is what `wrapping_rem` returns.
                // `checked_rem` would wrongly report that case as a failure.
                if b == 0 {
                    return Err(InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    });
                }

                self.stack.push(Val::I32(a.wrapping_rem(b)));

                ExecutionResult::Next
            }
            Instruction::I32And => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.bitand(b)));

                ExecutionResult::Next
            }
            Instruction::I32Or => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.bitor(b)));

                ExecutionResult::Next
            }
            Instruction::I32Xor => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.bitxor(b)));

                ExecutionResult::Next
            }
            // Shift and rotate counts are taken modulo the operand width, so a
            // count of 32 or more is well defined rather than a trap or UB. The
            // `wrapping_*`/`rotate_*` methods apply exactly that masking; the plain
            // `<<`/`>>` operators would instead panic in debug builds.
            Instruction::I32Shl => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32(a.wrapping_shl(b as u32)));

                ExecutionResult::Next
            }
            Instruction::I32ShrU => {
                let b = self.stack.pop().as_i32() as u32;
                let a = self.stack.pop().as_i32() as u32;

                // Logical shift: done on `u32` so the vacated high bits are zeros.
                self.stack.push(Val::I32(a.wrapping_shr(b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32ShrS => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                // Arithmetic shift: on `i32` the sign bit is replicated.
                self.stack.push(Val::I32(a.wrapping_shr(b as u32)));

                ExecutionResult::Next
            }
            Instruction::I32Rotl => {
                let b = self.stack.pop().as_i32() as u32;
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::I32(a.rotate_left(b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32Rotr => {
                let b = self.stack.pop().as_i32() as u32;
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::I32(a.rotate_right(b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32Eq => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32((a == b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32Ne => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32((a != b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32LtU => {
                let b = self.stack.pop().as_i32() as u32;
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::I32((a < b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32LtS => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32((a < b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32GtU => {
                let b = self.stack.pop().as_i32() as u32;
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::I32((a > b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32GtS => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32((a > b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32LeU => {
                let b = self.stack.pop().as_i32() as u32;
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::I32((a <= b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32LeS => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32((a <= b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32GeU => {
                let b = self.stack.pop().as_i32() as u32;
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::I32((a >= b) as i32));

                ExecutionResult::Next
            }
            Instruction::I32GeS => {
                let b = self.stack.pop().as_i32();
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I32((a >= b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64Clz => {
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.leading_zeros() as i64));

                ExecutionResult::Next
            }
            Instruction::I64Ctz => {
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.trailing_zeros() as i64));

                ExecutionResult::Next
            }
            Instruction::I64Popcnt => {
                let a = self.stack.pop().as_i64();

                // See `I32Popcnt`. The count is at most 64, but the result type is
                // `i64` — unary integer ops keep their operand's width, unlike the
                // comparisons.
                self.stack.push(Val::I64(a.count_ones() as i64));

                ExecutionResult::Next
            }
            Instruction::I64Eqz => {
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I32(if a == 0 { 1 } else { 0 }));

                ExecutionResult::Next
            }
            Instruction::I64Extend8S => {
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a as i8 as i64));

                ExecutionResult::Next
            }
            Instruction::I64Extend16S => {
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a as i16 as i64));

                ExecutionResult::Next
            }
            Instruction::I64Extend32S => {
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a as i32 as i64));

                ExecutionResult::Next
            }
            Instruction::I64ExtendI32U => {
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::I64(a as i64));

                ExecutionResult::Next
            }
            Instruction::I64ExtendI32S => {
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::I64(a as i64));

                ExecutionResult::Next
            }
            Instruction::I64TruncF32U => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = self.stack.pop().as_f32() as f64;
                let truncated = Self::trunc_float_to_int(a, 0.0, U64_TRUNC_HIGH, "u64")?;

                // As with the `i32` forms, the result is the unsigned bit pattern
                // held in a signed value.
                self.stack.push(Val::I64(truncated as u64 as i64));

                ExecutionResult::Next
            }
            Instruction::I64TruncF32S => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = self.stack.pop().as_f32() as f64;
                let truncated = Self::trunc_float_to_int(a, I64_TRUNC_LOW, I64_TRUNC_HIGH, "i64")?;

                self.stack.push(Val::I64(truncated as i64));

                ExecutionResult::Next
            }
            Instruction::I64TruncF64U => {
                let a = self.stack.pop().as_f64();
                let truncated = Self::trunc_float_to_int(a, 0.0, U64_TRUNC_HIGH, "u64")?;

                self.stack.push(Val::I64(truncated as u64 as i64));

                ExecutionResult::Next
            }
            Instruction::I64TruncF64S => {
                let a = self.stack.pop().as_f64();
                let truncated = Self::trunc_float_to_int(a, I64_TRUNC_LOW, I64_TRUNC_HIGH, "i64")?;

                self.stack.push(Val::I64(truncated as i64));

                ExecutionResult::Next
            }
            Instruction::I64TruncSatF32U => {
                // Saturate to `u64`, the *target* width — clamping at `u32::MAX`
                // first would lose every value an `i64` can still represent.
                let a = self.stack.pop().as_f32() as u64;

                self.stack.push(Val::I64(a as i64));

                ExecutionResult::Next
            }
            Instruction::I64TruncSatF32S => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::I64(a as i64));

                ExecutionResult::Next
            }
            Instruction::I64TruncSatF64U => {
                let a = self.stack.pop().as_f64() as u64;

                self.stack.push(Val::I64(a as i64));

                ExecutionResult::Next
            }
            Instruction::I64TruncSatF64S => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::I64(a as i64));

                ExecutionResult::Next
            }
            Instruction::I64ReinterpretF64 => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::I64(a.to_bits() as i64));

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
            Instruction::I64DivU => {
                let b = self.stack.pop().as_i64() as u64;
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::I64(a.checked_div(b).ok_or(
                    InstructionExecutionError::Division {
                        num: a.to_string(),
                        deno: b.to_string(),
                    },
                )? as i64));

                ExecutionResult::Next
            }
            Instruction::I64DivS => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.checked_div(b).ok_or(
                    InstructionExecutionError::Division {
                        num: a.to_string(),
                        deno: b.to_string(),
                    },
                )?));

                ExecutionResult::Next
            }
            Instruction::I64RemU => {
                let b = self.stack.pop().as_i64() as u64;
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::I64(a.checked_rem(b).ok_or(
                    InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    },
                )? as i64));

                ExecutionResult::Next
            }
            Instruction::I64RemS => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                // See `I32RemS`: only a zero divisor traps; `i64::MIN % -1` is `0`.
                if b == 0 {
                    return Err(InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    });
                }

                self.stack.push(Val::I64(a.wrapping_rem(b)));

                ExecutionResult::Next
            }
            Instruction::I64And => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.bitand(b)));

                ExecutionResult::Next
            }
            Instruction::I64Or => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.bitor(b)));

                ExecutionResult::Next
            }
            Instruction::I64Xor => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.bitxor(b)));

                ExecutionResult::Next
            }
            // As for `i32`, but masked modulo 64. The count arrives as an `i64` and
            // the shift methods take `u32`, so it is narrowed first — harmless,
            // since only the low 6 bits survive the masking anyway.
            Instruction::I64Shl => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I64(a.wrapping_shl(b as u32)));

                ExecutionResult::Next
            }
            Instruction::I64ShrU => {
                let b = self.stack.pop().as_i64() as u64;
                let a = self.stack.pop().as_i64() as u64;

                // Logical shift: done on `u64` so the vacated high bits are zeros.
                self.stack.push(Val::I64(a.wrapping_shr(b as u32) as i64));

                ExecutionResult::Next
            }
            Instruction::I64ShrS => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                // Arithmetic shift: on `i64` the sign bit is replicated.
                self.stack.push(Val::I64(a.wrapping_shr(b as u32)));

                ExecutionResult::Next
            }
            Instruction::I64Rotl => {
                let b = self.stack.pop().as_i64() as u64;
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::I64(a.rotate_left(b as u32) as i64));

                ExecutionResult::Next
            }
            Instruction::I64Rotr => {
                let b = self.stack.pop().as_i64() as u64;
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::I64(a.rotate_right(b as u32) as i64));

                ExecutionResult::Next
            }
            Instruction::I64Eq => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I32((a == b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64Ne => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I32((a != b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64LtU => {
                let b = self.stack.pop().as_i64() as u64;
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::I32((a < b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64LtS => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I32((a < b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64GtU => {
                let b = self.stack.pop().as_i64() as u64;
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::I32((a > b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64GtS => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I32((a > b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64LeU => {
                let b = self.stack.pop().as_i64() as u64;
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::I32((a <= b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64LeS => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I32((a <= b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64GeU => {
                let b = self.stack.pop().as_i64() as u64;
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::I32((a >= b) as i32));

                ExecutionResult::Next
            }
            Instruction::I64GeS => {
                let b = self.stack.pop().as_i64();
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::I32((a >= b) as i32));

                ExecutionResult::Next
            }
            Instruction::F32Abs => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a.abs()));

                ExecutionResult::Next
            }
            Instruction::F32Neg => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a.neg()));

                ExecutionResult::Next
            }
            Instruction::F32Ceil => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a.ceil()));

                ExecutionResult::Next
            }
            Instruction::F32Floor => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a.floor()));

                ExecutionResult::Next
            }
            Instruction::F32Trunc => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a.trunc()));

                ExecutionResult::Next
            }
            Instruction::F32Sqrt => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a.sqrt()));

                ExecutionResult::Next
            }
            Instruction::F32Nearest => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a.round_ties_even()));

                ExecutionResult::Next
            }
            Instruction::F32ConvertI32U => {
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::F32(a as f32));

                ExecutionResult::Next
            }
            Instruction::F32ConvertI32S => {
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::F32(a as f32));

                ExecutionResult::Next
            }
            Instruction::F32ConvertI64U => {
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::F32(a as f32));

                ExecutionResult::Next
            }
            Instruction::F32ConvertI64S => {
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::F32(a as f32));

                ExecutionResult::Next
            }
            Instruction::F32DemoteF64 => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F32(a as f32));

                ExecutionResult::Next
            }
            Instruction::F32ReinterpretI32 => {
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::F32(f32::from_bits(a)));

                ExecutionResult::Next
            }
            Instruction::F32Add => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a + b));

                ExecutionResult::Next
            }
            Instruction::F32Sub => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a - b));

                ExecutionResult::Next
            }
            Instruction::F32Mul => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F32(a * b));

                ExecutionResult::Next
            }
            Instruction::F32Div => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                // Unlike the integer divides this never traps: IEEE 754 gives
                // `±inf` for a non-zero numerator over zero, and NaN for `0.0/0.0`.
                self.stack.push(Val::F32(a / b));

                ExecutionResult::Next
            }
            Instruction::F32Eq => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::I32((a == b) as i32));

                ExecutionResult::Next
            }
            Instruction::F32Ne => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::I32((a != b) as i32));

                ExecutionResult::Next
            }
            Instruction::F32Lt => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::I32((a < b) as i32));

                ExecutionResult::Next
            }
            Instruction::F32Gt => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::I32((a > b) as i32));

                ExecutionResult::Next
            }
            Instruction::F32Le => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::I32((a <= b) as i32));

                ExecutionResult::Next
            }
            Instruction::F32Ge => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::I32((a >= b) as i32));

                ExecutionResult::Next
            }
            Instruction::F32Min => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                let r = if a.is_nan() || b.is_nan() {
                    f32::NAN
                } else if a == b {
                    // -0.0 and +0.0 compare equal, so pick by sign: min wants -0.0
                    if a.is_sign_negative() { a } else { b }
                } else if a < b {
                    a
                } else {
                    b
                };

                self.stack.push(Val::F32(r));

                ExecutionResult::Next
            }
            Instruction::F32Max => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                let r = if a.is_nan() || b.is_nan() {
                    f32::NAN
                } else if a == b {
                    // -0.0 and +0.0 compare equal, so pick by sign: max wants +0.0
                    if a.is_sign_positive() { a } else { b }
                } else if a > b {
                    a
                } else {
                    b
                };

                self.stack.push(Val::F32(r));

                ExecutionResult::Next
            }
            Instruction::F32Copysign => {
                let b = self.stack.pop().as_f32();
                let a = self.stack.pop().as_f32();

                // Purely a sign-bit transplant: the magnitude of `a` with the sign
                // of `b`. Defined even when either operand is NaN — the sign is
                // copied without inspecting the payload — so unlike `min`/`max`
                // this needs no NaN special case, and Rust's method matches.
                self.stack.push(Val::F32(a.copysign(b)));

                ExecutionResult::Next
            }
            Instruction::F64Abs => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a.abs()));

                ExecutionResult::Next
            }
            Instruction::F64Neg => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a.neg()));

                ExecutionResult::Next
            }
            Instruction::F64Ceil => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a.ceil()));

                ExecutionResult::Next
            }
            Instruction::F64Floor => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a.floor()));

                ExecutionResult::Next
            }
            Instruction::F64Trunc => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a.trunc()));

                ExecutionResult::Next
            }
            Instruction::F64Sqrt => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a.sqrt()));

                ExecutionResult::Next
            }
            Instruction::F64Nearest => {
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a.round_ties_even()));

                ExecutionResult::Next
            }
            Instruction::F64ConvertI32U => {
                let a = self.stack.pop().as_i32() as u32;

                self.stack.push(Val::F64(a as f64));

                ExecutionResult::Next
            }
            Instruction::F64ConvertI32S => {
                let a = self.stack.pop().as_i32();

                self.stack.push(Val::F64(a as f64));

                ExecutionResult::Next
            }
            Instruction::F64ConvertI64U => {
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::F64(a as f64));

                ExecutionResult::Next
            }
            Instruction::F64ConvertI64S => {
                let a = self.stack.pop().as_i64();

                self.stack.push(Val::F64(a as f64));

                ExecutionResult::Next
            }
            Instruction::F64PromoteF32 => {
                let a = self.stack.pop().as_f32();

                self.stack.push(Val::F64(a as f64));

                ExecutionResult::Next
            }
            Instruction::F64ReinterpretI64 => {
                let a = self.stack.pop().as_i64() as u64;

                self.stack.push(Val::F64(f64::from_bits(a)));

                ExecutionResult::Next
            }
            Instruction::F64Add => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a + b));

                ExecutionResult::Next
            }
            Instruction::F64Sub => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a - b));

                ExecutionResult::Next
            }
            Instruction::F64Mul => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::F64(a * b));

                ExecutionResult::Next
            }
            Instruction::F64Div => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                // See `F32Div`: division by zero yields an infinity or NaN, never
                // a trap.
                self.stack.push(Val::F64(a / b));

                ExecutionResult::Next
            }
            Instruction::F64Eq => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::I32((a == b) as i32));

                ExecutionResult::Next
            }
            Instruction::F64Ne => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::I32((a != b) as i32));

                ExecutionResult::Next
            }
            Instruction::F64Lt => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::I32((a < b) as i32));

                ExecutionResult::Next
            }
            Instruction::F64Gt => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::I32((a > b) as i32));

                ExecutionResult::Next
            }
            Instruction::F64Le => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::I32((a <= b) as i32));

                ExecutionResult::Next
            }
            Instruction::F64Ge => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                self.stack.push(Val::I32((a >= b) as i32));

                ExecutionResult::Next
            }
            Instruction::F64Min => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                let r = if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else if a == b {
                    // -0.0 and +0.0 compare equal, so pick by sign: min wants -0.0
                    if a.is_sign_negative() { a } else { b }
                } else if a < b {
                    a
                } else {
                    b
                };

                self.stack.push(Val::F64(r));

                ExecutionResult::Next
            }
            Instruction::F64Max => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                let r = if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else if a == b {
                    // -0.0 and +0.0 compare equal, so pick by sign: max wants +0.0
                    if a.is_sign_positive() { a } else { b }
                } else if a > b {
                    a
                } else {
                    b
                };

                self.stack.push(Val::F64(r));

                ExecutionResult::Next
            }
            Instruction::F64Copysign => {
                let b = self.stack.pop().as_f64();
                let a = self.stack.pop().as_f64();

                // See `F32Copysign`: magnitude of `a`, sign of `b`, NaN included.
                self.stack.push(Val::F64(a.copysign(b)));

                ExecutionResult::Next
            }
            Instruction::LocalGet { index } => {
                self.stack.push(self.get_local(*index));

                ExecutionResult::Next
            }
            Instruction::LocalSet { index } => {
                let val = self.stack.pop();

                self.set_local(*index, val);

                ExecutionResult::Next
            }
            Instruction::LocalTee { index } => {
                let val = self.stack.tee();

                self.set_local(*index, val);

                ExecutionResult::Next
            }
            Instruction::GlobalGet { index } => {
                self.stack.push(self.instance.global_vals[index.0 as usize]);

                ExecutionResult::Next
            }
            Instruction::GlobalSet { index } => {
                let val = self.stack.pop();

                self.instance.global_vals[index.0 as usize] = val;

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
            Instruction::Br {
                target_index,
                arity,
                recorded_height,
            } => {
                // Unwind to the target label's absolute height (frame base + its
                // recorded height) while keeping the top `arity` values, then jump.
                self.stack.truncate_by_preserving_arity(
                    *recorded_height + self.frame_base_height,
                    *arity,
                );

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
                        *recorded_height + self.frame_base_height,
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
                    branch.recorded_height + self.frame_base_height,
                    branch.arity,
                );

                ExecutionResult::JumpTo(branch.target_index)
            }
            Instruction::Return {
                target_index,
                arity,
                recorded_height,
            } => {
                self.stack.truncate_by_preserving_arity(
                    *recorded_height + self.frame_base_height,
                    *arity,
                );

                ExecutionResult::JumpTo(*target_index)
            }
            Instruction::End {
                arity,
                recorded_height,
            } => {
                // Sanity check the height model: when a block closes, the stack must
                // hold exactly its `arity` results above the label's recorded height.
                // Both are frame-relative, so shift by this frame's base to compare
                // against the shared stack's absolute height.
                debug_assert!(
                    self.stack.height() == *recorded_height + *arity + self.frame_base_height
                );

                ExecutionResult::Next
            }
        };

        // Resolve the control-flow decision into the next `pc` here, in the same
        // function that produced it — see the note on this fn for why.
        Ok(match res {
            ExecutionResult::Next => pc + 1,
            ExecutionResult::JumpTo(target) => target,
        })
    }
}
