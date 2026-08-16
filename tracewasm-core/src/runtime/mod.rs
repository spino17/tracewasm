//! The TraceWasm interpreter: a tree-walker over the flat, pre-lowered
//! instruction list produced by [`crate::instruction`].
//!
//! ## Execution model
//!
//! Each function body is a `Vec<Instruction>` with control flow already
//! resolved to absolute instruction indices and operand-stack heights
//! precomputed. Execution is a simple `pc` loop: [`TraceVM::execute_instruction`]
//! runs one instruction and reports what the driver should do next via [`Step`]
//! — advance (`Next`), jump (`JumpTo`), enter a callee (`Call`), or (implicitly,
//! by advancing past the final `End`) return from the function.
//!
//! [`TraceVM::execute_on_native_stack`] is that driver. A `Call` makes it recurse
//! into itself, so one native frame is live per active wasm frame; the depth is
//! bounded by [`Config::max_call_stack_depth`](crate::instance::config::Config)
//! rather than by the host stack, since overflowing the latter aborts the process
//! instead of unwinding.
//!
//! ## One shared stack across all frames — locals included
//!
//! Rather than giving every call its own operand stack, the whole call tree
//! shares a single `Stack`. A call does not allocate a new stack; the callee
//! simply builds its operands on top of the caller's. Recursion still uses the
//! native Rust call stack (one [`TraceVM::execute_on_native_stack`] frame per
//! active wasm call), but the potentially-large value stack is allocated exactly
//! once.
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
//! Instruction indices, by contrast, are per-function: each
//! [`TraceVM::execute_on_native_stack`] invocation has its own `instructions`
//! slice and `pc`.

use crate::{
    error::{
        CallIndirectError::{self, FunctionCall},
        FuncCallError, InstructionExecutionError, MemoryAccessKind, MemoryError, TraceRecord,
        TraceRecordKind, TraceWasmError,
    },
    instance::{
        Instance,
        traits::{ImportRegistry, ParamVals, ResultVals},
    },
    instruction::{
        Instruction, RuntimeFrame,
        stack::{StackBrTableTarget, StackInstruction},
    },
    memory::Memory,
    module::{FuncIndex, FuncKind, LocalIndex, Module, TableIndex, ValType, formatted_val_types},
    runtime::{
        stack::Stack,
        value::{DataVal, Val, Value},
    },
};
use smallvec::{SmallVec, smallvec};
use std::{
    ops::{BitAnd, BitOr, BitXor, Neg},
    sync::Arc,
};

pub(crate) mod reg;
pub(crate) mod stack;
pub mod value;

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

/// What [`TraceVM::execute_instruction`] tells its driver to do next.
///
/// Everything an instruction can do to the operand stack, memory, globals and
/// tables it does itself, against the [`Instance`]. Only the things the *driver*
/// owns — the program counter and the call stack — come back through here.
enum Step {
    /// Continue at the following instruction.
    Next,
    /// Enter a locally-defined callee. An imported callee never produces this:
    /// it runs to completion inside the dispatch and yields [`Self::Next`].
    Call {
        func_index: FuncIndex,
        /// How many operands the callee takes; they are already on the stack and
        /// become its leading local slots.
        params_count: u32,
        /// `Some(table)` when the call came from `call_indirect`, recorded so a
        /// trap can say which table it went through.
        is_indirect: Option<TableIndex>,
    },
    /// Continue at this index into the current function's instruction list.
    JumpTo(u32),
}

/// A caller suspended at a `call`, saved by the driver that keeps its call stack
/// in a `Vec` rather than on the native stack.
///
/// Holds everything needed to resume the caller, plus what the trace needs to
/// describe the call if the callee traps. The slice references borrow from the
/// [`Module`], so saving a frame copies pointers rather than instructions.
struct Frame<'a> {
    func_index: FuncIndex,
    instructions: &'a [StackInstruction],
    /// Index of the `call` itself, not the instruction after it; resuming adds
    /// one, and a trace records the call site.
    pc: u32,
    caller_base_height: u32,
    frame_base_height: u32,
    br_table_targets: &'a [StackBrTableTarget],
    instruction_offsets: &'a [u32],
    /// Result count of the callee, needed on return to know how much of its
    /// region to keep when truncating back to `caller_base_height`.
    callee_results_len: u32,
    /// Who this frame called, and through which table if indirect. Only the trace
    /// reads these.
    callee_func_index: FuncIndex,
    callee_is_indirect: Option<TableIndex>,
}

/// Namespace for the interpreter's entry points. Carries no state of its own:
/// everything mutable lives on the [`Instance`], and a frame's own data lives in
/// the locals of the driver running it.
pub(crate) struct TraceVM;

impl TraceVM {
    /*pub(crate) fn run_<M: Memory, I: ImportRegistry, Instr: Instruction>(
        func_index: FuncIndex,
        params: &[Val],
        instance: &mut Instance<M, I, Instr>,
        module: &Arc<Module<Instr>>,
    ) -> Result<ResultVals, FuncCallError> {
        let mut call_stack_depth = 0;

        instance.frame.reset();

        for param in params {
            instance.frame.push(param.into());
        }

        // The reset above put the stack at height 0, so this frame's base is 0.
        Self::execute_on_native_stack_::<_, _, Instr>(
            func_index,
            instance,
            module,
            &mut call_stack_depth,
        )
        .map_err(|trace| func_call_err_from_unwind(func_index, trace, module))?;

        // How many result values the function leaves on the stack.
        let func_decl = &module.func_decls[func_index.0 as usize];
        let results_ty = &module.types[func_decl.ty.0 as usize].results;
        let results_len = results_ty.len() as u32;

        let results = instance.frame.pop_results(results_len);

        let mut s: SmallVec<[Val; 3]> = smallvec![];

        for (i, result) in results.iter().enumerate() {
            s.push(result.into_val(&results_ty[i]));
        }

        debug_assert!(instance.frame.height() == 0);

        Ok(ResultVals::new(s))
    }

    fn func_body<Instr: Instruction>(
        func_index: FuncIndex,
        module: &Module<Instr>,
    ) -> &FuncBody<Instr> {
        todo!()
    }

    pub(crate) fn execute_on_native_stack_<M: Memory, I: ImportRegistry, Instr: Instruction>(
        func_index: FuncIndex,
        instance: &mut Instance<M, I, Instr>,
        module: &Arc<Module<Instr>>,
        call_stack_depth: &mut u32,
    ) -> Result<(), Unwind> {
        // `func_bodies` holds only locally-defined functions, so shift the global
        // function index down by the number of imports to index into it.
        let imported_func_count = module.imported_func_count;

        debug_assert!(func_index.0 >= imported_func_count);

        let func_decl = &module.func_decls[func_index.0 as usize];
        let ty = &module.types[func_decl.ty.0 as usize];
        let params_len = ty.params.len();
        let results_len = ty.results.len();
        let func_body: &FuncBody<Instr> = Self::func_body(func_index, module);
        let instructions = &func_body.instructions;
        let instruction_offsets = &func_body.instruction_offsets;
        // Annotated as a slice so the `Box` is dereferenced here rather than at the
        // call in the loop. The dispatch takes `&[TargetBranch]`, and coercing a
        // `&Box<[_]>` to it re-reads the pointer and length out of the `Box` — two
        // loads for a pair that is fixed for the whole frame.
        let br_table_targets: &[Instr::BrTableTarget] = &func_body.br_table_targets;

        // `locals` in the body is laid out params-first, then declared locals,
        // and `locals_ty[i]` is the declared type of local slot `i`. So
        // `locals_len >= params_len` always, and the subtraction below cannot
        // underflow.
        let locals_ty = &func_body.locals;
        let locals_len = locals_ty.len();

        // Frame setup. The arguments are already the topmost values on the shared
        // stack, so the locals region starts just below them and the args become
        // slots `0..params_len` without being copied anywhere.
        let caller_base_height = instance.frame.height() - params_len as u32;

        // The declared locals follow the params and must start at the zero value
        // of their type, per the spec. Pushing them here is what makes the locals
        // region contiguous, so `get_local` can index it directly.
        for i in 0..(locals_len - params_len) {
            let ty = locals_ty[i + params_len];

            instance.frame.push(Value::zero_of_ty(ty));
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

        // Both are fixed for the whole frame, so they are computed here rather
        // than in the call below: an expression in the argument list is an add and
        // a load per instruction executed, not per frame.
        let frame_base_height = caller_base_height + locals_len as u32;

        /*loop {
            let instr = &instructions[pc];

            let step = Self::execute_instruction(
                instr,
                caller_base_height,
                frame_base_height,
                module,
                instance,
                br_table_targets,
                imported_func_count,
            );

            let step = match step {
                Ok(step) => step,
                Err(err) => return Err(trap_here(func_index, pc, instruction_offsets, *err)),
            };

            match step {
                Step::Next => {
                    pc += 1;
                }
                Step::JumpTo(target_index) => {
                    pc = target_index as usize;
                }
                Step::Call {
                    func_index: callee_func_index,
                    params_count: _,
                    is_indirect,
                } => {
                    // Checked *before* recursing: the callee increments the counter
                    // on entry, so testing after would admit one frame past the
                    // limit. Every wasm frame costs a native one here, so the
                    // ceiling is what keeps a deep guest from overflowing the host
                    // stack, which aborts rather than unwinding.
                    let max_depth = instance.config().get_max_call_stack_depth();

                    if *call_stack_depth >= max_depth {
                        return Err(trap_here(
                            func_index,
                            pc,
                            instruction_offsets,
                            InstructionExecutionError::CallStackExhausted(max_depth),
                        ));
                    }

                    if let Err(mut trace) = TraceVM::execute_on_native_stack(
                        callee_func_index,
                        instance,
                        module,
                        call_stack_depth,
                    ) {
                        // This frame is one link in the chain that led to the
                        // trap, so it appends itself as the error passes through.
                        trace.push(TraceRecord {
                            func_index,
                            instr_index: pc,
                            kind: TraceRecordKind::Call {
                                callee_index: callee_func_index,
                                is_indirect,
                            },
                            instr_offset: instruction_offsets[pc],
                        });

                        return Err(trace);
                    }

                    // The callee left its results in the caller's argument slots,
                    // so there is nothing to do but resume past the call.
                    pc += 1;
                }
            }

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
        instance
            .frame
            .truncate_by_preserving_arity(caller_base_height, results_len as u32);*/

        Ok(())
    }*/
}

impl TraceVM {
    /// Top-level entry point: runs a locally-defined function and returns its
    /// result values (in declaration order).
    ///
    /// Resets [`Instance::stack`] to empty, pushes `params`, and delegates to
    /// [`Self::execute_on_native_stack`], which leaves the results on that stack
    /// for this function to pop.
    ///
    /// Resetting on entry rather than on exit is what makes an instance reusable
    /// after a trap: a failing call returns early with values still on the stack,
    /// and the next call clears them.
    ///
    /// # Errors
    ///
    /// A [`FuncCallError`] for any trap or host-function failure, carrying the
    /// backtrace assembled during the unwind. This is the only place the
    /// interpreter's [`Unwind`] is turned into an error the caller sees, because
    /// it is the only frame that knows the entry function the call was made
    /// through.
    pub(crate) fn run<M: Memory, I: ImportRegistry>(
        func_index: FuncIndex,
        params: &[Val],
        instance: &mut Instance<M, I, StackInstruction>,
        module: &Arc<Module<StackInstruction>>,
    ) -> Result<ResultVals, FuncCallError<StackInstruction>> {
        let mut call_stack_depth = 0;

        instance.frame.reset();
        instance.frame.set_params(params);

        // The reset above put the stack at height 0, so this frame's base is 0.
        Self::execute_on_native_stack(func_index, instance, module, &mut call_stack_depth)
            .map_err(|trace| func_call_err_from_unwind(func_index, trace, module))?;

        // How many result values the function leaves on the stack.
        let func_decl = &module.func_decls[func_index.0 as usize];
        let results_ty = &module.types[func_decl.ty.0 as usize].results;
        let results_len = results_ty.len() as u32;

        let results = instance.frame.results(results_len);

        let mut s: SmallVec<[Val; 3]> = smallvec![];

        for (i, result) in results.iter().enumerate() {
            s.push(result.into_val(&results_ty[i]));
        }

        Ok(ResultVals::new(s))
    }

    /// Runs one (locally-defined) function to completion on the shared stack.
    ///
    /// Called both as the top-level entry (with an empty stack, so this frame's
    /// base is 0) and recursively from its own driver loop, which is where a
    /// [`Step::Call`] is acted on — the instruction arms themselves only report
    /// the call, they do not perform it.
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
    /// An [`Unwind`]: the trace of the failure so far, innermost frame first.
    /// The trapping instruction records itself, and each frame the error passes
    /// through on its way out appends its own `call`. Frame setup cannot fail,
    /// since `Module::compile` already rejected the local types the VM does not
    /// model.
    ///
    /// Returning the trace rather than taking a buffer to fill keeps it off the
    /// dispatch loop entirely: the loop has no accumulator to keep live, and the
    /// native unwind visits exactly the frames the trace needs, in the order it
    /// needs them. [`Self::run`] turns it into a [`FuncCallError`] once, where the
    /// entry function's name and the module are in reach.
    fn execute_on_native_stack<M: Memory, I: ImportRegistry>(
        func_index: FuncIndex,
        instance: &mut Instance<M, I, StackInstruction>,
        module: &Arc<Module<StackInstruction>>,
        call_stack_depth: &mut u32,
    ) -> Result<(), Unwind<StackInstruction>> {
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
        // Annotated as a slice so the `Box` is dereferenced here rather than at the
        // call in the loop. The dispatch takes `&[TargetBranch]`, and coercing a
        // `&Box<[_]>` to it re-reads the pointer and length out of the `Box` — two
        // loads for a pair that is fixed for the whole frame.
        let br_table_targets: &[StackBrTableTarget] = &func_body.frame_layout.br_targets_arena;

        // `locals` in the body is laid out params-first, then declared locals,
        // and `locals_ty[i]` is the declared type of local slot `i`. So
        // `locals_len >= params_len` always, and the subtraction below cannot
        // underflow.
        let locals_ty = &func_body.locals;
        let locals_len = locals_ty.len();

        // Frame setup. The arguments are already the topmost values on the shared
        // stack, so the locals region starts just below them and the args become
        // slots `0..params_len` without being copied anywhere.
        let caller_base_height = instance.frame.height() - params_len as u32;

        // The declared locals follow the params and must start at the zero value
        // of their type, per the spec. Pushing them here is what makes the locals
        // region contiguous, so `get_local` can index it directly.
        for i in 0..(locals_len - params_len) {
            let ty = locals_ty[i + params_len];

            instance.frame.push(Value::zero_of_ty(ty));
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

        // Both are fixed for the whole frame, so they are computed here rather
        // than in the call below: an expression in the argument list is an add and
        // a load per instruction executed, not per frame.
        let frame_base_height = caller_base_height + locals_len as u32;

        loop {
            let instr = &instructions[pc];

            let step = Self::execute_instruction(
                instr,
                caller_base_height,
                frame_base_height,
                module,
                instance,
                br_table_targets,
                imported_func_count,
            );

            let step = match step {
                Ok(step) => step,
                Err(err) => return Err(trap_here(func_index, pc, instruction_offsets, *err)),
            };

            match step {
                Step::Next => {
                    pc += 1;
                }
                Step::JumpTo(target_index) => {
                    pc = target_index as usize;
                }
                Step::Call {
                    func_index: callee_func_index,
                    params_count: _,
                    is_indirect,
                } => {
                    // Checked *before* recursing: the callee increments the counter
                    // on entry, so testing after would admit one frame past the
                    // limit. Every wasm frame costs a native one here, so the
                    // ceiling is what keeps a deep guest from overflowing the host
                    // stack, which aborts rather than unwinding.
                    let max_depth = instance.config().get_max_call_stack_depth();

                    if *call_stack_depth >= max_depth {
                        return Err(trap_here(
                            func_index,
                            pc,
                            instruction_offsets,
                            InstructionExecutionError::CallStackExhausted(max_depth),
                        ));
                    }

                    if let Err(mut trace) = TraceVM::execute_on_native_stack(
                        callee_func_index,
                        instance,
                        module,
                        call_stack_depth,
                    ) {
                        // This frame is one link in the chain that led to the
                        // trap, so it appends itself as the error passes through.
                        trace.push(TraceRecord {
                            func_index,
                            instr_index: pc,
                            kind: TraceRecordKind::Call {
                                callee_index: callee_func_index,
                                is_indirect,
                            },
                            instr_offset: instruction_offsets[pc],
                        });

                        return Err(trace);
                    }

                    // The callee left its results in the caller's argument slots,
                    // so there is nothing to do but resume past the call.
                    pc += 1;
                }
            }

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
        instance
            .frame
            .truncate_by_preserving_arity(caller_base_height, results_len as u32);

        Ok(())
    }

    /// The same interpreter, driving its call stack from a `Vec<Frame>` instead
    /// of recursing.
    ///
    /// **Not wired up** — [`Self::run`] uses [`Self::execute_on_native_stack`].
    /// It is kept because it is the only shape that decouples guest call depth
    /// from the host stack: nothing here grows the native frame per wasm call, so
    /// depth would be bounded by memory rather than by
    /// [`Config::max_call_stack_depth`](crate::instance::config::Config).
    ///
    /// It is measurably slower, and for a reason that is structural rather than
    /// incidental. The frame state it must carry across the dispatch loop —
    /// instruction slice, offsets, branch targets, both base heights — occupies
    /// registers that the recursive driver leaves free, because there they are
    /// fixed for the life of the call and here they change on every call and
    /// return. The opcode arms lose that contention: the operand-stack pointer
    /// stops living in a register, so a push no longer folds into the pop before
    /// it, and the arms' shared tail becomes a general push instead.
    ///
    /// Restructuring around an inner per-frame loop, and outlining the
    /// call/return transitions, were both tried and moved it very little; the
    /// contention is not in how those are written.
    ///
    /// Its depth check tests `frames.len()` rather than a separate counter, since
    /// the `Vec` length *is* the depth.
    fn _execute_on_frame_stack<M: Memory, I: ImportRegistry>(
        mut func_index: FuncIndex,
        instance: &mut Instance<M, I, StackInstruction>,
        module: &Arc<Module<StackInstruction>>,
    ) -> Result<(), FuncCallError<StackInstruction>> {
        let mut frames: Vec<Frame> = Vec::with_capacity(10);
        let imported_func_count = module.imported_func_count;

        debug_assert!(func_index.0 >= imported_func_count);

        let top_enclosing_func_index = func_index;
        let func_decl = &module.func_decls[func_index.0 as usize];
        let ty = &module.types[func_decl.ty.0 as usize];
        let params_len = ty.params.len();
        let results_len = ty.results.len();
        let func_body = &module.func_bodies[(func_index.0 - imported_func_count) as usize];
        let mut instructions = func_body.instructions.as_ref();
        let mut instruction_offsets = func_body.instruction_offsets.as_ref();
        let mut br_table_targets = func_body.frame_layout.br_targets_arena.as_ref();

        let locals_ty = &func_body.locals;
        let locals_len = locals_ty.len();
        let mut caller_base_height: u32 = 0;
        let mut frame_base_height: u32 = locals_len as u32;
        let mut pc = 0;

        for i in 0..(locals_len - params_len) {
            let ty = locals_ty[i + params_len];

            instance.frame.push(Value::zero_of_ty(ty));
        }

        loop {
            let instr = &instructions[pc];

            let step = Self::execute_instruction(
                instr,
                caller_base_height,
                frame_base_height,
                module,
                instance,
                br_table_targets,
                imported_func_count,
            );

            let step = match step {
                Ok(step) => step,
                Err(err) => {
                    return Err(func_call_err(
                        top_enclosing_func_index,
                        frames,
                        *err,
                        pc,
                        instruction_offsets[pc],
                        module,
                    ));
                }
            };

            match step {
                Step::Next => {
                    pc += 1;
                }
                Step::JumpTo(target_index) => {
                    pc = target_index as usize;
                }
                Step::Call {
                    func_index: callee_func_index,
                    params_count: callee_params_count,
                    is_indirect,
                } => {
                    let max_depth = instance.config().get_max_call_stack_depth();

                    if frames.len() as u32 >= max_depth {
                        return Err(func_call_err(
                            top_enclosing_func_index,
                            frames,
                            InstructionExecutionError::CallStackExhausted(max_depth),
                            pc,
                            instruction_offsets[pc],
                            module,
                        ));
                    }

                    // only local function calls come here
                    debug_assert!(callee_func_index.0 >= imported_func_count);

                    let callee_func_ty = module.func_decls[callee_func_index.0 as usize].ty;
                    let callee_results_len = module.types[callee_func_ty.0 as usize].results.len();
                    let callee_func_body =
                        &module.func_bodies[(callee_func_index.0 - imported_func_count) as usize];
                    let callee_instructions = &callee_func_body.instructions;
                    let callee_locals_ty = &callee_func_body.locals;
                    let callee_br_table_targets = &callee_func_body.frame_layout.br_targets_arena;
                    let callee_instruction_offsets = &callee_func_body.instruction_offsets;

                    // save current frame's state
                    frames.push(Frame {
                        func_index,
                        instructions,
                        pc: pc as u32,
                        caller_base_height,
                        frame_base_height,
                        br_table_targets,
                        instruction_offsets,
                        callee_results_len: callee_results_len as u32,
                        callee_func_index,
                        callee_is_indirect: is_indirect,
                    });

                    let callee_locals_len = callee_locals_ty.len();
                    let callee_params_len = callee_params_count as usize;

                    // override the current state with callee
                    func_index = callee_func_index;
                    instructions = callee_instructions;
                    pc = 0;
                    caller_base_height = instance.frame.height() - callee_params_count;
                    frame_base_height = caller_base_height + callee_locals_len as u32;
                    br_table_targets = callee_br_table_targets;
                    instruction_offsets = callee_instruction_offsets;

                    // setup the callee frame
                    for i in 0..(callee_locals_len - callee_params_len) {
                        let ty = callee_locals_ty[i + callee_params_len];

                        instance.frame.push(Value::zero_of_ty(ty));
                    }

                    continue;
                }
            }

            if pc == instructions.len() {
                if frames.is_empty() {
                    break;
                }

                // pop the frame!
                let frame = frames.pop().unwrap(); // safe to unwrap as checked above!

                // tear down the locals and operands of the frame to just retain the results on top of the
                // old frame's height
                instance
                    .frame
                    .truncate_by_preserving_arity(caller_base_height, frame.callee_results_len);

                // reset the state of the frame which executed call instruction
                func_index = frame.func_index;
                instructions = frame.instructions;
                pc = frame.pc as usize + 1; // next instruction past the call
                caller_base_height = frame.caller_base_height;
                frame_base_height = frame.frame_base_height;
                br_table_targets = frame.br_table_targets;
                instruction_offsets = frame.instruction_offsets;
            }
        }

        instance
            .frame
            .truncate_by_preserving_arity(caller_base_height, results_len as u32);

        Ok(())
    }

    /// Runs an imported callee to completion: pops its parameters, re-tags them
    /// from its declared types, invokes the host function, pushes its results.
    ///
    /// The caller decides which callees come here; a locally-defined one is
    /// reported to the driver as [`Step::Call`] at the instruction arm instead.
    ///
    /// **Returns `Result<(), _>` rather than a [`Step`], deliberately.** With the
    /// error boxed that is eight bytes — a null-niche pointer — so it comes back
    /// in a register. Returning a `Result<Step, _>` makes it sixteen, which this
    /// ABI hands back through a caller-provided pointer, and the pointer it would
    /// be given is the dispatch's own result slot. That slot's address escaping to
    /// a real call is enough to stop the compiler keeping it in registers, and
    /// then *every* instruction pays a store and a reload for it — not just the
    /// two arms that can reach here.
    ///
    /// `is_indirect` only labels the failure: it selects which error variant wraps
    /// a host error.
    ///
    /// # Errors
    ///
    /// Whatever the imported callee returned, wrapped to name the callee.
    fn call_imported<M: Memory, I: ImportRegistry>(
        callee_func_index: FuncIndex,
        callee_params_count: u32,
        module: &Module<StackInstruction>,
        instance: &mut Instance<M, I, StackInstruction>,
        is_indirect: Option<TableIndex>,
    ) -> Result<(), Box<InstructionExecutionError<StackInstruction>>> {
        let callee_func_decl = &module.func_decls[callee_func_index.0 as usize];
        let callee_params_ty = &module.types[callee_func_decl.ty.0 as usize].params;

        debug_assert!(matches!(callee_func_decl.kind, FuncKind::Imported { .. }));

        let FuncKind::Imported {
            module_name,
            imported_func_name,
        } = &callee_func_decl.kind
        else {
            unreachable!()
        };

        // The stack holds untagged `Value`s, so the host boundary re-tags them
        // from the callee's declared parameter types. Collected into a
        // `ParamVals` rather than a boxed slice: it is a `SmallVec`, so an
        // import of five or fewer parameters costs no allocation.
        let callee_params = ParamVals::new(
            instance
                .frame
                .pop_params(callee_params_count)
                .iter()
                .zip(callee_params_ty)
                .map(|(param, ty)| param.into_val(ty))
                .collect(),
        );

        // `execute` returns a stack-allocated `ResultVals` (no heap for <=3 results).
        let results = instance
            .import_registry
            .execute(
                module_name,
                imported_func_name,
                callee_params.as_ref(),
                &mut instance.memory,
            )
            .map_err(|err| {
                Box::new(match is_indirect {
                    Some(table_index) => InstructionExecutionError::CallIndirect(
                        table_index,
                        FunctionCall(callee_func_index, Box::new(err.into())),
                    ),
                    None => {
                        InstructionExecutionError::Call(callee_func_index, Box::new(err.into()))
                    }
                })
            })?;

        // push results to the stack
        for res in results {
            instance.frame.push(res.into());
        }

        Ok(())
    }

    #[inline(always)]
    fn get_local<M: Memory, I: ImportRegistry>(
        index: LocalIndex,
        caller_base_height: u32,
        instance: &Instance<M, I, StackInstruction>,
    ) -> Value {
        let slot = (index.0 + caller_base_height) as usize;

        // Mirrors the SAFETY argument below, so a broken link shows up as a failed
        // test rather than as a silent out-of-bounds read. Compiled out in release.
        debug_assert!(
            slot < instance.frame.inner.len(),
            "local slot {slot} is outside the operand stack's backing storage \
             (len {}) — one of the invariants in the SAFETY comment no longer holds",
            instance.frame.inner.len()
        );

        // SAFETY: `slot < inner.len()`, which needs four separate facts. Only the
        // first belongs to `wasmparser`; the other three are this crate's own and
        // are the ones that can rot:
        //
        // 1. `index.0 < locals_len` — guaranteed by validation, which runs over the
        //    whole module in `Module::compile` before any lowering.
        // 2. `stack_pointer == caller_base_height + locals_len` once the frame is
        //    set up: whichever driver entered the frame derives
        //    `caller_base_height` by subtracting the arguments already on the
        //    stack, then pushes the remaining declared locals. Changing that setup
        //    invalidates this.
        // 3. `stack_pointer <= inner.len()` — the operand-stack invariant documented
        //    in `vm::stack`.
        // 4. `inner.len()` never shrinks. Nothing truncates, clears, resizes or
        //    shrinks it; `pop`/`truncate`/`reset` only move `stack_pointer`. Adding
        //    any such call would break this.
        //
        // Together: `caller_base_height + index.0 < stack_pointer <= inner.len()`.
        //
        // Constant expressions cannot reach here at all — they run on the much
        // smaller `Stack::for_const_expr_evaluation`, and
        // `emit_instruction_for_const_expr` accepts a closed whitelist of operators
        // that excludes `local.get`/`local.set`/`local.tee`.
        unsafe { *instance.frame.inner.get_unchecked(slot) }
    }

    /// Writes local slot `index` of the frame based at `caller_base_height`.
    ///
    /// The mirror of [`Self::get_local`], and it rests on the same four
    /// invariants; see the SAFETY comment there.
    #[inline(always)]
    fn set_local<M: Memory, I: ImportRegistry>(
        index: LocalIndex,
        val: Value,
        caller_base_height: u32,
        instance: &mut Instance<M, I, StackInstruction>,
    ) {
        let slot = (index.0 + caller_base_height) as usize;

        debug_assert!(
            slot < instance.frame.inner.len(),
            "local slot {slot} is outside the operand stack's backing storage \
             (len {}) — one of the invariants in `get_local`'s SAFETY comment no \
             longer holds",
            instance.frame.inner.len()
        );

        // SAFETY: identical to [`Self::get_local`] — see the four invariants
        // enumerated there. Writing rather than reading needs nothing extra: the
        // slot is inside this frame's locals region, which the operand discipline
        // never touches for the life of the frame.
        unsafe {
            *instance.frame.inner.get_unchecked_mut(slot) = val;
        }
    }

    /// Pops a load/store's dynamic address and adds the instruction's static
    /// `memarg` offset, giving the byte offset to access.
    ///
    /// Shared by every memory instruction, which differ only in width.
    ///
    /// # Errors
    ///
    /// [`MemoryError::EffectiveAddressOverflow`] if the sum leaves the 32-bit
    /// address space. Checked rather than wrapped: wrapping would fold a
    /// far-out-of-bounds address back to a valid one, turning a trap into a
    /// silently wrong access. Being out of range is not itself the trap here —
    /// the access that follows is bounds-checked anyway — but the addition must
    /// not lose that fact.
    #[inline(always)]
    fn pop_effective_address<M: Memory, I: ImportRegistry>(
        memarg_offset: u32,
        instance: &mut Instance<M, I, StackInstruction>,
    ) -> Result<usize, MemoryError> {
        let addr = instance.frame.pop().as_i32() as u32;

        let effective_offset = addr
            .checked_add(memarg_offset)
            .ok_or(MemoryError::EffectiveAddressOverflow(addr, memarg_offset))?;

        Ok(effective_offset as usize)
    }

    /// Executes one instruction against `instance` and reports what the driver
    /// should do next.
    ///
    /// The core dispatch: one arm per [`Instruction`] variant. Effects on the
    /// operand stack, memory, globals and tables happen here; only the program
    /// counter and the call stack are handed back, as a [`Step`].
    ///
    /// Takes its frame's context as loose arguments rather than a `&self` or a
    /// state struct so that the driver can hold them in registers across the loop.
    /// Every argument is constant for the life of a frame, so the driver computes
    /// them once on entry — an expression evaluated in the argument list would be
    /// evaluated once per instruction executed instead.
    ///
    /// `#[inline(always)]` because the call boundary is not worth its price here:
    /// inlined, the arms share the driver's registers and the whole dispatch
    /// becomes a jump table inside one function. The cost is frame size — every
    /// arm's spill slots land in the driver's frame, which is why
    /// [`Config::max_call_stack_depth`](crate::instance::config::Config) is tied
    /// to the size of that frame.
    ///
    /// # Errors
    ///
    /// The trap the instruction raised, untagged: it says what went wrong, and
    /// the driver adds where.
    #[inline(always)]
    fn execute_instruction<M: Memory, I: ImportRegistry>(
        instr: &StackInstruction,
        caller_base_height: u32,
        frame_base_height: u32,
        module: &Module<StackInstruction>,
        instance: &mut Instance<M, I, StackInstruction>,
        br_table_targets: &[StackBrTableTarget],
        imported_func_count: u32,
    ) -> Result<Step, Box<InstructionExecutionError<StackInstruction>>> {
        let res = match instr {
            StackInstruction::Call {
                func_index: callee_func_index,
                params_count: callee_params_count,
            } => {
                // Which callee kind this is decides who runs it: a local one is
                // handed to the driver to enter, an imported one runs to
                // completion here and control simply falls through.
                if callee_func_index.0 >= imported_func_count {
                    Step::Call {
                        func_index: *callee_func_index,
                        params_count: *callee_params_count,
                        is_indirect: None,
                    }
                } else {
                    Self::call_imported(
                        *callee_func_index,
                        *callee_params_count,
                        module,
                        instance,
                        None,
                    )?;

                    Step::Next
                }
            }
            StackInstruction::CallIndirect {
                ty_index,
                table_index,
            } => {
                let table = &instance.table_vals[table_index.0 as usize];
                // The index operand is an unsigned i32; a negative value becomes a
                // large `usize` and fails the bounds check below.
                let slot = instance.frame.pop().as_i32() as u32 as usize;

                // Trap if the index is past the table's end (wasm: "undefined element").
                let Some(func_ref) = table.table.get(slot).copied() else {
                    return Err(Box::new(InstructionExecutionError::CallIndirect(
                        *table_index,
                        CallIndirectError::TableSlotOutOfBounds,
                    )));
                };

                // Trap on a null element (wasm: "uninitialized element").
                let Some(callee_func_index) = func_ref else {
                    return Err(Box::new(InstructionExecutionError::CallIndirect(
                        *table_index,
                        CallIndirectError::NullElementInTable,
                    )));
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
                    return Err(Box::new(signature_mismatch(
                        *table_index,
                        declared_params,
                        declared_results,
                        params,
                        results,
                    )));
                }

                // As for `Call` above; the table index rides along so a failure can
                // say which table the call went through.
                if callee_func_index.0 >= imported_func_count {
                    Step::Call {
                        func_index: callee_func_index,
                        params_count: declared_params.len() as u32,
                        is_indirect: Some(*table_index),
                    }
                } else {
                    Self::call_imported(
                        callee_func_index,
                        declared_params.len() as u32,
                        module,
                        instance,
                        Some(*table_index),
                    )?;

                    Step::Next
                }
            }
            StackInstruction::Unreachable => {
                return Err(Box::new(InstructionExecutionError::Unreachable));
            }
            StackInstruction::Nop => Step::Next,
            StackInstruction::I32Const { value } => {
                instance.frame.push(Value::from_i32(*value));

                Step::Next
            }
            StackInstruction::I64Const { value } => {
                instance.frame.push(Value::from_i64(*value));

                Step::Next
            }
            StackInstruction::F32Const { value } => {
                instance.frame.push(Value::from_f32(*value));

                Step::Next
            }
            StackInstruction::F64Const { value } => {
                instance.frame.push(Value::from_f64(*value));

                Step::Next
            }
            StackInstruction::RefNull => {
                instance.frame.push(Value::from_ref(None));

                Step::Next
            }
            StackInstruction::RefFunc { function_index } => {
                instance.frame.push(Value::from_ref(Some(*function_index)));

                Step::Next
            }
            StackInstruction::RefIsNull => {
                let func_ref = instance.frame.pop().as_ref();

                if func_ref.is_none() {
                    instance.frame.push(Value::from_i32(1));
                } else {
                    instance.frame.push(Value::from_i32(0));
                }

                Step::Next
            }
            StackInstruction::MemorySize => {
                instance
                    .frame
                    .push(Value::from_i32(instance.memory.size_in_pages() as i32));

                Step::Next
            }
            StackInstruction::MemoryGrow => {
                // The delta is an unsigned i32; going through `u32` keeps a
                // high-bit-set value from sign-extending into a different number.
                let delta_in_pages = instance.frame.pop().as_i32() as u32 as u64;
                let max_pages = instance.config.get_max_memory_size_in_pages();

                // `instantiate` already narrowed this to the module's declared
                // maximum, so the configured cap is the effective ceiling here.
                match instance.memory.grow(delta_in_pages, max_pages) {
                    Ok(old_page) => instance.frame.push(Value::from_i32(old_page as i32)),
                    // Per the spec `memory.grow` does not trap: a request it cannot
                    // satisfy reports `-1` and execution continues.
                    Err(_) => instance.frame.push(Value::from_i32(-1)),
                }

                Step::Next
            }
            StackInstruction::MemoryCopy => {
                let len = instance.frame.pop().as_i32() as usize;
                let src = instance.frame.pop().as_i32() as usize;
                let dest = instance.frame.pop().as_i32() as usize;

                instance.memory.copy_within(dest, src, len)?;

                Step::Next
            }
            StackInstruction::MemoryFill => {
                let len = instance.frame.pop().as_i32() as usize;
                let val = instance.frame.pop().as_i32() as u32;
                let dest = instance.frame.pop().as_i32() as usize;

                instance.memory.fill(dest, val, len)?;

                Step::Next
            }
            StackInstruction::MemoryInit { data_index } => {
                let len = instance.frame.pop().as_i32() as usize;
                let src = instance.frame.pop().as_i32() as usize;
                let dest = instance.frame.pop().as_i32() as usize;

                // A dropped segment reads as *empty*, not as an outright trap: the
                // spec replaces its bytes with the empty sequence, so a
                // zero-length `memory.init` after `data.drop` still succeeds while
                // any non-empty read fails the bounds check below.
                let segment: &[u8] = match &instance.data_vals[*data_index as usize] {
                    DataVal::Dropped => &[],
                    DataVal::Passive(segment) => &segment,
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
                instance.memory.write(dest, &segment[src..end])?;

                Step::Next
            }
            StackInstruction::DataDrop { data_index } => {
                instance.data_vals[*data_index as usize] = DataVal::Dropped;

                Step::Next
            }
            StackInstruction::I32Load { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_i32(effective_offset)?;

                instance.frame.push(Value::from_i32(val));

                Step::Next
            }
            StackInstruction::I32Load8U { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_u8(effective_offset)? as i32;

                instance.frame.push(Value::from_i32(val));

                Step::Next
            }
            StackInstruction::I32Load8S { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_i8(effective_offset)? as i32;

                instance.frame.push(Value::from_i32(val));

                Step::Next
            }
            StackInstruction::I32Load16U { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_u16(effective_offset)? as i32;

                instance.frame.push(Value::from_i32(val));

                Step::Next
            }
            StackInstruction::I32Load16S { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_i16(effective_offset)? as i32;

                instance.frame.push(Value::from_i32(val));

                Step::Next
            }
            StackInstruction::I64Load { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_i64(effective_offset)?;

                instance.frame.push(Value::from_i64(val));

                Step::Next
            }
            StackInstruction::I64Load8U { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_u8(effective_offset)? as i64;

                instance.frame.push(Value::from_i64(val));

                Step::Next
            }
            StackInstruction::I64Load8S { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_i8(effective_offset)? as i64;

                instance.frame.push(Value::from_i64(val));

                Step::Next
            }
            StackInstruction::I64Load16U { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_u16(effective_offset)? as i64;

                instance.frame.push(Value::from_i64(val));

                Step::Next
            }
            StackInstruction::I64Load16S { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_i16(effective_offset)? as i64;

                instance.frame.push(Value::from_i64(val));

                Step::Next
            }
            StackInstruction::I64Load32U { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_u32(effective_offset)? as i64;

                instance.frame.push(Value::from_i64(val));

                Step::Next
            }
            StackInstruction::I64Load32S { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_i32(effective_offset)? as i64;

                instance.frame.push(Value::from_i64(val));

                Step::Next
            }
            StackInstruction::F32Load { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_f32(effective_offset)?;

                instance.frame.push(Value::from_f32(val));

                Step::Next
            }
            StackInstruction::F64Load { offset, align: _ } => {
                let effective_offset = Self::pop_effective_address(*offset, instance)?;
                let val = instance.memory.read_f64(effective_offset)?;

                instance.frame.push(Value::from_f64(val));

                Step::Next
            }
            StackInstruction::I32Store { offset, align: _ } => {
                let val = instance.frame.pop().as_i32();
                let effective_offset = Self::pop_effective_address(*offset, instance)?;

                instance.memory.write_u32(effective_offset, val as u32)?;

                Step::Next
            }
            StackInstruction::I32Store8 { offset, align: _ } => {
                let val = instance.frame.pop().as_i32();
                let effective_offset = Self::pop_effective_address(*offset, instance)?;

                instance.memory.write_u8(effective_offset, val as u8)?;

                Step::Next
            }
            StackInstruction::I32Store16 { offset, align: _ } => {
                let val = instance.frame.pop().as_i32();
                let effective_offset = Self::pop_effective_address(*offset, instance)?;

                instance.memory.write_u16(effective_offset, val as u16)?;

                Step::Next
            }
            StackInstruction::I64Store { offset, align: _ } => {
                let val = instance.frame.pop().as_i64();
                let effective_offset = Self::pop_effective_address(*offset, instance)?;

                instance.memory.write_u64(effective_offset, val as u64)?;

                Step::Next
            }
            StackInstruction::I64Store8 { offset, align: _ } => {
                let val = instance.frame.pop().as_i64();
                let effective_offset = Self::pop_effective_address(*offset, instance)?;

                instance.memory.write_u8(effective_offset, val as u8)?;

                Step::Next
            }
            StackInstruction::I64Store16 { offset, align: _ } => {
                let val = instance.frame.pop().as_i64();
                let effective_offset = Self::pop_effective_address(*offset, instance)?;

                instance.memory.write_u16(effective_offset, val as u16)?;

                Step::Next
            }
            StackInstruction::I64Store32 { offset, align: _ } => {
                let val = instance.frame.pop().as_i64();
                let effective_offset = Self::pop_effective_address(*offset, instance)?;

                instance.memory.write_u32(effective_offset, val as u32)?;

                Step::Next
            }
            StackInstruction::F32Store { offset, align: _ } => {
                let val = instance.frame.pop().as_f32();
                let effective_offset = Self::pop_effective_address(*offset, instance)?;

                instance.memory.write_f32(effective_offset, val)?;

                Step::Next
            }
            StackInstruction::F64Store { offset, align: _ } => {
                let val = instance.frame.pop().as_f64();
                let effective_offset = Self::pop_effective_address(*offset, instance)?;

                instance.memory.write_f64(effective_offset, val)?;

                Step::Next
            }
            StackInstruction::I32Clz => {
                let a = instance.frame.pop().as_i32();

                instance
                    .frame
                    .push(Value::from_i32(a.leading_zeros() as i32));

                Step::Next
            }
            StackInstruction::I32Ctz => {
                let a = instance.frame.pop().as_i32();

                instance
                    .frame
                    .push(Value::from_i32(a.trailing_zeros() as i32));

                Step::Next
            }
            StackInstruction::I32Popcnt => {
                let a = instance.frame.pop().as_i32();

                // Counts set bits in the two's-complement representation, so a
                // negative operand counts its sign bits too — which is what the
                // spec's bit-level definition asks for.
                instance.frame.push(Value::from_i32(a.count_ones() as i32));

                Step::Next
            }
            StackInstruction::I32Eqz => {
                let a = instance.frame.pop().as_i32();

                instance
                    .frame
                    .push(Value::from_i32(if a == 0 { 1 } else { 0 }));

                Step::Next
            }
            StackInstruction::I32Extend8S => {
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32(a as i8 as i32));

                Step::Next
            }
            StackInstruction::I32Extend16S => {
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32(a as i16 as i32));

                Step::Next
            }
            StackInstruction::I32WrapI64 => {
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i32(a as i32));

                Step::Next
            }
            StackInstruction::I32TruncF32U => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = instance.frame.pop().as_f32() as f64;
                let truncated = trunc_float_to_int(a, 0.0, U32_TRUNC_HIGH, "u32")?;

                // The result is the `u32` bit pattern held in an `i32`, so values
                // above `i32::MAX` come back out negative.
                instance
                    .frame
                    .push(Value::from_i32(truncated as u32 as i32));

                Step::Next
            }
            StackInstruction::I32TruncF32S => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = instance.frame.pop().as_f32() as f64;
                let truncated = trunc_float_to_int(a, I32_TRUNC_LOW, I32_TRUNC_HIGH, "i32")?;

                instance.frame.push(Value::from_i32(truncated as i32));

                Step::Next
            }
            StackInstruction::I32TruncF64U => {
                let a = instance.frame.pop().as_f64();
                let truncated = trunc_float_to_int(a, 0.0, U32_TRUNC_HIGH, "u32")?;

                instance
                    .frame
                    .push(Value::from_i32(truncated as u32 as i32));

                Step::Next
            }
            StackInstruction::I32TruncF64S => {
                let a = instance.frame.pop().as_f64();
                let truncated = trunc_float_to_int(a, I32_TRUNC_LOW, I32_TRUNC_HIGH, "i32")?;

                instance.frame.push(Value::from_i32(truncated as i32));

                Step::Next
            }
            StackInstruction::I32TruncSatF32U => {
                let a = instance.frame.pop().as_f32() as u32;

                instance.frame.push(Value::from_i32(a as i32));

                Step::Next
            }
            StackInstruction::I32TruncSatF32S => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_i32(a as i32));

                Step::Next
            }
            StackInstruction::I32TruncSatF64U => {
                // Saturate to `u32`, the *target* width — going through `u64` here
                // would clamp at the wrong bound and then wrap on the way down.
                let a = instance.frame.pop().as_f64() as u32;

                instance.frame.push(Value::from_i32(a as i32));

                Step::Next
            }
            StackInstruction::I32TruncSatF64S => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_i32(a as i32));

                Step::Next
            }
            StackInstruction::I32ReinterpretF32 => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_i32(a.to_bits() as i32));

                Step::Next
            }
            StackInstruction::I32Add => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32(a.wrapping_add(b)));

                Step::Next
            }
            StackInstruction::I32Sub => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32(a.wrapping_sub(b)));

                Step::Next
            }
            StackInstruction::I32Mul => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32(a.wrapping_mul(b)));

                Step::Next
            }
            StackInstruction::I32DivU => {
                let b = instance.frame.pop().as_i32() as u32;
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_i32(a.checked_div(b).ok_or(
                    InstructionExecutionError::Division {
                        num: a.to_string(),
                        deno: b.to_string(),
                    },
                )? as i32));

                Step::Next
            }
            StackInstruction::I32DivS => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32(a.checked_div(b).ok_or(
                    InstructionExecutionError::Division {
                        num: a.to_string(),
                        deno: b.to_string(),
                    },
                )?));

                Step::Next
            }
            StackInstruction::I32RemU => {
                let b = instance.frame.pop().as_i32() as u32;
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_i32(a.checked_rem(b).ok_or(
                    InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    },
                )? as i32));

                Step::Next
            }
            StackInstruction::I32RemS => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                // A zero divisor is the *only* trap here. Unlike `i32.div_s`,
                // `rem_s` does not trap on overflow: the spec defines
                // `i32::MIN % -1` as `0`, which is what `wrapping_rem` returns.
                // `checked_rem` would wrongly report that case as a failure.
                if b == 0 {
                    return Err(Box::new(InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    }));
                }

                instance.frame.push(Value::from_i32(a.wrapping_rem(b)));

                Step::Next
            }
            StackInstruction::I32And => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32(a.bitand(b)));

                Step::Next
            }
            StackInstruction::I32Or => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32(a.bitor(b)));

                Step::Next
            }
            StackInstruction::I32Xor => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32(a.bitxor(b)));

                Step::Next
            }
            // Shift and rotate counts are taken modulo the operand width, so a
            // count of 32 or more is well defined rather than a trap or UB. The
            // `wrapping_*`/`rotate_*` methods apply exactly that masking; the plain
            // `<<`/`>>` operators would instead panic in debug builds.
            StackInstruction::I32Shl => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance
                    .frame
                    .push(Value::from_i32(a.wrapping_shl(b as u32)));

                Step::Next
            }
            StackInstruction::I32ShrU => {
                let b = instance.frame.pop().as_i32() as u32;
                let a = instance.frame.pop().as_i32() as u32;

                // Logical shift: done on `u32` so the vacated high bits are zeros.
                instance
                    .frame
                    .push(Value::from_i32(a.wrapping_shr(b) as i32));

                Step::Next
            }
            StackInstruction::I32ShrS => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                // Arithmetic shift: on `i32` the sign bit is replicated.
                instance
                    .frame
                    .push(Value::from_i32(a.wrapping_shr(b as u32)));

                Step::Next
            }
            StackInstruction::I32Rotl => {
                let b = instance.frame.pop().as_i32() as u32;
                let a = instance.frame.pop().as_i32() as u32;

                instance
                    .frame
                    .push(Value::from_i32(a.rotate_left(b) as i32));

                Step::Next
            }
            StackInstruction::I32Rotr => {
                let b = instance.frame.pop().as_i32() as u32;
                let a = instance.frame.pop().as_i32() as u32;

                instance
                    .frame
                    .push(Value::from_i32(a.rotate_right(b) as i32));

                Step::Next
            }
            StackInstruction::I32Eq => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32((a == b) as i32));

                Step::Next
            }
            StackInstruction::I32Ne => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32((a != b) as i32));

                Step::Next
            }
            StackInstruction::I32LtU => {
                let b = instance.frame.pop().as_i32() as u32;
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_i32((a < b) as i32));

                Step::Next
            }
            StackInstruction::I32LtS => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32((a < b) as i32));

                Step::Next
            }
            StackInstruction::I32GtU => {
                let b = instance.frame.pop().as_i32() as u32;
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_i32((a > b) as i32));

                Step::Next
            }
            StackInstruction::I32GtS => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32((a > b) as i32));

                Step::Next
            }
            StackInstruction::I32LeU => {
                let b = instance.frame.pop().as_i32() as u32;
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_i32((a <= b) as i32));

                Step::Next
            }
            StackInstruction::I32LeS => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32((a <= b) as i32));

                Step::Next
            }
            StackInstruction::I32GeU => {
                let b = instance.frame.pop().as_i32() as u32;
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_i32((a >= b) as i32));

                Step::Next
            }
            StackInstruction::I32GeS => {
                let b = instance.frame.pop().as_i32();
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i32((a >= b) as i32));

                Step::Next
            }
            StackInstruction::I64Clz => {
                let a = instance.frame.pop().as_i64();

                instance
                    .frame
                    .push(Value::from_i64(a.leading_zeros() as i64));

                Step::Next
            }
            StackInstruction::I64Ctz => {
                let a = instance.frame.pop().as_i64();

                instance
                    .frame
                    .push(Value::from_i64(a.trailing_zeros() as i64));

                Step::Next
            }
            StackInstruction::I64Popcnt => {
                let a = instance.frame.pop().as_i64();

                // See `I32Popcnt`. The count is at most 64, but the result type is
                // `i64` — unary integer ops keep their operand's width, unlike the
                // comparisons.
                instance.frame.push(Value::from_i64(a.count_ones() as i64));

                Step::Next
            }
            StackInstruction::I64Eqz => {
                let a = instance.frame.pop().as_i64();

                instance
                    .frame
                    .push(Value::from_i32(if a == 0 { 1 } else { 0 }));

                Step::Next
            }
            StackInstruction::I64Extend8S => {
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a as i8 as i64));

                Step::Next
            }
            StackInstruction::I64Extend16S => {
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a as i16 as i64));

                Step::Next
            }
            StackInstruction::I64Extend32S => {
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a as i32 as i64));

                Step::Next
            }
            StackInstruction::I64ExtendI32U => {
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_i64(a as i64));

                Step::Next
            }
            StackInstruction::I64ExtendI32S => {
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_i64(a as i64));

                Step::Next
            }
            StackInstruction::I64TruncF32U => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = instance.frame.pop().as_f32() as f64;
                let truncated = trunc_float_to_int(a, 0.0, U64_TRUNC_HIGH, "u64")?;

                // As with the `i32` forms, the result is the unsigned bit pattern
                // held in a signed value.
                instance
                    .frame
                    .push(Value::from_i64(truncated as u64 as i64));

                Step::Next
            }
            StackInstruction::I64TruncF32S => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = instance.frame.pop().as_f32() as f64;
                let truncated = trunc_float_to_int(a, I64_TRUNC_LOW, I64_TRUNC_HIGH, "i64")?;

                instance.frame.push(Value::from_i64(truncated as i64));

                Step::Next
            }
            StackInstruction::I64TruncF64U => {
                let a = instance.frame.pop().as_f64();
                let truncated = trunc_float_to_int(a, 0.0, U64_TRUNC_HIGH, "u64")?;

                instance
                    .frame
                    .push(Value::from_i64(truncated as u64 as i64));

                Step::Next
            }
            StackInstruction::I64TruncF64S => {
                let a = instance.frame.pop().as_f64();
                let truncated = trunc_float_to_int(a, I64_TRUNC_LOW, I64_TRUNC_HIGH, "i64")?;

                instance.frame.push(Value::from_i64(truncated as i64));

                Step::Next
            }
            StackInstruction::I64TruncSatF32U => {
                // Saturate to `u64`, the *target* width — clamping at `u32::MAX`
                // first would lose every value an `i64` can still represent.
                let a = instance.frame.pop().as_f32() as u64;

                instance.frame.push(Value::from_i64(a as i64));

                Step::Next
            }
            StackInstruction::I64TruncSatF32S => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_i64(a as i64));

                Step::Next
            }
            StackInstruction::I64TruncSatF64U => {
                let a = instance.frame.pop().as_f64() as u64;

                instance.frame.push(Value::from_i64(a as i64));

                Step::Next
            }
            StackInstruction::I64TruncSatF64S => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_i64(a as i64));

                Step::Next
            }
            StackInstruction::I64ReinterpretF64 => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_i64(a.to_bits() as i64));

                Step::Next
            }
            StackInstruction::I64Add => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a.wrapping_add(b)));

                Step::Next
            }
            StackInstruction::I64Sub => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a.wrapping_sub(b)));

                Step::Next
            }
            StackInstruction::I64Mul => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a.wrapping_mul(b)));

                Step::Next
            }
            StackInstruction::I64DivU => {
                let b = instance.frame.pop().as_i64() as u64;
                let a = instance.frame.pop().as_i64() as u64;

                instance.frame.push(Value::from_i64(a.checked_div(b).ok_or(
                    InstructionExecutionError::Division {
                        num: a.to_string(),
                        deno: b.to_string(),
                    },
                )? as i64));

                Step::Next
            }
            StackInstruction::I64DivS => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a.checked_div(b).ok_or(
                    InstructionExecutionError::Division {
                        num: a.to_string(),
                        deno: b.to_string(),
                    },
                )?));

                Step::Next
            }
            StackInstruction::I64RemU => {
                let b = instance.frame.pop().as_i64() as u64;
                let a = instance.frame.pop().as_i64() as u64;

                instance.frame.push(Value::from_i64(a.checked_rem(b).ok_or(
                    InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    },
                )? as i64));

                Step::Next
            }
            StackInstruction::I64RemS => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                // See `I32RemS`: only a zero divisor traps; `i64::MIN % -1` is `0`.
                if b == 0 {
                    return Err(Box::new(InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    }));
                }

                instance.frame.push(Value::from_i64(a.wrapping_rem(b)));

                Step::Next
            }
            StackInstruction::I64And => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a.bitand(b)));

                Step::Next
            }
            StackInstruction::I64Or => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a.bitor(b)));

                Step::Next
            }
            StackInstruction::I64Xor => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i64(a.bitxor(b)));

                Step::Next
            }
            // As for `i32`, but masked modulo 64. The count arrives as an `i64` and
            // the shift methods take `u32`, so it is narrowed first — harmless,
            // since only the low 6 bits survive the masking anyway.
            StackInstruction::I64Shl => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance
                    .frame
                    .push(Value::from_i64(a.wrapping_shl(b as u32)));

                Step::Next
            }
            StackInstruction::I64ShrU => {
                let b = instance.frame.pop().as_i64() as u64;
                let a = instance.frame.pop().as_i64() as u64;

                // Logical shift: done on `u64` so the vacated high bits are zeros.
                instance
                    .frame
                    .push(Value::from_i64(a.wrapping_shr(b as u32) as i64));

                Step::Next
            }
            StackInstruction::I64ShrS => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                // Arithmetic shift: on `i64` the sign bit is replicated.
                instance
                    .frame
                    .push(Value::from_i64(a.wrapping_shr(b as u32)));

                Step::Next
            }
            StackInstruction::I64Rotl => {
                let b = instance.frame.pop().as_i64() as u64;
                let a = instance.frame.pop().as_i64() as u64;

                instance
                    .frame
                    .push(Value::from_i64(a.rotate_left(b as u32) as i64));

                Step::Next
            }
            StackInstruction::I64Rotr => {
                let b = instance.frame.pop().as_i64() as u64;
                let a = instance.frame.pop().as_i64() as u64;

                instance
                    .frame
                    .push(Value::from_i64(a.rotate_right(b as u32) as i64));

                Step::Next
            }
            StackInstruction::I64Eq => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i32((a == b) as i32));

                Step::Next
            }
            StackInstruction::I64Ne => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i32((a != b) as i32));

                Step::Next
            }
            StackInstruction::I64LtU => {
                let b = instance.frame.pop().as_i64() as u64;
                let a = instance.frame.pop().as_i64() as u64;

                instance.frame.push(Value::from_i32((a < b) as i32));

                Step::Next
            }
            StackInstruction::I64LtS => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i32((a < b) as i32));

                Step::Next
            }
            StackInstruction::I64GtU => {
                let b = instance.frame.pop().as_i64() as u64;
                let a = instance.frame.pop().as_i64() as u64;

                instance.frame.push(Value::from_i32((a > b) as i32));

                Step::Next
            }
            StackInstruction::I64GtS => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i32((a > b) as i32));

                Step::Next
            }
            StackInstruction::I64LeU => {
                let b = instance.frame.pop().as_i64() as u64;
                let a = instance.frame.pop().as_i64() as u64;

                instance.frame.push(Value::from_i32((a <= b) as i32));

                Step::Next
            }
            StackInstruction::I64LeS => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i32((a <= b) as i32));

                Step::Next
            }
            StackInstruction::I64GeU => {
                let b = instance.frame.pop().as_i64() as u64;
                let a = instance.frame.pop().as_i64() as u64;

                instance.frame.push(Value::from_i32((a >= b) as i32));

                Step::Next
            }
            StackInstruction::I64GeS => {
                let b = instance.frame.pop().as_i64();
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_i32((a >= b) as i32));

                Step::Next
            }
            StackInstruction::F32Abs => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a.abs()));

                Step::Next
            }
            StackInstruction::F32Neg => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a.neg()));

                Step::Next
            }
            StackInstruction::F32Ceil => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a.ceil()));

                Step::Next
            }
            StackInstruction::F32Floor => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a.floor()));

                Step::Next
            }
            StackInstruction::F32Trunc => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a.trunc()));

                Step::Next
            }
            StackInstruction::F32Sqrt => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a.sqrt()));

                Step::Next
            }
            StackInstruction::F32Nearest => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a.round_ties_even()));

                Step::Next
            }
            StackInstruction::F32ConvertI32U => {
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_f32(a as f32));

                Step::Next
            }
            StackInstruction::F32ConvertI32S => {
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_f32(a as f32));

                Step::Next
            }
            StackInstruction::F32ConvertI64U => {
                let a = instance.frame.pop().as_i64() as u64;

                instance.frame.push(Value::from_f32(a as f32));

                Step::Next
            }
            StackInstruction::F32ConvertI64S => {
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_f32(a as f32));

                Step::Next
            }
            StackInstruction::F32DemoteF64 => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f32(a as f32));

                Step::Next
            }
            StackInstruction::F32ReinterpretI32 => {
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_f32(f32::from_bits(a)));

                Step::Next
            }
            StackInstruction::F32Add => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a + b));

                Step::Next
            }
            StackInstruction::F32Sub => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a - b));

                Step::Next
            }
            StackInstruction::F32Mul => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f32(a * b));

                Step::Next
            }
            StackInstruction::F32Div => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                // Unlike the integer divides this never traps: IEEE 754 gives
                // `±inf` for a non-zero numerator over zero, and NaN for `0.0/0.0`.
                instance.frame.push(Value::from_f32(a / b));

                Step::Next
            }
            StackInstruction::F32Eq => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_i32((a == b) as i32));

                Step::Next
            }
            StackInstruction::F32Ne => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_i32((a != b) as i32));

                Step::Next
            }
            StackInstruction::F32Lt => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_i32((a < b) as i32));

                Step::Next
            }
            StackInstruction::F32Gt => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_i32((a > b) as i32));

                Step::Next
            }
            StackInstruction::F32Le => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_i32((a <= b) as i32));

                Step::Next
            }
            StackInstruction::F32Ge => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_i32((a >= b) as i32));

                Step::Next
            }
            StackInstruction::F32Min => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

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

                instance.frame.push(Value::from_f32(r));

                Step::Next
            }
            StackInstruction::F32Max => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

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

                instance.frame.push(Value::from_f32(r));

                Step::Next
            }
            StackInstruction::F32Copysign => {
                let b = instance.frame.pop().as_f32();
                let a = instance.frame.pop().as_f32();

                // Purely a sign-bit transplant: the magnitude of `a` with the sign
                // of `b`. Defined even when either operand is NaN — the sign is
                // copied without inspecting the payload — so unlike `min`/`max`
                // this needs no NaN special case, and Rust's method matches.
                instance.frame.push(Value::from_f32(a.copysign(b)));

                Step::Next
            }
            StackInstruction::F64Abs => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a.abs()));

                Step::Next
            }
            StackInstruction::F64Neg => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a.neg()));

                Step::Next
            }
            StackInstruction::F64Ceil => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a.ceil()));

                Step::Next
            }
            StackInstruction::F64Floor => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a.floor()));

                Step::Next
            }
            StackInstruction::F64Trunc => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a.trunc()));

                Step::Next
            }
            StackInstruction::F64Sqrt => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a.sqrt()));

                Step::Next
            }
            StackInstruction::F64Nearest => {
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a.round_ties_even()));

                Step::Next
            }
            StackInstruction::F64ConvertI32U => {
                let a = instance.frame.pop().as_i32() as u32;

                instance.frame.push(Value::from_f64(a as f64));

                Step::Next
            }
            StackInstruction::F64ConvertI32S => {
                let a = instance.frame.pop().as_i32();

                instance.frame.push(Value::from_f64(a as f64));

                Step::Next
            }
            StackInstruction::F64ConvertI64U => {
                let a = instance.frame.pop().as_i64() as u64;

                instance.frame.push(Value::from_f64(a as f64));

                Step::Next
            }
            StackInstruction::F64ConvertI64S => {
                let a = instance.frame.pop().as_i64();

                instance.frame.push(Value::from_f64(a as f64));

                Step::Next
            }
            StackInstruction::F64PromoteF32 => {
                let a = instance.frame.pop().as_f32();

                instance.frame.push(Value::from_f64(a as f64));

                Step::Next
            }
            StackInstruction::F64ReinterpretI64 => {
                let a = instance.frame.pop().as_i64() as u64;

                instance.frame.push(Value::from_f64(f64::from_bits(a)));

                Step::Next
            }
            StackInstruction::F64Add => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a + b));

                Step::Next
            }
            StackInstruction::F64Sub => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a - b));

                Step::Next
            }
            StackInstruction::F64Mul => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_f64(a * b));

                Step::Next
            }
            StackInstruction::F64Div => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                // See `F32Div`: division by zero yields an infinity or NaN, never
                // a trap.
                instance.frame.push(Value::from_f64(a / b));

                Step::Next
            }
            StackInstruction::F64Eq => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_i32((a == b) as i32));

                Step::Next
            }
            StackInstruction::F64Ne => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_i32((a != b) as i32));

                Step::Next
            }
            StackInstruction::F64Lt => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_i32((a < b) as i32));

                Step::Next
            }
            StackInstruction::F64Gt => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_i32((a > b) as i32));

                Step::Next
            }
            StackInstruction::F64Le => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_i32((a <= b) as i32));

                Step::Next
            }
            StackInstruction::F64Ge => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                instance.frame.push(Value::from_i32((a >= b) as i32));

                Step::Next
            }
            StackInstruction::F64Min => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

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

                instance.frame.push(Value::from_f64(r));

                Step::Next
            }
            StackInstruction::F64Max => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

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

                instance.frame.push(Value::from_f64(r));

                Step::Next
            }
            StackInstruction::F64Copysign => {
                let b = instance.frame.pop().as_f64();
                let a = instance.frame.pop().as_f64();

                // See `F32Copysign`: magnitude of `a`, sign of `b`, NaN included.
                instance.frame.push(Value::from_f64(a.copysign(b)));

                Step::Next
            }
            StackInstruction::LocalGet { index } => {
                instance
                    .frame
                    .push(Self::get_local(*index, caller_base_height, instance));

                Step::Next
            }
            StackInstruction::LocalSet { index } => {
                let val = instance.frame.pop();

                Self::set_local(*index, val, caller_base_height, instance);

                Step::Next
            }
            StackInstruction::LocalTee { index } => {
                let val = instance.frame.tee();

                Self::set_local(*index, val, caller_base_height, instance);

                Step::Next
            }
            StackInstruction::GlobalGet { index } => {
                instance
                    .frame
                    .push(instance.global_vals[index.0 as usize].into());

                Step::Next
            }
            StackInstruction::GlobalSet { index } => {
                let val = instance.frame.pop();
                let ty = module.globals[index.0 as usize].ty.content_type();

                instance.global_vals[index.0 as usize] = val.into_val(&ty);

                Step::Next
            }
            StackInstruction::Drop => {
                let _ = instance.frame.pop();

                Step::Next
            }
            StackInstruction::Select => {
                let cond = instance.frame.pop().as_i32();
                let b = instance.frame.pop();
                let a = instance.frame.pop();

                // true condition
                if cond != 0 {
                    instance.frame.push(a);
                } else {
                    instance.frame.push(b);
                }

                Step::Next
            }
            StackInstruction::Block => Step::Next,
            StackInstruction::Loop => Step::Next,
            StackInstruction::If {
                else_index,
                end_index,
            } => {
                let cond = instance.frame.pop().as_i32();

                if cond != 0 {
                    Step::Next
                } else {
                    if let Some(else_index) = else_index {
                        Step::JumpTo(*else_index + 1) // first instruction of the else branch
                    } else {
                        Step::JumpTo(*end_index)
                    }
                }
            }
            // this instruction would be encountered only when control flow is coming after completing `if` branch
            // because if the condition was `false` and the control went to `else` branch, it jumps to the first
            // instruction of `else` branch and not the `else` instruction.
            StackInstruction::Else { if_end_index } => Step::JumpTo(*if_end_index),
            StackInstruction::Br {
                target_index,
                arity,
                recorded_height,
            } => {
                // Unwind to the target label's absolute height (frame base + its
                // recorded height) while keeping the top `arity` values, then jump.
                instance
                    .frame
                    .truncate_by_preserving_arity(*recorded_height + frame_base_height, *arity);

                Step::JumpTo(*target_index)
            }
            StackInstruction::BrIf {
                target_index,
                arity,
                recorded_height,
            } => {
                let cond = instance.frame.pop().as_i32();

                if cond != 0 {
                    instance
                        .frame
                        .truncate_by_preserving_arity(*recorded_height + frame_base_height, *arity);

                    Step::JumpTo(*target_index)
                } else {
                    Step::Next
                }
            }
            StackInstruction::BrTable { start_index, len } => {
                // Widen before adding: the sum is bounded by the function's target
                // count, but doing it in `u32` would make an overflow a debug panic
                // rather than a wider add.
                let start = *start_index as usize;
                let targets = &br_table_targets[start..start + *len as usize];

                // the branch index is an unsigned i32; go through u32 so a
                // high-bit-set value maps to a large index (→ default), not a
                // sign-extended one.
                let index = instance.frame.pop().as_i32() as u32 as usize;
                let target_count = targets.len() - 1;

                let branch = if target_count <= index {
                    &targets[target_count] // always the last element of targets
                } else {
                    &targets[index]
                };

                instance.frame.truncate_by_preserving_arity(
                    branch.recorded_height + frame_base_height,
                    branch.arity,
                );

                Step::JumpTo(branch.target_index)
            }
            StackInstruction::Return {
                target_index,
                arity,
                recorded_height,
            } => {
                instance
                    .frame
                    .truncate_by_preserving_arity(*recorded_height + frame_base_height, *arity);

                Step::JumpTo(*target_index)
            }
            StackInstruction::End {
                arity,
                recorded_height,
            } => {
                // Sanity check the height model: when a block closes, the stack must
                // hold exactly its `arity` results above the label's recorded height.
                // Both are frame-relative, so shift by this frame's base to compare
                // against the shared stack's absolute height.
                debug_assert!(
                    instance.frame.height() == *recorded_height + *arity + frame_base_height
                );

                Step::Next
            }
        };

        Ok(res)
    }

    /// Evaluates a constant-expression instruction sequence to its single
    /// resulting [`Val`], on a small dedicated stack. Used to compute
    /// global/table/data/element initializers at instantiation.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::Unsupported`] if the sequence contains an
    /// instruction not permitted in a constant expression.
    pub(crate) fn const_expr_evaluator(instructions: &[StackInstruction], globals: &[Val]) -> Val {
        let mut stack: Stack<Val> = Stack::for_const_expr_evaluation();

        for instr in instructions {
            match instr {
                StackInstruction::I32Const { value } => {
                    stack.push(Val::I32(*value));
                }
                StackInstruction::I64Const { value } => {
                    stack.push(Val::I64(*value));
                }
                StackInstruction::F32Const { value } => {
                    stack.push(Val::F32(*value));
                }
                StackInstruction::F64Const { value } => stack.push(Val::F64(*value)),
                StackInstruction::GlobalGet { index } => {
                    stack.push(globals[index.0 as usize]);
                }
                StackInstruction::RefNull => stack.push(Val::Ref(None)),
                StackInstruction::RefFunc { function_index } => {
                    stack.push(Val::Ref(Some(*function_index)));
                }
                StackInstruction::I32Add => {
                    let b = stack.pop().as_i32();
                    let a = stack.pop().as_i32();

                    stack.push(Val::I32(a.wrapping_add(b)));
                }
                StackInstruction::I32Sub => {
                    let b = stack.pop().as_i32();
                    let a = stack.pop().as_i32();

                    stack.push(Val::I32(a.wrapping_sub(b)));
                }
                StackInstruction::I32Mul => {
                    let b = stack.pop().as_i32();
                    let a = stack.pop().as_i32();

                    stack.push(Val::I32(a.wrapping_mul(b)));
                }
                StackInstruction::I64Add => {
                    let b = stack.pop().as_i64();
                    let a = stack.pop().as_i64();

                    stack.push(Val::I64(a.wrapping_add(b)));
                }
                StackInstruction::I64Sub => {
                    let b = stack.pop().as_i64();
                    let a = stack.pop().as_i64();

                    stack.push(Val::I64(a.wrapping_sub(b)));
                }
                StackInstruction::I64Mul => {
                    let b = stack.pop().as_i64();
                    let a = stack.pop().as_i64();

                    stack.push(Val::I64(a.wrapping_mul(b)));
                }
                StackInstruction::End { .. } => {}
                _ => {
                    panic!("instruction `{:?}` in const expression evaluator", instr)
                }
            }
        }

        let val = stack.pop();

        val
    }
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
fn trunc_float_to_int<Instr: Instruction>(
    operand: f64,
    low: f64,
    high: f64,
    target: &str,
) -> Result<f64, InstructionExecutionError<Instr>> {
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

/// Builds the `call_indirect` signature-mismatch trap.
///
/// Outlined because the two `format!` calls expand `core::fmt` inline, and this
/// arm is inlined into the driver loop along with the rest of dispatch — where
/// that expansion inflates the loop's frame for a path taken only on a trap.
#[inline(never)]
fn signature_mismatch<Instr: Instruction>(
    table_index: TableIndex,
    declared_params: &[ValType],
    declared_results: &[ValType],
    params: &[ValType],
    results: &[ValType],
) -> InstructionExecutionError<Instr> {
    InstructionExecutionError::CallIndirect(
        table_index,
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
    )
}

/// A trap on its way back up the native call stack, carrying the trace built so
/// far, innermost frame first.
///
/// This is the error type of [`TraceVM::execute_on_native_stack`], which assembles
/// it during the unwind rather than reconstructing it afterwards: the trapping
/// instruction seeds it, and each enclosing frame appends its `call` as the error
/// passes through. [`TraceVM::run`] closes it into a [`FuncCallError`].
type Unwind<Instr> = Vec<TraceRecord<Instr>>;

/// Seeds an [`Unwind`] at the instruction that trapped.
///
/// Outlined so the allocation and the record's construction stay out of the
/// dispatch loop, which would otherwise carry their working values across every
/// instruction it executes.
#[inline(never)]
#[cold]
fn trap_here<Instr: Instruction>(
    func_index: FuncIndex,
    pc: usize,
    instruction_offsets: &[u32],
    err: InstructionExecutionError<Instr>,
) -> Unwind<Instr> {
    vec![TraceRecord {
        func_index,
        instr_index: pc,
        kind: TraceRecordKind::NonCall(err),
        instr_offset: instruction_offsets[pc],
    }]
}

/// Closes a completed [`Unwind`] into the error the caller sees, naming the entry
/// function the call was made through.
#[inline(never)]
fn func_call_err_from_unwind<Instr: Instruction>(
    func_index: FuncIndex,
    trace: Unwind<Instr>,
    module: &Arc<Module<Instr>>,
) -> FuncCallError<Instr> {
    // A `TypedFunc` is only handed out for an export, so the lookup resolves;
    // falling back rather than unwrapping keeps a failure while *reporting* a
    // trap from replacing it with a panic.
    let func_name = module
        .exported_func_name(func_index)
        .map(String::as_str)
        .unwrap_or("<unknown>");

    FuncCallError::new(
        func_name.to_string(),
        trace.into_boxed_slice(),
        module.clone(),
    )
}

/// Builds the caller-visible error for [`TraceVM::_execute_on_frame_stack`],
/// reconstructing the trace from its saved frames.
///
/// The counterpart to [`func_call_err_from_unwind`]. That driver keeps its call
/// stack in a `Vec` and unwinds in one step, so it has no per-frame moment at
/// which to append a record; the whole trace is walked out here instead —
/// `err` at `instr_index` as the innermost record, then `frames` in reverse, so
/// the result is innermost-first either way.
///
/// Outlined so none of this sits in the dispatch loop's frame.
#[inline(never)]
fn func_call_err<Instr: Instruction>(
    func_index: FuncIndex,
    frames: Vec<Frame>,
    err: InstructionExecutionError<Instr>,
    instr_index: usize,
    instr_offset: u32,
    module: &Arc<Module<Instr>>,
) -> FuncCallError<Instr> {
    let func_name = module
        .exported_func_name(func_index)
        .map(String::as_str)
        .unwrap_or("<unknown>");

    let mut trace: Vec<TraceRecord<Instr>> = vec![];

    trace.push(TraceRecord {
        func_index: if frames.is_empty() {
            func_index
        } else {
            frames.last().unwrap().callee_func_index
        },
        instr_index,
        kind: TraceRecordKind::NonCall(err),
        instr_offset,
    });

    let frames_len = frames.len();

    for i in 0..frames_len {
        let frame = &frames[frames_len - 1 - i];

        trace.push(TraceRecord {
            func_index: frame.func_index,
            instr_index: frame.pc as usize,
            kind: TraceRecordKind::Call {
                callee_index: frame.callee_func_index,
                is_indirect: frame.callee_is_indirect,
            },
            instr_offset: frame.instruction_offsets[frame.pc as usize],
        });
    }

    FuncCallError::new(
        func_name.to_string(),
        trace.into_boxed_slice(),
        module.clone(),
    )
}
