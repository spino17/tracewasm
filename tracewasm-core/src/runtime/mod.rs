//! The TraceWasm interpreter: a tree-walker over the flat, pre-lowered
//! instruction list produced by [`crate::instruction`].
//!
//! ## Execution model
//!
//! Each function body is a `Vec<Instr>` with control flow already resolved to
//! absolute instruction indices. Execution is a simple `pc` loop:
//! [`Instruction::execute`] runs one
//! instruction and reports what the driver should do next via [`Step`]
//! — advance (`Next`), jump (`JumpTo`), enter a callee (`Call`), or (implicitly,
//! by advancing past the final `End`) return from the function.
//!
//! [`TraceVM::execute_on_native_stack`] is that driver. A `Call` makes it recurse
//! into itself, so one native frame is live per active wasm frame; the depth is
//! bounded by [`Config::max_call_stack_depth`](crate::instance::config::Config)
//! rather than by the host stack, since overflowing the latter aborts the process
//! instead of unwinding.
//!
//! ## Where a frame's values live
//!
//! The driver itself owns no value storage. Each machine supplies its own through
//! [`RuntimeFrame`], reached as
//! [`Instruction::RuntimeFrame`](crate::instruction::Instruction), and the driver
//! only asks it to enter a frame, exit one, and hand back results. What follows
//! describes the stack machine's layout, since that is the one whose base
//! arithmetic the lowered instructions encode; the register machine sizes a
//! register file from its [`FrameLayout`](crate::instruction::FrameLayout) instead.
//!
//! Rather than giving every call its own operand stack, the whole call tree shares
//! a single [`Stack`]. A call does not allocate a new stack; the callee builds its
//! operands on top of the caller's. Recursion still uses the native Rust call
//! stack — one [`TraceVM::execute_on_native_stack`] frame per active wasm call —
//! but the potentially-large value stack is allocated exactly once.
//!
//! A frame's **locals live on that same shared stack**, not in a per-activation
//! vector. The layout of one frame, from its base upwards, is:
//!
//! ```text
//!   base_height ─┐
//!                ▼
//!   ... caller's stack | p0 p1 … | l0 l1 … | operands … →
//!                        └ params ┘└ decl. ┘▲
//!                        └──── locals ─────┘└─ callee_frame_base_height
//! ```
//!
//! The arguments are *already* on the stack when the callee is entered, so they
//! are left in place and become local slots `0..params_len` rather than being
//! popped and copied. Frame setup then pushes zero values for the declared locals
//! (`params_len..locals_len`), so `local.get`/`local.set` for slot `i` is a direct
//! index at `base_height + i`.
//!
//! ## The two base heights
//!
//! Because both locals and operands share the stack, a frame needs *two* bases,
//! and conflating them corrupts memory:
//!
//! - **`base_height`** — the bottom of the locals region, i.e. the height the
//!   stack had on entry minus the arguments. Used only by
//!   `get_local`/`set_local`, and as the truncation target on frame exit.
//! - **`callee_frame_base_height`** — the bottom of the *operand* region, i.e.
//!   `base_height + locals_len`. Used by every height-sensitive control
//!   operation.
//!
//! Both are fields of
//! [`StackCallerBaseData`](crate::instruction::stack::StackCallerBaseData), the
//! stack machine's
//! [`CallerBaseData`] — the trait through which
//! the driver carries a callee's base without knowing what it consists of.
//!
//! The lowered instructions store **frame-relative** operand heights
//! (`recorded_height`), computed as if the function ran on an empty operand stack
//! with its locals held separately. The interpreter converts relative → absolute
//! with a single rule:
//!
//! ```text
//! absolute_height = callee_frame_base_height + recorded_height (+ arity)
//! ```
//!
//! On frame exit — [`RuntimeFrame::exit_frame`] — the stack is truncated to
//! `base_height` preserving the result arity, which drops the locals and leaves
//! the results exactly where the caller's arguments were, so the caller does
//! nothing after the call returns. Instruction indices, by contrast, are
//! per-function: each [`TraceVM::execute_on_native_stack`] invocation has its own
//! `instructions` slice and `pc`.

use crate::{
    InstrOf, VirtualMachine,
    error::{
        CallIndirectError::{self, FunctionCall},
        FuncCallError, InstructionExecutionError, TraceRecord, TraceRecordKind,
    },
    instance::{
        Instance,
        traits::{ImportRegistry, ParamVals, ResultVals},
    },
    instruction::{CallerBaseData, Instruction, RuntimeFrame, stack::StackInstruction},
    memory::Memory,
    module::{FuncIndex, FuncKind, Module, TableIndex, ValType, formatted_val_types},
    runtime::{
        stack::Stack,
        value::{Val, Value},
    },
};
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;

// No outer doc comments on these: each module carries its own `//!` docs, and an
// outer `///` at the declaration site re-scopes the intra-doc links inside it to
// this module, silently breaking every one that resolved in its own scope.
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
pub(crate) const I32_TRUNC_LOW: f64 = i32::MIN as f32 as f64; // -2^31 = -2147483648
pub(crate) const I32_TRUNC_HIGH: f64 = i32::MAX as f32 as f64; // 2^31 = 2147483648
pub(crate) const U32_TRUNC_HIGH: f64 = u32::MAX as f32 as f64; // 2^32 = 4294967296
pub(crate) const I64_TRUNC_LOW: f64 = i64::MIN as f64; // -2^63 = -9223372036854775808
pub(crate) const I64_TRUNC_HIGH: f64 = i64::MAX as f64; // 2^63 = 9223372036854775808
pub(crate) const U64_TRUNC_HIGH: f64 = u64::MAX as f64; // 2^64 = 18446744073709551616

/// What [`Instruction::execute`] tells
/// its driver to do next.
///
/// Everything an instruction can do to the operand stack, memory, globals and
/// tables it does itself, against the [`Instance`]. Only the things the *driver*
/// owns — the program counter and the call stack — come back through here.
pub enum Step<Instr: Instruction> {
    /// Continue at the following instruction.
    Next,
    /// Enter a locally-defined callee. An imported callee never produces this:
    /// it runs to completion inside the dispatch and yields [`Self::Next`].
    Call {
        /// The callee, as a module-wide function index — so it counts imports, and
        /// the driver shifts it down by `imported_func_count` to reach a body.
        func_index: FuncIndex,
        /// Where the **callee's** frame begins inside the caller's storage, computed
        /// by the call instruction because only it knows how many arguments it left
        /// behind. The driver passes it straight through as the callee's own base.
        caller_base_data: Instr::CallerBaseData,
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
///
/// Built only by [`TraceVM::_execute_on_frame_stack`], which [`TraceVM::run`] does
/// not currently use — so nothing constructs one.
struct Frame<'a, Instr: Instruction> {
    func_index: FuncIndex,
    instructions: &'a [Instr],
    /// Index of the `call` itself, not the instruction after it; resuming adds
    /// one, and a trace records the call site.
    pc: u32,
    caller_base_data: Instr::CallerBaseData,
    frame_layout: &'a Instr::FrameLayout,
    instruction_offsets: &'a [u32],
    /// Result count of the callee, needed on return to know how much of its
    /// region to keep when truncating back to `caller_base_height`.
    callee_results_count: u32,
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
    /// Top-level entry point: runs a locally-defined function and returns its
    /// result values (in declaration order).
    ///
    /// Resets [`Instance::frame`](crate::instance::Instance) to empty, places
    /// `params` in it, and delegates to [`Self::execute_on_native_stack`], which
    /// leaves the results in the frame for this function to take.
    ///
    /// Generic over the lowering, so the driver itself is machine-agnostic: the
    /// frame comes from [`RuntimeFrame`] and its
    /// base from
    /// [`CallerBaseData::initial_data`].
    /// Whether a given lowering can actually be driven is
    /// [`Instruction::execute`]'s
    /// business — the register machine's is still unimplemented, so driving one
    /// panics rather than failing to compile.
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
    pub(crate) fn run<M: Memory, I: ImportRegistry, V: VirtualMachine>(
        func_index: FuncIndex,
        params: &[Val],
        instance: &mut Instance<M, I, V>,
        module: &Arc<Module<V>>,
    ) -> Result<ResultVals, FuncCallError> {
        let mut call_stack_depth = 0;

        instance.frame.reset();

        let caller_base_data = <InstrOf<V> as Instruction>::CallerBaseData::initial_data();

        instance.frame.set_initial_params(params);

        // The reset above put the stack at height 0, so this frame's base is 0.
        Self::execute_on_native_stack(
            func_index,
            instance,
            module,
            &mut call_stack_depth,
            caller_base_data,
        )
        .map_err(|trace| func_call_err_from_unwind(func_index, trace, module))?;

        // How many result values the function leaves on the stack.
        let func_decl = &module.func_decls[func_index.0 as usize];
        let results_ty = &module.types[func_decl.ty.0 as usize].results;
        let results_count = results_ty.len() as u32;

        let results = instance.frame.get_final_results(results_count);

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
    /// `stack.height() - params_count`. Results are likewise left on `stack` for
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
    fn execute_on_native_stack<M: Memory, I: ImportRegistry, V: VirtualMachine>(
        func_index: FuncIndex,
        instance: &mut Instance<M, I, V>,
        module: &Arc<Module<V>>,
        call_stack_depth: &mut u32,
        mut caller_base_data: <InstrOf<V> as Instruction>::CallerBaseData,
    ) -> Result<(), Unwind> {
        // `func_bodies` holds only locally-defined functions, so shift the global
        // function index down by the number of imports to index into it.
        let imported_func_count = module.imported_func_count;

        // Only an assertion because every way in already rejects an imported index:
        // `Module::get_typed_func` when it hands out the handle,
        // `Module::instantiate` before it runs a `start` function, and the driver's
        // `Step::Call` arm, which routes an import to `Self::call_imported`. If that
        // ever stops holding, the subtraction below wraps and the `func_bodies` index
        // panics rather than reporting anything.
        debug_assert!(func_index.0 >= imported_func_count);

        let func_decl = &module.func_decls[func_index.0 as usize];
        let ty = &module.types[func_decl.ty.0 as usize];
        let params_count = ty.params.len() as u32;
        let results_count = ty.results.len() as u32;
        let func_body = &module.func_bodies[(func_index.0 - imported_func_count) as usize];
        let instructions = &func_body.instructions;
        let instruction_offsets = &func_body.instruction_offsets;

        // `locals` in the body is laid out params-first, then declared locals, and
        // `locals_ty[i]` is the declared type of local slot `i`. So
        // `locals_ty.len() >= params_count` always, which is what `enter_frame` relies
        // on when it skips the params.
        let locals_ty = &func_body.locals;

        // The declared locals follow the params and must start at the zero value
        // of their type, per the spec. Pushing them here is what makes the locals
        // region contiguous, so `get_local` can index it directly.
        instance.frame.enter_frame(
            params_count,
            locals_ty,
            &mut caller_base_data,
            &func_body.frame_layout,
        );

        // Entering a frame. The matching decrement is after the driver loop; the
        // error paths out of the loop deliberately skip it, because an error always
        // propagates to the top-level `run` (nothing in the interpreter catches
        // one) and the counter is a fresh local per `run`. The depth *limit* is
        // enforced by the driver loop's `Step::Call` arm below, not here, so that the
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

        loop {
            let instr = &instructions[pc];

            let step = instr.execute(
                module,
                instance,
                &func_body.frame_layout,
                &caller_base_data,
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
                    caller_base_data,
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
                        caller_base_data,
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
            .exit_frame(results_count, &caller_base_data, &func_body.frame_layout);

        Ok(())
    }

    /// The same interpreter, driving its call stack from a `Vec<Frame>` instead of
    /// recursing.
    ///
    /// **Not wired up** — [`Self::run`] uses [`Self::execute_on_native_stack`]. It is
    /// kept because it is the only shape that decouples guest call depth from the
    /// host stack: nothing here grows the native frame per wasm call, so depth is
    /// bounded by memory rather than by the native stack, and
    /// [`Config::max_call_stack_depth`](crate::instance::config::Config) becomes a
    /// policy limit rather than a guard against an uncatchable abort.
    ///
    /// It passes the whole suite in both profiles when swapped in for
    /// [`Self::execute_on_native_stack`], including the differential tests and the
    /// multi-frame backtrace and depth-limit cases.
    ///
    /// It is measurably slower, and for a reason that is structural rather than
    /// incidental. The frame state it must carry across the dispatch loop —
    /// instruction slice, offsets, branch targets, the base data — occupies registers
    /// that the recursive driver leaves free, because there they are fixed for the
    /// life of the call and here they change on every call and return. The opcode arms
    /// lose that contention: the operand-stack pointer stops living in a register, so
    /// a push no longer folds into the pop before it, and the arms' shared tail
    /// becomes a general push instead.
    ///
    /// Restructuring around an inner per-frame loop, and outlining the call/return
    /// transitions, were both tried and moved it very little; the contention is not in
    /// how those are written.
    ///
    /// Its depth counter is `frames.len() + 1`, not `frames.len()`: `frames` holds the
    /// *suspended* callers, so the frame running now is not among them.
    ///
    /// # Errors
    ///
    /// A [`FuncCallError`] directly, rather than the [`Unwind`] the recursive driver
    /// returns. There is no native unwind to hang a per-frame record on, so
    /// [`func_call_err`] reconstructs the whole trace from `frames` in one pass.
    fn _execute_on_frame_stack<M: Memory, I: ImportRegistry, V: VirtualMachine>(
        mut func_index: FuncIndex,
        instance: &mut Instance<M, I, V>,
        module: &Arc<Module<V>>,
    ) -> Result<(), FuncCallError> {
        let mut frames: Vec<Frame<InstrOf<V>>> = Vec::with_capacity(10);
        let imported_func_count = module.imported_func_count;

        debug_assert!(func_index.0 >= imported_func_count);

        let func_decl = &module.func_decls[func_index.0 as usize];
        let ty = &module.types[func_decl.ty.0 as usize];
        let params_count = ty.params.len() as u32;
        let results_count = ty.results.len() as u32;
        let func_body = &module.func_bodies[(func_index.0 - imported_func_count) as usize];
        let mut instructions = func_body.instructions.as_ref();
        let mut instruction_offsets = func_body.instruction_offsets.as_ref();
        let mut frame_layout = &func_body.frame_layout;

        let locals_ty = &func_body.locals;
        let mut pc = 0;
        let mut caller_base_data = <InstrOf<V> as Instruction>::CallerBaseData::initial_data();

        // The entry function is named in every trace this driver raises, and
        // `func_index` below tracks whichever function is *currently* running, so the
        // two cannot be the same variable.
        let entry_func_index = func_index;

        instance.frame.enter_frame(
            params_count,
            locals_ty,
            &mut caller_base_data,
            &func_body.frame_layout,
        );

        loop {
            let instr = &instructions[pc];

            let step = instr.execute(
                module,
                instance,
                frame_layout,
                &caller_base_data,
                imported_func_count,
            );

            let step = match step {
                Ok(step) => step,
                // Unlike the recursive driver, there is no native unwind to append a
                // record per frame on the way out — the whole call stack is still
                // here in `frames`, so the trace is built in one pass.
                Err(err) => {
                    return Err(func_call_err(
                        entry_func_index,
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
                    caller_base_data: callee_caller_base_data,
                    is_indirect,
                } => {
                    let max_depth = instance.config().get_max_call_stack_depth();

                    // `frames` holds the *suspended* callers, so the frame running
                    // now is not among them — the current depth is one more than its
                    // length. Testing the length alone would admit `max_depth + 1`
                    // frames and disagree with the recursive driver, whose counter is
                    // incremented on entry.
                    let depth = frames.len() as u32 + 1;

                    if depth >= max_depth {
                        return Err(func_call_err(
                            entry_func_index,
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
                    let callee_params_count =
                        module.types[callee_func_ty.0 as usize].params.len() as u32;
                    let callee_results_count =
                        module.types[callee_func_ty.0 as usize].results.len() as u32;
                    let callee_func_body =
                        &module.func_bodies[(callee_func_index.0 - imported_func_count) as usize];
                    let callee_instructions = &callee_func_body.instructions;
                    let callee_locals_ty = &callee_func_body.locals;
                    let callee_instruction_offsets = &callee_func_body.instruction_offsets;
                    let callee_frame_layout = &callee_func_body.frame_layout;

                    // save current frame's state
                    frames.push(Frame {
                        func_index,
                        instructions,
                        pc: pc as u32,
                        caller_base_data,
                        frame_layout,
                        instruction_offsets,
                        callee_results_count,
                        callee_func_index,
                        callee_is_indirect: is_indirect,
                    });

                    // override the current state with callee
                    func_index = callee_func_index;
                    instructions = callee_instructions;
                    pc = 0;
                    caller_base_data = callee_caller_base_data;
                    frame_layout = callee_frame_layout;
                    instruction_offsets = callee_instruction_offsets;

                    instance.frame.enter_frame(
                        callee_params_count,
                        callee_locals_ty,
                        &mut caller_base_data,
                        &callee_func_body.frame_layout,
                    );

                    continue;
                }
            }

            if pc == instructions.len() {
                if frames.is_empty() {
                    break;
                }

                // pop the frame!
                let frame = frames.pop().unwrap(); // safe to unwrap as checked above!

                // `frame_layout` still describes the callee here — it is replaced with
                // the caller's below, once its results have been moved down.
                instance.frame.exit_frame(
                    frame.callee_results_count,
                    &caller_base_data,
                    frame_layout,
                );

                // reset the state of the frame which executed call instruction
                func_index = frame.func_index;
                instructions = frame.instructions;
                pc = frame.pc as usize + 1; // next instruction past the call
                caller_base_data = frame.caller_base_data;
                frame_layout = frame.frame_layout;
                instruction_offsets = frame.instruction_offsets;
            }
        }

        instance
            .frame
            .exit_frame(results_count, &caller_base_data, frame_layout);

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
    pub(crate) fn call_imported<M: Memory, I: ImportRegistry, V: VirtualMachine>(
        callee_func_index: FuncIndex,
        module: &Module<V>,
        instance: &mut Instance<M, I, V>,
        is_indirect: Option<TableIndex>,
        caller_base_data: &<InstrOf<V> as Instruction>::CallerBaseData,
    ) -> Result<(), Box<InstructionExecutionError>> {
        let callee_func_decl = &module.func_decls[callee_func_index.0 as usize];
        let callee_params_ty = &module.types[callee_func_decl.ty.0 as usize].params;
        let callee_params_count = callee_params_ty.len() as u32;

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
                .get_params_for_import_call(callee_params_count, caller_base_data)
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

        instance
            .frame
            .set_results_from_import_call(results, caller_base_data);

        Ok(())
    }

    /// Evaluates a constant-expression instruction sequence to its single
    /// resulting [`Val`], on a small dedicated stack. Used to compute
    /// global/table/data/element initializers at instantiation.
    ///
    /// # Panics
    ///
    /// If the sequence contains an instruction not permitted in a constant
    /// expression. This used to be a `TraceWasmError::Unsupported`, and the
    /// callers in [`Module`] propagated it with `?`; the
    /// panic is a placeholder from the refactor, not a deliberate contract.
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

        stack.pop()
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
pub(crate) fn trunc_float_to_int(
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

/// Builds the `call_indirect` signature-mismatch trap.
///
/// Outlined because the two `format!` calls expand `core::fmt` inline, and this
/// arm is inlined into the driver loop along with the rest of dispatch — where
/// that expansion inflates the loop's frame for a path taken only on a trap.
#[inline(never)]
pub(crate) fn signature_mismatch(
    table_index: TableIndex,
    declared_params: &[ValType],
    declared_results: &[ValType],
    params: &[ValType],
    results: &[ValType],
) -> InstructionExecutionError {
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
type Unwind = Vec<TraceRecord>;

/// Seeds an [`Unwind`] at the instruction that trapped.
///
/// Outlined so the allocation and the record's construction stay out of the
/// dispatch loop, which would otherwise carry their working values across every
/// instruction it executes.
#[inline(never)]
#[cold]
fn trap_here(
    func_index: FuncIndex,
    pc: usize,
    instruction_offsets: &[u32],
    err: InstructionExecutionError,
) -> Unwind {
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
fn func_call_err_from_unwind<V: VirtualMachine>(
    func_index: FuncIndex,
    trace: Unwind,
    module: &Arc<Module<V>>,
) -> FuncCallError {
    // A `TypedFunc` is only handed out for an export, so the lookup resolves;
    // falling back rather than unwrapping keeps a failure while *reporting* a
    // trap from replacing it with a panic.
    let func_name = module
        .exported_func_name(func_index)
        .map(String::as_str)
        .unwrap_or("<unknown>");

    FuncCallError::new(func_name.to_string(), trace.into_boxed_slice(), module)
}

/// Builds the caller-visible error for [`TraceVM::_execute_on_frame_stack`],
/// reconstructing the trace from its saved frames. Unused while [`TraceVM::run`]
/// drives on the native stack.
///
/// The counterpart to [`func_call_err_from_unwind`]. That driver keeps its call
/// stack in a `Vec` and unwinds in one step, so it has no per-frame moment at
/// which to append a record; the whole trace is walked out here instead —
/// `err` at `instr_index` as the innermost record, then `frames` in reverse, so
/// the result is innermost-first either way.
///
/// `entry_func_index` names the call the caller made, which is what
/// [`FuncCallError`] reports. The *trapping* function is not that one but the
/// callee of the last saved frame — or `entry_func_index` itself when nothing is
/// saved, which is the same function.
///
/// Outlined so none of this sits in the dispatch loop's frame.
#[inline(never)]
fn func_call_err<V: VirtualMachine>(
    entry_func_index: FuncIndex,
    frames: Vec<Frame<InstrOf<V>>>,
    err: InstructionExecutionError,
    instr_index: usize,
    instr_offset: u32,
    module: &Arc<Module<V>>,
) -> FuncCallError {
    let func_name = module
        .exported_func_name(entry_func_index)
        .map(String::as_str)
        .unwrap_or("<unknown>");

    let mut trace: Vec<TraceRecord> = vec![];

    trace.push(TraceRecord {
        func_index: if frames.is_empty() {
            entry_func_index
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

    FuncCallError::new(func_name.to_string(), trace.into_boxed_slice(), module)
}
