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
        InstructionExecutionError, MemoryAccessKind, MemoryError, TraceWasmError,
    },
    instance::{
        config::Config,
        traits::{ImportRegistry, ResultVals},
    },
    instruction::Instruction,
    memory::Memory,
    module::{FuncIndex, FuncKind, LocalIndex, Module, formatted_val_types},
    vm::stack::{DataVal, Stack, TableVal, Val},
};
use smallvec::{SmallVec, smallvec};
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
    tables: &'a mut [TableVal],
    datas: &'a mut [DataVal],
}

impl<'a, M: Memory, I: ImportRegistry> TraceVMState<'a, M, I> {
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

    fn call_func(
        &mut self,
        func_index: FuncIndex,
        params_count: u32,
        module: &Module,
        config: &Config,
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
            let results = self.import_registry.execute(
                module_name,
                imported_func_name,
                params.as_ref(),
                self.memory,
            )?;

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
                self.datas,
                config,
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
        config: &Config,
    ) -> Result<ExecutionResult, InstructionExecutionError> {
        let res = match instruction {
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
                    .push(Val::I32(self.memory.size_in_pages() as i32));

                ExecutionResult::Next
            }
            Instruction::MemoryGrow => {
                // The delta is an unsigned i32; going through `u32` keeps a
                // high-bit-set value from sign-extending into a different number.
                let delta_in_pages = self.stack.pop().as_i32() as u32 as u64;

                // `instantiate` already narrowed this to the module's declared
                // maximum, so the configured cap is the effective ceiling here.
                match self
                    .memory
                    .grow(delta_in_pages, config.get_max_memory_size_in_pages())
                {
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

                self.memory.copy_within(dest, src, len)?;

                ExecutionResult::Next
            }
            Instruction::MemoryFill => {
                let len = self.stack.pop().as_i32() as usize;
                let val = self.stack.pop().as_i32() as u32;
                let dest = self.stack.pop().as_i32() as usize;

                self.memory.fill(dest, val, len)?;

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
                let segment: &[u8] = match &self.datas[*data_index as usize] {
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
                self.memory.write(dest, &segment[src..end])?;

                ExecutionResult::Next
            }
            Instruction::DataDrop { data_index } => {
                self.datas[*data_index as usize] = DataVal::Dropped;

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
            Instruction::I32Store { offset, align: _ } => {
                let val = self.stack.pop().as_i32();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.memory.write_u32(effective_offset, val as u32)?;

                ExecutionResult::Next
            }
            Instruction::I32Store8 { offset, align: _ } => {
                let val = self.stack.pop().as_i32();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.memory.write_u8(effective_offset, val as u8)?;

                ExecutionResult::Next
            }
            Instruction::I32Store16 { offset, align: _ } => {
                let val = self.stack.pop().as_i32();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.memory.write_u16(effective_offset, val as u16)?;

                ExecutionResult::Next
            }
            Instruction::I64Store { offset, align: _ } => {
                let val = self.stack.pop().as_i64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.memory.write_u64(effective_offset, val as u64)?;

                ExecutionResult::Next
            }
            Instruction::I64Store8 { offset, align: _ } => {
                let val = self.stack.pop().as_i64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.memory.write_u8(effective_offset, val as u8)?;

                ExecutionResult::Next
            }
            Instruction::I64Store16 { offset, align: _ } => {
                let val = self.stack.pop().as_i64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.memory.write_u16(effective_offset, val as u16)?;

                ExecutionResult::Next
            }
            Instruction::I64Store32 { offset, align: _ } => {
                let val = self.stack.pop().as_i64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.memory.write_u32(effective_offset, val as u32)?;

                ExecutionResult::Next
            }
            Instruction::F32Store { offset, align: _ } => {
                let val = self.stack.pop().as_f32();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.memory.write_f32(effective_offset, val)?;

                ExecutionResult::Next
            }
            Instruction::F64Store { offset, align: _ } => {
                let val = self.stack.pop().as_f64();
                let effective_offset = self.pop_effective_address(*offset)?;

                self.memory.write_f64(effective_offset, val)?;

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
            Instruction::Call {
                func_index: callee_func_index,
                params_count,
            } => {
                self.call_func(*callee_func_index, *params_count, module, config)
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

                self.call_func(
                    callee_func_index,
                    declared_params.len() as u32,
                    module,
                    config,
                )
                .map_err(|err| {
                    InstructionExecutionError::CallIndirect(
                        *table_index,
                        FunctionCall(callee_func_index, Box::new(err)),
                    )
                })?;

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
    /// Always a [`TraceWasmError::InstructionExecution`]: every failure — a trap,
    /// an error from a nested call, or one from an imported function — is tagged
    /// here with the instruction that raised it. Frame setup cannot fail, since
    /// `Module::compile` already rejected the local types the VM does not model.
    ///
    /// [`FuncCallError`](crate::error::FuncCallError) depends on this being the
    /// only error shape that escapes the interpreter.
    // Threads the whole shared interpreter state (stack, memory, globals, tables,
    // registry) down each recursive call; bundling it would add a borrow-splitting
    // problem without simplifying anything.
    #[allow(clippy::too_many_arguments)]
    fn execute<M: Memory, I: ImportRegistry>(
        func_index: FuncIndex,
        params: &[Val],
        module: &Module,
        stack: &mut Stack<Val>,
        memory: &mut M,
        caller_base_height: u32,
        import_registry: &mut I,
        globals: &mut [Val],
        tables: &mut [TableVal],
        datas: &mut [DataVal],
        config: &Config,
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
        let instruction_offsets = &func_body.instruction_offsets;

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

            locals.push(Val::zero_of_ty(ty));
        }

        let mut state = TraceVMState {
            stack,
            memory,
            locals: Locals::new(locals),
            import_registry,
            globals,
            tables,
            datas,
        };

        // Driver loop. `pc` indexes this function's instruction list only.
        let mut pc = 0;

        loop {
            let instr = &instructions[pc];

            let res = state
                .execute(instr, caller_base_height, module, config)
                .map_err(|err| {
                    err.into_tracewasm_err(pc, func_index, instr, instruction_offsets[pc])
                })?;

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
    // Mirrors `execute`'s parameter list — see the note there.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run<M: Memory, I: ImportRegistry>(
        func_index: FuncIndex,
        params: &[Val],
        module: &Module,
        memory: &mut M,
        import_registry: &mut I,
        global_vals: &mut [Val],
        table_vals: &mut [TableVal],
        data_vals: &mut [DataVal],
        config: &Config,
    ) -> Result<ResultVals, TraceWasmError> {
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
            global_vals,
            table_vals,
            data_vals,
            config,
        )?;

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
