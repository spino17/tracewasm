#![allow(warnings)]

use crate::{
    error::TraceWasmError,
    instruction::register::lazy::{
        Global, GlobalSlot, LazyArena, LazyEntryDropResult, LazyLocation, LazySlot, Local,
        LocalSlot, SpillArena,
    },
    module::{FuncDecl, FuncType},
    vm::stack::Stack,
};
use std::marker::PhantomData;
use wasmparser::{Operator, OperatorsReader};

pub mod lazy;

#[derive(Debug, Clone, Copy)]
pub enum Const {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

#[derive(Debug, Clone, Copy)]
pub enum Slot {
    Const(Const),
    Local(u32),
    Global(u32),
    Spilled(u32),
    Register(u32), // index into stack
}

impl Default for Slot {
    fn default() -> Self {
        Slot::Const(Const::I32(0))
    }
}

#[derive(Clone, Copy)]
enum StackSlot {
    Const(Const),
    Register(u32),
    Local(LocalSlot),
    Global(GlobalSlot),
}

pub struct Registers<const L: usize, T> {
    start: u32,
    phantom: PhantomData<T>,
}

impl<const L: usize, T> Registers<L, T> {
    pub fn registers<'a>(&self, arena: &'a [T]) -> &'a [T; L] {
        let start = self.start as usize;

        arena[start..(start + L)].try_into().unwrap()
    }
}

pub struct InstructionSignature<const I: usize, const O: usize> {
    input: Registers<I, Slot>,
    output: Registers<O, u32>,
}

pub struct DynInstructionSignature {
    input: u32,
    output: u32,
    len: u32,
}

impl DynInstructionSignature {
    pub fn intput_registers<'a>(&self, arena: &'a [Slot]) -> &'a [Slot] {
        let start = self.input as usize;

        &arena[start..(start + self.len as usize)]
    }

    pub fn output_registers<'a>(&self, arena: &'a [u32]) -> &'a [u32] {
        let start = self.output as usize;

        &arena[start..(start + self.len as usize)]
    }
}

struct SimulatedStack {
    stack: Stack<StackSlot>,
    curr_register_index: usize,
    max_registers: usize,
    lazy_locals: LazyArena<Local>,
    lazy_globals: LazyArena<Global>,
    spills: SpillArena,
    input_registers: Vec<Slot>,
    output_registers: Vec<u32>,
}

impl SimulatedStack {
    fn new(locals_count: u32, globals_count: u32) -> Self {
        SimulatedStack {
            stack: Stack::new_with_capacity(0),
            curr_register_index: 0,
            max_registers: 0,
            lazy_locals: LazyArena::new(locals_count),
            lazy_globals: LazyArena::new(globals_count),
            spills: SpillArena::default(),
            input_registers: vec![],
            output_registers: vec![],
        }
    }

    fn advanced_register_index(&mut self) {
        self.curr_register_index += 1;

        if self.curr_register_index > self.max_registers {
            self.max_registers += 1;
        }
    }

    fn recede_register_index(&mut self) {
        self.curr_register_index -= 1;
    }

    fn pop_lazy<T>(
        slot: LazySlot<T>,
        arena: &mut LazyArena<T>,
        spills: &mut SpillArena,
    ) -> LazyLocation {
        let location = slot.location(&arena);

        if matches!(slot.decrease_ref_count(arena), LazyEntryDropResult::Dropped) {
            match location {
                LazyLocation::Original(local_index) => arena.origin[local_index as usize] = None,
                LazyLocation::Spilled(spill_index) => spills.free_slot(spill_index),
            }
        }

        location
    }

    fn push_lazy<T>(location: u32, arena: &mut LazyArena<T>) -> LazySlot<T> {
        let slot = match arena.origin[location as usize] {
            Some(slot) => {
                slot.advanced_ref_count(arena);

                slot
            }
            None => {
                let slot = arena.allocate(location);
                arena.origin[location as usize] = Some(slot);

                slot
            }
        };

        slot
    }

    fn pop(&mut self) -> Slot {
        let val = self.stack.pop();

        let slot = match val {
            StackSlot::Const(val) => Slot::Const(val),
            StackSlot::Register(val) => {
                self.recede_register_index();

                Slot::Register(val)
            }
            StackSlot::Local(slot) => {
                let location = Self::pop_lazy(slot, &mut self.lazy_locals, &mut self.spills);

                match location {
                    LazyLocation::Original(local_index) => Slot::Local(local_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
            StackSlot::Global(slot) => {
                let location = Self::pop_lazy(slot, &mut self.lazy_globals, &mut self.spills);

                match location {
                    LazyLocation::Original(global_index) => Slot::Global(global_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
        };

        slot
    }

    fn push(&mut self, val: Slot) {
        let slot = match val {
            Slot::Const(val) => StackSlot::Const(val),
            Slot::Register(val) => {
                self.advanced_register_index();

                StackSlot::Register(val)
            }
            Slot::Local(index) => {
                let slot = Self::push_lazy(index, &mut self.lazy_locals);

                StackSlot::Local(slot)
            }
            Slot::Global(index) => {
                let slot = Self::push_lazy(index, &mut self.lazy_globals);

                StackSlot::Global(slot)
            }
            Slot::Spilled(val) => unreachable!("spilled slots are never produced for push!"),
        };

        self.stack.push(slot);
    }

    fn push_const(&mut self, val: Const) {
        self.push(Slot::Const(val));
    }

    fn push_local(&mut self, index: u32) {
        self.push(Slot::Local(index));
    }

    fn push_global(&mut self, index: u32) {
        self.push(Slot::Global(index));
    }

    fn registers_for<const I: usize, const O: usize>(&mut self) -> InstructionSignature<I, O> {
        let input_start = self.input_registers.len();

        self.input_registers
            .resize(input_start + I, Slot::default());

        for i in 0..I {
            self.input_registers[input_start + I - 1 - i] = self.pop();
        }

        let output_start = self.output_registers.len();

        self.output_registers.resize(output_start + O, 0);

        for i in 0..O {
            self.output_registers[output_start + i] = self.curr_register_index as u32;
            let out = Slot::Register(self.curr_register_index as u32);

            self.push(out);
        }

        InstructionSignature {
            input: Registers {
                start: input_start as u32,
                phantom: PhantomData,
            },
            output: Registers {
                start: output_start as u32,
                phantom: PhantomData,
            },
        }
    }

    fn set_lazy<T>(
        location: u32,
        arena: &mut LazyArena<T>,
        spills: &mut SpillArena,
    ) -> Option<u32> {
        let Some(slot) = arena.origin[location as usize] else {
            return None;
        };

        let spill_index = spills.reserve_slot();

        slot.spill(spill_index, arena);
        arena.origin[location as usize] = None;

        Some(spill_index)
    }

    fn tee(&self) -> Slot {
        let top_slot = &self.stack.top();

        match top_slot {
            StackSlot::Const(val) => Slot::Const(*val),
            StackSlot::Register(val) => Slot::Register(*val),
            StackSlot::Local(slot) => match slot.location(&self.lazy_locals) {
                LazyLocation::Original(local_index) => Slot::Local(local_index),
                LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
            },
            StackSlot::Global(slot) => match slot.location(&self.lazy_globals) {
                LazyLocation::Original(global_index) => Slot::Global(global_index),
                LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
            },
        }
    }
}

/// The storage one lowered body needs, in slot counts.
///
/// Both fields are high-water marks over the whole body rather than counts at any
/// one point, so a frame sized to them never has to grow mid-execution.
///
/// **Operands only.** Locals are not counted here: like the stack pass's
/// [`max_height`](crate::instruction::stack), these are measured from the frame's
/// operand base, so a consumer laying out storage needs
/// `locals_len + registers + spills`.
pub(crate) struct FrameLayout {
    /// Operand registers, i.e. the peak `curr_register_index`.
    pub registers: u32,
    /// Spill slots holding locals and globals rescued from a later write by
    /// [`RegInstruction::LocalSpill`] / [`RegInstruction::GlobalSpill`].
    ///
    /// Zero for a body that never overwrites a lazily-forwarded local or global,
    /// which is the common case.
    pub spills: u32,
    pub input_registers_arena: Box<[Slot]>,
    pub output_registers_arena: Box<[u32]>,
}

/// The two outputs of lowering one function body into register form: the
/// instruction list, and the frame required to execute it.
pub(crate) type LoweredRegFuncBody = (Vec<RegInstruction>, FrameLayout);

pub enum RegInstruction {
    I32Load(u32, InstructionSignature<1, 1>), // (memarg, registers)
    GlobalSet(u32, InstructionSignature<1, 0>),
    LocalSet(u32, InstructionSignature<1, 0>),
    LocalTee(u32, InstructionSignature<1, 0>),
    I32Store(u32, InstructionSignature<2, 0>),
    I32Add(InstructionSignature<2, 1>),
    I32Eqz(InstructionSignature<1, 1>),
    Select(InstructionSignature<3, 1>),
    LocalSpill(u32, u32),  // (local_index, spill_index)
    GlobalSpill(u32, u32), // (global_index, spill_index)
    Move(DynInstructionSignature),
}

// One `RegInstruction` per lowered operator, so this size is multiplied across every
// compiled module — the same budget, and the same reasoning, as `Instruction` in the
// stack pass.
//
// What holds it here is that operands live in the flat side tables rather than in the
// variant: a `Registers<I, O>` is a pair of `u32` starts (8 bytes) whatever `I` and
// `O` are, so the widest variant is `I32Load(u32, Registers<1, 1>)` at 12 bytes plus
// tag. Inlining the operands instead would put `Select(Registers<3, 1>)` alone at 56.
//
// The constraint this places on what comes next: an instruction whose arity is not a
// compile-time constant — `call`, `call_indirect`, the block param/result moves — must
// stay within the same 8-byte shape. Either derive both arities at execution from an
// index the variant already carries (as `CallIndirect` does with its `ty_index` in the
// stack pass), or store an explicit `len` and drop something else to pay for it.
const _: () = assert!(
    size_of::<RegInstruction>() <= 16,
    "RegInstruction grew past 16 bytes. Need to keep it compact."
);

impl RegInstruction {
    pub(crate) fn emit_instruction_for_func(
        mut operator_reader: OperatorsReader<'_>,
        params: u32,
        results: u32,
        types: &[FuncType],
        func_decls: &[FuncDecl],
        locals_count: u32,
        globals_count: u32,
    ) -> Result<LoweredRegFuncBody, TraceWasmError> {
        let mut instructions: Vec<RegInstruction> = vec![];
        let mut simulated_stack = SimulatedStack::new(locals_count, globals_count);

        while !operator_reader.eof() {
            let (operator, offset) = operator_reader.read_with_offset()?;

            match operator {
                Operator::GlobalGet { global_index } => {
                    simulated_stack.push_global(global_index);
                }
                Operator::GlobalSet { global_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        global_index,
                        &mut simulated_stack.lazy_globals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::GlobalSpill(global_index, spill_index));
                    }

                    let registers = simulated_stack.registers_for::<1, 0>();

                    instructions.push(RegInstruction::GlobalSet(global_index, registers));
                }
                Operator::LocalGet { local_index } => {
                    simulated_stack.push_local(local_index);
                }
                Operator::LocalSet { local_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        local_index,
                        &mut simulated_stack.lazy_locals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::LocalSpill(local_index, spill_index));
                    }

                    let registers = simulated_stack.registers_for::<1, 0>();

                    instructions.push(RegInstruction::LocalSet(local_index, registers));
                }
                Operator::LocalTee { local_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        local_index,
                        &mut simulated_stack.lazy_locals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::LocalSpill(local_index, spill_index));
                    }

                    let input_start = simulated_stack.input_registers.len();

                    simulated_stack.input_registers.push(simulated_stack.tee());

                    let registers = InstructionSignature {
                        input: Registers {
                            start: input_start as u32,
                            phantom: PhantomData,
                        },
                        output: Registers {
                            start: simulated_stack.output_registers.len() as u32,
                            phantom: PhantomData,
                        },
                    };

                    instructions.push(RegInstruction::LocalTee(local_index, registers));
                }
                Operator::I32Const { value } => {
                    simulated_stack.push_const(Const::I32(value));
                }
                Operator::I32Load { memarg } => {
                    let registers = simulated_stack.registers_for::<1, 1>();

                    instructions.push(RegInstruction::I32Load(memarg.offset as u32, registers));
                }
                Operator::I32Store { memarg } => {
                    let registers = simulated_stack.registers_for::<2, 0>();

                    instructions.push(RegInstruction::I32Store(memarg.offset as u32, registers));
                }
                Operator::I32Add => {
                    let registers = simulated_stack.registers_for::<2, 1>();

                    instructions.push(RegInstruction::I32Add(registers));
                }
                Operator::I32Eqz => {
                    let registers = simulated_stack.registers_for::<1, 1>();

                    instructions.push(RegInstruction::I32Eqz(registers));
                }
                Operator::Nop => {
                    continue;
                }
                Operator::Select => {
                    let registers = simulated_stack.registers_for::<3, 1>();

                    instructions.push(RegInstruction::Select(registers));
                }
                Operator::Drop => {
                    simulated_stack.pop();

                    continue;
                }
                // TODO - add blocks and branch instructions!
                _ => {
                    return Err(TraceWasmError::Unsupported(format!(
                        "instruction `{:?}`",
                        operator
                    )));
                }
            }
        }

        // Both counts are already high-water marks — `max_registers` is maintained
        // by `advanced_register_index` and `allocation_len` only grows when no
        // freed spill slot can be reused — so they are read off directly here
        // rather than recomputed from the instruction list.
        let frame = FrameLayout {
            // Bounded by the operand-stack depth, which a function body's size in
            // the binary already bounds well below `u32::MAX`, so this cannot
            // truncate for any module that could be loaded at all.
            registers: simulated_stack.max_registers as u32,
            spills: simulated_stack.spills.allocation_len(),
            input_registers_arena: simulated_stack.input_registers.into_boxed_slice(),
            output_registers_arena: simulated_stack.output_registers.into_boxed_slice(),
        };

        Ok((instructions, frame))
    }
}
