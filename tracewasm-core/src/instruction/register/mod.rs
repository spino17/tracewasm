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
enum Slot {
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

struct Registers<const I: usize, const O: usize> {
    input: [Slot; I],
    output: [usize; O],
}

struct SimulatedStack {
    stack: Stack<StackSlot>,
    curr_register_index: usize,
    max_registers: usize,
    lazy_locals: LazyArena<Local>,
    lazy_globals: LazyArena<Global>,
    spills: SpillArena,
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

    fn registers_for<const I: usize, const O: usize>(&mut self) -> Registers<I, O> {
        let mut input = [Slot::default(); I];
        let mut output = [0; O];

        for i in 0..I {
            input[I - 1 - i] = self.pop();
        }

        for i in 0..O {
            output[i] = self.curr_register_index;
            let out = Slot::Register(self.curr_register_index as u32);

            self.push(out);
        }

        Registers { input, output }
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
#[derive(Debug, Clone, Copy)]
pub struct FrameLayout {
    /// Operand registers, i.e. the peak `curr_register_index`.
    pub registers: u32,
    /// Spill slots holding locals and globals rescued from a later write by
    /// [`RegInstruction::LocalSpill`] / [`RegInstruction::GlobalSpill`].
    ///
    /// Zero for a body that never overwrites a lazily-forwarded local or global,
    /// which is the common case.
    pub spills: u32,
}

/// The two outputs of lowering one function body into register form: the
/// instruction list, and the frame required to execute it.
type LoweredRegFuncBody = (Vec<RegInstruction>, FrameLayout);

pub enum RegInstruction {
    I32Load(Box<(u32, Registers<1, 1>)>), // (memarg, registers)
    GlobalSet(Box<(u32, Registers<1, 0>)>),
    LocalSet(Box<(u32, Registers<1, 0>)>),
    LocalTee(Box<(u32, Registers<1, 0>)>),
    I32Store(Box<(u32, Registers<2, 0>)>),
    I32Add(Box<Registers<2, 1>>),
    I32Eqz(Box<Registers<1, 1>>),
    Select(Box<Registers<3, 1>>),
    LocalSpill(u32, u32),  // (local_index, spill_index)
    GlobalSpill(u32, u32), // (global_index, spill_index)
}

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

                    instructions.push(RegInstruction::GlobalSet(Box::new((
                        global_index,
                        registers,
                    ))));
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

                    instructions.push(RegInstruction::LocalSet(Box::new((local_index, registers))));
                }
                Operator::LocalTee { local_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        local_index,
                        &mut simulated_stack.lazy_locals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::LocalSpill(local_index, spill_index));
                    }

                    let registers = Registers {
                        input: [simulated_stack.tee()],
                        output: [],
                    };

                    instructions.push(RegInstruction::LocalTee(Box::new((local_index, registers))));
                }
                Operator::I32Const { value } => {
                    simulated_stack.push_const(Const::I32(value));
                }
                Operator::I32Load { memarg } => {
                    let registers = simulated_stack.registers_for::<1, 1>();

                    instructions.push(RegInstruction::I32Load(Box::new((
                        memarg.offset as u32,
                        registers,
                    ))));
                }
                Operator::I32Store { memarg } => {
                    let registers = simulated_stack.registers_for::<2, 0>();

                    instructions.push(RegInstruction::I32Store(Box::new((
                        memarg.offset as u32,
                        registers,
                    ))));
                }
                Operator::I32Add => {
                    let registers = simulated_stack.registers_for::<2, 1>();

                    instructions.push(RegInstruction::I32Add(Box::new(registers)));
                }
                Operator::I32Eqz => {
                    let registers = simulated_stack.registers_for::<1, 1>();

                    instructions.push(RegInstruction::I32Eqz(Box::new(registers)));
                }
                Operator::Nop => {
                    continue;
                }
                Operator::Select => {
                    let registers = simulated_stack.registers_for::<3, 1>();

                    instructions.push(RegInstruction::Select(Box::new(registers)));
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
        };

        Ok((instructions, frame))
    }
}
