#![allow(warnings)]

use crate::{
    error::TraceWasmError,
    instruction::{
        Block, BlockKind, params_and_results_from_blockty,
        register::lazy::{
            Global, GlobalSlot, LazyArena, LazyEntryDropResult, LazyLocation, LazySlot, Local,
            LocalSlot, SpillArena,
        },
    },
    module::{FuncDecl, FuncType},
    vm::stack::Stack,
};
use std::marker::PhantomData;
use wasmparser::{BlockType, Operator, OperatorsReader};

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

pub struct Signature<const I: usize, const O: usize> {
    input: Registers<I, Slot>,
    output: Registers<O, u32>,
}

pub struct DynSignature {
    input: u32,
    output: u32,
    len: u32,
}

impl DynSignature {
    pub fn input_registers<'a>(&self, arena: &'a [Slot]) -> &'a [Slot] {
        let start = self.input as usize;

        &arena[start..(start + self.len as usize)]
    }

    pub fn output_registers<'a>(&self, arena: &'a [u32]) -> &'a [u32] {
        let start = self.output as usize;

        &arena[start..(start + self.len as usize)]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

struct PopsPushesResult {
    input_start: u32,
    output_start: u32,
}

#[derive(Default)]
struct ControlStack {
    stack: Vec<Block>,
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
    control_stack: ControlStack,
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
            control_stack: ControlStack::default(),
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

    fn add_block(
        &mut self,
        kind: BlockKind,
        blockty: &BlockType,
        types: &[FuncType],
    ) -> (u32, u32) {
        let (params, results) = params_and_results_from_blockty(blockty, types);

        let is_unreachable_traversing = self
            .control_stack
            .stack
            .last()
            .is_some_and(|b| b.is_unreachable_traversing);

        if is_unreachable_traversing {
            self.control_stack.stack.push(Block {
                kind,
                recorded_height: 0, // this won't be used at runtime because of unreachablity
                params,
                results,
                is_unreachable_traversing,
                has_inherited: true,
                attached_breaks: vec![],
            });

            return (params, results);
        }

        let recorded_height = match kind {
            BlockKind::Func => 0,
            BlockKind::Block { .. } => self.stack.height() - params,
            BlockKind::Loop { .. } => self.stack.height() - params,
            BlockKind::If { .. } => {
                // top is the `if` condition and then params
                self.stack.height() - params - 1
            }
        };

        self.control_stack.stack.push(Block {
            kind,
            recorded_height,
            params,
            results,
            is_unreachable_traversing: false,
            has_inherited: false,
            attached_breaks: vec![],
        });

        (params, results)
    }

    fn get_curr_block(&self) -> &Block {
        debug_assert!(!self.control_stack.stack.is_empty());

        &self.control_stack.stack[self.control_stack.stack.len() - 1]
    }

    fn get_curr_block_mut(&mut self) -> &mut Block {
        debug_assert!(!self.control_stack.stack.is_empty());
        let len = self.control_stack.stack.len();

        &mut self.control_stack.stack[len - 1]
    }

    fn get_block_mut(&mut self, index: usize) -> &mut Block {
        &mut self.control_stack.stack[index]
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

    fn pops_and_pushes(&mut self, pops: u32, pushes: u32) -> PopsPushesResult {
        let pops = pops as usize;
        let pushes = pushes as usize;
        let input_start = self.input_registers.len();
        let output_start = self.output_registers.len();

        self.input_registers
            .resize(input_start + pops, Slot::default());

        for i in 0..pops {
            self.input_registers[input_start + pops - 1 - i] = self.pop();
        }

        self.output_registers.resize(output_start + pushes, 0);

        for i in 0..pushes {
            self.output_registers[output_start + i] = self.curr_register_index as u32;
            let out = Slot::Register(self.curr_register_index as u32);

            self.push(out);
        }

        PopsPushesResult {
            input_start: input_start as u32,
            output_start: output_start as u32,
        }
    }

    fn registers_for<const I: usize, const O: usize>(&mut self) -> Signature<I, O> {
        let result = self.pops_and_pushes(I as u32, O as u32);

        Signature {
            input: Registers {
                start: result.input_start as u32,
                phantom: PhantomData,
            },
            output: Registers {
                start: result.output_start as u32,
                phantom: PhantomData,
            },
        }
    }

    fn materialize_stack_slots_in_registers(&mut self, depth: u32) -> DynSignature {
        let result = self.pops_and_pushes(depth, depth);

        DynSignature {
            input: result.input_start,
            output: result.output_start,
            len: depth,
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
    I32Load(u32, Signature<1, 1>), // (memarg, registers)
    GlobalSet(u32, Signature<1, 0>),
    LocalSet(u32, Signature<1, 0>),
    LocalTee(u32, Signature<1, 0>),
    I32Store(u32, Signature<2, 0>),
    I32Add(Signature<2, 1>),
    I32Eqz(Signature<1, 1>),
    Select(Signature<3, 1>),
    LocalSpill(u32, u32),  // (local_index, spill_index)
    GlobalSpill(u32, u32), // (global_index, spill_index)
    Move(DynSignature),
    If {
        cond: Signature<1, 0>,
        else_index: Option<u32>,
        end_index: u32,
    },
    Else {
        end_index: u32,
    },
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
    size_of::<RegInstruction>() <= 20,
    "RegInstruction grew past 20 bytes. Need to keep it compact."
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

        simulated_stack.control_stack.stack.push(Block {
            kind: BlockKind::Func,
            recorded_height: 0, // functions always have recorded height to be 0, so they leave stack with just its results
            params,
            results,
            is_unreachable_traversing: false,
            has_inherited: false,
            attached_breaks: vec![],
        });

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

                    let registers = Signature {
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
                Operator::If { blockty } => {
                    // the simulated stack would have layout like this at `if` instruction: [...other...][...params...][cond]
                    // to obtain recorded_height, we should pop params + 1 number of stack slots, and measure the `curr_register_index`
                    // which will be the recorded_height. So after the `end` of this `if` we should leave the stack
                    // at: recorded_height + results. We should materalize all the popped values by pushing it back on the stack
                    // making them materialized in registers. Same would be for results. So no matter what branch control flow takes
                    // the layout of frame in the start of the instruction and at the end of the instruction is same.

                    let (block_params, _) = simulated_stack.add_block(
                        BlockKind::If {
                            index: instructions.len() as u32 + 1, // at instruction.len, mov instruction be placed and then if instruction.
                            else_index: None,
                        },
                        &blockty,
                        types,
                    );

                    let move_registers =
                        simulated_stack.materialize_stack_slots_in_registers(block_params + 1);

                    instructions.push(RegInstruction::Move(move_registers));

                    let registers = simulated_stack.registers_for::<1, 0>(); // condition

                    instructions.push(RegInstruction::If {
                        cond: registers,
                        else_index: None,
                        end_index: u32::MAX,
                    });
                }
                Operator::Else => {
                    let if_block = simulated_stack.get_curr_block_mut();
                    let recorded_height = if_block.recorded_height;
                    let block_params = if_block.params;
                    let block_results = if_block.results;

                    let BlockKind::If {
                        index: _,
                        else_index,
                    } = &mut if_block.kind
                    else {
                        unreachable!(
                            "hitting this means TraceWasm has a bug recording the instructions"
                        )
                    };

                    *else_index = Some(if block_results == 0 {
                        instructions.len() as u32
                    } else {
                        instructions.len() as u32 + 1 // mov instruction also emitted!
                    });

                    if block_results != 0 {
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(block_results);

                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    // reset the frame layout with params on top for else instructions.
                    let result = simulated_stack.pops_and_pushes(
                        simulated_stack.stack.height() - recorded_height,
                        block_params,
                    );

                    instructions.push(RegInstruction::Else {
                        end_index: u32::MAX,
                    }); // TODO: backpatched when end is visited
                }
                Operator::End => {
                    todo!()
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
