#![allow(warnings)]

use crate::{
    error::TraceWasmError,
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
    Register(u32), // index into stack
}

impl Default for Slot {
    fn default() -> Self {
        Slot::Const(Const::I32(0))
    }
}

impl Slot {
    fn is_register(&self) -> bool {
        matches!(self, Slot::Register(_))
    }
}

struct Registers<const I: usize, const O: usize> {
    input: [Slot; I],
    output: [usize; O],
}

struct SimulatedStack {
    stack: Stack<Slot>,

    // below values are for real stack (registers) and not
    // simulated stack
    curr_register_index: usize,
    max_registers: usize,
}

impl Default for SimulatedStack {
    fn default() -> Self {
        SimulatedStack {
            stack: Stack::new_with_capacity(0),
            curr_register_index: 0,
            max_registers: 0,
        }
    }
}

impl SimulatedStack {
    fn advanced_register_index(&mut self) {
        self.curr_register_index += 1;

        if self.curr_register_index > self.max_registers {
            self.max_registers += 1;
        }
    }

    fn recede_register_index(&mut self) {
        self.curr_register_index -= 1;
    }

    fn pop(&mut self) -> Slot {
        let val = self.stack.pop();

        if val.is_register() {
            self.recede_register_index();
        }

        val
    }

    fn push(&mut self, val: Slot) {
        if val.is_register() {
            self.advanced_register_index();
        }

        self.stack.push(val);
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
}

pub enum RegInstruction {
    I32Load(Box<(u32, Registers<1, 1>)>), // (memarg, registers)
    GlobalSet(Box<(u32, Registers<1, 0>)>),
    LocalSet(Box<(u32, Registers<1, 0>)>),
    I32Store(Box<(u32, Registers<2, 0>)>),
    I32Add(Box<Registers<2, 1>>),
    I32Eqz(Box<Registers<1, 1>>),
}

impl RegInstruction {
    pub(crate) fn emit_instruction_for_func(
        mut operator_reader: OperatorsReader<'_>,
        params: u32,
        results: u32,
        types: &[FuncType],
        func_decls: &[FuncDecl],
    ) -> Result<Vec<RegInstruction>, TraceWasmError> {
        let mut instructions: Vec<RegInstruction> = vec![];
        let mut simulated_stack = SimulatedStack::default();

        while !operator_reader.eof() {
            let (operator, offset) = operator_reader.read_with_offset()?;

            match operator {
                Operator::GlobalGet { global_index } => {
                    simulated_stack.push_global(global_index);
                }
                Operator::GlobalSet { global_index } => {
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
                    let registers = simulated_stack.registers_for::<1, 0>();

                    instructions.push(RegInstruction::LocalSet(Box::new((local_index, registers))));
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
                // TODO - add blocks and branch instructions!
                _ => {
                    return Err(TraceWasmError::Unsupported(format!(
                        "instruction `{:?}`",
                        operator
                    )));
                }
            }
        }

        Ok(instructions)
    }
}
