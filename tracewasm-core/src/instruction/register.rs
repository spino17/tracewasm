#![allow(warnings)]

use crate::{
    error::TraceWasmError,
    module::{FuncDecl, FuncType},
    vm::stack::Stack,
};
use wasmparser::{Operator, OperatorsReader};

pub enum RegInstruction {} // register instructions

#[derive(Clone, Copy)]
pub enum Const {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

#[derive(Clone, Copy)]
enum Slot {
    Const(Const),
    Local(u32),
    Global(u32),
    Register(u32), // index into stack
}

impl Slot {
    fn is_register(&self) -> bool {
        matches!(self, Slot::Register(_))
    }
}

struct Registers11 {
    input: Slot,
    output: Slot,
}

struct Registers01 {
    output: Slot,
}

struct Registers10 {
    input: Slot,
}

struct Registers20 {
    input: (Slot, Slot),
}

struct Registers21 {
    input: (Slot, Slot),
    output: Slot,
}

struct Registers30 {
    input: (Slot, Slot, Slot),
}

struct Registers31 {
    input: (Slot, Slot, Slot),
    output: Slot,
}

struct SimulatedStack {
    stack: Stack<Slot>,

    // below values are for real stack (registers) and not
    // simulated stack
    curr_register_index: u32,
    max_registers: u32,
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

    fn registers_for_11(&mut self) -> Registers11 {
        let val = self.pop();

        let out = Slot::Register(self.curr_register_index);

        self.push(out);

        Registers11 {
            input: val,
            output: out,
        }
    }

    fn registers_for_01(&mut self) -> Registers01 {
        let out = Slot::Register(self.curr_register_index);

        self.push(out);

        Registers01 { output: out }
    }

    fn registers_for_10(&mut self) -> Registers10 {
        let val = self.pop();

        Registers10 { input: val }
    }

    fn registers_for_21(&mut self) -> Registers21 {
        let val2 = self.pop();
        let val1 = self.pop();

        let out = Slot::Register(self.curr_register_index);

        self.push(out);

        Registers21 {
            input: (val1, val2),
            output: out,
        }
    }

    fn registers_for_20(&mut self) -> Registers20 {
        let val2 = self.pop();
        let val1 = self.pop();

        Registers20 {
            input: (val1, val2),
        }
    }

    fn registers_for_30(&mut self) -> Registers30 {
        let val3 = self.pop();
        let val2 = self.pop();
        let val1 = self.pop();

        Registers30 {
            input: (val1, val2, val3),
        }
    }

    fn registers_for_31(&mut self) -> Registers31 {
        let val3 = self.pop();
        let val2 = self.pop();
        let val1 = self.pop();

        let out = Slot::Register(self.curr_register_index);

        self.push(out);

        Registers31 {
            input: (val1, val2, val3),
            output: out,
        }
    }
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
                Operator::I32Const { value } => {
                    simulated_stack.push_const(Const::I32(value));
                }
                Operator::GlobalGet { global_index } => {
                    simulated_stack.push_global(global_index);
                }
                Operator::LocalGet { local_index } => {
                    simulated_stack.push_local(local_index);
                }
                Operator::I32Load { memarg } => {
                    let registers = simulated_stack.registers_for_11();
                }
                Operator::GlobalSet { global_index } => {
                    let registers = simulated_stack.registers_for_10();
                }
                Operator::LocalSet { local_index } => {
                    let registers = simulated_stack.registers_for_10();
                }
                Operator::I32Store { memarg } => {
                    let registers = simulated_stack.registers_for_20();
                }
                Operator::I32Add => {
                    let registers = simulated_stack.registers_for_21();
                }
                Operator::I32Eqz => {
                    let registers = simulated_stack.registers_for_11();
                }
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
