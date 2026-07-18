use crate::parser::{FuncIndex, GlobalIndex};
use wasmparser::{Ieee32, Ieee64, Operator, OperatorsReader};

pub enum Instruction {
    I32Const { value: i32 },
    I64Const { value: i64 },
    F32Const { value: Ieee32 },
    F64Const { value: Ieee64 },
    GlobalGet { index: GlobalIndex },
    RefNull,
    RefFunc { index: FuncIndex },
}

impl Instruction {
    pub(crate) fn from_operator(operator: Operator<'_>) -> Result<Instruction, anyhow::Error> {
        let instruction = match operator {
            Operator::I32Const { value } => Instruction::I32Const { value },
            Operator::I64Const { value } => Instruction::I64Const { value },
            Operator::F32Const { value } => Instruction::F32Const { value },
            Operator::F64Const { value } => Instruction::F64Const { value },
            Operator::GlobalGet { global_index } => Instruction::GlobalGet {
                index: GlobalIndex(global_index),
            },
            Operator::RefNull { .. } => Instruction::RefNull,
            Operator::RefFunc { function_index } => Instruction::RefFunc {
                index: FuncIndex(function_index),
            },
            _ => todo!(),
        };

        Ok(instruction)
    }

    pub(crate) fn emit_instruction_from_operator_reader(
        mut operator_reader: OperatorsReader<'_>,
    ) -> Result<Vec<Instruction>, anyhow::Error> {
        let mut instructions = vec![];

        while !operator_reader.eof() {
            let operator = operator_reader.read()?;
            instructions.push(Instruction::from_operator(operator)?);
        }

        Ok(instructions)
    }
}
