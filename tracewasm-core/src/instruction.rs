use wasmparser::{Operator, OperatorsReader};

pub enum Instruction {
    Unreachable,
    Nop,
}

impl Instruction {
    pub(crate) fn from_operator(operator: Operator<'_>) -> Result<Instruction, anyhow::Error> {
        let instruction = match operator {
            Operator::Unreachable => Instruction::Unreachable,
            Operator::Nop => Instruction::Nop,
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
