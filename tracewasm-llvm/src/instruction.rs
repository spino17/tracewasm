use crate::{cfg::BasicBlockId, value::I1Value};

pub enum Instruction {
    UnconditionalBr {
        label: BasicBlockId,
    },
    ConditionalBr {
        cond: I1Value,
        true_label: BasicBlockId,
        false_label: BasicBlockId,
    },
}

impl Instruction {
    pub fn new_unconditional_br(label: BasicBlockId) -> Self {
        Instruction::UnconditionalBr { label }
    }

    pub fn new_conditional_br(
        cond: I1Value,
        true_label: BasicBlockId,
        false_label: BasicBlockId,
    ) -> Self {
        Instruction::ConditionalBr {
            cond,
            true_label,
            false_label,
        }
    }
}
