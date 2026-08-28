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

impl Instruction {}
