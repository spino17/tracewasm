use crate::instruction::register::{RegFrameLayout, RegInstruction};
use tracewasm_llvm::{
    cfg::{ControlFlowGraph, builder::Builder, context::Context},
    emit::Emitter,
};

pub(crate) struct RegInstructionEmitter {}

pub(crate) struct RegInstructionCtx {
    func_name: String,
    frame_layout: RegFrameLayout,
}

impl Emitter for RegInstructionEmitter {
    type SourceInstr = RegInstruction;
    type SourceInstrCtx = RegInstructionCtx;

    fn emit_cfg(
        self,
        stream: &[RegInstruction],
        ctx: &RegInstructionCtx,
    ) -> Result<ControlFlowGraph, anyhow::Error> {
        todo!()
    }
}
