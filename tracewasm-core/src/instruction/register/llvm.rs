use crate::instruction::register::{RegFrameLayout, RegInstruction};
use tracewasm_llvm::{cfg::ControlFlowGraph, emit::Emitter};

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
        _stream: &[RegInstruction],
        _ctx: &RegInstructionCtx,
    ) -> Result<ControlFlowGraph, anyhow::Error> {
        todo!()
    }
}
