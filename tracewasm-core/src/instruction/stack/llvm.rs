use crate::instruction::stack::{StackFrameLayout, StackInstruction};
use tracewasm_llvm::{cfg::ControlFlowGraph, emit::Emitter};

pub(crate) struct StackInstructionEmitter {}

pub(crate) struct StackInstructionContext {
    func_name: String,
    frame_layout: StackFrameLayout,
}

impl Emitter for StackInstructionEmitter {
    type SourceInstr = StackInstruction;
    type SourceInstrCtx = StackInstructionContext;

    fn emit_cfg(
        self,
        _stream: &[StackInstruction],
        _ctx: &StackInstructionContext,
    ) -> Result<ControlFlowGraph, anyhow::Error> {
        todo!()
    }
}
