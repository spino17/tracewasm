use crate::instruction::register::{RegFrameLayout, RegInstruction};
use tracewasm_llvm::{
    cfg::{Builder, Context, ControlFlowGraph},
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
        let mut context = Context::default();
        let mut builder = Builder::new("".to_string(), "".to_string());

        let func = builder.add_function(ctx.func_name.to_string(), &mut context);
        let entry = func.add_basic_block("entry".to_string(), &mut context);

        let cursor = builder.cursor_at_block(entry, &mut context);

        for instr in stream {}

        Ok(builder.build())
    }
}
