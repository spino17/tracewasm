use crate::cfg::ControlFlowGraph;

pub trait Emitter {
    type SourceInstr;
    type SourceInstrCtx;

    fn emit_cfg(
        self,
        stream: &[Self::SourceInstr],
        ctx: &Self::SourceInstrCtx,
    ) -> Result<ControlFlowGraph, anyhow::Error>;
}
