use crate::cfg::ControlFlowGraph;

pub trait Emitter<I> {
    type SourceInstr;
    type SourceInstrCtx;

    fn emit_cfg(
        self,
        stream: &[Self::SourceInstr],
        ctx: &Self::SourceInstrCtx,
    ) -> Result<ControlFlowGraph<I>, anyhow::Error>;
}
