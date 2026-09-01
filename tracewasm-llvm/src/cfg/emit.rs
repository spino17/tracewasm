use crate::cfg::{ControlFlowGraph, context::Context};

impl ControlFlowGraph {
    fn emit_headers(ir: &mut str) {}

    pub fn emit_ll(&self, ctx: &Context) -> String {
        let mut ir = String::default();
        let module = &self.module;
        let funcs = &module.functions;

        Self::emit_headers(&mut ir);

        todo!()
    }
}
