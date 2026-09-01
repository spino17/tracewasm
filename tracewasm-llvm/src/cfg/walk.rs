use crate::{
    cfg::{context::Context, module::Module},
    instruction::{
        AllocaOperands, ConditionalBrOperands, GetElementPtrOperands, InstructionKind,
        LoadOperands, PhiInstruction, StoreOperands, UnconditionalBrOperands,
    },
    value::Value,
};

pub trait CfgVisitor {
    type Result;

    fn visit_phi(&mut self, instr: &PhiInstruction, ctx: &Context) -> Self::Result;

    fn visit_unconditional_br(
        &mut self,
        operands: &UnconditionalBrOperands,
        ctx: &Context,
    ) -> Self::Result;

    fn visit_conditional_br(
        &mut self,
        operands: &ConditionalBrOperands,
        ctx: &Context,
    ) -> Self::Result;

    fn visit_load(&mut self, operands: &LoadOperands, value: &Value, ctx: &Context)
    -> Self::Result;

    fn visit_store(&mut self, operands: &StoreOperands, ctx: &Context) -> Self::Result;

    fn visit_alloca(
        &mut self,
        operands: &AllocaOperands,
        value: &Value,
        ctx: &Context,
    ) -> Self::Result;

    fn visit_get_element_ptr(
        &mut self,
        operands: &GetElementPtrOperands,
        value: &Value,
        ctx: &Context,
    ) -> Self::Result;

    fn walk_module(&mut self, module: &Module, ctx: &mut Context) -> Self::Result {
        let funcs = &module.functions;

        for func_id in funcs {
            let func = ctx.get_func(*func_id);
            let _func_name = func.name;
            let blocks = &func.blocks;

            for block_id in blocks {
                let block = ctx.get_block(*block_id);
                let phis = &block.phis;
                let instructions = &block.instructions;

                for phi in phis {
                    self.visit_phi(phi, ctx);
                }

                for instr in instructions {
                    let instr_kind = &instr.kind;
                    let val = instr.value.as_ref();

                    match instr_kind {
                        InstructionKind::UnconditionalBr(operands) => {
                            self.visit_unconditional_br(operands, ctx);
                        }
                        InstructionKind::ConditionalBr(operands) => {
                            self.visit_conditional_br(operands, ctx);
                        }
                        InstructionKind::Alloca(operands) => {
                            self.visit_alloca(operands, val.unwrap(), ctx);
                        }
                        InstructionKind::Load(operands) => {
                            self.visit_load(operands, val.unwrap(), ctx);
                        }
                        InstructionKind::Store(operands) => {
                            self.visit_store(operands, ctx);
                        }
                        InstructionKind::GetElementPtr(operands) => {
                            self.visit_get_element_ptr(operands, val.unwrap(), ctx);
                        }
                    }
                }
            }
        }

        todo!()
    }
}
