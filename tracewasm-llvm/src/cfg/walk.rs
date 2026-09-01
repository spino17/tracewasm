use crate::{
    cfg::{
        basic_block::{BasicBlock, BasicBlockId},
        context::Context,
        function::{FuncId, Function},
        module::Module,
    },
    instruction::{
        AllocaOperands, ConditionalBrOperands, GetElementPtrOperands, InstructionKind,
        LoadOperands, PhiInstruction, StoreOperands, UnconditionalBrOperands,
    },
    value::Value,
};

pub trait CfgVisitor {
    type OkType;
    type ErrType;

    fn visit_phi(
        &mut self,
        instr: &PhiInstruction,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn visit_unconditional_br(
        &mut self,
        operands: &UnconditionalBrOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn visit_conditional_br(
        &mut self,
        operands: &ConditionalBrOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn visit_load(
        &mut self,
        operands: &LoadOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn visit_store(
        &mut self,
        operands: &StoreOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn visit_alloca(
        &mut self,
        operands: &AllocaOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn visit_get_element_ptr(
        &mut self,
        operands: &GetElementPtrOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn visit_basic_block(
        &mut self,
        block: &BasicBlock,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn visit_func(&mut self, func: &Function, ctx: &Context)
    -> Result<Self::OkType, Self::ErrType>;

    fn visit_module(
        &mut self,
        module: &Module,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn walk_basic_block(
        &mut self,
        id: BasicBlockId,
        block: &BasicBlock,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        let phis = &block.phis;
        let instructions = &block.instructions;
        let mut phi_results = vec![];
        let mut instr_results = vec![];

        let _res = self.visit_basic_block(block, ctx)?;

        for phi in phis {
            phi_results.push(self.visit_phi(phi, ctx)?);
        }

        for instr in instructions {
            let instr_kind = &instr.kind;
            let val = instr.value.as_ref();

            instr_results.push(match instr_kind {
                InstructionKind::UnconditionalBr(operands) => {
                    self.visit_unconditional_br(operands, ctx)?
                }
                InstructionKind::ConditionalBr(operands) => {
                    self.visit_conditional_br(operands, ctx)?
                }
                InstructionKind::Alloca(operands) => {
                    self.visit_alloca(operands, val.unwrap(), ctx)?
                }
                InstructionKind::Load(operands) => self.visit_load(operands, val.unwrap(), ctx)?,
                InstructionKind::Store(operands) => self.visit_store(operands, ctx)?,
                InstructionKind::GetElementPtr(operands) => {
                    self.visit_get_element_ptr(operands, val.unwrap(), ctx)?
                }
            });
        }

        self.post_block_visit(block.func_id, id, phi_results, instr_results)
    }

    fn walk_func(
        &mut self,
        func_id: FuncId,
        func: &Function,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        let blocks = &func.blocks;
        let mut block_results = vec![];

        let _res = self.visit_func(func, ctx)?;

        for block_id in blocks {
            let block = ctx.get_block(*block_id);
            let block_res = self.walk_basic_block(*block_id, block, ctx)?;

            block_results.push(block_res);
        }

        self.post_func_visit(func_id, block_results)
    }

    fn walk_module(
        &mut self,
        module: &Module,
        ctx: &mut Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        let funcs = &module.functions;
        let mut func_results = vec![];

        let _res = self.visit_module(module, ctx)?;

        for func_id in funcs {
            let func = ctx.get_func(*func_id);

            func_results.push(self.walk_func(*func_id, func, ctx)?);
        }

        self.post_module_visit(func_results)
    }

    fn post_block_visit(
        &mut self,
        func: FuncId,
        block: BasicBlockId,
        phi_results: Vec<Self::OkType>,
        instr_results: Vec<Self::OkType>,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn post_func_visit(
        &mut self,
        func: FuncId,
        block_results: Vec<Self::OkType>,
    ) -> Result<Self::OkType, Self::ErrType>;

    fn post_module_visit(
        &mut self,
        func_results: Vec<Self::OkType>,
    ) -> Result<Self::OkType, Self::ErrType>;
}
