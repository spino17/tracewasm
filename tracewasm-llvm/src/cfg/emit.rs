use crate::cfg::{ControlFlowGraph, context::Context, walk::CfgVisitor};

pub struct IREmitter {
    ir: String,
    indentation: bool,
}

impl IREmitter {
    pub fn emit(cfg: ControlFlowGraph, ctx: &Context) -> Result<String, anyhow::Error> {
        let mut emitter = IREmitter {
            ir: String::default(),
            indentation: false,
        };

        emitter.walk_cfg(&cfg, ctx)?;

        Ok(emitter.ir)
    }

    fn set_indentation(&mut self) {
        self.indentation = true;
    }

    fn unset_indentation(&mut self) {
        self.indentation = false;
    }

    fn push_str(&mut self, s: &str) {
        self.ir.push_str(&format!(
            "{}{}",
            if self.indentation { "    " } else { "" },
            s
        ));
    }
}

impl CfgVisitor for IREmitter {
    type OkType = ();
    type ErrType = anyhow::Error;

    fn visit_unconditional_br(
        &mut self,
        operands: &crate::instruction::UnconditionalBrOperands,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }

    fn visit_conditional_br(
        &mut self,
        operands: &crate::instruction::ConditionalBrOperands,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }

    fn visit_alloca(
        &mut self,
        operands: &crate::instruction::AllocaOperands,
        value: &crate::value::Value,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }

    fn visit_load(
        &mut self,
        operands: &crate::instruction::LoadOperands,
        value: &crate::value::Value,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }

    fn visit_store(
        &mut self,
        operands: &crate::instruction::StoreOperands,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }

    fn visit_get_element_ptr(
        &mut self,
        operands: &crate::instruction::GetElementPtrOperands,
        value: &crate::value::Value,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }

    fn visit_phi(
        &mut self,
        instr: &crate::instruction::PhiInstruction,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }

    fn visit_func(
        &mut self,
        func: &super::function::Function,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }

    fn visit_basic_block(
        &mut self,
        block: &super::basic_block::BasicBlock,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }

    fn visit_cfg(
        &mut self,
        module: &super::ControlFlowGraph,
        ctx: &super::context::Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        todo!()
    }
}
