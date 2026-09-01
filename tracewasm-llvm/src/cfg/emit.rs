use crate::{
    cfg::{
        ControlFlowGraph, basic_block::BasicBlock, basic_block::BasicBlockId, context::Context,
        function::Function, walk::CfgVisitor,
    },
    instruction::{
        AllocaOperands, ConditionalBrOperands, GetElementPtrOperands, LoadOperands, PhiInstruction,
        StoreOperands, UnconditionalBrOperands,
    },
    value::{ConstValue, Value, ValueKind},
};
use anyhow::bail;

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

    /// One whole line, since `push_str` prepends the indentation to every call —
    /// building a line in pieces would indent each piece.
    fn push_line(&mut self, s: &str) {
        self.push_str(&format!("{s}\n"));
    }

    /// A register's `%name`, or the literal a constant is spelled as.
    fn operand(value: &Value, ctx: &Context) -> Result<String, anyhow::Error> {
        match value.kind() {
            ValueKind::Reg(reg) => Ok(format!("%{}", ctx.str_interner.value(reg.name.0))),
            ValueKind::Const(id) => Ok(Self::constant(ctx.const_interner.value(id.raw()))),
            // Nothing builds one yet, so refusing is honest: emitting a placeholder
            // would put text into the module that `llvm-as` cannot parse.
            ValueKind::ConstExpr(_) => {
                bail!("a constant expression operand is not emitted yet")
            }
        }
    }

    /// `<type> <operand>`, the form an operand takes almost everywhere in LLVM.
    fn typed_operand(value: &Value, ctx: &Context) -> Result<String, anyhow::Error> {
        Ok(format!(
            "{} {}",
            ctx.display(value.ty()),
            Self::operand(value, ctx)?
        ))
    }

    /// Floats are written as hex, always.
    ///
    /// LLVM only accepts the decimal form when the literal is *exactly* the value the
    /// type can hold: `float 0.1` is refused with "floating point constant invalid for
    /// type", while `double 0.1` is fine. The hex form is exact by construction, so it
    /// sidesteps the distinction — and for a `float` the digits are the f32 value
    /// widened to f64, which is the encoding LLVM reads back.
    fn constant(value: &ConstValue) -> String {
        match value {
            ConstValue::I1(v) => {
                if *v == 0 {
                    "false".to_string()
                } else {
                    "true".to_string()
                }
            }
            ConstValue::I8(v) => v.to_string(),
            ConstValue::I16(v) => v.to_string(),
            ConstValue::I32(v) => v.to_string(),
            ConstValue::I64(v) => v.to_string(),
            ConstValue::Float(v) => {
                format!("0x{:016X}", (v.into_inner() as f64).to_bits())
            }
            ConstValue::Double(v) => format!("0x{:016X}", v.into_inner().to_bits()),
            ConstValue::NullPtr => "null".to_string(),
        }
    }

    /// `label %name`, which is how a branch and a phi both name a block.
    fn label(id: BasicBlockId, ctx: &Context) -> String {
        format!("%{}", Self::block_name(id, ctx))
    }

    fn block_name(id: BasicBlockId, ctx: &Context) -> String {
        let block = ctx.get_block(id);

        ctx.str_interner.value(block.name.0).to_string()
    }

    /// The `%x = ` an instruction that defines a register is prefixed with.
    fn assignment(value: &Value, ctx: &Context) -> Result<String, anyhow::Error> {
        Ok(format!("{} = ", Self::operand(value, ctx)?))
    }

    fn alignment(align: Option<u32>) -> String {
        match align {
            Some(align) => format!(", align {align}"),
            None => String::new(),
        }
    }
}

impl CfgVisitor for IREmitter {
    type OkType = ();
    type ErrType = anyhow::Error;

    fn visit_cfg(
        &mut self,
        module: &ControlFlowGraph,
        _ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        self.unset_indentation();

        // Both are optional in a module, and `Builder::new` accepts empty strings for
        // them — an empty `target triple = ""` line is not what that means.
        if !module.module.data_layout.is_empty() {
            self.push_line(&format!(
                "target datalayout = \"{}\"",
                module.module.data_layout
            ));
        }

        if !module.module.triple.is_empty() {
            self.push_line(&format!("target triple = \"{}\"", module.module.triple));
        }

        if !module.module.data_layout.is_empty() || !module.module.triple.is_empty() {
            self.push_line("");
        }

        Ok(())
    }

    fn visit_func(
        &mut self,
        func: &Function,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        let mut params = vec![];

        for param in &func.params {
            params.push(Self::typed_operand(param, ctx)?);
        }

        let name = ctx.str_interner.value(func.name.0);

        self.unset_indentation();

        self.push_line(&format!(
            "define {} @{}({}) {{",
            ctx.display(func.result),
            name,
            params.join(", ")
        ));

        Ok(())
    }

    fn visit_basic_block(
        &mut self,
        block: &BasicBlock,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        // A label sits at the margin; the instructions under it are indented.
        self.unset_indentation();
        self.push_line(&format!("{}:", ctx.str_interner.value(block.name.0)));
        self.set_indentation();

        Ok(())
    }

    fn visit_phi(
        &mut self,
        instr: &PhiInstruction,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        let mut branches = vec![];

        for (block, value) in &instr.branches {
            branches.push(format!(
                "[ {}, {} ]",
                Self::operand(value, ctx)?,
                Self::label(*block, ctx)
            ));
        }

        self.push_line(&format!(
            "{}phi {} {}",
            Self::assignment(&instr.value, ctx)?,
            ctx.display(instr.ref_ty),
            branches.join(", ")
        ));

        Ok(())
    }

    fn visit_unconditional_br(
        &mut self,
        operands: &UnconditionalBrOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        self.push_line(&format!("br label {}", Self::label(operands.label, ctx)));

        Ok(())
    }

    fn visit_conditional_br(
        &mut self,
        operands: &ConditionalBrOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        // The condition is an `I1Value`, so the `i1` is known without asking the pool.
        let cond = match operands.cond.kind {
            ValueKind::Reg(reg) => format!("%{}", ctx.str_interner.value(reg.name.0)),
            ValueKind::Const(id) => Self::constant(ctx.const_interner.value(id.raw())),
            ValueKind::ConstExpr(_) => {
                bail!("a constant expression condition is not emitted yet")
            }
        };

        self.push_line(&format!(
            "br i1 {}, label {}, label {}",
            cond,
            Self::label(operands.true_label, ctx),
            Self::label(operands.false_label, ctx)
        ));

        Ok(())
    }

    fn visit_alloca(
        &mut self,
        operands: &AllocaOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        let count = match &operands.count {
            Some(count) => format!(", {}", Self::typed_operand(count, ctx)?),
            None => String::new(),
        };

        self.push_line(&format!(
            "{}alloca {}{}{}",
            Self::assignment(value, ctx)?,
            ctx.display(operands.ty),
            count,
            Self::alignment(operands.align)
        ));

        Ok(())
    }

    fn visit_load(
        &mut self,
        operands: &LoadOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        self.push_line(&format!(
            "{}load {}, {}{}",
            Self::assignment(value, ctx)?,
            ctx.display(operands.ty),
            Self::typed_operand(&operands.ptr, ctx)?,
            Self::alignment(operands.align)
        ));

        Ok(())
    }

    fn visit_store(
        &mut self,
        operands: &StoreOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        self.push_line(&format!(
            "store {}, {}{}",
            Self::typed_operand(&operands.value, ctx)?,
            Self::typed_operand(&operands.ptr, ctx)?,
            Self::alignment(operands.align)
        ));

        Ok(())
    }

    fn visit_get_element_ptr(
        &mut self,
        operands: &GetElementPtrOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        let mut indices = vec![];

        for index in operands.indices.iter() {
            indices.push(Self::typed_operand(index, ctx)?);
        }

        // Every index is a separate trailing operand, and an empty list is legal:
        // `getelementptr i32, ptr %p` assembles.
        let indices = if indices.is_empty() {
            String::new()
        } else {
            format!(", {}", indices.join(", "))
        };

        self.push_line(&format!(
            "{}getelementptr {}{}, {}{}",
            Self::assignment(value, ctx)?,
            if operands.inbounds { "inbounds " } else { "" },
            ctx.display(operands.source_ty),
            Self::typed_operand(&operands.ptr, ctx)?,
            indices
        ));

        Ok(())
    }

    fn post_func_visit(
        &mut self,
        _func: crate::cfg::function::FuncId,
        _block_results: Vec<Self::OkType>,
    ) -> Result<Self::OkType, Self::ErrType> {
        self.unset_indentation();
        self.push_line("}");
        self.push_line("");

        Ok(())
    }
}
