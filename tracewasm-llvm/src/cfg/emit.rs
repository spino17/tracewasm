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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        interner::TyId,
        test_support::fixture,
        value::{NullPtr, Type},
    };

    /// Every instruction the builder can produce, in one module, compared against the
    /// exact text.
    ///
    /// The expected string is not hand-written: it is what the emitter produced, then
    /// checked with `llvm-as`, which reported "parsed and verified". So this pins the
    /// output against a known-assembling module rather than against my reading of the
    /// LangRef — a spacing or keyword-order slip that still looks plausible would show
    /// up here as a diff.
    #[test]
    fn a_module_emits_ir_that_llvm_assembles() {
        let (mut ctx, mut builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f32_ty = ctx.f32_ty();
        let f64_ty = ctx.f64_ty();
        let ptr_ty = ctx.ptr_ty();

        let array_ty: TyId = ctx
            .ty_interner
            .intern(Type::Array {
                size: 4,
                element_ty: f64_ty,
            })
            .into();

        let struct_ty: TyId = ctx
            .ty_interner
            .intern(Type::Struct {
                fields: Box::new([i32_ty, array_ty]),
                packed: false,
            })
            .into();

        let f = builder
            .add_function(
                "main".to_string(),
                &[(i32_ty, Some("n".to_string())), (ptr_ty, None)],
                i32_ty,
                &mut ctx,
            )
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let exit = f.add_basic_block("exit".to_string(), &mut ctx).unwrap();

        let in_entry = builder.cursor_at_block(entry);

        let slot = in_entry
            .add_alloca(struct_ty, None, Some(8), Some("s"), &mut ctx)
            .unwrap();

        let count = Value::from_const(4i32, None, &mut ctx).unwrap();

        in_entry
            .add_alloca(i64_ty, Some((count, None)), None, None, &mut ctx)
            .unwrap();

        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();
        let one = Value::from_const(1i32, None, &mut ctx).unwrap();
        let two = Value::from_const(2i32, None, &mut ctx).unwrap();

        let elem = in_entry
            .add_get_element_ptr(
                None,
                slot,
                vec![zero, one, two],
                Some(true),
                Some("e"),
                &mut ctx,
            )
            .unwrap();

        let loaded = in_entry
            .add_load(f64_ty, elem.clone(), Some(8), Some("d"), &mut ctx)
            .unwrap();

        in_entry
            .add_store(loaded.clone(), elem, Some(8), None, &mut ctx)
            .unwrap();

        // `0.1f32` is the case that forces the hex encoding: `float 0.1` is refused by
        // `llvm-as` with "floating point constant invalid for type".
        let a_float = Value::from_const(0.1f32, None, &mut ctx).unwrap();

        let float_slot = in_entry
            .add_alloca(f32_ty, None, None, Some("fs"), &mut ctx)
            .unwrap();

        in_entry
            .add_store(a_float, float_slot, None, None, &mut ctx)
            .unwrap();

        let null = Value::from_const(NullPtr, None, &mut ctx).unwrap();

        let ptr_slot = in_entry
            .add_alloca(ptr_ty, None, None, Some("np"), &mut ctx)
            .unwrap();

        in_entry
            .add_store(null, ptr_slot, None, None, &mut ctx)
            .unwrap();
        in_entry.add_unconditional_br(body, &mut ctx);

        let in_body = builder.cursor_at_block(body);

        let (phi_handler, phi) = in_body
            .add_phi(&[(entry, loaded)], Some("m"), &mut ctx)
            .unwrap();

        // `body` reaches itself, so that edge needs its own incoming value — LLVM
        // requires one phi entry per predecessor.
        phi_handler.add_branch((body, phi), &mut ctx).unwrap();

        let cond = Value::from_const(true, None, &mut ctx)
            .unwrap()
            .into_i1(&ctx)
            .unwrap();

        in_body.add_conditional_br(cond, body, exit, &mut ctx);

        // No `ret` exists in the builder yet, so `exit` terminates with a self-loop.
        let in_exit = builder.cursor_at_block(exit);
        in_exit.add_unconditional_br(exit, &mut ctx);

        let ir = IREmitter::emit(builder.build(), &ctx).unwrap();

        let expected = concat!(
            "target triple = \"arm64-apple-macosx\"\n",
            "\n",
            "define i32 @main(i32 %n, ptr %0) {\n",
            "entry:\n",
            "    %s = alloca { i32, [4 x double] }, align 8\n",
            "    %1 = alloca i64, i32 4\n",
            "    %e = getelementptr inbounds { i32, [4 x double] }, ptr %s, i32 0, i32 1, i32 2\n",
            "    %d = load double, ptr %e, align 8\n",
            "    store double %d, ptr %e, align 8\n",
            "    %fs = alloca float\n",
            "    store float 0x3FB99999A0000000, ptr %fs\n",
            "    %np = alloca ptr\n",
            "    store ptr null, ptr %np\n",
            "    br label %body\n",
            "body:\n",
            "    %m = phi double [ %d, %entry ], [ %m, %body ]\n",
            "    br i1 true, label %body, label %exit\n",
            "exit:\n",
            "    br label %exit\n",
            "}\n",
            "\n",
        );

        assert_eq!(ir, expected, "\n--- emitted ---\n{ir}");
    }

    /// A `float` and a `double` holding the same number encode differently: the hex
    /// digits for a `float` are its value *widened to f64*, which is the form
    /// `llvm-as` reads back. Writing the f32 bits directly would be a different
    /// number.
    #[test]
    fn a_float_constant_is_encoded_as_its_widened_bits() {
        assert_eq!(
            IREmitter::constant(&ConstValue::Float(0.1f32.into())),
            "0x3FB99999A0000000"
        );

        assert_eq!(
            IREmitter::constant(&ConstValue::Double(0.1f64.into())),
            "0x3FB999999999999A",
            "the same decimal is a different constant at double width"
        );

        // The one place the two agree, so a test that only used `1.0` would not tell
        // the encodings apart.
        assert_eq!(
            IREmitter::constant(&ConstValue::Float(1.0f32.into())),
            IREmitter::constant(&ConstValue::Double(1.0f64.into()))
        );
    }

    /// `i1` renders as `true`/`false`, and a null pointer as `null`.
    #[test]
    fn scalar_constants_render_as_llvm_spells_them() {
        assert_eq!(IREmitter::constant(&ConstValue::I1(1)), "true");
        assert_eq!(IREmitter::constant(&ConstValue::I1(0)), "false");
        assert_eq!(IREmitter::constant(&ConstValue::I32(-7)), "-7");
        assert_eq!(
            IREmitter::constant(&ConstValue::I64(1 << 40)),
            "1099511627776"
        );
        assert_eq!(IREmitter::constant(&ConstValue::NullPtr), "null");
    }

    /// An empty triple and data layout are omitted rather than emitted as empty
    /// strings — `target triple = ""` is not what "unset" means.
    #[test]
    fn an_unset_target_emits_no_target_lines() {
        let ctx = Context::default();
        let builder = crate::cfg::builder::Builder::new(String::new(), String::new());

        let ir = IREmitter::emit(builder.build(), &ctx).unwrap();

        assert_eq!(ir, "", "an empty module emits nothing at all");
    }
}
