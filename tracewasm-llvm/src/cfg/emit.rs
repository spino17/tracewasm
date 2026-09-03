//! Rendering a finished graph as textual LLVM IR.

use crate::{
    cfg::{
        ControlFlowGraph,
        basic_block::{BasicBlock, BasicBlockId},
        context::Context,
        function::Function,
        global::{GlobalVariable, Linkage, Visiblity},
        walk::CfgVisitor,
    },
    instruction::{
        AllocaOperands, CallOperands, ConditionalBrOperands, FArithmeticOperands, FCmpOperands,
        FNegOperands, GetElementPtrOperands, IArithmeticOperands, ICmpOperands, LoadOperands,
        PhiInstruction, RetOperands, StoreOperands, UnconditionalBrOperands,
    },
    value::{ConstExpr, ConstValue, FuncSignature, I1Value, Value, ValueKind},
};
use anyhow::bail;

/// Renders a [`ControlFlowGraph`] as textual `.ll`.
///
/// A [`CfgVisitor`] that appends to a string as it walks. The output is what
/// `llvm-as` accepts: the crate's own test builds a module using every instruction
/// and checks that `llvm-as` both parses **and** verifies it.
///
/// Nothing is validated here — a block without a terminator, or a phi missing an
/// entry for a predecessor, is emitted as-is and reported by `llvm-as`.
pub struct IREmitter {
    ir: String,
    indentation: bool,
}

impl IREmitter {
    /// Renders `cfg` and returns the IR.
    ///
    /// # Errors
    ///
    /// If the graph contains something the emitter cannot spell. Currently that is
    /// only a constant expression other than
    /// [`GetElementPtr`](crate::value::ConstExpr::GetElementPtr), since the others
    /// carry no operands — refused rather than written as a placeholder `llvm-as`
    /// could not parse.
    pub fn emit(cfg: ControlFlowGraph) -> Result<String, anyhow::Error> {
        let mut emitter = IREmitter {
            ir: String::default(),
            indentation: false,
        };

        emitter.walk_cfg(&cfg)?;

        Ok(emitter.ir)
    }

    /// Indents subsequent lines, for instructions under a label.
    fn set_indentation(&mut self) {
        self.indentation = true;
    }

    /// Returns to the margin, for labels and `define`.
    fn unset_indentation(&mut self) {
        self.indentation = false;
    }

    /// Appends text, prefixed by the current indentation.
    ///
    /// Because the prefix goes on *every* call, a line has to be built as one
    /// string — see [`push_line`](Self::push_line).
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

    /// A register's `%name`, the literal a constant is spelled as, or a constant
    /// expression written inline.
    fn operand(value: &Value, ctx: &Context) -> Result<String, anyhow::Error> {
        Self::operand_kind(value.kind(), ctx)
    }

    /// [`operand`](Self::operand) for a bare [`ValueKind`].
    ///
    /// Split out for [`I1Value`](crate::value::I1Value), which carries a kind without
    /// a whole [`Value`] to hand over — so a branch condition renders through exactly
    /// the same arms as every other operand rather than a parallel copy of them.
    fn operand_kind(kind: &ValueKind, ctx: &Context) -> Result<String, anyhow::Error> {
        match kind {
            ValueKind::Reg(reg) => Ok(format!("%{}", ctx.str_interner.value(reg.name.0))),
            ValueKind::ConstExpr(expr) => Self::const_expr(expr, ctx),
            // A global is referred to by name, whatever it names — a variable, a
            // defined function, a declaration. Its *value* is the address, which is
            // why `Value::from_global` types it `ptr`.
            ValueKind::Global(global) => {
                Ok(format!("@{}", ctx.str_interner.value(global.name().0)))
            }
        }
    }

    /// A constant expression, in the parenthesised form LLVM writes it.
    ///
    /// Unlike the instruction, a constant expression takes no `%x = ` and wraps its
    /// operands in parentheses: `getelementptr inbounds ([4 x i32], ptr @g, i32 0,
    /// i32 2)`. It appears wherever a constant may, so it is produced by
    /// [`operand`](Self::operand) rather than by a visit.
    fn const_expr(expr: &ConstExpr, ctx: &Context) -> Result<String, anyhow::Error> {
        match expr {
            ConstExpr::GetElementPtr(operands) => {
                let mut indices = vec![];

                for index in operands.indices.iter() {
                    indices.push(Self::typed_operand(index, ctx)?);
                }

                let indices = if indices.is_empty() {
                    String::new()
                } else {
                    format!(", {}", indices.join(", "))
                };

                Ok(format!(
                    "getelementptr {}({}, {}{})",
                    if operands.inbounds { "inbounds " } else { "" },
                    ctx.display(operands.source_ty),
                    Self::typed_operand(&operands.ptr, ctx)?,
                    indices
                ))
            }
            ConstExpr::Const(const_id) => {
                Ok(Self::constant(ctx.const_interner.value(const_id.raw())))
            }
            // These four carry no operands to render — `PtrToInt` and `IntToPtr` have
            // no fields at all, and `BitCast`/`Trunc` name only a target type with no
            // value to convert. Refusing is honest: a placeholder would put text into
            // the module that `llvm-as` cannot parse.
            other => bail!("`{other:?}` has no operands, so it cannot be emitted yet"),
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

    /// `%name`, how a branch or a phi names a block.
    fn label(id: BasicBlockId, ctx: &Context) -> String {
        format!("%{}", Self::block_name(id, ctx))
    }

    /// A block's label without the sigil.
    fn block_name(id: BasicBlockId, ctx: &Context) -> String {
        let block = ctx.get_block(id);

        ctx.str_interner.value(block.name.0).to_string()
    }

    /// The `%x = ` prefix for an instruction that defines a register.
    fn assignment(value: &Value, ctx: &Context) -> Result<String, anyhow::Error> {
        Ok(format!("{} = ", Self::operand(value, ctx)?))
    }

    /// `, align N`, or nothing when the ABI default is wanted.
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

    fn visit_cfg(&mut self, cfg: &ControlFlowGraph) -> Result<Self::OkType, Self::ErrType> {
        self.unset_indentation();

        // An unset data layout is the empty string, and an empty
        // `target datalayout = ""` line is not what "unset" means. A structured
        // `Triple` is always present, so that guard is belt-and-braces.
        if !cfg.context.module.data_layout.is_empty() {
            self.push_line(&format!(
                "target datalayout = \"{}\"",
                cfg.context.module.data_layout
            ));
        }

        if !cfg.context.module.triple.is_empty() {
            self.push_line(&format!(
                "target triple = \"{}\"",
                cfg.context.module.triple
            ));
        }

        if !cfg.context.module.data_layout.is_empty() || !cfg.context.module.triple.is_empty() {
            self.push_line("");
        }

        Ok(())
    }

    fn visit_global_variable(
        &mut self,
        name: &str,
        data: &GlobalVariable,
        linkage: Linkage,
        visiblity: Visiblity,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        // A *definition* with external linkage omits the keyword — that is the
        // default. Writing `external` is what makes it a declaration, and `llvm-as`
        // then refuses an initializer outright: `@g = external global i32 0` does not
        // parse.
        let linkage = if linkage == Linkage::External && data.initializer.is_some() {
            String::new()
        } else {
            format!("{linkage} ")
        };

        // `default` is the default, and a *local* symbol may have nothing else —
        // `@g = internal hidden global i32 0` is refused with "symbol with local
        // linkage must have default visibility".
        let visiblity = if visiblity == Visiblity::Default {
            String::new()
        } else {
            format!("{visiblity} ")
        };

        // A declaration names a type and stops; every other linkage needs a value.
        let initializer = match &data.initializer {
            Some(expr) => format!(" {}", Self::const_expr(expr, ctx)?),
            None => String::new(),
        };

        self.unset_indentation();

        self.push_line(&format!(
            "@{} = {}{}global {}{}",
            name,
            linkage,
            visiblity,
            ctx.display(data.ty),
            initializer
        ));

        Ok(())
    }

    fn visit_imported_func(
        &mut self,
        func_name: &str,
        func_sig: &FuncSignature,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        // A `declare` names types only — there is no body for a parameter name to
        // refer to. Names are legal but carry nothing, so they are left out.
        let params: Vec<String> = func_sig
            .params
            .iter()
            .map(|ty| ctx.display(*ty).to_string())
            .collect();

        self.unset_indentation();

        self.push_line(&format!(
            "declare {} @{}({})",
            ctx.display(func_sig.result),
            func_name,
            params.join(", ")
        ));

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

    fn visit_ret(
        &mut self,
        operands: &RetOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        // `ret void` carries no operand; every other result is `ret <ty> <val>`, and
        // the type comes from the instruction rather than the value so a `void`
        // return still spells its type.
        match &operands.value {
            Some(value) => {
                self.push_line(&format!("ret {}", Self::typed_operand(value, ctx)?));
            }
            None => {
                self.push_line(&format!("ret {}", ctx.display(operands.ty)));
            }
        }

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
        // The condition is an `I1Value`, so the `i1` is known without asking the pool
        // and only the operand itself has to be rendered.
        let cond = Self::operand_kind(&operands.cond.kind, ctx)?;

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

    fn visit_call(
        &mut self,
        operands: &CallOperands,
        value: Option<&Value>,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        let mut args = vec![];

        for param in &operands.params {
            args.push(Self::typed_operand(param, ctx)?);
        }

        // Only the return type is written, not the whole function type — the long
        // form is legal but only needed for a variadic callee.
        //
        // A `void` call defines no register and takes no `%x = ` prefix: `llvm-as`
        // refuses one with "instructions returning void cannot have a name".
        let assignment = match value {
            Some(value) => Self::assignment(value, ctx)?,
            None => String::new(),
        };

        self.push_line(&format!(
            "{}call {} @{}({})",
            assignment,
            ctx.display(operands.return_ty),
            ctx.str_interner.value(operands.func_name.0),
            args.join(", ")
        ));

        Ok(())
    }

    fn visit_icmp(
        &mut self,
        operands: &ICmpOperands,
        value: &I1Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        // The operand type is written once, between the predicate and the first
        // operand, and the operands themselves are untyped — `icmp eq i32 %a, %b`,
        // not `icmp eq i32 %a, i32 %b`. Both are known to share `operands.ty` by
        // construction, so there is nothing to reconcile here.
        //
        // The result is an `I1Value`, so the register renders through `operand_kind`
        // for the same reason a branch condition does.
        self.push_line(&format!(
            "{} = icmp {} {} {}, {}",
            Self::operand_kind(&value.kind, ctx)?,
            operands.cond,
            ctx.display(operands.ty),
            Self::operand(&operands.a, ctx)?,
            Self::operand(&operands.b, ctx)?
        ));

        Ok(())
    }

    fn visit_iarithmetic(
        &mut self,
        operands: &IArithmeticOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        // `<reg> = <op> <ty> <a>, <b>` — the type once, then two untyped operands, the
        // same shape as a comparison. What differs is the result type: an arithmetic
        // instruction defines a value of the *operand* type, not an `i1`.
        self.push_line(&format!(
            "{}{} {} {}, {}",
            Self::assignment(value, ctx)?,
            operands.op,
            ctx.display(operands.ty),
            Self::operand(&operands.a, ctx)?,
            Self::operand(&operands.b, ctx)?
        ));

        Ok(())
    }

    fn visit_fcmp(
        &mut self,
        operands: &FCmpOperands,
        value: &I1Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        // Same shape as `icmp`: the type once, then two untyped operands.
        self.push_line(&format!(
            "{} = fcmp {} {} {}, {}",
            Self::operand_kind(&value.kind, ctx)?,
            operands.cond,
            ctx.display(operands.ty),
            Self::operand(&operands.a, ctx)?,
            Self::operand(&operands.b, ctx)?
        ));

        Ok(())
    }

    fn visit_farithmetic(
        &mut self,
        operands: &FArithmeticOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        self.push_line(&format!(
            "{}{} {} {}, {}",
            Self::assignment(value, ctx)?,
            operands.op,
            ctx.display(operands.ty),
            Self::operand(&operands.a, ctx)?,
            Self::operand(&operands.b, ctx)?
        ));

        Ok(())
    }

    fn visit_fneg(
        &mut self,
        operands: &FNegOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType> {
        // One operand and no comma: `llvm-as` reads anything after one as metadata.
        self.push_line(&format!(
            "{}fneg {} {}",
            Self::assignment(value, ctx)?,
            ctx.display(operands.ty),
            Self::operand(&operands.value, ctx)?
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
        cfg::{
            builder::Builder,
            module::{DataLayout, DataLayoutSpec, Endianness, Mangling, Triple},
        },
        error::ContextError,
        instruction::{FCond, GetElementPtrOperands, ICond},
        interner::TyId,
        test_support::fixture,
        value::{ConstExpr, FuncSignature, NullPtr, Type},
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
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f32_ty = ctx.f32_ty();
        let f64_ty = ctx.f64_ty();
        let ptr_ty = ctx.ptr_ty();
        let void_ty = ctx.void_ty();

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

        // The callees come first: a call resolves against the functions added so far,
        // so a forward reference would not resolve.
        let helper = builder
            .define_function(
                "helper".to_string(),
                &[(i32_ty, Some("v".to_string()))],
                i32_ty,
                &mut ctx,
            )
            .unwrap();

        let helper_entry = helper
            .add_basic_block("entry".to_string(), &mut ctx)
            .unwrap();

        let passed = helper
            .nth_param(0, &ctx)
            .expect("helper takes one parameter")
            .clone();

        builder
            .cursor_at_block(helper_entry)
            .build_ret(Some(passed), Some(i32_ty), &mut ctx)
            .unwrap();

        let noop = builder
            .define_function("noop".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let noop_entry = noop.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        builder
            .cursor_at_block(noop_entry)
            .build_ret(None, Some(void_ty), &mut ctx)
            .unwrap();

        let f = builder
            .define_function(
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
            .build_alloca(struct_ty, None, Some(8), Some("s"), &mut ctx)
            .unwrap();

        let count = Value::from_const(4i32, None, &mut ctx).unwrap();

        in_entry
            .build_alloca(i64_ty, Some((count, None)), None, None, &mut ctx)
            .unwrap();

        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();
        let one = Value::from_const(1i32, None, &mut ctx).unwrap();
        let two = Value::from_const(2i32, None, &mut ctx).unwrap();

        let elem = in_entry
            .build_get_element_ptr(
                slot,
                None,
                vec![zero, one, two],
                Some(true),
                Some("e"),
                &mut ctx,
            )
            .unwrap();

        let loaded = in_entry
            .build_load(elem.clone(), Some(f64_ty), Some(8), Some("d"), &mut ctx)
            .unwrap();

        in_entry
            .build_store(elem, loaded.clone(), None, Some(8), &mut ctx)
            .unwrap();

        // `0.1f32` is the case that forces the hex encoding: `float 0.1` is refused by
        // `llvm-as` with "floating point constant invalid for type".
        let a_float = Value::from_const(0.1f32, None, &mut ctx).unwrap();

        let float_slot = in_entry
            .build_alloca(f32_ty, None, None, Some("fs"), &mut ctx)
            .unwrap();

        in_entry
            .build_store(float_slot, a_float, None, None, &mut ctx)
            .unwrap();

        let null = Value::from_const(NullPtr, None, &mut ctx).unwrap();

        let ptr_slot = in_entry
            .build_alloca(ptr_ty, None, None, Some("np"), &mut ctx)
            .unwrap();

        in_entry
            .build_store(ptr_slot, null, None, None, &mut ctx)
            .unwrap();
        in_entry.build_unconditional_br(body, &mut ctx).unwrap();

        let in_body = builder.cursor_at_block(body);

        let (phi_handler, phi) = in_body
            .build_phi(&[(entry, loaded)], Some("m"), &mut ctx)
            .unwrap();

        // `body` reaches itself, so that edge needs its own incoming value — LLVM
        // requires one phi entry per predecessor.
        phi_handler
            .add_branch((body, phi.clone()), &mut ctx)
            .unwrap();

        // The branch condition comes from a real comparison rather than a literal, so
        // the `icmp` line and the `i1` it feeds are both covered here.
        let limit = Value::from_const(10i32, None, &mut ctx).unwrap();
        let counter = f
            .nth_param(0, &ctx)
            .expect("main takes an i32 first parameter")
            .clone();

        let cond = in_body
            .build_icmp(ICond::Ult, None, counter.clone(), limit, Some("cmp"), &mut ctx)
            .unwrap();

        // An `fcmp` alongside it, so the float comparison is emitted and assembled
        // too. `ord` is the predicate with no integer analogue: it asks only whether
        // neither operand is a NaN.
        let half = Value::from_const(0.5f64, None, &mut ctx).unwrap();

        in_body
            .build_fcmp(FCond::Ord, None, phi.clone(), half.clone(), Some("fcmp"), &mut ctx)
            .unwrap();

        // One of each arithmetic shape, so all three emitters are assembled: an
        // integer op, a float op, and the unary `fneg`.
        let step = Value::from_const(1i32, None, &mut ctx).unwrap();

        in_body
            .build_iarithmetic(
                IArithmeticOp::Add,
                None,
                counter,
                step,
                Some("next"),
                &mut ctx,
            )
            .unwrap();

        in_body
            .build_farithmetic(
                FArithmeticOp::FMul,
                None,
                phi.clone(),
                half,
                Some("scaled"),
                &mut ctx,
            )
            .unwrap();

        in_body.build_fneg(phi, Some("neg"), &mut ctx).unwrap();

        in_body
            .build_conditional_br(cond, body, exit, &mut ctx)
            .unwrap();

        // Both call shapes: one returning a value, one `void`. `helper` was added
        // before `main`, since the callee has to already exist.
        let in_exit = builder.cursor_at_block(exit);
        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        let answer = in_exit
            .build_call(
                "helper".to_string(),
                &[(seven, None)],
                Some(i32_ty),
                Some("c"),
                &mut ctx,
            )
            .expect("helper takes an i32 and returns one")
            .expect("a non-void call defines a register");

        assert!(
            in_exit
                .build_call("noop".to_string(), &[], None, None, &mut ctx)
                .unwrap()
                .is_none(),
            "a void call defines nothing"
        );

        in_exit
            .build_ret(Some(answer), Some(i32_ty), &mut ctx)
            .unwrap();

        let ir = IREmitter::emit(builder.build(ctx)).unwrap();

        let expected = concat!(
            "target triple = \"arm64-apple-macosx\"\n",
            "\n",
            "define i32 @helper(i32 %v) {\n",
            "entry:\n",
            "    ret i32 %v\n",
            "}\n",
            "\n",
            "define void @noop() {\n",
            "entry:\n",
            "    ret void\n",
            "}\n",
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
            "    %cmp = icmp ult i32 %n, 10\n",
            "    %fcmp = fcmp ord double %m, 0x3FE0000000000000\n",
            "    %next = add i32 %n, 1\n",
            "    %scaled = fmul double %m, 0x3FE0000000000000\n",
            "    %neg = fneg double %m\n",
            "    br i1 %cmp, label %body, label %exit\n",
            "exit:\n",
            "    %c = call i32 @helper(i32 7)\n",
            "    call void @noop()\n",
            "    ret i32 %c\n",
            "}\n",
            "\n",
        );

        assert_eq!(ir, expected, "\n--- emitted ---\n{ir}");
    }

    /// A declared function is emitted as a `declare` line, and calling it produces
    /// IR that assembles.
    ///
    /// Both halves matter. Without the `declare`, `llvm-as` refuses the call with
    /// "use of undefined value" — the signature being recorded is enough for
    /// `build_call` to resolve, but not for the module to be valid.
    #[test]
    fn a_declared_function_is_emitted_and_callable() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();
        let void_ty = ctx.void_ty();

        builder
            .declare_function("host_add".to_string(), &[i32_ty, f64_ty], i32_ty, &mut ctx)
            .unwrap();

        builder
            .declare_function("host_noop".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let f = builder
            .define_function("f".to_string(), &[], i32_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();
        let half = Value::from_const(0.5f64, None, &mut ctx).unwrap();

        let result = cursor
            .build_call(
                "host_add".to_string(),
                &[(seven, None), (half, None)],
                None,
                Some("r"),
                &mut ctx,
            )
            .expect("a declared function is callable")
            .expect("it returns an i32");

        cursor
            .build_call("host_noop".to_string(), &[], None, None, &mut ctx)
            .unwrap();

        cursor
            .build_ret(Some(result), Some(i32_ty), &mut ctx)
            .unwrap();

        let ir = IREmitter::emit(builder.build(ctx)).unwrap();

        assert_eq!(
            ir,
            concat!(
                "target triple = \"arm64-apple-macosx\"\n",
                "\n",
                // Declarations come first, and carry types only — no parameter names,
                // since there is no body for one to refer to.
                "declare i32 @host_add(i32, double)\n",
                "declare void @host_noop()\n",
                "define i32 @f() {\n",
                "entry:\n",
                "    %r = call i32 @host_add(i32 7, double 0x3FE0000000000000)\n",
                "    call void @host_noop()\n",
                "    ret i32 %r\n",
                "}\n",
                "\n",
            ),
            "\n--- emitted ---\n{ir}"
        );
    }

    /// A declared global emits `@g = external global <ty>` — no initializer, since
    /// `external` names something defined elsewhere. `llvm-as` refuses one:
    /// `@g = external global i32 0` does not parse.
    #[test]
    fn a_global_variable_without_an_initializer_is_a_declaration() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();

        builder
            .declare_global_variable("counter".to_string(), Some(i32_ty), None, &mut ctx)
            .expect("an i32 is a legal global type");

        let ir = IREmitter::emit(builder.build(ctx)).unwrap();

        assert_eq!(
            ir,
            concat!(
                "target triple = \"arm64-apple-macosx\"\n",
                "\n",
                "@counter = external global i32\n",
            ),
            "\n--- emitted ---\n{ir}"
        );
    }

    /// With an initializer the `external` keyword is *omitted*, because that keyword
    /// is what makes a global a declaration — and a declaration may not be
    /// initialised. Writing both is refused by `llvm-as`.
    #[test]
    fn an_initialised_global_variable_omits_the_external_keyword() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let ptr_ty = ctx.ptr_ty();

        // A constant gep is the one initializer `ConstExpr` can express today.
        let null = Value::from_const(NullPtr, None, &mut ctx).unwrap();
        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();

        let init = ConstExpr::GetElementPtr(Box::new(GetElementPtrOperands {
            source_ty: i32_ty,
            ptr: null,
            indices: Box::new([zero]),
            inbounds: false,
        }));

        builder
            .declare_global_variable("p".to_string(), Some(ptr_ty), Some(init), &mut ctx)
            .unwrap();

        let ir = IREmitter::emit(builder.build(ctx)).unwrap();

        assert_eq!(
            ir,
            concat!(
                "target triple = \"arm64-apple-macosx\"\n",
                "\n",
                "@p = global ptr getelementptr (i32, ptr null, i32 0)\n",
            ),
            "\n--- emitted ---\n{ir}"
        );
    }

    /// A global is an address, so its value is a `ptr` and it is written by name.
    /// That is what lets one be stored, loaded through, or passed as an argument.
    #[test]
    fn a_global_is_an_operand_referred_to_by_name() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let ptr_ty = ctx.ptr_ty();
        let void_ty = ctx.void_ty();

        let counter = builder
            .declare_global_variable("counter".to_string(), Some(i32_ty), None, &mut ctx)
            .unwrap();

        let f = builder
            .define_function("f".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let address = Value::from_global(counter, &mut ctx);

        assert_eq!(address.ty(), ptr_ty, "a global's value is its address");

        // Its pointee is recoverable, so the load needs no explicit type.
        let loaded = cursor
            .build_load(address.clone(), None, None, Some("v"), &mut ctx)
            .expect("the global says what it points at");

        assert_eq!(loaded.ty(), i32_ty, "inferred from the global's type");

        cursor
            .build_store(address, loaded, None, None, &mut ctx)
            .unwrap();

        cursor.build_ret(None, Some(void_ty), &mut ctx).unwrap();

        let ir = IREmitter::emit(builder.build(ctx)).unwrap();

        assert!(
            ir.contains("    %v = load i32, ptr @counter\n"),
            "the global is named in the load, got:\n{ir}"
        );

        assert!(
            ir.contains("    store i32 %v, ptr @counter\n"),
            "and in the store, got:\n{ir}"
        );
    }

    /// A global holds a value, so its type needs a size. `llvm-as` refuses
    /// `@g = external global void` with "void type only allowed for function results"
    /// and a function-typed one with "invalid type for global variable" — the same
    /// pair excluded everywhere else. Aggregates and pointers are fine.
    #[test]
    fn a_global_variable_needs_a_sized_type() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();
        let void_ty = ctx.void_ty();

        let err = builder
            .declare_global_variable("a".to_string(), Some(void_ty), None, &mut ctx)
            .expect_err("`void` has no size");

        assert!(
            matches!(&err, ContextError::GlobalVariableTypeNotSized(t) if t == "void"),
            "the error must name the offending type, got: {err}"
        );

        let signature = FuncSignature::new(vec![i32_ty], i32_ty);
        let func_ty: TyId = ctx.ty_interner.intern(Type::Func(signature)).into();

        assert!(
            matches!(
                builder.declare_global_variable("b".to_string(), Some(func_ty), None, &mut ctx),
                Err(ContextError::GlobalVariableTypeNotSized(_))
            ),
            "a function type is not a global's type either"
        );

        // An aggregate is sized, so it is accepted.
        let struct_ty: TyId = ctx
            .ty_interner
            .intern(Type::Struct {
                fields: Box::new([i32_ty, f64_ty]),
                packed: false,
            })
            .into();

        assert!(
            builder
                .declare_global_variable("c".to_string(), Some(struct_ty), None, &mut ctx)
                .is_ok(),
            "an aggregate has a size"
        );

        assert_eq!(
            ctx.module.global_variables.len(),
            1,
            "only the accepted global was recorded"
        );
    }

    /// A plain constant initializer, now that `ConstExpr` can carry one — and with
    /// the type omitted, since the initializer already determines it.
    #[test]
    fn a_global_variable_infers_its_type_from_a_constant_initializer() {
        let (mut ctx, builder) = fixture();

        let seven = Value::from_const(7i32, None, &mut ctx).unwrap();

        let ValueKind::ConstExpr(init) = seven.kind() else {
            panic!("a constant is a constant expression")
        };

        builder
            .declare_global_variable("count".to_string(), None, Some(init.clone()), &mut ctx)
            .expect("the initializer says what the type is");

        let ir = IREmitter::emit(builder.build(ctx)).unwrap();

        assert_eq!(
            ir,
            concat!(
                "target triple = \"arm64-apple-macosx\"\n",
                "\n",
                "@count = global i32 7\n",
            ),
            "\n--- emitted ---\n{ir}"
        );
    }

    /// Given both, they have to agree — and **exactly**. `llvm-as` refuses
    /// `@g = global i32 true` with "constant expression type mismatch" even though an
    /// `i1` is an integer, and `@g = global double 0` with "integer constant must
    /// have integer type".
    #[test]
    fn a_global_initializer_must_match_the_declared_type_exactly() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();

        let const_of = |v: &Value| {
            let ValueKind::ConstExpr(expr) = v.kind() else {
                panic!("a constant is a constant expression")
            };

            expr.clone()
        };

        let a_bool = const_of(&Value::from_const(true, None, &mut ctx).unwrap());

        let err = builder
            .declare_global_variable("a".to_string(), Some(i32_ty), Some(a_bool), &mut ctx)
            .expect_err("an i1 does not initialise an i32");

        assert!(
            matches!(
                &err,
                ContextError::GlobalInitializerTypeMismatch(declared, given)
                    if declared == "i32" && given == "i1"
            ),
            "the error must name both types, got: {err}"
        );

        // An integer does not initialise a float either, in either direction.
        let an_int = const_of(&Value::from_const(0i32, None, &mut ctx).unwrap());

        assert!(
            matches!(
                builder.declare_global_variable(
                    "b".to_string(),
                    Some(f64_ty),
                    Some(an_int),
                    &mut ctx
                ),
                Err(ContextError::GlobalInitializerTypeMismatch(..))
            ),
            "an i32 does not initialise a double"
        );

        // The matching case still goes through.
        let an_i32 = const_of(&Value::from_const(1i32, None, &mut ctx).unwrap());

        assert!(
            builder
                .declare_global_variable("c".to_string(), Some(i32_ty), Some(an_i32), &mut ctx)
                .is_ok()
        );

        assert_eq!(
            ctx.module.global_variables.len(),
            1,
            "only the accepted global was recorded"
        );
    }

    /// Neither a type nor an initializer leaves nothing to declare and nothing to
    /// infer from — the same shape as a `ret` with neither.
    #[test]
    fn a_global_variable_needs_a_type_or_an_initializer() {
        let (mut ctx, builder) = fixture();

        let err = builder
            .declare_global_variable("g".to_string(), None, None, &mut ctx)
            .expect_err("nothing to go on");

        assert!(
            matches!(&err, ContextError::GlobalTypeAndInitializerBothAbsent),
            "got: {err}"
        );

        assert!(
            ctx.module.global_variables.is_empty(),
            "the refused global was not recorded"
        );
    }

    /// Globals and functions share one namespace, so a name is used once.
    #[test]
    fn a_global_variable_cannot_reuse_a_name() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        builder
            .declare_global_variable("g".to_string(), Some(i32_ty), None, &mut ctx)
            .unwrap();

        assert!(
            builder
                .declare_global_variable("g".to_string(), Some(i32_ty), None, &mut ctx)
                .is_err(),
            "a second global of that name collides"
        );

        assert!(
            builder
                .define_function("g".to_string(), &[], void_ty, &mut ctx)
                .is_err(),
            "and so does a function, since the namespace is shared"
        );
    }

    /// Every linkage and visibility spells itself the way `llvm-as` accepts.
    #[test]
    fn linkage_and_visibility_render_as_llvm_spells_them() {
        for (linkage, expected) in [
            (Linkage::External, "external"),
            (Linkage::Internal, "internal"),
            (Linkage::Private, "private"),
            (Linkage::Weak, "weak"),
            (Linkage::Linkonce, "linkonce"),
            (Linkage::LinkonceOdr, "linkonce_odr"),
            (Linkage::WeakOdr, "weak_odr"),
            (Linkage::Common, "common"),
            (Linkage::Appending, "appending"),
            (Linkage::AvailableExternally, "available_externally"),
            (Linkage::ExternWeak, "extern_weak"),
        ] {
            assert_eq!(linkage.to_string(), expected);
        }

        for (visiblity, expected) in [
            (Visiblity::Default, "default"),
            (Visiblity::Hidden, "hidden"),
            (Visiblity::Protected, "protected"),
        ] {
            assert_eq!(visiblity.to_string(), expected);
        }
    }

    /// A constant `getelementptr` is written inline, in the parenthesised form, and
    /// used wherever a constant may be — here as the address a `store` writes to.
    ///
    /// The instruction form and the constant form differ: the instruction takes a
    /// `%x = ` and no parentheses, the constant expression takes parentheses and no
    /// assignment.
    #[test]
    fn a_constant_getelementptr_is_emitted_inline() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();
        let ptr_ty = ctx.ptr_ty();

        let array_ty: TyId = ctx
            .ty_interner
            .intern(Type::Array {
                size: 4,
                element_ty: i32_ty,
            })
            .into();

        let f = builder
            .define_function("f".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        let null = Value::from_const(NullPtr, None, &mut ctx).unwrap();
        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();
        let two = Value::from_const(2i32, None, &mut ctx).unwrap();

        // `getelementptr inbounds ([4 x i32], ptr null, i32 0, i32 2)`
        let const_gep = Value::from_const_expr(
            ConstExpr::GetElementPtr(Box::new(GetElementPtrOperands {
                source_ty: array_ty,
                ptr: null,
                indices: Box::new([zero, two]),
                inbounds: true,
            })),
            &mut ctx,
        );

        assert_eq!(
            const_gep.ty(),
            ptr_ty,
            "a constant gep is a pointer, like the instruction"
        );

        let slot = cursor
            .build_alloca(ptr_ty, None, None, Some("s"), &mut ctx)
            .unwrap();

        cursor
            .build_store(slot, const_gep, None, None, &mut ctx)
            .expect("a constant expression is a valid store value");

        cursor.build_ret(None, Some(void_ty), &mut ctx).unwrap();

        let ir = IREmitter::emit(builder.build(ctx)).unwrap();

        assert!(
            ir.contains(
                "store ptr getelementptr inbounds ([4 x i32], ptr null, i32 0, i32 2), ptr %s"
            ),
            "the constant gep is written inline, got:\n{ir}"
        );
    }

    /// Without `inbounds` the keyword is simply absent.
    #[test]
    fn a_constant_getelementptr_omits_inbounds_when_unset() {
        let mut ctx = crate::test_support::ctx();

        let i32_ty = ctx.i32_ty();

        let array_ty: TyId = ctx
            .ty_interner
            .intern(Type::Array {
                size: 4,
                element_ty: i32_ty,
            })
            .into();

        let null = Value::from_const(NullPtr, None, &mut ctx).unwrap();
        let zero = Value::from_const(0i32, None, &mut ctx).unwrap();

        let expr = ConstExpr::GetElementPtr(Box::new(GetElementPtrOperands {
            source_ty: array_ty,
            ptr: null,
            indices: Box::new([zero]),
            inbounds: false,
        }));

        assert_eq!(
            IREmitter::const_expr(&expr, &ctx).unwrap(),
            "getelementptr ([4 x i32], ptr null, i32 0)"
        );
    }

    /// The four operand-less variants cannot be rendered, so the emitter refuses
    /// rather than writing something `llvm-as` could not parse.
    #[test]
    fn an_operandless_constant_expression_is_refused() {
        let ctx = crate::test_support::ctx();

        let err = IREmitter::const_expr(&ConstExpr::PtrToInt {}, &ctx)
            .expect_err("`ptrtoint` carries no operand");

        assert!(
            err.to_string().contains("no operands"),
            "the error must say why, got: {err}"
        );
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

    /// An unset data layout is omitted rather than written as an empty string —
    /// `target datalayout = ""` is not what "unset" means.
    ///
    /// Note that a [`Triple`](crate::cfg::module::Triple) is always present now that
    /// it is structured, so a module with no functions still emits its `target
    /// triple` line and the blank separator after it.
    #[test]
    fn an_unset_data_layout_emits_no_datalayout_line() {
        let ctx = crate::test_support::ctx();
        let ir = IREmitter::emit(Builder.build(ctx)).unwrap();

        assert_eq!(
            ir, "target triple = \"arm64-apple-macosx\"\n\n",
            "the triple is written, the absent layout is not"
        );

        assert!(
            !ir.contains("datalayout"),
            "an unset layout emits no line at all, not an empty one"
        );
    }

    /// And a layout that *is* set gets its own line, ahead of the triple.
    #[test]
    fn a_set_data_layout_is_emitted_before_the_triple() {
        let ctx = Context::new(
            Triple::new(
                "arm64".to_string(),
                "apple".to_string(),
                "macosx".to_string(),
                None,
            ),
            DataLayout::new(vec![
                DataLayoutSpec::Endianness(Endianness::Little),
                DataLayoutSpec::Mangling(Mangling::MachO),
                DataLayoutSpec::StackAlignment(128),
            ]),
        );

        let ir = IREmitter::emit(Builder.build(ctx)).unwrap();

        assert_eq!(
            ir,
            concat!(
                "target datalayout = \"e-m:o-S128\"\n",
                "target triple = \"arm64-apple-macosx\"\n",
                "\n",
            )
        );
    }
}
