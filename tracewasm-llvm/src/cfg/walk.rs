//! Traversing a finished graph.

use crate::{
    cfg::{
        ControlFlowGraph,
        basic_block::{BasicBlock, BasicBlockId},
        context::Context,
        function::{FuncId, Function},
        global::{GlobalKind, GlobalVariable, Linkage, Visibility},
    },
    instruction::{
        AllocaOperands, CallOperands, CastOperands, ConditionalBrOperands, FBinOpOperands,
        FCmpOperands, FNegOperands, GetElementPtrOperands, IBinOpOperands, ICmpOperands,
        InstructionKind, LoadOperands, PhiInstruction, RetOperands, StoreOperands,
        UnconditionalBrOperands,
    },
    value::{FuncSignature, I1Value, Value},
};

/// Walks a [`ControlFlowGraph`], visiting each construct in emission order.
///
/// Implement the `visit_*` methods; the `walk_*` methods drive the traversal and are
/// provided. [`IREmitter`](crate::cfg::emit::IREmitter) is the implementation that
/// renders text, but the trait is equally usable for analysis or validation.
///
/// # Order
///
/// A construct is visited **before** its children, and the matching `post_*_visit`
/// hook runs after them. That pairing is what lets an emitter write `define …{` in
/// `visit_func` and `}` in [`post_func_visit`](Self::post_func_visit) without
/// tracking depth itself.
///
/// ```text
/// visit_cfg
///   visit_imported_func     for each declaration
///   visit_func              for each defined function
///     visit_basic_block     for each block
///       visit_phi           for each phi, then
///       visit_<instr>       for each instruction
///     post_block_visit
///   post_func_visit
/// post_module_visit
/// ```
///
/// The `post_*` hooks default to returning `OkType::default()`, so an implementation
/// only overrides the ones it needs.
pub trait CfgVisitor {
    /// What each visit returns. Collected and handed to the `post_*` hooks, so a
    /// visitor that accumulates results can use them; an emitter that writes as it
    /// goes uses `()`.
    type OkType: Default;

    /// How a visit fails. The walk stops at the first error.
    type ErrType;

    /// Visits a phi node.
    fn visit_phi(
        &mut self,
        instr: &PhiInstruction,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a `ret`. A terminator: it ends its block.
    fn visit_ret(
        &mut self,
        operands: &RetOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a `br label`. A terminator.
    fn visit_unconditional_br(
        &mut self,
        operands: &UnconditionalBrOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a `br i1`. A terminator.
    fn visit_conditional_br(
        &mut self,
        operands: &ConditionalBrOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a `load`. `value` is the register it defines.
    fn visit_load(
        &mut self,
        operands: &LoadOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a `store`, which defines no register.
    fn visit_store(
        &mut self,
        operands: &StoreOperands,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits an `alloca`. `value` is the pointer it defines.
    fn visit_alloca(
        &mut self,
        operands: &AllocaOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a `getelementptr`. `value` is the pointer it defines.
    fn visit_get_element_ptr(
        &mut self,
        operands: &GetElementPtrOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a `call`.
    ///
    /// `value` is the register it defines, and is `None` for a `void` callee — the
    /// only visit here that takes an `Option`, since every other value-producing
    /// instruction always defines one.
    fn visit_call(
        &mut self,
        operands: &CallOperands,
        value: Option<&Value>,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits an `icmp`.
    ///
    /// The result is an [`I1Value`] rather than a `Value`, and never `None`: an
    /// `icmp` always defines a register, and that register is always an `i1`.
    fn visit_icmp(
        &mut self,
        operands: &ICmpOperands,
        value: &I1Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits an integer binary operation — arithmetic, bitwise or a shift.
    ///
    /// The result has the operands' type, not `i1`.
    fn visit_ibinop(
        &mut self,
        operands: &IBinOpOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits an `fcmp`.
    ///
    /// Like [`visit_icmp`](Self::visit_icmp), the result is an [`I1Value`] and never
    /// `None`: an `fcmp` always defines a register, and it is always an `i1`.
    fn visit_fcmp(
        &mut self,
        operands: &FCmpOperands,
        value: &I1Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a floating-point binary operation.
    fn visit_fbinop(
        &mut self,
        operands: &FBinOpOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits an `fneg`.
    ///
    /// One operand — which is why it is not an [`FBinOp`](crate::instruction::FBinOp).
    fn visit_fneg(
        &mut self,
        operands: &FNegOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a conversion. The result has the destination type.
    fn visit_cast(
        &mut self,
        operands: &CastOperands,
        value: &Value,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a block, before its phis and instructions.
    fn visit_basic_block(
        &mut self,
        block: &BasicBlock,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a global variable.
    ///
    /// `linkage` and `visibility` come alongside the [`GlobalVariable`] rather than
    /// being read from it, because they live on the enclosing `GlobalData` — and they
    /// are part of how a global is written, not decoration: emitting without them
    /// would turn an `internal` global into an externally visible one.
    ///
    /// Globals are visited before declarations and definitions, matching the order a
    /// module reads.
    fn visit_global_variable(
        &mut self,
        name: &str,
        data: &GlobalVariable,
        linkage: Linkage,
        visibility: Visibility,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a declared-but-not-defined function.
    ///
    /// Declarations come before definitions in the walk, matching how a module reads:
    /// what it links against, then what it provides. There is no [`FuncId`] and no
    /// body — only a name and a signature.
    fn visit_imported_func(
        &mut self,
        func_name: &str,
        func_sig: &FuncSignature,
        ctx: &Context,
    ) -> Result<Self::OkType, Self::ErrType>;

    /// Visits a function, before its blocks.
    fn visit_func(&mut self, func: &Function, ctx: &Context)
    -> Result<Self::OkType, Self::ErrType>;

    /// Visits the module, before its functions.
    fn visit_cfg(&mut self, module: &ControlFlowGraph) -> Result<Self::OkType, Self::ErrType>;

    /// Walks one block: the block itself, then its phis, then its instructions.
    ///
    /// Provided; override only to change the traversal, not to observe it.
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
                InstructionKind::Ret(operands) => self.visit_ret(operands, ctx)?,
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
                InstructionKind::Call(operands) => self.visit_call(operands, val, ctx)?,
                InstructionKind::ICmp(operands) => {
                    self.visit_icmp(operands, &val.unwrap().clone().into_i1(ctx).unwrap(), ctx)?
                }
                InstructionKind::IBinOp(operands) => {
                    self.visit_ibinop(operands, val.unwrap(), ctx)?
                }
                InstructionKind::FCmp(operands) => {
                    self.visit_fcmp(operands, &val.unwrap().clone().into_i1(ctx).unwrap(), ctx)?
                }
                InstructionKind::FBinOp(operands) => {
                    self.visit_fbinop(operands, val.unwrap(), ctx)?
                }
                InstructionKind::FNeg(operands) => self.visit_fneg(operands, val.unwrap(), ctx)?,
                InstructionKind::Cast(operands) => self.visit_cast(operands, val.unwrap(), ctx)?,
            });
        }

        self.post_block_visit(block.func_id, id, phi_results, instr_results)
    }

    /// Walks one function: the function itself, then its blocks in creation order.
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

    /// Walks the whole module. This is the entry point.
    fn walk_cfg(&mut self, cfg: &ControlFlowGraph) -> Result<Self::OkType, Self::ErrType> {
        let funcs = &cfg.context.module.functions;
        let mut func_results = vec![];
        let mut imported_func_results = vec![];
        let mut global_variable_results = vec![];

        let cfg_result = self.visit_cfg(cfg)?;

        for variable in &cfg.context.module.global_variables {
            let name = cfg.context.str_interner.value(variable.0);

            let data = cfg
                .context
                .module
                .globals
                .get(variable)
                .expect("hitting this means globals tracking logic by their name is incorrect");

            let linkage = data.linkage;
            let visibility = data.visibility;

            let GlobalKind::Variable(var) = &data.kind else {
                unreachable!("hitting this means globals tracking logic by their name is incorrect")
            };

            global_variable_results.push(self.visit_global_variable(
                name,
                var,
                linkage,
                visibility,
                &cfg.context,
            )?);
        }

        for &imported_func in &cfg.context.module.imported_functions {
            let func_name = cfg.context.str_interner.value(imported_func.0);

            let GlobalKind::Func(func_sig) =
                &cfg.context.module.globals.get(&imported_func).unwrap().kind
            else {
                unreachable!("hitting this means globals tracking logic by their name is incorrect")
            };

            imported_func_results.push(self.visit_imported_func(
                func_name,
                func_sig,
                &cfg.context,
            )?);
        }

        for func_id in funcs {
            let func = cfg.context.get_func(*func_id);

            func_results.push(self.walk_func(*func_id, func, &cfg.context)?);
        }

        self.post_module_visit(
            cfg_result,
            global_variable_results,
            imported_func_results,
            func_results,
        )
    }

    /// Runs after a block's phis and instructions, with what each visit returned.
    fn post_block_visit(
        &mut self,
        _func: FuncId,
        _block: BasicBlockId,
        _phi_results: Vec<Self::OkType>,
        _instr_results: Vec<Self::OkType>,
    ) -> Result<Self::OkType, Self::ErrType> {
        Ok(Self::OkType::default())
    }

    /// Runs after a function's blocks — where an emitter writes the closing `}`.
    fn post_func_visit(
        &mut self,
        _func: FuncId,
        _block_results: Vec<Self::OkType>,
    ) -> Result<Self::OkType, Self::ErrType> {
        Ok(Self::OkType::default())
    }

    /// Runs after every declaration and every function, with what each returned.
    /// Its return value is what [`walk_cfg`](Self::walk_cfg) yields.
    fn post_module_visit(
        &mut self,
        _cfg_visit_result: Self::OkType,
        _global_variable_results: Vec<Self::OkType>,
        _imported_func_results: Vec<Self::OkType>,
        _func_results: Vec<Self::OkType>,
    ) -> Result<Self::OkType, Self::ErrType> {
        Ok(Self::OkType::default())
    }
}
