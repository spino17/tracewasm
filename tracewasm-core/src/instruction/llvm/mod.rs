//! Lowering a lowered wasm body into an LLVM control-flow graph.
//!
//! This pass runs *after* [`stack`](crate::instruction::stack), over the
//! [`StackInstruction`](crate::instruction::stack::StackInstruction) stream rather
//! than over the operator stream. That is the point: by the time an instruction list
//! exists, every structured branch has already been resolved to an absolute index —
//! an `if` knows where its `else` and `end` are, a `br` knows the instruction it
//! targets. A CFG builder otherwise has to discover exactly those forward references
//! itself, so taking them from the interpreter's lowering means the two machines
//! cannot disagree about control flow.
//!
//! Dead code never arrives here either. The lowering drops unreachable operators
//! rather than marking them, so every instruction this pass sees can execute, and no
//! arm has to recognise a region it must not enter.
//!
//! # The three pieces of state
//!
//! | | Holds |
//! |---|---|
//! | [`SimulatedStack`] | the wasm operand stack, as LLVM values rather than numbers |
//! | [`InstrIndexToBasicBlockMap`] | each open label's `end` block and the phis waiting there |
//! | [`ControlStack`] | the open labels, so a `br` can tell which one it is leaving |
//!
//! The simulated stack is what makes the translation direct: wasm says "add the top
//! two operands", so the pass pops two [`ValueId`]s, emits an `add`, and pushes the
//! result. Where the interpreter would move a number, this moves a name.
//!
//! # Phis are built empty and filled as branches arrive
//!
//! When a label opens, its `end` block is created immediately and given one empty phi
//! per result. Every path that later reaches that label — the fall-through at `end`,
//! the then-arm at `else`, any `br` inside — adds its values to those phis through the
//! handles kept in [`PhiValBranches`].
//!
//! Building them up front rather than collecting values and emitting phis at the
//! `end` is what lets a `br` from arbitrary depth work without a second backpatching
//! pass: the phi it must feed already exists.
//!
//! **Branch values are in the label's own order — deepest first, top last** — which is
//! the order the result types are declared in and therefore the order the phis were
//! created in. A caller handing them over top-first would type-check only while every
//! result shares a type, and silently pair the wrong values when they do not.
//!
//! # Locals are memory, not registers
//!
//! Every wasm local becomes an `alloca` in the entry block, written on entry and
//! read by `local.get`. No attempt is made to build SSA here — LLVM's `mem2reg`
//! does that job better, and giving it the obvious shape is cheaper than getting
//! phi placement right twice.

use crate::{
    VirtualMachine,
    instruction::Instruction,
    module::{FuncIndex, Module, ValType},
    runtime::stack::Stack,
};
use rustc_hash::FxHashMap;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
    vec,
};
use tracewasm_llvm::{
    cfg::{
        ControlFlowGraph,
        basic_block::BasicBlockId,
        context::Context,
        global::{DeclaredFunc, DefinedFunc, GlobalId},
        module::{DataLayout, DataLayoutSpec, Endianness, Mangling, Triple},
    },
    error::PhiError,
    instruction::{
        PhiInstrHandler,
        cursor::{OperandTy, RegName},
    },
    interner::TyId,
    value::{Value, ValueId},
};

/// One open label's `end`: the block control lands in, and the phis waiting there.
///
/// The three vectors are parallel and in the label's own result order, deepest first.
/// `phi_handlers[i]` and `phi_vals[i]` are two ends of the same phi — the handle a
/// branch adds itself to, and the register everything downstream reads.
pub(crate) struct PhiValBranches {
    /// Where control lands. Created when the label opens, because a `br` inside it
    /// needs somewhere to jump long before the `end` is reached.
    pub(crate) basic_block: BasicBlockId,
    /// The label's result types, which fix how many phis there are and what each one
    /// accepts.
    pub(crate) phi_val_types: Vec<TyId>,
    /// The registers the phis define — what the label leaves on the operand stack.
    pub(crate) phi_vals: Vec<ValueId>,
    /// The handles branches add themselves to. Separate from `phi_vals` because a phi
    /// is written in two stages: created with the block, completed as paths arrive.
    pub(crate) phi_handlers: Vec<PhiInstrHandler>,
}

pub(crate) trait BranchTarget {
    fn name() -> &'static str;
}

pub(crate) struct End;

impl BranchTarget for End {
    fn name() -> &'static str {
        "end"
    }
}

pub(crate) struct Loop;

impl BranchTarget for Loop {
    fn name() -> &'static str {
        "loop"
    }
}

pub(crate) struct BranchTargetBasicBlockMap<T>(FxHashMap<u32, PhiValBranches>, PhantomData<T>);

impl<T> Default for BranchTargetBasicBlockMap<T> {
    fn default() -> Self {
        BranchTargetBasicBlockMap(FxHashMap::default(), PhantomData)
    }
}

impl<T: BranchTarget> BranchTargetBasicBlockMap<T> {
    /// Opens a label: records the block branches land in, and builds one empty phi
    /// per value it carries.
    ///
    /// Called when the label *opens*, not when it closes — a branch inside it needs
    /// the block and its phis to already exist. For an `end` the values are the
    /// label's results; for a `loop` header they are its params.
    ///
    /// # Errors
    ///
    /// Whatever [`build_phi`](tracewasm_llvm::instruction::cursor::Cursor::build_phi)
    /// reports; an empty phi is legal here only because the type is stated outright.
    pub fn open(
        &mut self,
        index: u32,
        phi_val_types: &[ValType],
        block: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<(), PhiError> {
        let phi_val_types: Vec<TyId> = phi_val_types
            .iter()
            .map(|x| llvm_ty_from_wasm(x, ctx))
            .collect();

        let mut cursor = ctx.cursor_at_block(block);
        let mut phi_handlers = vec![];
        let mut phi_vals = vec![];

        for (i, res) in phi_val_types.iter().enumerate() {
            // Named deliberately. These are created when the label *opens*, so an
            // unnamed one would take a number well before the block it sits in is
            // printed — and LLVM requires unnamed registers to run in textual order.
            // A named register draws nothing from that counter. See `RegName`.
            let (phi_handler, phi_val) = cursor.build_phi(
                &[],
                OperandTy::Asserted(*res),
                RegName::Named(format!("{}{}_res{}", T::name(), index, i)),
            )?;

            phi_handlers.push(phi_handler);
            phi_vals.push(phi_val);
        }

        self.0.insert(
            index,
            PhiValBranches {
                basic_block: block,
                phi_val_types,
                phi_vals,
                phi_handlers,
            },
        );

        Ok(())
    }

    pub fn add_branch(
        &mut self,
        index: u32,
        values: Vec<ValueId>,
        block: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<BasicBlockId, PhiError> {
        let data = self
            .0
            .get_mut(&index)
            .expect("this method should only be called after calling `insert`");

        if values.len() != data.phi_val_types.len() {
            panic!("values passed` should match the expected arity")
        }

        // `values` is in the label's own order — deepest first — so it lines up with
        // `phi_handlers` index for index. A caller that hands them over top-first
        // would type-check only while every result shares a type, and silently pair
        // the wrong values when they do.
        for (i, value) in values.iter().enumerate() {
            let phi_handler = data.phi_handlers[i];

            phi_handler.add_branch((block, *value), ctx)?;
        }

        Ok(data.basic_block)
    }

    pub fn phi_vals_and_block(&self, index: u32) -> Option<(&[ValueId], BasicBlockId)> {
        self.0
            .get(&index)
            .map(|x| (x.phi_vals.as_slice(), x.basic_block))
    }

    /// The block branches to this label land in, without touching its phis.
    ///
    /// [`add_branch`](Self::add_branch) hands the same block back, so this is for a
    /// caller that needs the destination without having a value to contribute.
    #[allow(dead_code, reason = "no caller yet; the read/remove half of the map, for the arms still to come")]
    pub fn get_basic_block(&self, index: u32) -> Option<BasicBlockId> {
        self.0.get(&index).map(|x| x.basic_block)
    }

    /// Closes a label, taking its block and the phis that were filled there.
    ///
    /// Nothing calls this yet, so an entry outlives the label it describes and the
    /// map grows for the length of a function. Harmless while a body is one pass, and
    /// the place to start if it stops being.
    #[allow(dead_code, reason = "no caller yet; the read/remove half of the map, for the arms still to come")]
    pub fn remove(&mut self, index: u32) -> Option<PhiValBranches> {
        self.0.remove(&index)
    }
}

/// What each still-open label has waiting, keyed by the instruction index that
/// closes it.
///
/// Both maps are keyed by an index the instruction stream already carries — an
/// `if` stores its `else_index` and `end_index`, a `br` its `target_index` — so a
/// jump is a lookup rather than a search back through the control stack.
#[derive(Default)]
pub(crate) struct InstrIndexToBasicBlockMap {
    /// Keyed by the `end`'s index. See [`PhiValBranches`].
    end_map: BranchTargetBasicBlockMap<End>,
    loop_map: BranchTargetBasicBlockMap<Loop>,
    /// Keyed by the `else`'s index: the block the false arm starts in, and the
    /// block's params to restore onto the simulated stack when it does.
    ///
    /// The params need no phi. They were live *before* the branch, so they dominate
    /// both arms and can be used exactly as they are.
    else_map: FxHashMap<u32, (BasicBlockId, Vec<ValueId>)>,
}

impl InstrIndexToBasicBlockMap {
    /// Records where an `if`'s false arm begins, and the params to restore when it
    /// does. Consumed by whichever reaches the else first — the `Else` instruction, or
    /// a `br` out of the then-arm that skips it.
    pub fn new_else(&mut self, index: u32, block: BasicBlockId, params: Vec<ValueId>) {
        self.else_map.insert(index, (block, params));
    }

    /// Takes the else arm's block and params. Taking rather than reading is the
    /// point: an arm is entered once, so a second call means the pass reached the
    /// same `else` twice.
    pub fn remove_else(&mut self, index: u32) -> Option<(BasicBlockId, Vec<ValueId>)> {
        self.else_map.remove(&index)
    }

    pub fn new_end(
        &mut self,
        index: u32,
        results: &[ValType],
        block: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<(), PhiError> {
        self.end_map.open(index, results, block, ctx)
    }

    pub fn add_end_branch(
        &mut self,
        index: u32,
        values: Vec<ValueId>,
        block: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<BasicBlockId, PhiError> {
        self.end_map.add_branch(index, values, block, ctx)
    }

    pub fn end_phi_vals_and_block(&self, index: u32) -> Option<(&[ValueId], BasicBlockId)> {
        self.end_map.phi_vals_and_block(index)
    }

    /// See [`BranchTargetBasicBlockMap::get_basic_block`].
    #[allow(dead_code, reason = "no caller yet; the read/remove half of the map, for the arms still to come")]
    pub fn get_end_basic_block(&self, index: u32) -> Option<BasicBlockId> {
        self.end_map.get_basic_block(index)
    }

    /// See [`BranchTargetBasicBlockMap::remove`].
    #[allow(dead_code, reason = "no caller yet; the read/remove half of the map, for the arms still to come")]
    pub fn remove_end(&mut self, index: u32) -> Option<PhiValBranches> {
        self.end_map.remove(index)
    }

    pub fn new_loop(
        &mut self,
        index: u32,
        params: &[ValType],
        block: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<(), PhiError> {
        self.loop_map.open(index, params, block, ctx)
    }

    pub fn add_loop_branch(
        &mut self,
        index: u32,
        values: Vec<ValueId>,
        block: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<BasicBlockId, PhiError> {
        self.loop_map.add_branch(index, values, block, ctx)
    }

    pub fn loop_phi_vals_and_block(&self, index: u32) -> Option<(&[ValueId], BasicBlockId)> {
        self.loop_map.phi_vals_and_block(index)
    }

    /// See [`BranchTargetBasicBlockMap::get_basic_block`].
    #[allow(dead_code, reason = "no caller yet; the read/remove half of the map, for the arms still to come")]
    pub fn get_loop_basic_block(&self, index: u32) -> Option<BasicBlockId> {
        self.loop_map.get_basic_block(index)
    }

    /// See [`BranchTargetBasicBlockMap::remove`].
    #[allow(dead_code, reason = "no caller yet; the read/remove half of the map, for the arms still to come")]
    pub fn remove_loop(&mut self, index: u32) -> Option<PhiValBranches> {
        self.loop_map.remove(index)
    }

    pub fn add_branch_to_target(
        &mut self,
        index: u32,
        values: Vec<ValueId>,
        block: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<BasicBlockId, PhiError> {
        if self.end_map.0.contains_key(&index) {
            self.add_end_branch(index, values, block, ctx)
        } else {
            self.add_loop_branch(index, values, block, ctx)
        }
    }
}

/// Wasm's operand stack, holding LLVM values instead of numbers.
///
/// The interpreter's [`Stack`] with a different element type, which is what makes the
/// translation a transcription rather than an analysis: wasm says "add the top two
/// operands", so this pops two ids, emits an `add`, and pushes the result. Nothing is
/// evaluated — a push records *which register* will hold the value at run time.
pub(crate) struct SimulatedStack {
    stack: Stack<ValueId>,
}

impl Default for SimulatedStack {
    fn default() -> Self {
        SimulatedStack {
            stack: Stack::new_with_capacity(0),
        }
    }
}

impl Deref for SimulatedStack {
    type Target = Stack<ValueId>;

    fn deref(&self) -> &Self::Target {
        &self.stack
    }
}

impl DerefMut for SimulatedStack {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stack
    }
}

/// What an open `if` needs beyond its `end`, so a `br` inside it knows where to
/// resume.
///
/// After a `br`, everything up to the label's `else` or `end` is gone from the
/// stream, so the next instruction is one of those two. Which one depends on where
/// the `br` was — hence both fields.
pub(crate) struct IfCtx {
    /// Where the false arm begins, if there is one.
    pub else_instr_index: Option<u32>,
    /// Whether the false arm has been entered. A `br` from the then-arm resumes at
    /// the `else`; a `br` from the else-arm has no arm left to enter and resumes at
    /// the `end`.
    pub is_else_ongoing: bool,
}

#[allow(
    dead_code,
    reason = "scaffolding for the `block`/`loop` arms, which are not emitted yet"
)]
/// Which construct a label came from.
///
/// Only `If` carries anything: it is the one label with two arms, so it is the one a
/// `br` can leave in more than one way.
pub(crate) enum LabelKind {
    /// An `if`, with the state its two arms need. See [`IfCtx`].
    If(IfCtx),
    /// A `loop`, whose branch target is its start rather than its `end`.
    Loop,
    /// A `block`.
    Block,
    /// The implicit label around the whole body, which `return` targets.
    Func,
}

#[allow(
    dead_code,
    reason = "scaffolding for the `block`/`loop` arms, which are not emitted yet"
)]
/// One open label, innermost last on the [`ControlStack`].
pub(crate) struct Label {
    /// Which construct opened it, and whatever that construct needs. See [`LabelKind`].
    pub kind: LabelKind,
    /// Index of the instruction that opened it.
    pub instr_index: usize,
    /// Index of the `end` that closes it — where a `br` leaving this label resumes.
    pub end_instr_index: usize,
}

/// The labels currently open, innermost last.
///
/// Distinct from the `end` map, which answers "where does label *N* close?". This
/// answers "which label am I in?" — the question a `br` asks, because after it the
/// rest of the enclosing label is unreachable and the pass has to know which one to
/// skip to.
///
/// Index 0 is the implicit function label, pushed by `compile_func`: a function body
/// has no opening instruction to push one for it.
#[derive(Default)]
pub(crate) struct ControlStack {
    stack: Vec<Label>,
}

impl ControlStack {
    /// Opens a label. Paired with [`leave_label`](Self::leave_label) at its `end`, or
    /// at a `br` that skips the `end` entirely.
    pub fn enter_label(&mut self, kind: LabelKind, instr_index: usize, end_instr_index: usize) {
        self.stack.push(Label {
            kind,
            instr_index,
            end_instr_index,
        });
    }

    /// The innermost label's `if` state, or `None` if it is not an `if`.
    ///
    /// Mutable because the one caller — a `br` leaving the then-arm — both reads
    /// `is_else_ongoing` and sets it, standing in for the `Else` instruction it
    /// skips over.
    pub fn try_curr_label_as_if_mut(&mut self) -> Option<&mut IfCtx> {
        let LabelKind::If(ctx) = &mut self.curr_label_mut().kind else {
            return None;
        };

        Some(ctx)
    }

    /// Closes the innermost label.
    ///
    /// # Panics
    ///
    /// If none is open, which means an `end` was reached without a matching opener —
    /// a bug in the pass, since the instruction stream is already balanced.
    pub fn leave_label(&mut self) -> Label {
        self.stack
            .pop()
            .expect("hitting this means the logic for control stack mutation is incorrect")
    }

    /// The innermost open label.
    pub fn curr_label(&self) -> &Label {
        &self.stack[self.stack.len() - 1]
    }

    /// The innermost open label, mutably.
    pub fn curr_label_mut(&mut self) -> &mut Label {
        let len = self.stack.len();

        &mut self.stack[len - 1]
    }

    #[allow(
        dead_code,
        reason = "scaffolding for the `block`/`loop` arms, which are not emitted yet"
    )]
    /// The function label's own span, for `return` — which targets the outermost
    /// label however deeply nested it appears.
    pub fn enclosing_func_instr_indices(&self) -> (usize, usize) {
        (self.stack[0].instr_index, self.stack[0].end_instr_index)
    }
}

/// The pass: everything one module's translation needs to carry between
/// instructions.
///
/// Reached through [`Module::build_cfg`](crate::module::Module::build_cfg) rather
/// than constructed directly. The three `pub(crate)` fields are the running state
/// described in the module docs; the two maps below are the function index space,
/// split the way wasm splits it.
#[derive(Default)]
pub struct WasmInstrLLVMPassManager {
    /// Imported functions, which become LLVM declarations — a signature and no body.
    declared_funcs: FxHashMap<FuncIndex, GlobalId<DeclaredFunc>>,
    /// Locally-defined functions, which become LLVM definitions.
    defined_funcs: FxHashMap<FuncIndex, GlobalId<DefinedFunc>>,
    pub(crate) instr_index_to_basic_block: InstrIndexToBasicBlockMap,
    pub(crate) simulated_stack: SimulatedStack,
    pub(crate) control_stack: ControlStack,
}

impl WasmInstrLLVMPassManager {
    /// Translates a whole module into one [`ControlFlowGraph`].
    ///
    /// Signatures first, bodies second, in two passes over the index space. A body
    /// may call any function — one declared later, or itself — and a call needs the
    /// callee's handle to exist, so no body is emitted until every signature is in.
    ///
    /// # Errors
    ///
    /// Whatever the builders report. A rejection here is a bug in this pass rather
    /// than bad input: the module has already been validated, so the IR it describes
    /// is well-formed by the time it gets here.
    pub fn compile<V: VirtualMachine>(
        mut self,
        module: &Arc<Module<V>>,
    ) -> Result<ControlFlowGraph, anyhow::Error> {
        // TODO: get this from the system on which this function is called!
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

        let mut builder = ctx.builder();

        let ty_decls = &module.types;
        let func_decls = &module.func_decls;
        let imported_func_count = module.imported_func_count; // this many functions will be declared! rest are defined!

        // Signatures first, bodies second. A body may call any function — one declared
        // later in the index space, or itself — and the callee's handle has to exist
        // before the call is built, so no body is emitted until every signature is in.
        for func_index in 0..func_decls.len() {
            let func_decl = &func_decls[func_index];
            let ty = func_decl.ty;
            let func_ty = &ty_decls[ty.0 as usize];
            let params = &func_ty.params;
            let results = &func_ty.results;

            let (llvm_params, llvm_result) =
                llvm_signature_from_wasm(params, results, &mut builder)?;

            if func_index < imported_func_count as usize {
                let func = builder.declare_function(
                    format!("fn{}", func_index),
                    &llvm_params,
                    llvm_result,
                )?;

                self.declared_funcs
                    .insert(FuncIndex(func_index as u32), func);
            } else {
                let mut llvm_param_decls = vec![];

                for (counter, param) in llvm_params.into_iter().enumerate() {
                    llvm_param_decls.push((param, RegName::Named(format!("param{}", counter))));
                }

                let func = builder.define_function(
                    format!("fn{}", func_index),
                    &llvm_param_decls,
                    llvm_result,
                )?;

                self.defined_funcs
                    .insert(FuncIndex(func_index as u32), func);
            }
        }

        for func_index in (imported_func_count as usize)..func_decls.len() {
            let func_index = FuncIndex(func_index as u32);
            // Copied out rather than borrowed: `compile_func` takes `&mut self`.
            let func = self.defined_funcs[&func_index];

            self.compile_func(func_index, func, module, &mut builder)?;
        }

        Ok(builder.build())
    }

    /// Translates one function body: its locals, then its instructions.
    ///
    /// Sets up three things the instruction arms rely on and cannot establish
    /// themselves — an `alloca` per local in the entry block, the function's own
    /// `end` label, and the implicit function label on the control stack. A body has
    /// no opening instruction to push that last one, so a `return` or a final `end`
    /// would otherwise find the control stack empty.
    ///
    /// The loop is index-driven rather than a `for` over the slice: an arm returns
    /// where to resume, which after a `br` is not the next instruction.
    fn compile_func<V: VirtualMachine>(
        &mut self,
        func_index: FuncIndex,
        func: GlobalId<DefinedFunc>,
        module: &Arc<Module<V>>,
        ctx: &mut Context,
    ) -> Result<(), anyhow::Error> {
        debug_assert!(func_index.0 >= module.imported_func_count);

        // `func_bodies` covers the defined functions only, so it is indexed by the
        // function index *shifted past the imports* — see `Module::imported_func_count`.
        let func_body = &module.func_bodies[(func_index.0 - module.imported_func_count) as usize];
        let local_types = &func_body.locals;
        let instructions = &func_body.instructions;
        let frame_layout = &func_body.frame_layout;
        let func_decl = &module.func_decls[func_index.0 as usize];
        let func_ty = &module.types[func_decl.ty.0 as usize];
        let results_ty = &func_ty.results;

        let entry = func.add_basic_block("entry", ctx)?;
        let params = func.params(ctx).to_vec();

        let runtime_ctx_ptr = *params
            .last()
            .expect("the context is always a function param");

        let mut entry_cursor = ctx.cursor_at_block(entry);
        let mut locals = vec![];

        for (counter, (i, local_ty)) in local_types.iter().enumerate().enumerate() {
            // last param is runtime ctx!
            let (ptr, val, alignment) = if i < params.len() - 1 {
                let param = &params[i];
                let ty = param.ty(&entry_cursor);
                let alignment = ty.alignment(&entry_cursor);

                (
                    entry_cursor.build_alloca(
                        ty,
                        None,
                        alignment,
                        RegName::Named(format!("local{}", counter)),
                    )?,
                    *param,
                    alignment,
                )
            } else {
                let ty = llvm_ty_from_wasm(local_ty, &mut entry_cursor);
                let val = Value::zero_of_ty(ty, &mut entry_cursor)
                    .expect("type for wasm locals are always basic type i.e. i32, i64, f32, f64");
                let alignment = ty.alignment(&entry_cursor);

                (
                    entry_cursor.build_alloca(
                        ty,
                        None,
                        alignment,
                        RegName::Named(format!("local{}", counter)),
                    )?,
                    val,
                    alignment,
                )
            };

            locals.push(ptr);
            entry_cursor.build_store(ptr, val, OperandTy::Inferred, alignment)?;
        }

        let func_end = func.add_basic_block("end", ctx)?;
        // The body's last instruction is the function's own `end`. It has no opening
        // instruction to enter the label from — `block`/`loop`/`if` each push theirs
        // when emitted — so the implicit outermost label is pushed here instead, and
        // popped by that `end` like any other.
        let func_end_instr_index = instructions.len() - 1;

        self.instr_index_to_basic_block.new_end(
            func_end_instr_index as u32,
            results_ty,
            func_end,
            ctx,
        )?;

        self.control_stack
            .enter_label(LabelKind::Func, 0, func_end_instr_index);

        let mut cursor = ctx.cursor_at_block(entry);
        let mut index = 0;

        loop {
            if index >= instructions.len() {
                break;
            }

            let instr = &instructions[index];

            let (next_block, next_instr_index) = instr.emit_llvm_ir(
                index,
                cursor,
                instructions,
                frame_layout,
                &locals,
                &runtime_ctx_ptr,
                func,
                self,
            )?;

            index = next_instr_index;
            cursor = ctx.cursor_at_block(next_block);
        }

        self.build_func_return(func, func_end_instr_index as u32, ctx)?;

        Ok(())
    }

    /// Emits the function's `ret` into its `end` block.
    ///
    /// Here rather than in the `End` arm, for the same reason the locals are set up
    /// here: this is the one place that runs once per function, whatever shape the
    /// body took. Parameters are moved into memory on the way in at a single point;
    /// results are assembled on the way out at a single point.
    ///
    /// The `End` arm is not that place. A body whose last instruction is `return`
    /// *skips* its own `end` — the branch arm resumes past it — so a `ret` emitted
    /// there would be missed exactly when the function always returns early. Every
    /// path out of the body still lands in this block, because `return` targets it
    /// too.
    ///
    /// # Panics
    ///
    /// If the function's `end` was never registered, which `compile_func` does
    /// before emitting a single instruction.
    fn build_func_return(
        &mut self,
        func: GlobalId<DefinedFunc>,
        func_end_instr_index: u32,
        ctx: &mut Context,
    ) -> Result<(), anyhow::Error> {
        let result_ty = func.return_ty(ctx);

        let (phi_vals, end_block) = self
            .instr_index_to_basic_block
            .end_phi_vals_and_block(func_end_instr_index)
            .expect("the function's `end` is registered before its body is emitted");

        // Copied out so the cursor below can borrow the context mutably. The results
        // are the `end`'s phis, deepest first — the order wasm declares them in.
        let phi_vals = phi_vals.to_vec();
        let mut end_cursor = ctx.cursor_at_block(end_block);

        // Wasm returns nothing, one value, or several. LLVM spells the last of those
        // as a struct, so these are three different instructions rather than one with
        // a varying operand.
        if result_ty.is_void(&end_cursor) {
            let void_ty = end_cursor.void_ty();

            end_cursor.build_ret(None, OperandTy::Asserted(void_ty))?;

            return Ok(());
        }

        // Only the field count is needed, so the borrow on the type pool ends here.
        let field_count = match result_ty.try_struct(&end_cursor) {
            Some((fields, _)) => fields.len(),
            None => {
                end_cursor.build_ret(Some(phi_vals[0]), OperandTy::Inferred)?;

                return Ok(());
            }
        };

        // Several results, assembled in memory: this crate has no `insertvalue`, so
        // each field is stored through a `getelementptr` and the whole struct loaded
        // back. LLVM folds that into the `insertvalue` chain it would have been — but
        // it takes `sroa` to do it, not `mem2reg` alone, because this `alloca` is not
        // in the entry block.
        let func_return_ptr = end_cursor.build_alloca(
            result_ty,
            None,
            result_ty.alignment(&end_cursor),
            RegName::Named("fn_return_ptr".to_string()),
        )?;

        // A `getelementptr` into a struct steps over the pointee first and descends
        // second, so every field index is preceded by this 0. See `build_get_element_ptr`.
        let zero_index = end_cursor.const_value(0i32, OperandTy::Inferred)?;

        for (i, field_val) in phi_vals.iter().take(field_count).enumerate() {
            // Field `i` and phi `i` are the same slot: wasm declares its results
            // deepest-first, and `add_end` builds the phis in that same order.
            let field_index = end_cursor.const_value(i as i32, OperandTy::Inferred)?;

            let field_ptr = end_cursor.build_get_element_ptr(
                func_return_ptr,
                OperandTy::Inferred,
                &[zero_index, field_index],
                Some(true),
                RegName::Named(format!("result{}_ptr", i)),
            )?;

            end_cursor.build_store(field_ptr, *field_val, OperandTy::Inferred, None)?;
        }

        let return_val = end_cursor.build_load(
            func_return_ptr,
            OperandTy::Inferred,
            None,
            RegName::Named("fn_return_val".to_string()),
        )?;

        end_cursor.build_ret(Some(return_val), OperandTy::Inferred)?;

        Ok(())
    }
}

/// The LLVM type a wasm value type becomes.
///
/// A reference becomes `ptr`, since that is what one is once it is an operand.
///
/// # Panics
///
/// On `v128`, which [`Module::compile`](crate::module::Module::compile) rejects at
/// section level — so reaching it here would mean the check was lost.
fn llvm_ty_from_wasm(ty: &ValType, ctx: &mut Context) -> TyId {
    match ty {
        ValType::I32 => ctx.i32_ty(),
        ValType::I64 => ctx.i64_ty(),
        ValType::F32 => ctx.f32_ty(),
        ValType::F64 => ctx.f64_ty(),
        ValType::Ref(_) => ctx.ptr_ty(),
        ValType::V128 => unreachable!("v128 is rejected at Module check time"),
    }
}

/// The LLVM signature a wasm function type becomes.
///
/// Two things differ from a straight mapping:
///
/// * **A runtime pointer is appended** to the parameters, through which a body
///   reaches the instance's memory and tables. It goes *last* precisely so that a
///   wasm local index and its LLVM parameter index stay equal.
/// * **Multiple results become a struct**, which is how LLVM returns more than one
///   value. No results becomes `void` — wasm spells that as an empty list, and it is
///   what most functions have.
fn llvm_signature_from_wasm(
    params: &[ValType],
    results: &[ValType],
    ctx: &mut Context,
) -> Result<(Vec<TyId>, TyId), anyhow::Error> {
    let mut llvm_params = vec![];

    for param_ty in params {
        llvm_params.push(llvm_ty_from_wasm(param_ty, ctx));
    }

    llvm_params.push(ctx.ptr_ty()); // pointer to runtime struct containing mmap memory pointers etc.

    // The runtime pointer goes on the *end* so a wasm local index and its LLVM
    // parameter index stay equal.
    let llvm_result = match results {
        // Wasm spells "returns nothing" as an empty result list; LLVM spells it
        // `void`, so this arm is not the degenerate case it looks like — it is
        // every function rustc emits for a unit return.
        [] => ctx.void_ty(),
        [result] => llvm_ty_from_wasm(result, ctx),
        _ => {
            let mut fields = vec![];

            for result in results {
                fields.push(llvm_ty_from_wasm(result, ctx));
            }

            ctx.struct_ty(&fields, false)?
        }
    };

    Ok((llvm_params, llvm_result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::stack::{LabelSignature, StackFrameLayout, StackInstruction};
    use tracewasm_llvm::{
        cfg::{
            builder::Builder,
            emit::IREmitter,
            module::{DataLayout, Triple},
        },
        instruction::cursor::OperandTy,
        value::NullPtr,
    };

    /// A body whose only label is an `if` at instruction 0, returning `results`.
    ///
    /// Lowering builds this from the operator stream; these tests hand-write the
    /// instruction slice, so they have to hand-write the layout that goes with it.
    fn layout_for_if_at_zero(results: &[ValType]) -> StackFrameLayout {
        let mut label_instr_index_to_signature = FxHashMap::default();

        label_instr_index_to_signature.insert(
            0,
            LabelSignature {
                params: vec![].into_boxed_slice(),
                results: results.to_vec().into_boxed_slice(),
            },
        );

        StackFrameLayout {
            br_targets_arena: vec![].into_boxed_slice(),
            label_instr_index_to_signature,
        }
    }

    /// These drive `emit_llvm_ir` over a hand-written instruction slice rather than
    /// going through `compile_func`, standing in for the operand producers by pushing
    /// onto the simulated stack between calls — most operators still hit the catch-all
    /// `todo!()` arm. That is enough to pin the control arms: the condition's
    /// comparison, the fall-through terminators, and the phis at the `end`.
    fn harness() -> (Builder, GlobalId<DefinedFunc>, BasicBlockId, ValueId) {
        let ctx = Context::new(
            Triple::new(
                "arm64".to_string(),
                "apple".to_string(),
                "macosx".to_string(),
                None,
            ),
            DataLayout::default(),
        );

        let mut builder = ctx.builder();
        let i32_ty = builder.i32_ty();

        let func = builder
            .define_function("f".to_string(), &[(i32_ty, "n".into())], i32_ty)
            .unwrap();

        let entry = func.add_basic_block("entry", &mut builder).unwrap();
        let n = func.nth_param(0, &builder).unwrap();

        (builder, func, entry, n)
    }

    /// Runs `instructions` from `entry`, calling `between` after each one so the test
    /// can push whatever that operator's body would have left on the stack.
    fn run(
        pass: &mut WasmInstrLLVMPassManager,
        builder: &mut Builder,
        func: GlobalId<DefinedFunc>,
        entry: BasicBlockId,
        instructions: &[StackInstruction],
        frame_layout: &StackFrameLayout,
        mut between: impl FnMut(&mut WasmInstrLLVMPassManager, &mut Builder, usize),
    ) -> BasicBlockId {
        let mut block = entry;
        let null_ptr = Value::from_const(NullPtr, OperandTy::Inferred, builder).unwrap();
        let mut index = 0;

        // Index-driven, like `compile_func`: an arm returns where to resume, which is
        // not always the next instruction.
        while index < instructions.len() {
            let cursor = builder.cursor_at_block(block);

            // No locals: these cases are about control flow, and none of the operators
            // under test reads a local slot.
            let (next_block, next_index) = instructions[index]
                .emit_llvm_ir(
                    index,
                    cursor,
                    instructions,
                    frame_layout,
                    &[],
                    &null_ptr,
                    func,
                    pass,
                )
                .unwrap();

            between(pass, builder, index);

            block = next_block;
            index = next_index;
        }

        block
    }

    #[test]
    fn an_if_else_joins_its_two_arms_with_one_phi() {
        let (mut builder, func, entry, n) = harness();
        let mut pass = WasmInstrLLVMPassManager::default();

        // `i32.const 1` would have pushed the condition.
        pass.simulated_stack.push(n);

        let instructions = [
            StackInstruction::If {
                else_index: Some(1),
                end_index: 2,
            },
            StackInstruction::Else { if_end_index: 2 },
            StackInstruction::End {
                arity: 1,
                recorded_height: 0,
            },
        ];

        let end = run(
            &mut pass,
            &mut builder,
            func,
            entry,
            &instructions,
            &layout_for_if_at_zero(&[ValType::I32]),
            |pass, builder, index| {
                // Each arm's body leaves one result behind.
                if index == 0 || index == 1 {
                    let val = builder
                        .const_value(if index == 0 { 10i32 } else { 20i32 }, OperandTy::Inferred)
                        .unwrap();

                    pass.simulated_stack.push(val);
                }
            },
        );

        let result = pass.simulated_stack.pop();

        builder
            .cursor_at_block(end)
            .build_ret(Some(result), OperandTy::Inferred)
            .unwrap();

        let ir = IREmitter::emit(builder.build()).unwrap();

        // The condition is compared, not relabelled — and the register is named, so
        // it cannot be numbered out of textual order the way an unnamed one would be.
        assert!(ir.contains("%if0_cond = icmp ne i32 %n, 0"), "{ir}");
        assert!(
            ir.contains("br i1 %if0_cond, label %if0_then, label %if0_else"),
            "{ir}"
        );

        // Nothing the pass defines draws from LLVM's unnamed counter.
        assert!(!ir.contains("%0"), "an unnamed register slipped in\n{ir}");

        // Both arms are closed rather than falling off the end of their block.
        assert_eq!(
            ir.matches("br label %if0_end").count(),
            2,
            "both arms should jump to the end\n{ir}"
        );

        // One phi, naming both predecessors — not one phi per predecessor.
        assert_eq!(ir.matches("phi").count(), 1, "{ir}");
        assert!(
            ir.contains("phi i32 [ 10, %if0_then ], [ 20, %if0_else ]"),
            "{ir}"
        );
    }

    #[test]
    fn an_if_without_an_else_still_reaches_the_end_from_both_edges() {
        let (mut builder, func, entry, n) = harness();
        let mut pass = WasmInstrLLVMPassManager::default();

        let param = builder.const_value(7i32, OperandTy::Inferred).unwrap();

        // `[i32] -> [i32]`: the block's param is also its result, which is the only
        // shape an `if` without an `else` can have.
        pass.simulated_stack.push(param);
        pass.simulated_stack.push(n);

        let instructions = [
            StackInstruction::If {
                else_index: None,
                end_index: 1,
            },
            StackInstruction::End {
                arity: 1,
                recorded_height: 0,
            },
        ];

        let end = run(
            &mut pass,
            &mut builder,
            func,
            entry,
            &instructions,
            &layout_for_if_at_zero(&[ValType::I32]),
            |pass, builder, index| {
                if index == 0 {
                    // The then-arm replaces the param with a result of its own.
                    pass.simulated_stack.pop();

                    let val = builder.const_value(99i32, OperandTy::Inferred).unwrap();

                    pass.simulated_stack.push(val);
                }
            },
        );

        let result = pass.simulated_stack.pop();

        builder
            .cursor_at_block(end)
            .build_ret(Some(result), OperandTy::Inferred)
            .unwrap();

        let ir = IREmitter::emit(builder.build()).unwrap();

        // The false edge skips straight to the end, so the phi has to name `entry`
        // alongside the then-arm or it is short a predecessor.
        assert!(
            ir.contains("phi i32 [ 99, %if0_then ], [ 7, %entry ]")
                || ir.contains("phi i32 [ 7, %entry ], [ 99, %if0_then ]"),
            "{ir}"
        );
    }

    #[test]
    fn a_multi_value_end_keeps_its_results_in_stack_order() {
        let (mut builder, func, entry, n) = harness();
        let mut pass = WasmInstrLLVMPassManager::default();

        pass.simulated_stack.push(n);

        let instructions = [
            StackInstruction::If {
                else_index: Some(1),
                end_index: 2,
            },
            StackInstruction::Else { if_end_index: 2 },
            StackInstruction::End {
                arity: 2,
                recorded_height: 0,
            },
        ];

        let end = run(
            &mut pass,
            &mut builder,
            func,
            entry,
            &instructions,
            &layout_for_if_at_zero(&[ValType::I32, ValType::I32]),
            |pass, builder, index| {
                if index == 0 || index == 1 {
                    // Two results per arm: `1`/`2` from the then-arm, `3`/`4` from the
                    // else-arm, pushed bottom-first — so `2` and `4` are the tops.
                    let base = if index == 0 { 1i32 } else { 3i32 };

                    for offset in 0..2 {
                        let val = builder
                            .const_value(base + offset, OperandTy::Inferred)
                            .unwrap();

                        pass.simulated_stack.push(val);
                    }
                }
            },
        );

        // One phi per result, not one per predecessor.
        assert_eq!(pass.simulated_stack.height(), 2);

        // Returning the *top* is what makes the stack order observable. Asserting only
        // on the phis' operands would not: reversing the slot index swaps which phi is
        // built first and which is pushed first, leaving both pairings intact.
        let top = pass.simulated_stack.pop();

        builder
            .cursor_at_block(end)
            .build_ret(Some(top), OperandTy::Inferred)
            .unwrap();

        let ir = IREmitter::emit(builder.build()).unwrap();

        // Each phi pairs the values that sat at the same stack slot — bottom with
        // bottom, top with top.
        assert!(
            ir.contains("phi i32 [ 1, %if0_then ], [ 3, %if0_else ]"),
            "{ir}"
        );
        assert!(
            ir.contains("phi i32 [ 2, %if0_then ], [ 4, %if0_else ]"),
            "{ir}"
        );

        // ...and the one holding the tops is the one left on top of the stack.
        let top_phi = ir
            .lines()
            .find(|line| line.contains("[ 2, %if0_then ]"))
            .expect("the phi joining the two arms' top values");
        let reg = top_phi
            .split_whitespace()
            .next()
            .expect("a phi line starts with the register it defines");

        assert!(
            ir.contains(&format!("ret i32 {reg}")),
            "expected the returned value to be {reg}\n{ir}"
        );
    }
}
