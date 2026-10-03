use crate::{
    instruction::stack::{
        FuncContext, StackInstruction,
        llvm::{
            IfCtx, LabelKind, WasmInstrLLVMPassManager,
            ctx::{OptionalU32, RuntimeContext, TableEntry},
        },
    },
    module::{Module, ValType},
};
use std::sync::Arc;
use tracewasm_llvm::{
    cfg::{basic_block::BasicBlockId, global::FuncRef},
    instruction::{
        Access, CastOp, ICond,
        cursor::{Cursor, OperandTy, RegName},
    },
    value::{FuncSignature, ValueId},
};
use wasmparser::ExternalKind::Table;

impl StackInstruction {
    // A CFG-building pass needs the cursor, the whole instruction stream, the frame
    // layout, the locals, the runtime pointer and the enclosing function — grouping
    // them into a context struct would only move the list.
    /// Translates this one instruction into LLVM IR, returning where to carry on.
    ///
    /// The return is `(block, next_index)` rather than nothing, because neither is
    /// implied by the instruction alone: an `if` leaves the cursor in its *then*
    /// block, and a `br` resumes at the enclosing label's `else` or `end` rather than
    /// at the following instruction. The driver in
    /// [`compile_func`](llvm::WasmInstrLLVMPassManager) does what it is told rather
    /// than tracking control flow a second time.
    ///
    /// `instructions` is the whole body, since a control instruction reads the
    /// operand arity and unwind height off the `end` it names. `locals` is one
    /// pointer per local, `runtime_ctx_ptr` the instance pointer threaded in as the
    /// last parameter.
    ///
    /// An inherent method rather than part of [`Instruction`]: only the stack
    /// machine is lowered to LLVM, so the register machine has nothing to implement.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_llvm_ir<'a>(
        &self,
        instr_index: usize,
        mut curr_cursor: Cursor<'a>,
        runtime_ctx_ptr: tracewasm_llvm::value::ValueId,
        module: &Arc<Module<crate::Stack>>,
        pass_manager: &mut WasmInstrLLVMPassManager,
        func_ctx: FuncContext<'_>,
    ) -> Result<(BasicBlockId, usize), anyhow::Error> {
        match self {
            StackInstruction::LocalGet { index } => {
                let index = index.0 as usize;
                let local_ptr = &func_ctx.locals[index];

                // Named, like every register this pass defines: an unnamed one takes
                // its number when it is *created*, but LLVM numbers by position in the
                // printed function, and blocks here are created long before they are
                // filled. See `RegName` — a named register draws nothing from that
                // counter, so it cannot be numbered out of order.
                let local_val = curr_cursor.build_load(
                    *local_ptr,
                    OperandTy::Inferred,
                    Access::Aligned,
                    RegName::Named(format!("local{}_val", index)),
                )?;

                pass_manager.simulated_stack.push(local_val);
            }
            StackInstruction::LocalSet { index } => {
                let index = index.0 as usize;
                let local_ptr = &func_ctx.locals[index];
                let val = pass_manager.simulated_stack.pop();

                curr_cursor.build_store(*local_ptr, val, OperandTy::Inferred, Access::Aligned)?;
            }
            StackInstruction::LocalTee { index } => {
                let index = index.0 as usize;
                let local_ptr = &func_ctx.locals[index];
                let top_val = pass_manager.simulated_stack.peek_from_top(0);

                curr_cursor.build_store(
                    *local_ptr,
                    *top_val,
                    OperandTy::Inferred,
                    Access::Aligned,
                )?;
            }
            StackInstruction::GlobalGet { index } => {
                let global_ptr = RuntimeContext::global_ptr(
                    runtime_ctx_ptr,
                    &mut curr_cursor,
                    RegName::Named(format!("global{}_ptr", instr_index)),
                )?;

                let index_val = curr_cursor.const_value(index.0 as i32, OperandTy::Inferred)?;
                let i64_ty = curr_cursor.i64_ty();
                let i32_ty = curr_cursor.i32_ty();

                let global_index_ptr = curr_cursor.build_get_element_ptr(
                    global_ptr,
                    OperandTy::Asserted(i64_ty),
                    &[index_val],
                    Some(true),
                    RegName::Named(format!("global{}_{}_ptr", instr_index, index.0)),
                )?;

                let global_val = curr_cursor.build_load(
                    global_index_ptr,
                    OperandTy::Asserted(i64_ty),
                    Access::Aligned,
                    RegName::Named(format!("global{}_{}_val", instr_index, index.0)),
                )?;

                let global_ty = &module.globals[index.0 as usize].ty.content_type();

                let global_val = match global_ty {
                    ValType::I32 => curr_cursor.build_cast(
                        CastOp::Trunc,
                        global_val,
                        OperandTy::Inferred,
                        i32_ty,
                        RegName::Named(format!("global{}_{}_val_as_i32", instr_index, index.0)),
                    )?,
                    ValType::I64 => global_val,
                    // Floats are stored by bit pattern in the low bits, so the value is
                    // recovered with a `bitcast` of the same width: `f32` through the
                    // low 32 bits, `f64` from the whole slot.
                    ValType::F32 => {
                        let bits = curr_cursor.build_cast(
                            CastOp::Trunc,
                            global_val,
                            OperandTy::Inferred,
                            i32_ty,
                            RegName::Named(format!("global{}_{}_val_bits", instr_index, index.0)),
                        )?;

                        let f32_ty = curr_cursor.f32_ty();

                        curr_cursor.build_cast(
                            CastOp::Bitcast,
                            bits,
                            OperandTy::Inferred,
                            f32_ty,
                            RegName::Named(format!("global{}_{}_val_as_f32", instr_index, index.0)),
                        )?
                    }
                    ValType::F64 => {
                        let f64_ty = curr_cursor.f64_ty();

                        curr_cursor.build_cast(
                            CastOp::Bitcast,
                            global_val,
                            OperandTy::Inferred,
                            f64_ty,
                            RegName::Named(format!("global{}_{}_val_as_f64", instr_index, index.0)),
                        )?
                    }
                    ValType::Ref(_) | ValType::V128 => {
                        unreachable!("globals with function ref or v128 not allowed!")
                    }
                };

                pass_manager.simulated_stack.push(global_val);
            }
            StackInstruction::GlobalSet { index } => {
                let val = pass_manager.simulated_stack.pop();

                let global_ptr = RuntimeContext::global_ptr(
                    runtime_ctx_ptr,
                    &mut curr_cursor,
                    RegName::Named(format!("global{}_ptr", instr_index)),
                )?;

                let index_val = curr_cursor.const_value(index.0 as i32, OperandTy::Inferred)?;
                let i64_ty = curr_cursor.i64_ty();
                let i32_ty = curr_cursor.i32_ty();

                let global_index_ptr = curr_cursor.build_get_element_ptr(
                    global_ptr,
                    OperandTy::Asserted(i64_ty),
                    &[index_val],
                    Some(true),
                    RegName::Named(format!("global{}_{}_ptr", instr_index, index.0)),
                )?;

                let global_ty = &module.globals[index.0 as usize].ty.content_type();

                // Into the slot's `i64`, with the same bits `From<Val> for GlobalVal`
                // writes: an `i32` zero-extended (upper bits clear), a float by bit
                // pattern in the low bits.
                let casted_val = match global_ty {
                    ValType::I32 => curr_cursor.build_cast(
                        CastOp::Zext,
                        val,
                        OperandTy::Inferred,
                        i64_ty,
                        RegName::Named(format!("global{}_{}_casted_val", instr_index, index.0)),
                    )?,
                    ValType::I64 => val,
                    ValType::F32 => {
                        let bits = curr_cursor.build_cast(
                            CastOp::Bitcast,
                            val,
                            OperandTy::Inferred,
                            i32_ty,
                            RegName::Named(format!("global{}_{}_val_bits", instr_index, index.0)),
                        )?;

                        curr_cursor.build_cast(
                            CastOp::Zext,
                            bits,
                            OperandTy::Inferred,
                            i64_ty,
                            RegName::Named(format!("global{}_{}_casted_val", instr_index, index.0)),
                        )?
                    }
                    ValType::F64 => curr_cursor.build_cast(
                        CastOp::Bitcast,
                        val,
                        OperandTy::Inferred,
                        i64_ty,
                        RegName::Named(format!("global{}_{}_casted_val", instr_index, index.0)),
                    )?,
                    ValType::Ref(_) => todo!(),
                    ValType::V128 => unreachable!("globals with v128 not allowed!"),
                };

                curr_cursor.build_store(
                    global_index_ptr,
                    casted_val,
                    OperandTy::Asserted(i64_ty),
                    Access::Aligned,
                )?;
            }
            StackInstruction::Call {
                func_index,
                params_count,
            } => {
                let callee_func = pass_manager
                    .get_func(func_index)
                    .expect("function always exist if it has made up till this (compile) phase!");

                let (_, signature) = callee_func.name_and_sig(&curr_cursor)?;
                let is_void = signature.result.is_void(&curr_cursor);

                let mut params: Vec<(ValueId, OperandTy)> = pass_manager
                    .simulated_stack
                    .pops_and_reverse(*params_count)
                    .iter()
                    .map(|x| (*x, OperandTy::Inferred))
                    .collect();

                params.push((runtime_ctx_ptr, OperandTy::Inferred));

                if is_void {
                    curr_cursor.build_void_call(callee_func, &params)?;

                    return Ok((curr_cursor.basic_block(), instr_index + 1));
                }

                let result = curr_cursor.build_call(
                    callee_func,
                    &params,
                    OperandTy::Inferred,
                    RegName::Named(format!("func{}_{}_result", func_index.0, instr_index)),
                )?;

                let results =
                    &module.types[module.func_decls[func_index.0 as usize].ty.0 as usize].results;

                let results_count = results.len();

                if results_count == 1 {
                    pass_manager.simulated_stack.push(result);

                    return Ok((curr_cursor.basic_block(), instr_index + 1));
                }

                for i in 0..results_count {
                    let field_val = curr_cursor.build_extract_value(
                        result,
                        &[i as u32],
                        RegName::Named(format!(
                            "func{}_{}_result_{}",
                            func_index.0, instr_index, i
                        )),
                    )?;

                    pass_manager.simulated_stack.push(field_val);
                }
            }
            StackInstruction::CallIndirect {
                ty_index,
                table_index,
            } => {
                let table_ptr = RuntimeContext::table_ptr(
                    runtime_ctx_ptr,
                    &mut curr_cursor,
                    RegName::Named(format!("table{}_ptr", instr_index)),
                )?;

                let i32_ty = curr_cursor.i32_ty();
                let i8_ty = curr_cursor.i8_ty();
                let ptr_ty = curr_cursor.ptr_ty();

                let index_val =
                    curr_cursor.const_value(table_index.0 as i32, OperandTy::Inferred)?;
                let table_entry_ty = TableEntry::llvm_ty(&mut curr_cursor);

                let zero_index = curr_cursor.const_value(0i32, OperandTy::Inferred)?;
                let first_index = curr_cursor.const_value(1i32, OperandTy::Inferred)?;

                let table_entry_ptr_ptr = curr_cursor.build_get_element_ptr(
                    table_ptr,
                    OperandTy::Asserted(table_entry_ty),
                    &[index_val, zero_index],
                    Some(true),
                    RegName::Named(format!(
                        "table_entry{}_{}_ptr_ptr",
                        instr_index, table_index.0,
                    )),
                )?;

                let table_entry_ptr = curr_cursor.build_load(
                    table_entry_ptr_ptr,
                    OperandTy::Inferred,
                    Access::Aligned,
                    RegName::Named(format!("table_entry{}_{}_ptr", instr_index, table_index.0)),
                )?;

                let slot = pass_manager.simulated_stack.pop();
                let optional_u32_ty = OptionalU32::llvm_ty(&mut curr_cursor);

                // TODO: bounds-check `slot` before this GEP. Load the entry's `table_len`
                // (GEP `[table_index, 1]`, next to the `table_ptr` above) and branch to a
                // trap block unless `icmp ult slot, table_len` — unsigned, as wasm reads
                // the index. `inbounds` past the end is poison, so the slot GEP and the
                // `val`/`tag` loads below must sit in the in-bounds branch.
                let func_ref_ptr = curr_cursor.build_get_element_ptr(
                    table_entry_ptr,
                    OperandTy::Asserted(optional_u32_ty),
                    &[slot],
                    Some(true),
                    RegName::Named(format!("func_ref{}_ptr", instr_index)),
                )?;

                let func_ref_val_ptr = curr_cursor.build_get_element_ptr(
                    func_ref_ptr,
                    OperandTy::Asserted(optional_u32_ty),
                    &[zero_index, zero_index],
                    Some(true),
                    RegName::Named(format!("func_ref{}_val_ptr", instr_index)),
                )?;

                let func_ref_val = curr_cursor.build_load(
                    func_ref_val_ptr,
                    OperandTy::Asserted(i32_ty),
                    Access::Aligned,
                    RegName::Named(format!("func_ref{}_val", instr_index)),
                )?;

                let func_ref_tag_ptr = curr_cursor.build_get_element_ptr(
                    func_ref_ptr,
                    OperandTy::Asserted(optional_u32_ty),
                    &[zero_index, first_index],
                    Some(true),
                    RegName::Named(format!("func_ref{}_tag_ptr", instr_index)),
                )?;

                let func_ref_tag = curr_cursor.build_load(
                    func_ref_tag_ptr,
                    OperandTy::Asserted(i8_ty),
                    Access::Aligned,
                    RegName::Named(format!("func_ref{}_tag", instr_index)),
                )?;

                let func_ty = &module.types[ty_index.0 as usize];
                let callee_params = &func_ty.params;
                let callee_results = &func_ty.results;
                // TODO: null-check the slot: trap if `func_ref_tag` is
                // `OptionalU32::TAG_NULL` (0). The `fn_table` GEP and load below must sit in
                // the non-null branch: a null slot's `val` is 0, which is out of bounds when
                // the module has no functions (`fn_table` is `[0 x ptr]`).
                let func_table = curr_cursor.global_value(func_ctx.func_table);

                let func_ptr = curr_cursor.build_get_element_ptr(
                    func_table,
                    OperandTy::Inferred,
                    &[zero_index, func_ref_val],
                    Some(true),
                    RegName::Named(format!("func{}_ptr", instr_index)),
                )?;

                let func = curr_cursor.build_load(
                    func_ptr,
                    OperandTy::Asserted(ptr_ty),
                    Access::Aligned,
                    RegName::Named(format!("func{}", instr_index)),
                )?;

                let (llvm_params, llvm_result) =
                    WasmInstrLLVMPassManager::llvm_signature_from_wasm(
                        callee_params,
                        callee_results,
                        &mut curr_cursor,
                    )?;

                let sig = FuncSignature::new(&llvm_params, llvm_result);
                let is_void = sig.result.is_void(&curr_cursor);
                // TODO: check the callee's signature before calling: trap unless its type
                // matches `ty_index`. `sig` is only what this `call_indirect` *expects* — the
                // pointer carries no type, and calling through a mismatched signature is
                // undefined behaviour in LLVM, where wasm requires a trap. Needs each
                // function's type at run time: e.g. a `fn_type_table` of canonical type ids
                // (`[N x i32]`, built beside `fn_table` in `compile`), compared with
                // `icmp eq` against the canonical id of `ty_index`. Canonical, because wasm
                // matches function types structurally, not by index.
                //
                // TODO: once the three checks above exist, the straight-line code becomes
                // blocks: entry → in_bounds → non_null → call, each check branching to one
                // shared trap block (`unreachable`), and the `call` block returned as the
                // continuation.
                let callee_func_ref = FuncRef::Pointer { ptr: func, sig };

                let mut params: Vec<(ValueId, OperandTy)> = pass_manager
                    .simulated_stack
                    .pops_and_reverse(callee_params.len() as u32)
                    .iter()
                    .map(|x| (*x, OperandTy::Inferred))
                    .collect();

                params.push((runtime_ctx_ptr, OperandTy::Inferred));

                if is_void {
                    curr_cursor.build_void_call(callee_func_ref, &params)?;

                    return Ok((curr_cursor.basic_block(), instr_index + 1));
                }

                let result = curr_cursor.build_call(
                    callee_func_ref,
                    &params,
                    OperandTy::Inferred,
                    RegName::Named(format!("indirect_func{}_result", instr_index)),
                )?;

                let results_count = callee_results.len();

                if results_count == 1 {
                    pass_manager.simulated_stack.push(result);

                    return Ok((curr_cursor.basic_block(), instr_index + 1));
                }

                for i in 0..results_count {
                    let field_val = curr_cursor.build_extract_value(
                        result,
                        &[i as u32],
                        RegName::Named(format!("indirect_func{}_result_{}", instr_index, i)),
                    )?;

                    pass_manager.simulated_stack.push(field_val);
                }

                todo!()
            }
            StackInstruction::Block { end_index } => {
                pass_manager.control_stack.enter_label(
                    LabelKind::Block,
                    instr_index,
                    *end_index as usize,
                );

                let block = func_ctx
                    .func
                    .add_basic_block(format!("block{}", instr_index), &mut curr_cursor)?;

                let end = func_ctx
                    .func
                    .add_basic_block(format!("block{}_end", instr_index), &mut curr_cursor)?;

                let label_sig = func_ctx.frame_layout
            .label_instr_index_to_signature
            .get(&(instr_index as u32)).expect("hitting this means tracking of label instr index to its signature mapping while lowering is incorrect");

                pass_manager.instr_index_to_basic_block.new_end(
                    *end_index,
                    &label_sig.results,
                    end,
                    &mut curr_cursor,
                )?;

                curr_cursor.build_unconditional_br(block)?;

                return Ok((block, instr_index + 1));
            }
            StackInstruction::Loop { end_index } => {
                pass_manager.control_stack.enter_label(
                    LabelKind::Loop,
                    instr_index,
                    *end_index as usize,
                );

                let loop_block = func_ctx
                    .func
                    .add_basic_block(format!("loop{}", instr_index), &mut curr_cursor)?;

                let end_block = func_ctx
                    .func
                    .add_basic_block(format!("loop{}_end", instr_index), &mut curr_cursor)?;

                let label_sig = func_ctx.frame_layout
            .label_instr_index_to_signature
            .get(&(instr_index as u32)).expect("hitting this means tracking of label instr index to its signature mapping while lowering is incorrect");

                let param_types = &label_sig.params;
                let params_count = param_types.len() as u32;
                let result_types = &label_sig.results;
                let start_index = pass_manager.simulated_stack.height() - params_count;
                let mut params = vec![];

                for i in start_index..pass_manager.simulated_stack.height() {
                    params.push(pass_manager.simulated_stack.stack[i as usize]);
                }

                pass_manager.instr_index_to_basic_block.new_loop(
                    instr_index as u32,
                    param_types,
                    loop_block,
                    &mut curr_cursor,
                )?;

                pass_manager.instr_index_to_basic_block.add_loop_branch(
                    instr_index as u32,
                    params,
                    curr_cursor.basic_block(),
                    &mut curr_cursor,
                )?;

                pass_manager.instr_index_to_basic_block.new_end(
                    *end_index,
                    result_types,
                    end_block,
                    &mut curr_cursor,
                )?;

                curr_cursor.build_unconditional_br(loop_block)?;

                let (phi_vals, _) = pass_manager
            .instr_index_to_basic_block
            .loop_phi_vals_and_block(instr_index as u32).expect("hitting this means logic for tracking target index of labels in lowering is incorrect");

                for i in 0..params_count {
                    pass_manager.simulated_stack.stack[(start_index + i) as usize] =
                        phi_vals[i as usize];
                }

                return Ok((loop_block, instr_index + 1));
            }
            StackInstruction::If {
                else_index,
                end_index,
            } => {
                pass_manager.control_stack.enter_label(
                    LabelKind::If(IfCtx {
                        else_instr_index: *else_index,
                        is_else_ongoing: false,
                    }),
                    instr_index,
                    *end_index as usize,
                );

                let (recorded_height, _) = Self::recorded_height_and_arity_from_end_instruction(
                    *end_index,
                    func_ctx.instructions,
                );

                let label_sig = func_ctx.frame_layout
            .label_instr_index_to_signature
            .get(&(instr_index as u32)).expect("hitting this means tracking of label instr index to its signature mapping while lowering is incorrect");

                let results_ty = &label_sig.results;

                // Wasm branches on "non-zero", and LLVM's `br` takes an `i1`, so the
                // condition needs a real comparison. Retyping the i32 in place would
                // emit `br i1 %x` against a register defined as i32 — IR that does not
                // assemble.
                let cond_val = pass_manager.simulated_stack.pop();
                let zero = curr_cursor.const_value(0i32, OperandTy::Inferred)?;

                let cond = curr_cursor.build_icmp(
                    ICond::Ne,
                    OperandTy::Inferred,
                    cond_val,
                    zero,
                    RegName::Named(format!("if{}_cond", instr_index)),
                )?;

                let if_then = func_ctx
                    .func
                    .add_basic_block(format!("if{}_then", instr_index), &mut curr_cursor)?;

                // The block's params are live *before* the branch, so they dominate both
                // arms and can be used as they are — no phi. Collected bottom-up, which
                // is the order they are pushed back in.
                let mut params = vec![];

                for i in recorded_height..pass_manager.simulated_stack.height() {
                    params.push(pass_manager.simulated_stack.stack[i as usize]);
                }

                let if_else = if let Some(else_index) = else_index {
                    let if_else = func_ctx
                        .func
                        .add_basic_block(format!("if{}_else", instr_index), &mut curr_cursor)?;

                    pass_manager.instr_index_to_basic_block.new_else(
                        *else_index,
                        if_else,
                        params.clone(),
                    );

                    Some(if_else)
                } else {
                    None
                };

                let if_end = func_ctx
                    .func
                    .add_basic_block(format!("if{}_end", instr_index), &mut curr_cursor)?;

                pass_manager.instr_index_to_basic_block.new_end(
                    *end_index,
                    results_ty,
                    if_end,
                    &mut curr_cursor,
                )?;

                let false_label = if let Some(if_else) = if_else {
                    if_else
                } else {
                    // No `else`, so the false edge jumps straight to the `end` and
                    // carries the block's params through as its results — wasm requires
                    // the two to match for an `if` without an else arm. Recording it here
                    // is what keeps the phi at `if_end` from being short a predecessor.
                    // `params` was collected deepest-first, which is already the order
                    // the phis are in.
                    pass_manager.instr_index_to_basic_block.add_end_branch(
                        *end_index,
                        params,
                        curr_cursor.basic_block(),
                        &mut curr_cursor,
                    )?;

                    if_end
                };

                curr_cursor.build_conditional_br(cond, if_then, false_label)?;

                return Ok((if_then, instr_index + 1));
            }
            StackInstruction::Else { if_end_index } => {
                let if_ctx = pass_manager
                    .control_stack
                    .try_curr_label_as_if_mut()
                    .expect("if-else not balanced!");

                if_ctx.is_else_ongoing = true;

                let curr_basic_block = curr_cursor.basic_block();

                let (recorded_height, arity) = Self::recorded_height_and_arity_from_end_instruction(
                    *if_end_index,
                    func_ctx.instructions,
                );

                debug_assert!(pass_manager.simulated_stack.height() - recorded_height == arity);

                // `pops_and_reverse` hands them back deepest-first, which is the order
                // the `end`'s phis are in. Popping one at a time would give the reverse.
                let results = pass_manager.simulated_stack.pops_and_reverse(arity);

                let if_end = pass_manager.instr_index_to_basic_block.add_end_branch(
                    *if_end_index,
                    results,
                    curr_basic_block,
                    &mut curr_cursor,
                )?;

                curr_cursor.build_unconditional_br(if_end)?;

                // restore the stack with original params
                let (else_block, params) = pass_manager
                    .instr_index_to_basic_block
                    .remove_else(instr_index as u32)
                    .expect("hitting this means logic for tracking `else` index is incorrect");

                for param in params {
                    pass_manager.simulated_stack.push(param);
                }

                return Ok((else_block, instr_index + 1));
            }
            StackInstruction::Return {
                target_index,
                arity,
                recorded_height: _,
            }
            | StackInstruction::Br {
                target_index,
                arity,
                recorded_height: _,
            } => {
                // Deepest first, top last — the order the target's phis are in. A `br`
                // only reads them: the stack is unwound at the label it lands in.
                let mut results = vec![];
                let start_index = pass_manager.simulated_stack.height() - *arity;

                for i in start_index..pass_manager.simulated_stack.height() {
                    results.push(pass_manager.simulated_stack.stack[i as usize]);
                }

                // can be loop or end
                let target_block = pass_manager
                    .instr_index_to_basic_block
                    .add_branch_to_target(
                        *target_index,
                        results,
                        curr_cursor.basic_block(),
                        &mut curr_cursor,
                    )?;

                curr_cursor.build_unconditional_br(target_block)?;

                let curr_label_end_index = pass_manager.control_stack.curr_label().end_instr_index;

                let (recorded_height, _) =
                    StackInstruction::recorded_height_and_arity_from_end_instruction(
                        curr_label_end_index as u32,
                        func_ctx.instructions,
                    );

                pass_manager.simulated_stack.truncate(recorded_height);

                let (next_block, next_instr_index) = if let Some(if_ctx) =
                    pass_manager.control_stack.try_curr_label_as_if_mut()
                    && !if_ctx.is_else_ongoing
                    && let Some(curr_label_else_index) = if_ctx.else_instr_index
                {
                    // restore the stack with original params
                    let (else_block, params) = pass_manager
                        .instr_index_to_basic_block
                        .remove_else(curr_label_else_index)
                        .expect("hitting this means logic for tracking `else` index is incorrect");

                    for param in params {
                        pass_manager.simulated_stack.push(param);
                    }

                    if_ctx.is_else_ongoing = true;

                    (else_block, curr_label_else_index as usize + 1)
                } else {
                    let (phi_vals, end_block) = pass_manager
                .instr_index_to_basic_block
                .end_phi_vals_and_block(curr_label_end_index as u32)
                .expect("hitting this means logic for tracking target index of labels in lowering is incorrect");

                    for val in phi_vals {
                        pass_manager.simulated_stack.push(*val);
                    }

                    pass_manager.control_stack.leave_label();

                    (end_block, curr_label_end_index + 1)
                };

                return Ok((next_block, next_instr_index));
            }
            StackInstruction::BrIf {
                target_index,
                arity,
                recorded_height: _recorded_height,
            } => {
                let cond_val = pass_manager.simulated_stack.pop();
                let zero = curr_cursor.const_value(0i32, OperandTy::Inferred)?;

                let cond = curr_cursor.build_icmp(
                    ICond::Ne,
                    OperandTy::Inferred,
                    cond_val,
                    zero,
                    RegName::Named(format!("br_if{}_cond", instr_index)),
                )?;

                let br_if_false = func_ctx
                    .func
                    .add_basic_block(format!("br_if{}_false", instr_index), &mut curr_cursor)?;

                let start_index = pass_manager.simulated_stack.height() - *arity;
                let mut results = vec![];

                for i in start_index..pass_manager.simulated_stack.height() {
                    results.push(pass_manager.simulated_stack.stack[i as usize]);
                }

                let target_block = pass_manager
                    .instr_index_to_basic_block
                    .add_branch_to_target(
                        *target_index,
                        results,
                        curr_cursor.basic_block(),
                        &mut curr_cursor,
                    )?;

                curr_cursor.build_conditional_br(cond, target_block, br_if_false)?;

                return Ok((br_if_false, instr_index + 1));
            }
            StackInstruction::BrTable { start_index, len } => {
                let targets = &func_ctx.frame_layout.br_table_targets()
                    [*start_index as usize..(*start_index + *len) as usize];

                let index = pass_manager.simulated_stack.pop();
                let index_ty = index.ty(&curr_cursor);
                let mut cases = vec![];
                let target_count = targets.len() - 1;
                let mut default_block = None;

                for (i, target) in targets.iter().enumerate() {
                    let arity = target.arity;
                    let target_index = target.target_index;

                    let start_index = pass_manager.simulated_stack.height() - arity;
                    let mut results = vec![];

                    for j in start_index..pass_manager.simulated_stack.height() {
                        results.push(pass_manager.simulated_stack.stack[j as usize]);
                    }

                    let block = pass_manager
                        .instr_index_to_basic_block
                        .add_branch_to_target(
                            target_index,
                            results,
                            curr_cursor.basic_block(),
                            &mut curr_cursor,
                        )?;

                    if i == target_count {
                        default_block = Some(block);
                    } else {
                        cases.push((
                            curr_cursor
                                .const_literal(i as u32 as i32, OperandTy::Asserted(index_ty))?,
                            block,
                        ));
                    }
                }

                let default_block = default_block.unwrap();

                curr_cursor.build_switch(index, OperandTy::Inferred, default_block, &cases)?;

                let curr_label_end_index = pass_manager.control_stack.curr_label().end_instr_index;

                let (recorded_height, _) =
                    StackInstruction::recorded_height_and_arity_from_end_instruction(
                        curr_label_end_index as u32,
                        func_ctx.instructions,
                    );

                pass_manager.simulated_stack.truncate(recorded_height);

                let (next_block, next_instr_index) = if let Some(if_ctx) =
                    pass_manager.control_stack.try_curr_label_as_if_mut()
                    && !if_ctx.is_else_ongoing
                    && let Some(curr_label_else_index) = if_ctx.else_instr_index
                {
                    // restore the stack with original params
                    let (else_block, params) = pass_manager
                        .instr_index_to_basic_block
                        .remove_else(curr_label_else_index)
                        .expect("hitting this means logic for tracking `else` index is incorrect");

                    for param in params {
                        pass_manager.simulated_stack.push(param);
                    }

                    if_ctx.is_else_ongoing = true;

                    (else_block, curr_label_else_index as usize + 1)
                } else {
                    let (phi_vals, end_block) = pass_manager
                .instr_index_to_basic_block
                .end_phi_vals_and_block(curr_label_end_index as u32)
                .expect("hitting this means logic for tracking target index of labels in lowering is incorrect");

                    for val in phi_vals {
                        pass_manager.simulated_stack.push(*val);
                    }

                    pass_manager.control_stack.leave_label();

                    (end_block, curr_label_end_index + 1)
                };

                return Ok((next_block, next_instr_index));
            }
            StackInstruction::End {
                arity,
                recorded_height,
            } => {
                pass_manager.control_stack.leave_label();

                let curr_basic_block = curr_cursor.basic_block();

                debug_assert!(pass_manager.simulated_stack.height() - recorded_height == *arity);

                // Deepest-first, matching the phi order established by `add_end`.
                let results = pass_manager.simulated_stack.pops_and_reverse(*arity);

                pass_manager.instr_index_to_basic_block.add_end_branch(
                    instr_index as u32,
                    results,
                    curr_basic_block,
                    &mut curr_cursor,
                )?;

                let (phi_vals, end_block) = pass_manager
                    .instr_index_to_basic_block
                    .end_phi_vals_and_block(instr_index as u32)
                    .expect("hitting this means logic for tracking `end` index is incorrect");

                for val in phi_vals {
                    pass_manager.simulated_stack.push(*val);
                }

                // `end_cursor` borrows from `curr_cursor`, so the fall-through jump has
                // to come after the phis are in — which is also why the block being left
                // is closed last rather than first.
                curr_cursor.build_unconditional_br(end_block)?;

                return Ok((end_block, instr_index + 1));
            }
            _ => todo!(),
        };

        Ok((curr_cursor.basic_block(), instr_index + 1))
    }
}
