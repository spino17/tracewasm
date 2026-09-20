use crate::{
    VirtualMachine,
    instruction::Instruction,
    module::{FuncIndex, Module, ValType},
    runtime::stack::Stack,
};
use rustc_hash::FxHashMap;
use std::{
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
    value::Value,
};

pub(crate) struct EndBasicBlockBranches {
    pub(crate) basic_block: BasicBlockId,
    pub(crate) arity: u32,
    pub(crate) phi_vals: Vec<Value>,
    pub(crate) phi_handlers: Vec<PhiInstrHandler>,
}

#[derive(Default)]
pub(crate) struct InstrIndexToBasicBlockMap {
    end_map: FxHashMap<u32, EndBasicBlockBranches>,
    else_map: FxHashMap<u32, (BasicBlockId, Vec<Value>)>,
    // loop_map: FxHashMap<u32, Vec<PhiInstrHandler>>,
}

impl InstrIndexToBasicBlockMap {
    pub fn add_end(&mut self, index: u32, arity: u32, block: BasicBlockId) {
        self.end_map.insert(
            index,
            EndBasicBlockBranches {
                basic_block: block,
                arity,
                phi_vals: vec![],
                phi_handlers: vec![],
            },
        );
    }

    pub fn add_branch_to_end(
        &mut self,
        index: u32,
        values: Vec<Value>,
        block: BasicBlockId,
        ctx: &mut Context,
    ) -> Result<(), PhiError> {
        let end_data = self
            .end_map
            .get_mut(&index)
            .expect("this method should only be called after calling `add_end`");

        if values.len() != end_data.arity as usize {
            panic!("values passed to `end` should match the expected arity")
        }

        let mut end_cursor = ctx.cursor_at_block(end_data.basic_block);

        if end_data.phi_vals.is_empty() && end_data.arity != 0 {
            let mut phi_vals = vec![];
            let mut phi_handlers = vec![];

            for value in &values {
                let (phi_handler, phi_val) =
                    end_cursor.build_phi(&[(block, value.clone())], RegName::Unnamed)?;

                phi_vals.push(phi_val);
                phi_handlers.push(phi_handler);
            }

            end_data.phi_vals = phi_vals;
            end_data.phi_handlers = phi_handlers;
        } else {
            for (i, value) in values.iter().enumerate() {
                let phi_handler = end_data.phi_handlers[i];

                phi_handler.add_branch((block, value.clone()), ctx)?;
            }
        }

        Ok(())
    }

    pub fn phi_vals_and_branch_for_end(&self, index: u32) -> Option<(&[Value], BasicBlockId)> {
        self.end_map
            .get(&index)
            .map(|x| (x.phi_vals.as_slice(), x.basic_block))
    }

    pub fn add_else(&mut self, index: u32, block: BasicBlockId, params: Vec<Value>) {
        self.else_map.insert(index, (block, params));
    }

    pub fn take_else_data(&mut self, index: u32) -> Option<(BasicBlockId, Vec<Value>)> {
        self.else_map.remove(&index)
    }

    pub fn take_end_data(&mut self, index: u32) -> Option<EndBasicBlockBranches> {
        self.end_map.remove(&index)
    }

    /// The block a label's `end` was opened with, without consuming its branches.
    ///
    /// [`take_end_data`](Self::take_end_data) is for the `end` itself, which is done
    /// with the entry; this is for everything that has to *jump* there while the label
    /// is still open — the then-arm falling into an `else`, and every `br` inside.
    pub fn end_basic_block(&self, index: u32) -> BasicBlockId {
        self.end_map
            .get(&index)
            .expect("this method should only be called after calling `add_end`")
            .basic_block
    }
}

pub(crate) struct SimulatedStack {
    stack: Stack<Value>,
}

impl Default for SimulatedStack {
    fn default() -> Self {
        SimulatedStack {
            stack: Stack::new_with_capacity(0),
        }
    }
}

impl Deref for SimulatedStack {
    type Target = Stack<Value>;

    fn deref(&self) -> &Self::Target {
        &self.stack
    }
}

impl DerefMut for SimulatedStack {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stack
    }
}

#[derive(Default)]
pub struct WasmInstrLLVMPassManager {
    declared_funcs: FxHashMap<FuncIndex, GlobalId<DeclaredFunc>>,
    defined_funcs: FxHashMap<FuncIndex, GlobalId<DefinedFunc>>,
    pub(crate) instr_index_to_bb: InstrIndexToBasicBlockMap,
    pub(crate) simulated_stack: SimulatedStack,
}

impl WasmInstrLLVMPassManager {
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
                Self::llvm_signature_from_wasm(params, results, &mut builder)?;

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
                let mut counter = 0;

                for param in llvm_params {
                    llvm_param_decls.push((param, RegName::Named(format!("param{}", counter))));

                    counter += 1;
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
        let results_count = module.types[module.func_decls[func_index.0 as usize].ty.0 as usize]
            .results
            .len();

        let entry = func.add_basic_block("entry", ctx)?;
        let params = func.params(ctx).to_vec();
        let runtime_ctx_ptr = params
            .last()
            .expect("the context is always a function param")
            .clone();
        let mut entry_cursor = ctx.cursor_at_block(entry);

        let mut counter = 0;
        let mut locals = vec![];

        for (i, local_ty) in local_types.iter().enumerate() {
            // last param is runtime ctx!
            let (ptr, val, alignment) = if i < params.len() - 1 {
                let param = &params[i];
                let ty = param.ty();
                let alignment = ty.alignment(&entry_cursor);

                (
                    entry_cursor.build_alloca(
                        ty,
                        None,
                        alignment,
                        RegName::Named(format!("local{}", counter)),
                    )?,
                    param.clone(),
                    alignment,
                )
            } else {
                let ty = Self::llvm_ty_from_wasm(local_ty, &mut entry_cursor);
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

            locals.push(ptr.clone());
            entry_cursor.build_store(&ptr, &val, OperandTy::Inferred, alignment)?;

            counter += 1;
        }

        let func_end = func.add_basic_block("end", ctx)?;

        self.instr_index_to_bb.add_end(
            instructions.len() as u32 - 1,
            results_count as u32,
            func_end,
        );

        let mut cursor = ctx.cursor_at_block(entry);

        for (instr_index, instr) in instructions.iter().enumerate() {
            // match on the instr!
            // for simple instructions, map it to LLVM instruction
            // for branching instructions like if-else
            let next_block = instr.emit_llvm_ir(
                instr_index,
                cursor,
                instructions,
                &locals,
                &runtime_ctx_ptr,
                func,
                self,
            )?;

            cursor = ctx.cursor_at_block(next_block);
        }

        Ok(())
    }

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

    fn llvm_signature_from_wasm(
        params: &[ValType],
        results: &[ValType],
        ctx: &mut Context,
    ) -> Result<(Vec<TyId>, TyId), anyhow::Error> {
        let mut llvm_params = vec![];

        for param_ty in params {
            llvm_params.push(Self::llvm_ty_from_wasm(param_ty, ctx));
        }

        llvm_params.push(ctx.ptr_ty()); // pointer to runtime struct containing mmap memory pointers etc.

        // The runtime pointer goes on the *end* so a wasm local index and its LLVM
        // parameter index stay equal.
        let llvm_result = match results {
            // Wasm spells "returns nothing" as an empty result list; LLVM spells it
            // `void`, so this arm is not the degenerate case it looks like — it is
            // every function rustc emits for a unit return.
            [] => ctx.void_ty(),
            [result] => Self::llvm_ty_from_wasm(result, ctx),
            _ => {
                let mut fields = vec![];

                for result in results {
                    fields.push(Self::llvm_ty_from_wasm(result, ctx));
                }

                ctx.struct_ty(&fields, false)?
            }
        };

        Ok((llvm_params, llvm_result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::stack::StackInstruction;
    use tracewasm_llvm::{
        cfg::{
            builder::Builder,
            emit::IREmitter,
            module::{DataLayout, Triple},
        },
        instruction::cursor::OperandTy,
        value::NullPtr,
    };

    /// The pass is not drivable yet — `compile_func` ends in `todo!()` and every
    /// non-control operator hits the catch-all arm — so these drive `emit_llvm_ir`
    /// over a hand-written instruction slice instead, standing in for the operand
    /// producers by pushing onto the simulated stack between calls. That is enough to
    /// pin the three control arms: the condition's comparison, the fall-through
    /// terminators, and the phis at the `end`.
    fn harness() -> (Builder, GlobalId<DefinedFunc>, BasicBlockId, Value) {
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
        mut between: impl FnMut(&mut WasmInstrLLVMPassManager, &mut Builder, usize),
    ) -> BasicBlockId {
        let mut block = entry;
        let null_ptr = Value::from_const(NullPtr, OperandTy::Inferred, builder).unwrap();

        for (index, instr) in instructions.iter().enumerate() {
            let cursor = builder.cursor_at_block(block);

            // No locals: these cases are about control flow, and none of the operators
            // under test reads a local slot.
            block = instr
                .emit_llvm_ir(index, cursor, instructions, &[], &null_ptr, func, pass)
                .unwrap();

            between(pass, builder, index);
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
            .build_ret(Some(&result), OperandTy::Inferred)
            .unwrap();

        let ir = IREmitter::emit(builder.build()).unwrap();

        // The condition is compared, not relabelled.
        assert!(ir.contains("icmp ne i32 %n, 0"), "{ir}");
        assert!(
            ir.contains("br i1 %0, label %if0_then, label %if0_else"),
            "{ir}"
        );

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
            .build_ret(Some(&result), OperandTy::Inferred)
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
            .build_ret(Some(&top), OperandTy::Inferred)
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
            .trim()
            .split_whitespace()
            .next()
            .expect("a phi line starts with the register it defines");

        assert!(
            ir.contains(&format!("ret i32 {reg}")),
            "expected the returned value to be {reg}\n{ir}"
        );
    }
}
