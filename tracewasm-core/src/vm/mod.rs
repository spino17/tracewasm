use crate::{
    ast::{FuncIndex, Module},
    error::TraceWasmError,
    instruction::Instruction,
    memory::Memory,
    vm::stack::{Locals, Stack, Val},
};

pub mod stack;

enum ExecutionResult {
    JumpTo(usize),
    Next,
}

pub struct TraceVMState<'a, M> {
    stack: Stack<Val>,
    memory: &'a mut M,
    locals: Locals,
}

impl<'a, M: Memory> TraceVMState<'a, M> {
    fn execute(
        &mut self,
        instruction: &Instruction,
        func_index: FuncIndex,
    ) -> Result<ExecutionResult, TraceWasmError> {
        let res = match instruction {
            // TODO - give complete stack trace! - with range information of the
            // original WASM instruction
            Instruction::Unreachable => {
                return Err(TraceWasmError::Execution(
                    func_index.0,
                    "unreachable panic!".to_string(),
                ));
            }
            Instruction::Nop => ExecutionResult::Next,
            Instruction::Block {
                blockty: _blockty,
                end_index: _end_index,
            } => ExecutionResult::Next,
            Instruction::Loop { blockty: _blockty } => ExecutionResult::Next,
            Instruction::If {
                blockty: _blockty,
                else_index,
                end_index,
            } => {
                let cond = self.stack.pop().as_i32();

                if cond != 0 {
                    ExecutionResult::Next
                } else {
                    if let Some(else_index) = else_index {
                        ExecutionResult::JumpTo(*else_index + 1) // first instruction of the else branch
                    } else {
                        ExecutionResult::JumpTo(*end_index)
                    }
                }
            }
            // this instruction would be encountered only when control flow is coming after completing `if` branch
            // because if the condition was `false` and the control went to `else` branch, it jumps to the first
            // instruction of `else` branch and not the `else` instruction.
            Instruction::Else { if_end_index } => ExecutionResult::JumpTo(*if_end_index),
            Instruction::End {
                arity,
                recorded_height,
            } => {
                debug_assert!(self.stack.height() as u32 == *recorded_height + *arity);
                ExecutionResult::Next
            }
            Instruction::Br {
                target_index,
                arity,
                recorded_height,
            } => {
                self.stack
                    .truncate_by_preserving_arity(*recorded_height, *arity);

                ExecutionResult::JumpTo(*target_index)
            }
            Instruction::BrIf {
                target_index,
                arity,
                recorded_height,
            } => todo!(),
            Instruction::BrTable { targets } => todo!(),
            Instruction::Return {
                target_index,
                arity,
                recorded_height,
            } => todo!(),
            Instruction::Call { func_index } => todo!(),
        };

        Ok(res)
    }
}

pub struct TraceVM;

impl TraceVM {
    /// Stateless top-level API of the TraceWasm VM to execute a local function of the WASM module.
    pub(crate) fn execute<M: Memory>(
        func_index: FuncIndex,
        params: &[Val],
        module: &Module,
        memory: &mut M,
    ) -> Result<Box<[Val]>, TraceWasmError> {
        let imported_func_count = module.imported_func_count;
        let func_decl = &module.func_decls[func_index.0 as usize];
        let ty = &module.types[func_decl.ty_index.0 as usize];
        let params_ty = &ty.params;
        let results_len = ty.results.len() as u32;
        let func_body = &module.func_bodies[(func_index.0 - imported_func_count) as usize];
        let instructions = &func_body.instructions;
        let locals_ty = &func_body.locals;

        if params.len() != params_ty.len() {
            return Err(TraceWasmError::Execution(
                func_index.0,
                format!(
                    "expected params `{}`, got `{}`",
                    params_ty.len(),
                    params.len()
                ),
            ));
        }

        let mut locals: Vec<Val> = Vec::with_capacity(locals_ty.len());

        // set the params in locals
        for i in 0..params.len() {
            debug_assert!(params[i].is_ty(locals_ty[i])?); // types should match the value for params!

            locals.push(params[i]);
        }

        // WASM spec tells to set the declared locals with the zero value of their respective type
        for i in params.len()..locals_ty.len() {
            let ty = locals_ty[i];

            locals.push(Val::zero_of_ty(ty)?);
        }

        let mut state = TraceVMState {
            stack: Stack::default(),
            memory,
            locals: Locals::new(locals),
        };

        let mut pc = 0;

        loop {
            let instr = &instructions[pc];

            match state.execute(instr, func_index)? {
                ExecutionResult::JumpTo(next_pc) => {
                    pc = next_pc;

                    continue;
                }
                ExecutionResult::Next => {
                    pc += 1;

                    if pc == instructions.len() {
                        break;
                    }

                    continue;
                }
            }
        }

        // at this point, stack would have result values!
        // pop all the values of the stack and return
        Ok(state.stack.pops_and_reverse(results_len).into_boxed_slice())
    }
}
