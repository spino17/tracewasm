use crate::{
    ast::{FuncIndex, Module},
    error::TraceWasmError,
    instruction::Instruction,
    memory::Memory,
    vm::stack::{Locals, Stack, Val},
};

pub mod stack;

pub struct TraceVMState<'a, M> {
    stack: Stack,
    memory: &'a mut M,
    locals: Locals,
}

impl<'a, M: Memory> TraceVMState<'a, M> {
    fn execute(&mut self, instruction: &Instruction) -> ExecutionResult {
        // match on instruction
        // mutate the state of stack, memory and locals
        // and return the next pc or end the execution
        todo!()
    }
}

enum ExecutionResult {
    NextInstruction(usize),
    End,
}

pub struct TraceVM;

impl TraceVM {
    /// Stateless top-level API of the TraceWasm VM to execute a functio of the WASM module.
    pub(crate) fn execute<M: Memory>(
        func_index: FuncIndex,
        params: &[Val],
        module: &Module,
        memory: &mut M,
    ) -> Result<Box<[Val]>, TraceWasmError> {
        // if the call is directed to TraceVM's execute then func_index is for a local function
        let imported_func_count = module.imported_func_count;
        let func_decl = &module.func_decls[func_index.0 as usize];
        let ty = &module.types[func_decl.ty_index.0 as usize];
        let params_ty = &ty.params;
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

            match state.execute(instr) {
                ExecutionResult::NextInstruction(next_pc) => {
                    pc = next_pc;

                    continue;
                }
                ExecutionResult::End => break,
            }
        }

        // at this point, stack would have result values!
        // pop all the values of the stack and return

        todo!()
    }
}
