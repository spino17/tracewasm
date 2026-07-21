use crate::{
    ast::{FuncIndex, Module},
    error::TraceWasmError,
    instruction::Instruction,
    memory::Memory,
};
use wasmparser::ValType;

pub const VM_STACK_INITIAL_ALLOCATION_SIZE: usize = 512 * 1024; // 512Kib

#[derive(Debug, Copy, Clone)]
pub enum Val {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Ref(Option<FuncIndex>),
}

impl Val {
    fn i32_zero() -> Self {
        Val::I32(0)
    }

    fn i64_zero() -> Self {
        Val::I64(0)
    }

    fn f32_zero() -> Self {
        Val::F32(0.0)
    }

    fn f64_zero() -> Self {
        Val::F64(0.0)
    }

    fn ref_zero() -> Self {
        Val::Ref(None)
    }

    fn zero_of_ty(ty: ValType) -> Result<Self, TraceWasmError> {
        let val = match ty {
            ValType::I32 => Self::i32_zero(),
            ValType::I64 => Self::i64_zero(),
            ValType::F32 => Self::f32_zero(),
            ValType::F64 => Self::f64_zero(),
            ValType::Ref(_) => Self::ref_zero(),
            ValType::V128 => return Err(TraceWasmError::Unsupported("v128 type".to_string())),
        };

        Ok(val)
    }

    fn is_ty(&self, ty: ValType) -> Result<bool, TraceWasmError> {
        let val = match ty {
            ValType::I32 => matches!(self, Val::I32(_)),
            ValType::I64 => matches!(self, Val::I64(_)),
            ValType::F32 => matches!(self, Val::F32(_)),
            ValType::F64 => matches!(self, Val::F64(_)),
            ValType::Ref(_) => matches!(self, Val::Ref(_)),
            ValType::V128 => return Err(TraceWasmError::Unsupported("v128 type".to_string())),
        };

        Ok(val)
    }
}

struct Stack {
    inner: Vec<Val>,
    stack_pointer: usize, // points to the top of the stack
}

impl Default for Stack {
    fn default() -> Self {
        Stack {
            inner: Vec::with_capacity(VM_STACK_INITIAL_ALLOCATION_SIZE),
            stack_pointer: 0,
        }
    }
}

impl Stack {
    fn push(&mut self, val: Val) {
        if self.stack_pointer < self.inner.len() {
            self.inner[self.stack_pointer] = val;
        } else {
            // self.stack_pointer == self.inner.len()
            self.inner.push(val);
        }

        self.stack_pointer += 1;
    }

    fn pop(&mut self) -> Val {
        let val = self.inner[self.stack_pointer - 1];
        self.stack_pointer -= 1;

        val
    }

    fn truncate(&mut self, new_height: usize) {
        self.stack_pointer = new_height;
    }
}

struct Locals {
    inner: Vec<Val>, // size = params + declared locals
}

pub struct TraceVMState<'a> {
    stack: Stack,
    memory: &'a mut Memory,
    locals: Locals,
}

impl<'a> TraceVMState<'a> {
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
    pub fn execute(
        func_index: FuncIndex,
        params: &[Val],
        module: &Module,
        memory: &mut Memory,
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
            locals: Locals { inner: locals },
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
