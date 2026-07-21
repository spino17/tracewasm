use crate::{
    ast::{FuncIndex, Module},
    error::TraceWasmError,
    instruction::Instruction,
};
use wasmparser::ValType;

#[derive(Debug, Copy, Clone)]
pub enum Val {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
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

    fn zero_of_ty(ty: ValType) -> Result<Self, TraceWasmError> {
        let val = match ty {
            ValType::I32 => Self::i32_zero(),
            ValType::I64 => Self::i64_zero(),
            ValType::F32 => Self::f32_zero(),
            ValType::F64 => Self::f64_zero(),
            _ => return Err(TraceWasmError::Unsupported(format!("type `{}`", ty))),
        };

        Ok(val)
    }

    fn is_ty(&self, ty: ValType) -> bool {
        match ty {
            ValType::I32 => matches!(self, Val::I32(_)),
            ValType::I64 => matches!(self, Val::I64(_)),
            ValType::F32 => matches!(self, Val::F32(_)),
            ValType::F64 => matches!(self, Val::F64(_)),
            _ => false,
        }
    }
}

struct Stack {}

impl Default for Stack {
    fn default() -> Self {
        // should allocate starting size for stack according to WASM spec
        Stack {}
    }
}

struct Memory {}

impl Default for Memory {
    fn default() -> Self {
        // should allocate starting size for memory according to WASM spec
        Memory {}
    }
}

struct Locals {
    inner: Vec<Val>, // size = params + declared locals
}

pub struct TraceVMState {
    stack: Stack,
    memory: Memory,
    locals: Locals,
}

impl TraceVMState {
    fn execute(&mut self, instruction: &Instruction) -> ExecutionResult {
        todo!()
    }
}

enum ExecutionResult {
    NextInstruction(usize),
    End,
}

pub struct TraceVM;

impl TraceVM {
    pub fn execute(
        func_index: FuncIndex,
        params: &[Val],
        module: &Module,
    ) -> Result<Box<[Val]>, TraceWasmError> {
        // if the call is directed to TraceVM's execute then func_index is for a local function
        let imported_func_count = module.imported_func_count;
        let func_body = &module.func_bodies[(func_index.0 - imported_func_count) as usize];
        let instructions = &func_body.instructions;
        let locals_ty = &func_body.locals;

        if params.len() > locals_ty.len() {
            return Err(TraceWasmError::Execution(
                func_index.0,
                "provided params are more than the locals of the function".to_string(),
            ));
        }

        let mut locals: Vec<Val> = Vec::with_capacity(locals_ty.len());

        // set the params in locals
        for i in 0..params.len() {
            debug_assert!(params[i].is_ty(locals_ty[i])); // types should match the value for params!

            locals[i] = params[i];
        }

        // WASM spec tells to set the declared locals with the zero value of their respective type
        for i in params.len()..locals.len() {
            let ty = locals_ty[i];

            locals[i] = Val::zero_of_ty(ty)?;
        }

        let mut state = TraceVMState {
            stack: Stack::default(),
            memory: Memory::default(),
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
