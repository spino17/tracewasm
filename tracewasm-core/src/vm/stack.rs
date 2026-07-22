use crate::{ast::FuncIndex, error::TraceWasmError};
use wasmparser::ValType;

pub const VM_STACK_INITIAL_ALLOCATION_SIZE: usize = 512 * 1024; // 512Kib

#[derive(Debug, Copy, Clone)]
pub(crate) enum Val {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Ref(Option<FuncIndex>),
}

impl Val {
    pub fn i32_zero() -> Self {
        Val::I32(0)
    }

    pub fn i64_zero() -> Self {
        Val::I64(0)
    }

    pub fn f32_zero() -> Self {
        Val::F32(0.0)
    }

    pub fn f64_zero() -> Self {
        Val::F64(0.0)
    }

    pub fn ref_zero() -> Self {
        Val::Ref(None)
    }

    pub fn zero_of_ty(ty: ValType) -> Result<Self, TraceWasmError> {
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

    pub fn is_ty(&self, ty: ValType) -> Result<bool, TraceWasmError> {
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

pub(crate) struct Locals {
    inner: Vec<Val>, // size = params + declared locals
}

impl Locals {
    pub fn new(locals: Vec<Val>) -> Self {
        Locals { inner: locals }
    }

    pub fn set(&mut self, index: usize, val: Val) {
        self.inner[index] = val;
    }

    pub fn get(&self, index: usize) -> Val {
        self.inner[index]
    }
}

pub(crate) struct Stack {
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
    pub fn push(&mut self, val: Val) {
        if self.stack_pointer < self.inner.len() {
            self.inner[self.stack_pointer] = val;
        } else {
            self.inner.push(val);
        }

        self.stack_pointer += 1;
    }

    pub fn pop(&mut self) -> Val {
        let val = self.inner[self.stack_pointer - 1];
        self.stack_pointer -= 1;

        val
    }

    pub fn truncate(&mut self, new_height: usize) {
        self.stack_pointer = new_height;
    }

    pub fn truncate_by_preserving_arity(&mut self, new_height: usize, arity: u32) {
        let arity = arity as usize;

        for i in 0..arity {
            self.inner[new_height + arity as usize - 1 - i] =
                self.inner[self.stack_pointer as usize - 1 - i];
        }

        self.stack_pointer = new_height + arity;
    }
}
