use crate::{
    instruction::RuntimeFrame,
    runtime::{stack::VM_STACK_INITIAL_ALLOCATION_SIZE, value::Value},
};
use smallvec::smallvec;

pub struct RegFrame {
    inner: Vec<Value>,
}

impl Default for RegFrame {
    fn default() -> Self {
        RegFrame {
            inner: Vec::with_capacity(VM_STACK_INITIAL_ALLOCATION_SIZE),
        }
    }
}

impl RuntimeFrame for RegFrame {
    fn set_params(&mut self, params: &[super::value::Val]) {
        for (i, param) in params.iter().enumerate() {
            self.inner[i] = param.into();
        }
    }

    fn results(&mut self, results_count: u32) -> smallvec::SmallVec<[Value; 3]> {
        let mut s = smallvec![];

        for i in 0..results_count {
            s.push(self.inner[i as usize]);
        }

        s
    }

    fn reset(&mut self) {}
}
