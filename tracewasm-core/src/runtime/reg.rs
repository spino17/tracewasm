use crate::{
    instruction::{CallerBaseData, RuntimeFrame, register::RegCallerBaseData},
    runtime::{stack::VM_STACK_INITIAL_ALLOCATION_SIZE, value::Value},
};
use smallvec::{SmallVec, smallvec};

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
    type CallerBaseData = RegCallerBaseData;

    fn set_params(&mut self, params: &[super::value::Val]) {
        for (i, param) in params.iter().enumerate() {
            self.inner[i] = param.into();
        }
    }

    fn get_params(
        &mut self,
        params_count: u32,
        caller_base_data: &RegCallerBaseData,
    ) -> SmallVec<[Value; 5]> {
        let base_register_index = caller_base_data.base_offset() as usize;
        let mut s = smallvec![];

        for i in 0..(params_count as usize) {
            s.push(self.inner[base_register_index + i]);
        }

        s
    }

    fn set_results<R: IntoIterator<Item = super::value::Val>>(
        &mut self,
        results: R,
        caller_base_data: &Self::CallerBaseData,
    ) {
        let base_register_index = caller_base_data.base_offset() as usize;

        for (i, res) in results.into_iter().enumerate() {
            self.inner[base_register_index + i] = res.into();
        }
    }

    fn results(&mut self, results_count: u32) -> SmallVec<[Value; 3]> {
        let mut s = smallvec![];

        for i in 0..results_count {
            s.push(self.inner[i as usize]);
        }

        s
    }

    fn reset(&mut self) {}

    fn set_zero_values_in_locals_after_params(
        &mut self,
        params_count: u32,
        locals_ty: &[crate::module::ValType],
        caller_base_data: &RegCallerBaseData,
    ) {
        // base_register_index, base_register_index+1...base_register_index + params_count - 1 is filled with params
        let base_register_index = caller_base_data.base_offset() as usize;
        let locals_len = locals_ty.len();
        let params_count = params_count as usize;

        for i in 0..(locals_len - params_count) {
            let ty = locals_ty[i + params_count];

            self.inner[base_register_index + params_count + i] = Value::zero_of_ty(ty);
        }
    }

    fn tear_callee_frame_and_set_results(
        &mut self,
        results_count: u32,
        caller_base_data: &Self::CallerBaseData,
    ) {
        // memmove from caller_base_data + locals to caller_base_data
        todo!()
    }
}
