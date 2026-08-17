//! The register machine's frame store — the counterpart of
//! [`Stack<Value>`](crate::runtime::stack::Stack).
//!
//! A flat register file rather than a stack: every access is an index relative to
//! the activation's base, taken from
//! [`CallerBaseData::base_offset`](crate::instruction::CallerBaseData::base_offset),
//! and nothing is consumed by reading it. That is the opposite of the stack
//! machine's convention on both counts, which the
//! [`RuntimeFrame`](crate::instruction::RuntimeFrame) trait docs spell out.
//!
//! **Unfinished.** The register backend is not yet executable, and this file is
//! where that shows: `inner` is never sized (see [`RegFrame`]),
//! `tear_callee_frame_and_set_results` is unimplemented, and `reset` does
//! nothing.

use crate::{
    instruction::{CallerBaseData, RuntimeFrame, register::RegCallerBaseData},
    runtime::{stack::VM_STACK_INITIAL_ALLOCATION_SIZE, value::Value},
};
use smallvec::{SmallVec, smallvec};

/// One instance's register file, shared by every activation in a call chain.
///
/// **`inner` is currently only reserved, never sized.** [`Default`] uses
/// `Vec::with_capacity`, which leaves `len() == 0`, and no method here pushes or
/// resizes — so every indexed write below is out of bounds until something gives
/// the file a length. The stack machine gets away with the same constructor
/// because it reaches its storage through `push`.
pub struct RegFrame {
    /// The register file. Register `n` of an activation based at `b` is
    /// `inner[b + n]`.
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

    /// Reads a finished call's results.
    ///
    /// **Reads from register 0, not from the activation's base**, because the
    /// trait method takes no `CallerBaseData` to offset against — correct only
    /// for the outermost frame.
    fn results(&mut self, results_count: u32) -> SmallVec<[Value; 3]> {
        let mut s = smallvec![];

        for i in 0..results_count {
            s.push(self.inner[i as usize]);
        }

        s
    }

    /// Currently a no-op, so a trapped call's values survive into the next one.
    /// The stack machine's `reset` moves its pointer back to zero.
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
