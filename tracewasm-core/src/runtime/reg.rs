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
//! where that shows: `inner` is never sized (see [`RegFrame`]) and `reset` does
//! nothing.

use crate::{
    instruction::{
        CallerBaseData, RuntimeFrame,
        register::{RegBrTableTarget, RegCallerBaseData, RegFrameLayout},
    },
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
    type BrTableTarget = RegBrTableTarget;
    type FrameLayout = RegFrameLayout;

    fn set_initial_params(&mut self, params: &[super::value::Val]) {
        for param in params {
            self.inner.push(param.into());
        }
    }

    fn get_params_for_import_call(
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

    fn set_results_from_import_call<R: IntoIterator<Item = super::value::Val>>(
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
    fn get_final_results(&mut self, results_count: u32) -> SmallVec<[Value; 3]> {
        let mut s = smallvec![];

        for i in 0..results_count {
            s.push(self.inner[i as usize]);
        }

        s
    }

    /// Currently a no-op, so a trapped call's values survive into the next one.
    /// The stack machine's `reset` moves its pointer back to zero.
    fn reset(&mut self) {}

    fn enter_frame(
        &mut self,
        params_count: u32,
        locals_ty: &[crate::module::ValType],
        caller_base_data: &RegCallerBaseData,
        frame_layout: &RegFrameLayout,
    ) {
        // base_register_index, base_register_index+1...base_register_index + params_count - 1 is filled with params
        let base_register_index = caller_base_data.base_offset() as usize;
        let locals_len = locals_ty.len();
        let params_count = params_count as usize;
        let total_register_capacity =
            base_register_index + locals_len + frame_layout.registers as usize;

        if self.inner.len() < total_register_capacity {
            self.inner.resize(total_register_capacity, Value::default());
        }

        for i in 0..(locals_len - params_count) {
            let ty = locals_ty[i + params_count];

            self.inner[base_register_index + params_count + i] = Value::zero_of_ty(ty);
        }
    }

    /// Moves the callee's results down over its locals, so they land where the
    /// caller staged the arguments.
    ///
    /// The two bases bracket the frame. Results are produced in the callee's
    /// *operand* region, which begins at `callee_frame_base_register_index`
    /// (`base + locals`), because
    /// [`RegFrameLayout`](crate::instruction::register::RegFrameLayout) numbers
    /// registers from the operand base — register `r` is `inner[operand_base + r]`,
    /// and a body's `end` materialises its results into registers `0..results`.
    /// They are copied to `base_register_index`, which is the absolute position of
    /// the `caller_base` the calling
    /// [`RegInstruction::Call`](crate::instruction::register::RegInstruction::Call)
    /// staged its arguments at — so the caller needs to do nothing when the call
    /// returns.
    ///
    /// Gathered into a temporary before being written back, since the two ranges
    /// overlap whenever the callee has fewer locals than results. The copy only
    /// ever moves *downward* (operands sit above locals), so a plain forward loop
    /// would also be safe today — the temporary keeps that from being load-bearing.
    /// A callee with no locals makes this a self-copy, which is a harmless no-op.
    ///
    /// Nothing is reclaimed: a positional machine has no pointer to move, so the
    /// space above the results is simply reused by the next call.
    fn exit_frame(&mut self, results_count: u32, caller_base_data: &RegCallerBaseData) {
        let mut temp: SmallVec<[Value; 3]> = smallvec![];
        let base_register_index = caller_base_data.base_register_index as usize;
        let callee_frame_base_index = caller_base_data.callee_frame_base_register_index as usize;

        for i in 0..results_count as usize {
            temp.push(self.inner[callee_frame_base_index + i]);
        }

        for i in 0..results_count as usize {
            self.inner[base_register_index + i] = temp[i];
        }
    }
}

/// Tests for the register file's sizing.
///
/// A frame spans `[base, base + locals_len + registers)` — params, then declared
/// locals, then the operand registers numbered from the operand base. Getting that
/// span wrong is not caught by anything else: the lowering never sees the file, and
/// the fault surfaces as an out-of-bounds index deep in an unrelated instruction,
/// or not at all when a deeper call has already grown the file past the mistake.
///
/// So these assert the **length directly** rather than exercising a frame and
/// waiting for a panic. Each also checks the exact length, so over-allocating is a
/// failure too — the file is shared by the whole call chain, and slack in every
/// frame compounds with depth.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{module::ValType, runtime::value::Val};

    /// A layout declaring `registers` operand registers and nothing else. The
    /// arenas are what lowering fills for the *instructions* to index; sizing never
    /// reads them.
    fn layout(registers: u32) -> RegFrameLayout {
        RegFrameLayout {
            registers,
            spills: 0,
            input_registers_arena: Box::new([]),
            output_registers_arena: Box::new([]),
            br_targets_arena: Box::new([]),
        }
    }

    /// Enters a frame based at `base`, and reports the resulting file length.
    fn enter(
        frame: &mut RegFrame,
        base: u32,
        params: &[ValType],
        declared: &[ValType],
        registers: u32,
    ) -> usize {
        let locals_ty: Vec<ValType> = params.iter().chain(declared).copied().collect();

        let caller_base_data = RegCallerBaseData {
            base_register_index: base,
            callee_frame_base_register_index: u32::MAX,
        };

        frame.enter_frame(
            params.len() as u32,
            &locals_ty,
            &caller_base_data,
            &layout(registers),
        );

        frame.inner.len()
    }

    #[test]
    fn a_frame_reaches_its_highest_register() {
        // the shape that caught the off-by-`params_count`: one param, one declared
        // local, two registers — so the file must reach index 3
        let mut frame = RegFrame::default();

        frame.set_initial_params(&[Val::I32(7)]);

        let len = enter(&mut frame, 0, &[ValType::I32], &[ValType::I32], 2);

        assert_eq!(len, 4, "1 param + 1 local + 2 registers");

        // the operand base is above the locals, so the last register is the last slot
        let operand_base = 0 + 2;

        assert!(
            operand_base + 2 <= len,
            "register 1 at index {} must be in bounds of {len}",
            operand_base + 1
        );
    }

    #[test]
    fn params_alone_still_need_room_for_registers() {
        // no declared locals: `locals_len - params_count` is zero, which must not
        // underflow and must not shorten the frame
        let mut frame = RegFrame::default();

        frame.set_initial_params(&[Val::I32(1), Val::I32(2)]);

        let len = enter(&mut frame, 0, &[ValType::I32, ValType::I32], &[], 3);

        assert_eq!(len, 5, "2 params + 0 locals + 3 registers");
    }

    #[test]
    fn a_frame_with_no_registers_is_sized_for_its_locals() {
        let mut frame = RegFrame::default();

        let len = enter(
            &mut frame,
            0,
            &[ValType::I64],
            &[ValType::I64, ValType::I64],
            0,
        );

        assert_eq!(len, 3, "1 param + 2 locals + 0 registers");
    }

    #[test]
    fn a_nested_frame_is_sized_from_its_own_base() {
        // a callee based at 4 needs the file to reach 4 + locals + registers, not
        // just its own span — the caller's frame below it stays live
        let mut frame = RegFrame::default();

        let len = enter(&mut frame, 4, &[ValType::I32], &[ValType::I32], 2);

        assert_eq!(len, 8, "base 4 + 1 param + 1 local + 2 registers");
    }

    #[test]
    fn a_shallow_frame_does_not_truncate_the_file_above_it() {
        // A deep call grows the file; returning to a shallow frame must not shrink
        // it, or the values of every frame in between would be dropped. The guard
        // in `enter_frame` is what prevents that, and this is the case that would
        // catch its removal.
        let mut frame = RegFrame::default();

        let deep = enter(&mut frame, 16, &[ValType::I32], &[ValType::I32], 4);

        assert_eq!(deep, 22);

        let shallow = enter(&mut frame, 0, &[ValType::I32], &[ValType::I32], 2);

        assert_eq!(
            shallow, deep,
            "re-entering a frame at base 0 must leave the file at its high-water mark"
        );
    }

    #[test]
    fn declared_locals_are_zeroed_and_params_are_left_alone() {
        // the sibling arithmetic: the zeroing loop runs over `locals_len -
        // params_count` slots starting *after* the params, so a param must survive
        // frame entry while a declared local must come out zero
        let mut frame = RegFrame::default();

        frame.set_initial_params(&[Val::I64(-9)]);

        enter(
            &mut frame,
            0,
            &[ValType::I64],
            &[ValType::I64, ValType::I64],
            1,
        );

        assert_eq!(frame.inner[0].as_i64(), -9, "the param is not overwritten");
        assert_eq!(frame.inner[1].as_i64(), 0, "declared local 0 is zeroed");
        assert_eq!(frame.inner[2].as_i64(), 0, "declared local 1 is zeroed");
    }
}
