//! The register machine's frame store — the counterpart of
//! [`Stack<Value>`](crate::runtime::stack::Stack).
//!
//! A flat register file rather than a stack: every access is an index relative to
//! the activation's base, taken from
//! [`CallerBaseData::base_offset`],
//! and nothing is consumed by reading it. That is the opposite of the stack
//! machine's convention on both counts, which the
//! [`RuntimeFrame`] trait docs spell out.
//!
//! Two regions, each a stack of per-frame slices with its own base:
//!
//! * the **register file**, holding a frame's params, declared locals and operand
//!   registers contiguously. Its base comes from the caller's position, so the
//!   space a returning frame occupied is reused without being reclaimed.
//! * the **spill area**, holding the locals and globals a frame materialised ahead
//!   of a write (see [`lazy`](crate::instruction::register::lazy)). Its base is the
//!   area's current length, so unlike the register file it *has* to be truncated on
//!   frame exit — nothing else would ever hand a slot back.
//!
//! **Unfinished.** The register backend is not yet executable: `RegInstruction`'s
//! `execute` is unimplemented, so nothing yet reads a spill slot or a register
//! back out.

use crate::{
    instruction::{
        CallerBaseData, RuntimeFrame,
        register::{RegBrTableTarget, RegCallerBaseData, RegFrameLayout},
    },
    runtime::{stack::VM_STACK_INITIAL_ALLOCATION_SIZE, value::Value},
};
use smallvec::{SmallVec, smallvec};

/// One instance's value storage, shared by every activation in a call chain.
///
/// Both regions are sized by [`RuntimeFrame::enter_frame`] from the callee's
/// [`RegFrameLayout`], and both are emptied by [`RuntimeFrame::reset`] at the start
/// of every call — which is what makes an instance reusable after a trap, since a
/// trap unwinds without reaching [`RuntimeFrame::exit_frame`].
pub(crate) struct RegFrame {
    /// The register file. Register `n` of an activation based at `b` is
    /// `inner[b + n]`, where the frame spans its params, its declared locals, and
    /// then the layout's operand registers.
    ///
    /// `reset` must clear it, not merely truncate to a base:
    /// [`RuntimeFrame::set_initial_params`] positions the entry function's params
    /// by `push`, so they only land at register 0 — where
    /// [`CallerBaseData::initial_data`] says they are — while this is empty.
    pub registers: Vec<Value>,
}

impl Default for RegFrame {
    fn default() -> Self {
        RegFrame {
            registers: Vec::with_capacity(VM_STACK_INITIAL_ALLOCATION_SIZE),
        }
    }
}

impl RuntimeFrame for RegFrame {
    type CallerBaseData = RegCallerBaseData;
    type BrTableTarget = RegBrTableTarget;
    type FrameLayout = RegFrameLayout;

    fn set_initial_params(&mut self, params: &[super::value::Val]) {
        for param in params {
            self.registers.push(param.into());
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
            s.push(self.registers[base_register_index + i]);
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
            self.registers[base_register_index + i] = res.into();
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
            s.push(self.registers[i as usize]);
        }

        s
    }

    /// Empties both regions, so the next call starts from a base of 0 in each.
    ///
    /// Needed on every call, not only after a trap: `set_initial_params` pushes, so
    /// the entry frame's params land at register 0 only while the file is empty.
    /// The spill area additionally *has* to be cleared here, because a trap returns
    /// without reaching `exit_frame` and its base is a length — an orphaned region
    /// would push every later frame's base up for the life of the instance.
    fn reset(&mut self) {
        self.registers.clear();
    }

    fn enter_frame(
        &mut self,
        params_count: u32,
        locals_ty: &[crate::module::ValType],
        caller_base_data: &mut RegCallerBaseData,
        frame_layout: &RegFrameLayout,
    ) {
        // base_register_index, base_register_index+1...base_register_index + params_count - 1 is filled with params
        let base_register_index = caller_base_data.base_offset() as usize;
        let locals_count = locals_ty.len();
        let params_count = params_count as usize;
        let total_register_capacity =
            base_register_index + (frame_layout.registers + frame_layout.spills) as usize;

        if self.registers.len() < total_register_capacity {
            self.registers
                .resize(total_register_capacity, Value::default());
        }

        for i in 0..(locals_count - params_count) {
            let ty = locals_ty[i + params_count];

            self.registers[base_register_index + params_count + i] = Value::zero_of_ty(ty);
        }

        debug_assert!(frame_layout.locals_count == locals_ty.len() as u32);
    }

    /// Moves the callee's results down over its locals, so they land where the
    /// caller staged the arguments.
    ///
    /// The two bases bracket the frame. Results are produced in the callee's
    /// *operand* region, which begins at `callee_frame_base_register_index`
    /// (`base + locals`), because
    /// [`RegFrameLayout`] numbers
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
    fn exit_frame(
        &mut self,
        results_count: u32,
        caller_base_data: &RegCallerBaseData,
        frame_layout: &RegFrameLayout,
    ) {
        let mut temp: SmallVec<[Value; 3]> = smallvec![];
        let base_register_index = caller_base_data.base_register_index as usize;
        let callee_frame_base_index = base_register_index + frame_layout.locals_count as usize;

        for i in 0..results_count as usize {
            temp.push(self.registers[callee_frame_base_index + i]);
        }

        for i in 0..results_count as usize {
            self.registers[base_register_index + i] = temp[i];
        }
    }
}

/// Tests for the register file's sizing.
///
/// A frame spans `[base, base + locals_count + registers)` — params, then declared
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
    use std::u32;

    use super::*;
    use crate::{module::ValType, runtime::value::Val};

    /// A layout declaring `registers` operand registers and nothing else. The
    /// arenas are what lowering fills for the *instructions* to index; sizing never
    /// reads them.
    /// `registers` is the frame's **total** width — its locals *and* its operand
    /// registers — because that is what `enter_frame` reserves and what the lowering
    /// records: `max_registers` starts at `locals_count` and counts up from there.
    /// The `enter*` helpers below take an operand count and add the locals, so a test
    /// can still say "this body needs two registers" and mean it.
    fn layout(registers: u32, locals_count: u32) -> RegFrameLayout {
        RegFrameLayout {
            registers,
            locals_count,
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

        let mut caller_base_data = RegCallerBaseData {
            base_register_index: base,
        };

        frame.enter_frame(
            params.len() as u32,
            &locals_ty,
            &mut caller_base_data,
            &layout(locals_ty.len() as u32 + registers, locals_ty.len() as u32),
        );

        frame.registers.len()
    }

    /// [`enter`], handing back the base data so a test can read the bases
    /// `enter_frame` recorded into it, or pass it to `exit_frame`.
    fn enter_full(
        frame: &mut RegFrame,
        base: u32,
        params: &[ValType],
        declared: &[ValType],
        registers: u32,
        spills: u32,
    ) -> RegCallerBaseData {
        let locals_ty: Vec<ValType> = params.iter().chain(declared).copied().collect();

        let mut caller_base_data = RegCallerBaseData {
            base_register_index: base,
        };

        let mut frame_layout = layout(locals_ty.len() as u32 + registers, locals_ty.len() as u32);

        frame_layout.spills = spills;

        frame.enter_frame(
            params.len() as u32,
            &locals_ty,
            &mut caller_base_data,
            &frame_layout,
        );

        caller_base_data
    }

    // -----------------------------------------------------------------------
    // spills, as a region inside the frame
    // -----------------------------------------------------------------------
    //
    // Spills used to be a second `Vec` with its own base, which had to be truncated
    // on exit because that base was a length and only ever moved up. They now sit
    // above the frame's registers in the one file, so the base arithmetic and the
    // truncation are both gone: the next frame at a lower base simply overwrites
    // them, exactly as it does the registers.

    /// The reservation has to cover the spills as well as the registers, or a spill
    /// write lands past the end of the file.
    #[test]
    fn a_frames_spill_region_is_reserved_above_its_registers() {
        let mut frame = RegFrame::default();

        // 1 param, so the layout's total width is 1 local + 1 register = 2, plus 2
        // spill slots above that
        let _ = enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 2);

        assert_eq!(
            frame.registers.len(),
            4,
            "2 for the locals and registers, 2 more for the spills"
        );
    }

    /// A callee based above its caller's frame gets a spill region above its own
    /// registers, and it cannot reach back into the caller's.
    #[test]
    fn a_nested_frames_spills_do_not_collide_with_its_callers() {
        let mut frame = RegFrame::default();

        let caller = enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 2);
        // the caller occupies [0, 2) for locals+registers and [2, 4) for spills
        let callee = enter_full(&mut frame, 4, &[ValType::I32], &[], 1, 3);

        assert_eq!(caller.base_register_index, 0);
        assert_eq!(callee.base_register_index, 4);

        assert_eq!(
            frame.registers.len(),
            4 + 2 + 3,
            "the callee's own registers and spills sit above everything the caller holds"
        );
    }

    /// Repeated calls at one depth reuse the same space rather than growing the file,
    /// which is the property the old spill vector needed an explicit truncation for.
    #[test]
    fn repeated_calls_at_the_same_depth_reuse_the_same_space() {
        let mut frame = RegFrame::default();

        let _caller = enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 1);
        let mut lens = vec![];

        for _ in 0..4 {
            let _ = enter_full(&mut frame, 3, &[ValType::I32], &[], 1, 2);
            lens.push(frame.registers.len());
        }

        assert!(
            lens.windows(2).all(|w| w[0] == w[1]),
            "the file must not grow across calls at one depth: {lens:?}"
        );
    }

    #[test]
    fn reset_clears_the_whole_file() {
        let mut frame = RegFrame::default();

        enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 3);
        assert!(frame.registers.len() >= 4);

        // A trap unwinds without reaching `exit_frame`, so `reset` is what makes the
        // instance reusable — and it must *clear*, not truncate: `set_initial_params`
        // pushes, so the entry frame's params only land at index 0 while it is empty.
        frame.reset();

        assert_eq!(
            frame.registers.len(),
            0,
            "the file is emptied, not truncated"
        );
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
        let operand_base = 2;

        assert!(
            operand_base + 2 <= len,
            "register 1 at index {} must be in bounds of {len}",
            operand_base + 1
        );
    }

    #[test]
    fn params_alone_still_need_room_for_registers() {
        // no declared locals: `locals_count - params_count` is zero, which must not
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

    /// `set_initial_params` positions by `push`, so it lands at `inner.len()` —
    /// but `initial_data` says the entry frame is based at 0. The two only agree
    /// while the file is empty, which is what `reset` has to guarantee on every
    /// call, not just the first.
    #[test]
    fn a_second_call_puts_its_params_back_at_zero() {
        let mut frame = RegFrame::default();

        frame.set_initial_params(&[Val::I32(7)]);

        enter(&mut frame, 0, &[ValType::I32], &[ValType::I32], 2);

        assert_eq!(
            frame.registers[0].as_i32(),
            7,
            "first call's param lands at 0"
        );

        // the driver resets at the start of every call, which is what makes an
        // instance reusable — including after a trap left the file grown
        frame.reset();
        frame.set_initial_params(&[Val::I32(9)]);

        assert_eq!(
            frame.registers[0].as_i32(),
            9,
            "the second call's param must land at 0 too, not above the first call's \
             high-water mark"
        );
    }

    #[test]
    fn declared_locals_are_zeroed_and_params_are_left_alone() {
        // the sibling arithmetic: the zeroing loop runs over `locals_count -
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

        assert_eq!(
            frame.registers[0].as_i64(),
            -9,
            "the param is not overwritten"
        );
        assert_eq!(frame.registers[1].as_i64(), 0, "declared local 0 is zeroed");
        assert_eq!(frame.registers[2].as_i64(), 0, "declared local 1 is zeroed");
    }
}
