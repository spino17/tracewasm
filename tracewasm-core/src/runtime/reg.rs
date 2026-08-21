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
    /// The spill area, one region per live frame, based at
    /// [`RegCallerBaseData::spills_base_index`].
    ///
    /// Kept apart from `inner` so a spill index stays independent of a register
    /// index. The cost is that its base is this vector's length rather than a
    /// position derived from the caller, so a frame's region is only released by
    /// `exit_frame` truncating to the recorded base.
    pub spills: Vec<Value>,
}

impl Default for RegFrame {
    fn default() -> Self {
        RegFrame {
            registers: Vec::with_capacity(VM_STACK_INITIAL_ALLOCATION_SIZE),
            spills: vec![],
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
        self.spills.clear();
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
            base_register_index + locals_count + frame_layout.registers as usize;

        if self.registers.len() < total_register_capacity {
            self.registers
                .resize(total_register_capacity, Value::default());
        }

        for i in 0..(locals_count - params_count) {
            let ty = locals_ty[i + params_count];

            self.registers[base_register_index + params_count + i] = Value::zero_of_ty(ty);
        }

        let spills_base_index = self.spills.len();

        caller_base_data.set_spills_base_index(spills_base_index as u32);

        self.spills.resize(
            spills_base_index + frame_layout.spills as usize,
            Value::default(),
        );

        caller_base_data.callee_frame_base_register_index =
            caller_base_data.base_register_index + locals_ty.len() as u32;
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
    fn exit_frame(&mut self, results_count: u32, caller_base_data: &RegCallerBaseData) {
        let mut temp: SmallVec<[Value; 3]> = smallvec![];
        let base_register_index = caller_base_data.base_register_index as usize;
        let callee_frame_base_index = caller_base_data.callee_frame_base_register_index as usize;

        for i in 0..results_count as usize {
            temp.push(self.registers[callee_frame_base_index + i]);
        }

        for i in 0..results_count as usize {
            self.registers[base_register_index + i] = temp[i];
        }

        self.spills.resize(
            caller_base_data.spills_base_index as usize,
            Value::default(),
        );
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

        let mut caller_base_data = RegCallerBaseData {
            base_register_index: base,
            callee_frame_base_register_index: u32::MAX,
            spills_base_index: u32::MAX,
        };

        frame.enter_frame(
            params.len() as u32,
            &locals_ty,
            &mut caller_base_data,
            &layout(registers),
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
            callee_frame_base_register_index: u32::MAX,
            spills_base_index: u32::MAX,
        };

        let mut frame_layout = layout(registers);

        frame_layout.spills = spills;

        frame.enter_frame(
            params.len() as u32,
            &locals_ty,
            &mut caller_base_data,
            &frame_layout,
        );

        caller_base_data
    }

    #[test]
    fn a_frames_spill_region_is_appended_and_its_base_recorded() {
        let mut frame = RegFrame::default();
        let base_data = enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 2);

        assert_eq!(
            base_data.spills_base_index, 0,
            "the first frame's spills start at 0"
        );
        assert_eq!(frame.spills.len(), 2, "its two slots are appended");
    }

    #[test]
    fn a_nested_frames_spills_sit_above_its_callers() {
        let mut frame = RegFrame::default();
        let caller = enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 2);
        let callee = enter_full(&mut frame, 2, &[ValType::I32], &[], 1, 3);

        assert_eq!(caller.spills_base_index, 0);

        assert_eq!(
            callee.spills_base_index, 2,
            "the callee's region starts where the caller's ended"
        );

        assert_eq!(frame.spills.len(), 5, "2 + 3 slots are live at this depth");
    }

    /// Unlike the register file — whose base comes from the caller's position and so
    /// reuses space naturally — a spill base is `spills.len()`, which only moves up.
    /// Truncating on exit is therefore what keeps a loop of calls from growing it
    /// without bound.
    #[test]
    fn exiting_a_frame_releases_its_spill_region() {
        let mut frame = RegFrame::default();
        let caller = enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 2);
        let callee = enter_full(&mut frame, 2, &[ValType::I32], &[], 1, 3);

        assert_eq!(frame.spills.len(), 5);

        // a callee with no results, so only the spill half of the teardown runs
        frame.exit_frame(0, &callee);

        assert_eq!(
            frame.spills.len(),
            2,
            "the callee's region is released, the caller's survives"
        );

        frame.exit_frame(0, &caller);

        assert_eq!(frame.spills.len(), 0, "and the caller's in turn");
    }

    #[test]
    fn repeated_calls_at_the_same_depth_reuse_the_same_spill_base() {
        let mut frame = RegFrame::default();
        let caller = enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 1);
        let mut bases = vec![];

        for _ in 0..4 {
            let callee = enter_full(&mut frame, 2, &[ValType::I32], &[], 1, 2);

            bases.push(callee.spills_base_index);
            frame.exit_frame(0, &callee);
        }

        assert!(
            bases.windows(2).all(|w| w[0] == w[1]),
            "every call at this depth must get the same base, not a fresh one: {bases:?}"
        );

        assert_eq!(frame.spills.len(), 1, "only the caller's region is left");

        let _ = caller;
    }

    #[test]
    fn a_frame_with_no_spills_still_records_a_base() {
        let mut frame = RegFrame::default();
        let base_data = enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 0);

        assert_eq!(
            base_data.spills_base_index, 0,
            "the base must be recorded even when the region is empty, since \
             `exit_frame` truncates to it"
        );

        assert_eq!(frame.spills.len(), 0);
    }

    #[test]
    fn reset_clears_the_spill_area_too() {
        let mut frame = RegFrame::default();

        enter_full(&mut frame, 0, &[ValType::I32], &[], 1, 3);

        assert_eq!(frame.spills.len(), 3);

        // a trap unwinds without reaching `exit_frame`, so the next call's reset is
        // what releases the region — otherwise every spill base above it shifts up
        // for the life of the instance
        frame.reset();

        assert_eq!(frame.spills.len(), 0, "spills are released");
        assert_eq!(frame.registers.len(), 0, "and the register file with them");
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
