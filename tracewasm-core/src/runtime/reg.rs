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
//! # A frame's four regions
//!
//! One `Vec<Value>` holds everything, and a frame is four consecutive regions of
//! it. Their order is what makes the layout work, so it is fixed:
//!
//! ```text
//!   base + 0                                  locals — params first, then declared
//!   base + locals_count                       constants — the body's interned pool
//!   base + locals_count + consts              spills
//!   base + locals_count + consts + spills     operand registers
//!   base + registers + consts + spills        end of frame
//! ```
//!
//! An operand carries one absolute frame index and no tag, so a read is
//! `registers[base + index]` — a single add, whichever region the index falls in.
//! Only the region *sizes* say which one that is, which is what
//! [`Slot::render`](crate::instruction::register::Slot) consults and what nothing on
//! the execution path needs to.
//!
//! Note that [`RegFrameLayout::registers`] counts from the frame base and so
//! **includes the locals**: a frame is `registers + spills + consts` slots wide, not
//! `locals + registers + …`.
//!
//! # Why constants and spills sit below the registers
//!
//! A callee is based at its caller's `caller_base`, which is an *operand register*
//! index — so the callee's frame begins at or above the start of the caller's
//! operand region. Everything the caller needs to survive the call therefore has to
//! live below that point:
//!
//! * its **locals**, which the body reads after the call returns;
//! * its **constants**, which are read wherever they appear in the body;
//! * its **spills**, whose entire purpose is holding a local or global that a later
//!   write or call would otherwise have clobbered (see
//!   [`lazy`](crate::instruction::register::lazy)).
//!
//! Ordering them below the operand registers makes that hold by construction rather
//! than by arithmetic: no callee frame can reach them, because no `caller_base` is
//! that low.
//!
//! It also keeps a call free of copies. The caller stages the arguments at
//! `caller_base` and they *are* the callee's first locals, so
//! [`RuntimeFrame::enter_frame`] leaves them where they are — it only zeroes the
//! declared locals above them and writes the constant pool above those.
//!
//! # Reuse without reclamation
//!
//! A frame's base comes from the caller's position, not from the file's length, so
//! the space a returning frame occupied is reused by the next call at that depth
//! without anything being truncated. [`RuntimeFrame::exit_frame`] therefore only
//! moves results down; it does not shrink the file.

use crate::{
    instruction::{
        CallerBaseData, RuntimeFrame,
        register::{Const, RegBrTableTarget, RegCallerBaseData, RegFrameLayout},
    },
    runtime::{stack::VM_STACK_INITIAL_ALLOCATION_SIZE, value::Value},
};
use smallvec::{SmallVec, smallvec};

/// One instance's value storage, shared by every activation in a call chain.
///
/// Sized by [`RuntimeFrame::enter_frame`] from the callee's [`RegFrameLayout`], and
/// emptied by [`RuntimeFrame::reset`] at the start of every call — which is what
/// makes an instance reusable after a trap, since a trap unwinds without reaching
/// [`RuntimeFrame::exit_frame`].
pub(crate) struct RegFrame {
    /// Every live value in the call chain, one frame's four regions after another's.
    ///
    /// Slot `n` of an activation based at `b` is `registers[b + n]`, where `n` ranges
    /// over the frame's locals, constants, spills and operand registers in that
    /// order — see the module docs for the boundaries.
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

    /// Stages the entry function's arguments, which `reset` has just made slots
    /// `0..params.len()` — where [`CallerBaseData::initial_data`] says the outermost
    /// frame is based.
    fn set_initial_params(&mut self, params: &[super::value::Val]) {
        for param in params {
            self.registers.push(param.into());
        }
    }

    /// Reads an imported callee's arguments, which the caller staged at
    /// `caller_base` exactly as it would for a local call.
    ///
    /// Indexed from the frame *base*, not the operand region: the arguments are the
    /// callee's first locals, so they sit at the bottom of its frame.
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

    /// Writes a host function's results back over the arguments it was given.
    ///
    /// The same slots [`Self::exit_frame`] would have moved a local callee's results
    /// down to, so the caller reads an imported call's results the same way.
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

    /// Empties the file, so the next call starts from a base of 0.
    ///
    /// Needed on every call, not only after a trap: `set_initial_params` pushes, so
    /// the entry frame's params land at register 0 only while the file is empty.
    fn reset(&mut self) {
        self.registers.clear();
    }

    /// Prepares the callee's frame: sizes the file for it, zeroes its declared
    /// locals, and materialises its constant pool.
    ///
    /// The arguments are **not** touched. The caller staged them contiguously at
    /// `caller_base`, which is this frame's base, so they are already the callee's
    /// params — slots `0..params_count`. That is what makes a call copy-free, and it
    /// is why the zeroing loop starts at `params_count` rather than 0.
    ///
    /// Above the locals comes the constant pool. It is rewritten on every entry
    /// because these are frame slots like any other: the body may reuse the region
    /// across a recursive call, and a previous activation at this depth left its own
    /// values behind.
    ///
    /// The spill region needs nothing here — a spill is written before it is read, by
    /// the [`LocalSpill`](crate::instruction::register::RegInstruction::LocalSpill) or
    /// [`GlobalSpill`](crate::instruction::register::RegInstruction::GlobalSpill) that
    /// the lowering placed above every path that reads it — so entry only has to
    /// reserve room for it.
    fn enter_frame(
        &mut self,
        params_count: u32,
        locals_ty: &[crate::module::ValType],
        caller_base_data: &mut RegCallerBaseData,
        frame_layout: &RegFrameLayout,
    ) {
        let base_register_index = caller_base_data.base_offset() as usize;
        let locals_count = locals_ty.len();
        let params_count = params_count as usize;

        // `registers` counts from the frame base, so it already covers the locals;
        // the two other regions are what widen the frame beyond it.
        let total_register_capacity = base_register_index
            + (frame_layout.registers + frame_layout.spills) as usize
            + frame_layout.consts.len();

        // Only grow. A deeper call may have pushed the file past this frame's end,
        // and those slots belong to frames still live below the deepest one.
        if self.registers.len() < total_register_capacity {
            self.registers
                .resize(total_register_capacity, Value::default());
        }

        // Wasm requires declared locals to start at zero, and the slots may hold
        // whatever a previous activation at this depth left in them.
        for i in 0..(locals_count - params_count) {
            let ty = locals_ty[i + params_count];

            self.registers[base_register_index + params_count + i] = Value::zero_of_ty(ty);
        }

        // Constant `i` is at `locals_count + i`, matching the frame index the
        // lowering's constant backpatch resolved each operand to.
        for i in 0..frame_layout.consts.len() {
            let const_val = frame_layout.consts[i];

            self.registers[base_register_index + locals_count + i] = match const_val {
                Const::I32(val) => Value::from_i32(val),
                Const::I64(val) => Value::from_i64(val),
                Const::F32(val) => Value::from_f32(val.into()),
                Const::F64(val) => Value::from_f64(val.into()),
                Const::Ref(r) => Value::from_ref(r),
            };
        }
    }

    /// Moves the callee's results down to its frame base, so they land where the
    /// caller staged the arguments.
    ///
    /// A body's `end` materialises its results into its first operand registers, so
    /// they are read from the top region — `base + locals + consts + spills` — and
    /// written to `base`, the absolute position the calling
    /// [`RegInstruction::Call`](crate::instruction::register::RegInstruction::Call)
    /// staged its arguments at. That is why the caller needs to do nothing when a
    /// call returns: the results are already in the slots it will read them from.
    ///
    /// The layout is needed to locate the operand region, and it must be the
    /// *callee's* — passing the caller's would read the results from the wrong
    /// offset and silently return the wrong values.
    ///
    /// Gathered into a temporary before being written back, since the two ranges
    /// overlap whenever a frame has fewer locals, constants and spills than results.
    /// The copy only ever moves *downward*, so a plain forward loop would also be
    /// safe today — the temporary keeps that from being load-bearing. A frame with
    /// none of those three regions makes this a self-copy, which is a harmless no-op.
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
        let locals_count = frame_layout.locals_count as usize;
        let consts_len = frame_layout.consts.len();
        let spills = frame_layout.spills as usize;

        for i in 0..results_count as usize {
            temp.push(self.registers[base_register_index + locals_count + consts_len + spills + i]);
        }

        for i in 0..results_count as usize {
            self.registers[base_register_index + i] = temp[i];
        }
    }
}

/// Tests for the register file's layout and sizing.
///
/// A frame is four consecutive regions, in this order:
///
/// ```text
///   base + 0                                   locals (params first)
///   base + locals                              constants
///   base + locals + consts                     spills
///   base + locals + consts + spills            operand registers
///   base + registers + consts + spills         end of frame
/// ```
///
/// Two facts make that arithmetic easy to get wrong, so both are pinned below rather
/// than left to a reader of `enter_frame`:
///
/// - [`RegFrameLayout::registers`] **includes the locals**: lowering starts both
///   `curr_register_index` and `max_registers` at `locals_count`. So the frame's
///   width is `registers + spills + consts`, *not* `locals + registers + ...`.
/// - Constants and spills sit *below* the operand registers, which is what lets a
///   callee be based at its caller's `caller_base` without destroying them. Placing
///   them above the registers — where a `caller_base` points below them — is what
///   produced garbage addresses like `0xFFFF0000` at execution.
///
/// Getting the span wrong is caught by nothing else: the lowering never sees the
/// file, and the fault surfaces as an out-of-bounds index deep in an unrelated
/// instruction, or not at all when a deeper call already grew the file past the
/// mistake. So these assert lengths and **absolute indices** directly. Each length
/// is exact, so over-allocating fails too — the file is shared by the whole call
/// chain, and slack in every frame compounds with depth.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{module::ValType, runtime::value::Val};

    /// A layout for a frame with `locals` locals, `operands` operand registers,
    /// `spills` spill slots and `consts` constants.
    ///
    /// `operands` is the count *above* the locals; this adds `locals` to it to get
    /// the `registers` field, so a test reads in the units lowering thinks in while
    /// the field keeps the meaning the runtime expects. The arenas are what lowering
    /// fills for the *instructions* to index; sizing never reads them.
    fn layout(locals: u32, operands: u32, spills: u32, consts: &[Const]) -> RegFrameLayout {
        RegFrameLayout {
            registers: locals + operands,
            spills,
            locals_count: locals,
            consts: consts.to_vec().into_boxed_slice(),
            input_registers_arena: Box::new([]),
            output_registers_arena: Box::new([]),
            br_targets_arena: Box::new([]),
        }
    }

    /// The absolute index of each region's first slot, from the same formula
    /// `enter_frame` and `exit_frame` use.
    fn consts_base(base: u32, locals: u32) -> usize {
        (base + locals) as usize
    }

    fn spills_base(base: u32, locals: u32, consts: u32) -> usize {
        (base + locals + consts) as usize
    }

    fn operand_base(base: u32, locals: u32, consts: u32, spills: u32) -> usize {
        (base + locals + consts + spills) as usize
    }

    /// Enters a frame based at `base`, returning the resulting file length.
    fn enter(
        frame: &mut RegFrame,
        base: u32,
        params: &[ValType],
        declared: &[ValType],
        operands: u32,
        spills: u32,
        consts: &[Const],
    ) -> usize {
        let locals_ty: Vec<ValType> = params.iter().chain(declared).copied().collect();
        let mut caller_base_data = RegCallerBaseData {
            base_register_index: base,
        };

        frame.enter_frame(
            params.len() as u32,
            &locals_ty,
            &mut caller_base_data,
            &layout(locals_ty.len() as u32, operands, spills, consts),
        );

        frame.registers.len()
    }

    /// A frame with no constants and no spills: width is `locals + operands`, and
    /// the operand region begins immediately above the locals.
    #[test]
    fn a_plain_frame_is_locals_then_operand_registers() {
        let mut frame = RegFrame::default();

        frame.set_initial_params(&[Val::I32(7)]);

        // 1 param + 1 declared local + 2 operand registers
        let len = enter(&mut frame, 0, &[ValType::I32], &[ValType::I32], 2, 0, &[]);

        assert_eq!(len, 4, "1 param + 1 local + 2 operand registers");
        assert_eq!(
            operand_base(0, 2, 0, 0),
            2,
            "operands start above the locals"
        );
    }

    /// The width formula is `registers + spills + consts`, and `registers` already
    /// carries the locals. Adding `locals` again — the shape the old layout implied —
    /// would over-allocate by exactly `locals_count`.
    #[test]
    fn the_registers_count_already_includes_the_locals() {
        let mut frame = RegFrame::default();

        // registers == locals_count exactly, i.e. a body that needs no operand
        // register at all. The frame is then just wide enough for its locals.
        let mut caller_base_data = RegCallerBaseData {
            base_register_index: 0,
        };
        let frame_layout = RegFrameLayout {
            registers: 3,
            spills: 0,
            locals_count: 3,
            consts: Box::new([]),
            input_registers_arena: Box::new([]),
            output_registers_arena: Box::new([]),
            br_targets_arena: Box::new([]),
        };

        frame.enter_frame(
            0,
            &[ValType::I32, ValType::I32, ValType::I32],
            &mut caller_base_data,
            &frame_layout,
        );

        assert_eq!(
            frame.registers.len(),
            3,
            "width is `registers + spills + consts`, not `locals + registers`"
        );
    }

    /// All four regions at once, with every boundary asserted as an absolute index.
    #[test]
    fn the_four_regions_are_laid_out_in_order() {
        let mut frame = RegFrame::default();

        frame.set_initial_params(&[Val::I32(11)]);

        // 1 param + 1 declared local | 2 consts | 1 spill | 3 operand registers
        let len = enter(
            &mut frame,
            0,
            &[ValType::I32],
            &[ValType::I32],
            3,
            1,
            &[Const::I32(101), Const::I32(102)],
        );

        // registers = locals(2) + operands(3) = 5; width = 5 + spills(1) + consts(2)
        assert_eq!(len, 8, "registers(5) + spills(1) + consts(2)");

        assert_eq!(consts_base(0, 2), 2);
        assert_eq!(spills_base(0, 2, 2), 4);
        assert_eq!(operand_base(0, 2, 2, 1), 5);

        // the regions do not overlap and cover the frame exactly
        assert_eq!(operand_base(0, 2, 2, 1) + 3, len, "operands reach the end");

        assert_eq!(frame.registers[0].as_i32(), 11, "param at the frame base");
        assert_eq!(frame.registers[1].as_i32(), 0, "declared local zeroed");
        assert_eq!(frame.registers[2].as_i32(), 101, "const 0 at consts_base");
        assert_eq!(frame.registers[3].as_i32(), 102, "const 1 above it");
    }

    /// Constants are materialised on every entry, at `base + locals_count + id` —
    /// the same index the lowering's constant backpatch produces.
    #[test]
    fn constants_are_materialised_at_the_const_region() {
        let mut frame = RegFrame::default();

        let len = enter(
            &mut frame,
            0,
            &[],
            &[ValType::I64],
            1,
            0,
            &[Const::I64(-7), Const::I64(1 << 40), Const::I64(0)],
        );

        // registers = locals(1) + operands(1) = 2; width = 2 + 0 + consts(3)
        assert_eq!(len, 5);

        let cb = consts_base(0, 1);

        assert_eq!(cb, 1);
        assert_eq!(frame.registers[cb].as_i64(), -7);
        assert_eq!(frame.registers[cb + 1].as_i64(), 1 << 40);
        assert_eq!(frame.registers[cb + 2].as_i64(), 0);
    }

    /// The regression test for the clobbering bug.
    ///
    /// A callee is based at its caller's `caller_base`, which lowering guarantees is
    /// at or above the caller's operand base. Every region the caller needs to
    /// survive the call — its locals, constants and spills — therefore sits strictly
    /// below the callee's frame. When constants lived *above* the registers instead,
    /// the callee's frame overlapped them and the caller resumed reading whatever the
    /// callee had left behind.
    #[test]
    fn a_nested_frame_cannot_reach_its_callers_constants() {
        let mut frame = RegFrame::default();

        frame.set_initial_params(&[Val::I32(11)]);

        // caller: 2 locals | 2 consts | 1 spill | 3 operands  =>  width 8
        let caller_len = enter(
            &mut frame,
            0,
            &[ValType::I32],
            &[ValType::I32],
            3,
            1,
            &[Const::I32(101), Const::I32(102)],
        );

        assert_eq!(caller_len, 8);

        // A legal `caller_base`: at or above the caller's own operand base, which is
        // what the register backpatch's `+ spills + consts_len` shift guarantees.
        let callee_base = operand_base(0, 2, 2, 1) as u32;

        assert_eq!(callee_base, 5);

        // callee: 1 local | 1 const | 0 spills | 2 operands
        enter(
            &mut frame,
            callee_base,
            &[ValType::I32],
            &[],
            2,
            0,
            &[Const::I32(999)],
        );

        // the callee's own constant landed in the callee's region...
        assert_eq!(
            frame.registers[consts_base(callee_base, 1)].as_i32(),
            999,
            "the callee's constant is materialised in the callee's frame"
        );

        // ...and the caller's constants are still there.
        assert_eq!(
            frame.registers[consts_base(0, 2)].as_i32(),
            101,
            "the caller's first constant must survive the call"
        );
        assert_eq!(
            frame.registers[consts_base(0, 2) + 1].as_i32(),
            102,
            "the caller's second constant must survive the call"
        );

        // and so are its locals
        assert_eq!(
            frame.registers[0].as_i32(),
            11,
            "the caller's param survives"
        );
    }

    /// `exit_frame` copies results down from the callee's *operand* base to the
    /// frame base, so they land exactly where the caller staged the arguments.
    #[test]
    fn exit_frame_moves_results_from_the_operand_base_to_the_frame_base() {
        let mut frame = RegFrame::default();

        // 2 locals | 1 const | 1 spill | 2 operands  =>  operand base at 4, width 6
        let len = enter(
            &mut frame,
            0,
            &[],
            &[ValType::I32, ValType::I32],
            2,
            1,
            &[Const::I32(55)],
        );

        assert_eq!(len, 6);

        let ob = operand_base(0, 2, 1, 1);

        assert_eq!(ob, 4);

        // the body's `end` materialises its results into the first operand registers
        frame.registers[ob] = Value::from_i32(71);
        frame.registers[ob + 1] = Value::from_i32(72);

        let caller_base_data = RegCallerBaseData {
            base_register_index: 0,
        };

        frame.exit_frame(2, &caller_base_data, &layout(2, 2, 1, &[Const::I32(55)]));

        assert_eq!(
            frame.registers[0].as_i32(),
            71,
            "result 0 at the frame base"
        );
        assert_eq!(frame.registers[1].as_i32(), 72, "result 1 above it");
    }

    /// A callee with no locals, constants or spills makes the operand base the frame
    /// base, so the result copy is a self-copy — which must be a no-op, not a shift.
    #[test]
    fn exit_frame_is_a_no_op_when_the_operand_base_is_the_frame_base() {
        let mut frame = RegFrame::default();

        let len = enter(&mut frame, 0, &[], &[], 2, 0, &[]);

        assert_eq!(len, 2);
        assert_eq!(operand_base(0, 0, 0, 0), 0);

        frame.registers[0] = Value::from_i32(5);
        frame.registers[1] = Value::from_i32(6);

        let caller_base_data = RegCallerBaseData {
            base_register_index: 0,
        };

        frame.exit_frame(2, &caller_base_data, &layout(0, 2, 0, &[]));

        assert_eq!(frame.registers[0].as_i32(), 5);
        assert_eq!(frame.registers[1].as_i32(), 6);
    }

    #[test]
    fn params_alone_still_need_room_for_registers() {
        // no declared locals: `locals_count - params_count` is zero, which must not
        // underflow and must not shorten the frame
        let mut frame = RegFrame::default();

        frame.set_initial_params(&[Val::I32(1), Val::I32(2)]);

        let len = enter(&mut frame, 0, &[ValType::I32, ValType::I32], &[], 3, 0, &[]);

        assert_eq!(len, 5, "2 params + 3 operand registers");
    }

    #[test]
    fn a_frame_with_no_operand_registers_is_sized_for_its_locals() {
        let mut frame = RegFrame::default();

        let len = enter(
            &mut frame,
            0,
            &[ValType::I64],
            &[ValType::I64, ValType::I64],
            0,
            0,
            &[],
        );

        assert_eq!(len, 3, "1 param + 2 locals, no operands");
    }

    #[test]
    fn a_nested_frame_is_sized_from_its_own_base() {
        // a callee based at 4 needs the file to reach its own end, not just its own
        // width — the caller's frame below it stays live
        let mut frame = RegFrame::default();
        let len = enter(&mut frame, 4, &[ValType::I32], &[ValType::I32], 2, 0, &[]);

        assert_eq!(len, 8, "base 4 + registers(4)");
    }

    /// The spill and constant regions widen a nested frame too, so a callee based at
    /// `b` reaches `b + registers + spills + consts` — the case that would catch the
    /// capacity computation dropping either term.
    #[test]
    fn a_nested_frames_consts_and_spills_widen_it() {
        let mut frame = RegFrame::default();

        let len = enter(
            &mut frame,
            10,
            &[ValType::I32],
            &[],
            2,
            2,
            &[Const::I32(1), Const::I32(2), Const::I32(3)],
        );

        // registers = locals(1) + operands(2) = 3; width = 3 + spills(2) + consts(3)
        assert_eq!(len, 18, "base 10 + registers(3) + spills(2) + consts(3)");
    }

    #[test]
    fn a_shallow_frame_does_not_truncate_the_file_above_it() {
        // A deep call grows the file; returning to a shallow frame must not shrink
        // it, or the values of every frame in between would be dropped. The guard
        // in `enter_frame` is what prevents that, and this is the case that would
        // catch its removal.
        let mut frame = RegFrame::default();
        let deep = enter(&mut frame, 16, &[ValType::I32], &[ValType::I32], 4, 0, &[]);

        assert_eq!(deep, 22);

        let shallow = enter(&mut frame, 0, &[ValType::I32], &[ValType::I32], 2, 0, &[]);

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

        enter(&mut frame, 0, &[ValType::I32], &[ValType::I32], 2, 0, &[]);

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

    /// A trap unwinds without reaching `exit_frame`, so the next call's reset is what
    /// releases the file — otherwise every base above it stays shifted up for the
    /// life of the instance.
    #[test]
    fn reset_empties_the_register_file() {
        let mut frame = RegFrame::default();

        enter(&mut frame, 0, &[ValType::I32], &[], 3, 2, &[Const::I32(1)]);

        assert_eq!(
            frame.registers.len(),
            7,
            "registers(4) + spills(2) + consts(1)"
        );

        frame.reset();

        assert_eq!(frame.registers.len(), 0);
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
            0,
            &[],
        );

        assert_eq!(
            frame.registers[0].as_i64(),
            -9,
            "the param is not overwritten"
        );
        assert_eq!(frame.registers[1].as_i64(), 0, "declared local 0 is zeroed");
        assert_eq!(frame.registers[2].as_i64(), 0, "declared local 1 is zeroed");
    }

    /// Zeroing the declared locals must not reach into the constant region above
    /// them: a stale local left behind by a previous, deeper frame has to be cleared,
    /// while the constants written after it must survive.
    #[test]
    fn zeroing_locals_stops_below_the_constants() {
        let mut frame = RegFrame::default();

        // dirty the file so the zeroing has something to actually clear
        let deep = enter(&mut frame, 0, &[], &[], 8, 0, &[]);

        assert_eq!(deep, 8);

        for i in 0..8 {
            frame.registers[i] = Value::from_i32(-1);
        }

        enter(
            &mut frame,
            0,
            &[],
            &[ValType::I32, ValType::I32],
            1,
            0,
            &[Const::I32(77)],
        );

        assert_eq!(
            frame.registers[0].as_i32(),
            0,
            "declared local 0 is cleared"
        );
        assert_eq!(
            frame.registers[1].as_i32(),
            0,
            "declared local 1 is cleared"
        );
        assert_eq!(
            frame.registers[consts_base(0, 2)].as_i32(),
            77,
            "the constant above them is written, not zeroed"
        );
    }
}
