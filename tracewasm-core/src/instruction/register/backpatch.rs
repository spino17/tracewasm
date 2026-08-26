use crate::instruction::register::{
    Const, DynSignature, RegFrameLayout, RegInstruction, Signature, Slot, interner::InternedId,
    lazy::SpillIndex,
};
use rustc_hash::FxHashMap;
use std::slice::Iter;

/// An operand as lowering knows it, before the frame layout is final.
///
/// The provisional counterpart of [`Slot`]. Three of the four cases cannot be turned
/// into a frame index while the body is still being walked, because the constant and
/// spill region sizes are not yet known — so they travel as what they *are* and are
/// resolved in the end-of-body pass (see the module docs). The fourth is already
/// final: a local or a global cannot move.
#[derive(Clone, Copy)]
pub(crate) enum BackPatchableSlot {
    /// An operand register, as a *provisional* frame index — counted from the frame
    /// base, so at or above `locals_count`, but not yet clear of the constant and
    /// spill regions. Resolved by shifting up `consts + spills`.
    Register(u16),
    /// A spill slot, by pool index. Resolved to `locals_count + consts + slot`.
    Spill(SpillIndex),
    /// An interned constant, by pool id. Resolved to `locals_count + id`.
    Const(InternedId<Const>),
    /// An operand whose location is already final: a local, or a global.
    Slot(Slot),
}

impl BackPatchableSlot {
    /// Whether this operand occupies an operand register.
    ///
    /// A variant test, so it is correct at any point in lowering — unlike
    /// [`Slot::is_register`], which compares an index against `locals_count` and so
    /// only holds while indices are provisional. This is the one the register-height
    /// bookkeeping uses.
    pub fn is_register(&self) -> bool {
        matches!(self, BackPatchableSlot::Register(_))
    }

    fn absolute_slot_in_frame(&self, locals: u16, consts: u16, spills: u16) -> Slot {
        match self {
            BackPatchableSlot::Slot(slot) => *slot,
            BackPatchableSlot::Const(const_id) => Slot(const_id.raw() + locals),
            BackPatchableSlot::Spill(spill_index) => {
                Slot(spill_index.raw_value() + locals + consts)
            }
            BackPatchableSlot::Register(index) => Slot(index + consts + spills),
        }
    }
}

impl From<Slot> for BackPatchableSlot {
    /// Adopts an already-resolved operand — a local or a global, the two that need
    /// no backpatch.
    fn from(value: Slot) -> Self {
        BackPatchableSlot::Slot(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum InstructionSource {
    Emit,
    BrIfCond,
    BrIfMov,
    BrTableIndex,
    BrTableMov,
    CallIndirectSlot,
    CallIndirectMov,
}

#[derive(Default)]
pub(crate) struct BackpatchMap(
    pub FxHashMap<usize, Vec<(InstructionSource, Vec<BackPatchableSlot>)>>,
); // instr_index -> source -> input slots

impl BackpatchMap {
    fn apply_to_input_registers(
        patches: &mut Iter<'_, (InstructionSource, Vec<BackPatchableSlot>)>,
        inputs: &mut [Slot],
        locals: u16,
        consts: u16,
        spills: u16,
        expected_source: InstructionSource,
    ) {
        let Some((source, patch)) = patches.next() else {
            return;
        };

        debug_assert!(*source == expected_source);

        let inputs_len = inputs.len();

        debug_assert!(patch.len() == inputs_len);

        for (i, slot) in patch.iter().enumerate() {
            inputs[i] = slot.absolute_slot_in_frame(locals, consts, spills);
        }
    }

    fn apply_to_output_registers(output_start: &mut u16, _locals: u16, consts: u16, spills: u16) {
        *output_start = *output_start + consts + spills;
    }

    fn apply_to_signature<const I: usize, const O: usize>(
        patches: &mut Iter<'_, (InstructionSource, Vec<BackPatchableSlot>)>,
        sig: &mut Signature<I, O>,
        locals: u16,
        consts: u16,
        spills: u16,
        expected_source: InstructionSource,
    ) {
        Self::apply_to_input_registers(
            patches,
            &mut sig.input.registers,
            locals,
            consts,
            spills,
            expected_source,
        );

        Self::apply_to_output_registers(&mut sig.output.start, locals, consts, spills);
    }

    fn apply_to_dyn_signature(
        patches: &mut Iter<'_, (InstructionSource, Vec<BackPatchableSlot>)>,
        sig: &mut DynSignature,
        locals: u16,
        consts: u16,
        spills: u16,
        expected_source: InstructionSource,
    ) {
        Self::apply_to_input_registers(
            patches,
            &mut sig.input,
            locals,
            consts,
            spills,
            expected_source,
        );

        Self::apply_to_output_registers(&mut sig.output_start, locals, consts, spills);
    }

    pub fn apply(
        &mut self,
        instructions: &mut [RegInstruction],
        locals: u16,
        consts: u16,
        spills: u16,
        frame_layout: &mut RegFrameLayout,
    ) {
        for (instr_index, patches) in &self.0 {
            let instr = &mut instructions[*instr_index];

            match instr {
                RegInstruction::LocalSet { index: _, input } => {
                    let mut patch_iter = patches.iter();

                    Self::apply_to_input_registers(
                        &mut patch_iter,
                        &mut input.registers,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                RegInstruction::LocalTee { index: _, input } => {
                    debug_assert!(patches.len() == 1);

                    let (source, patch) = &patches[0];

                    debug_assert!(matches!(source, InstructionSource::Emit));
                    debug_assert!(patch.len() == 1);

                    let patch = patch[0];

                    input.registers[0] = patch.absolute_slot_in_frame(locals, consts, spills)
                }
                RegInstruction::GlobalGet { index: _, output } => {
                    Self::apply_to_output_registers(&mut output.start, locals, consts, spills);
                }
                RegInstruction::GlobalSet { index: _, input } => {
                    let mut patch_iter = patches.iter();

                    Self::apply_to_input_registers(
                        &mut patch_iter,
                        &mut input.registers,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                RegInstruction::If(id) => {
                    let entry = frame_layout.if_arena.get_mut(*id);
                    let mut patch_iter = patches.iter();

                    Self::apply_to_input_registers(
                        &mut patch_iter,
                        &mut entry.cond.registers,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                RegInstruction::BrIf(id) => {
                    let entry = frame_layout.br_if_arena.get_mut(*id);
                    let mut patch_iter = patches.iter();

                    Self::apply_to_input_registers(
                        &mut patch_iter,
                        &mut entry.cond.registers,
                        locals,
                        consts,
                        spills,
                        InstructionSource::BrIfCond,
                    );

                    Self::apply_to_dyn_signature(
                        &mut patch_iter,
                        &mut entry.mov,
                        locals,
                        consts,
                        spills,
                        InstructionSource::BrIfMov,
                    );
                }
                RegInstruction::BrTable(id) => {
                    let entry = frame_layout.br_table_arena.get_mut(*id);
                    let mut patch_iter = patches.iter();

                    Self::apply_to_input_registers(
                        &mut patch_iter,
                        &mut entry.index.registers,
                        locals,
                        consts,
                        spills,
                        InstructionSource::BrTableIndex,
                    );

                    for target in &mut entry.br_targets {
                        Self::apply_to_dyn_signature(
                            &mut patch_iter,
                            &mut target.mov,
                            locals,
                            consts,
                            spills,
                            InstructionSource::BrTableMov,
                        );
                    }
                }
                RegInstruction::CallIndirect(id) => {
                    let entry = frame_layout.call_indirect_arena.get_mut(*id);
                    let mut patch_iter = patches.iter();

                    Self::apply_to_input_registers(
                        &mut patch_iter,
                        &mut entry.slot.registers,
                        locals,
                        consts,
                        spills,
                        InstructionSource::CallIndirectSlot,
                    );

                    Self::apply_to_dyn_signature(
                        &mut patch_iter,
                        &mut entry.operands,
                        locals,
                        consts,
                        spills,
                        InstructionSource::CallIndirectMov,
                    );
                }
                RegInstruction::Select(id) => {
                    let entry = frame_layout.select_arena.get_mut(*id);
                    let mut patch_iter = patches.iter();

                    Self::apply_to_signature(
                        &mut patch_iter,
                        &mut entry.0,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                RegInstruction::MemoryInit(id) => {
                    let entry = frame_layout.memory_init_arena.get_mut(*id);
                    let mut patch_iter = patches.iter();

                    Self::apply_to_input_registers(
                        &mut patch_iter,
                        &mut entry.operands.registers,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                // Every fixed-arity signature resolves the same way, so these are grouped by
                // operand shape rather than by family: an or-pattern binds one type, and
                // each body is a single call. A load joins the `Signature<1, 1>` group
                // because its `offset` is an interned id rather than an operand, so it
                // needs no resolution.
                RegInstruction::I32Load { sig, .. }
                | RegInstruction::I32Load8S { sig, .. }
                | RegInstruction::I32Load8U { sig, .. }
                | RegInstruction::I32Load16S { sig, .. }
                | RegInstruction::I32Load16U { sig, .. }
                | RegInstruction::I64Load { sig, .. }
                | RegInstruction::I64Load8S { sig, .. }
                | RegInstruction::I64Load8U { sig, .. }
                | RegInstruction::I64Load16S { sig, .. }
                | RegInstruction::I64Load16U { sig, .. }
                | RegInstruction::I64Load32S { sig, .. }
                | RegInstruction::I64Load32U { sig, .. }
                | RegInstruction::F32Load { sig, .. }
                | RegInstruction::F64Load { sig, .. }
                | RegInstruction::I32Clz(sig)
                | RegInstruction::I32Ctz(sig)
                | RegInstruction::I32Eqz(sig)
                | RegInstruction::I32Extend16S(sig)
                | RegInstruction::I32Extend8S(sig)
                | RegInstruction::I32Popcnt(sig)
                | RegInstruction::I32ReinterpretF32(sig)
                | RegInstruction::I32TruncF32S(sig)
                | RegInstruction::I32TruncF32U(sig)
                | RegInstruction::I32TruncF64S(sig)
                | RegInstruction::I32TruncF64U(sig)
                | RegInstruction::I32TruncSatF32S(sig)
                | RegInstruction::I32TruncSatF32U(sig)
                | RegInstruction::I32TruncSatF64S(sig)
                | RegInstruction::I32TruncSatF64U(sig)
                | RegInstruction::I32WrapI64(sig)
                | RegInstruction::I64Clz(sig)
                | RegInstruction::I64Ctz(sig)
                | RegInstruction::I64Eqz(sig)
                | RegInstruction::I64Extend16S(sig)
                | RegInstruction::I64Extend32S(sig)
                | RegInstruction::I64Extend8S(sig)
                | RegInstruction::I64ExtendI32S(sig)
                | RegInstruction::I64ExtendI32U(sig)
                | RegInstruction::I64Popcnt(sig)
                | RegInstruction::I64ReinterpretF64(sig)
                | RegInstruction::I64TruncF32S(sig)
                | RegInstruction::I64TruncF32U(sig)
                | RegInstruction::I64TruncF64S(sig)
                | RegInstruction::I64TruncF64U(sig)
                | RegInstruction::I64TruncSatF32S(sig)
                | RegInstruction::I64TruncSatF32U(sig)
                | RegInstruction::I64TruncSatF64S(sig)
                | RegInstruction::I64TruncSatF64U(sig)
                | RegInstruction::F32Abs(sig)
                | RegInstruction::F32Ceil(sig)
                | RegInstruction::F32ConvertI32S(sig)
                | RegInstruction::F32ConvertI32U(sig)
                | RegInstruction::F32ConvertI64S(sig)
                | RegInstruction::F32ConvertI64U(sig)
                | RegInstruction::F32DemoteF64(sig)
                | RegInstruction::F32Floor(sig)
                | RegInstruction::F32Nearest(sig)
                | RegInstruction::F32Neg(sig)
                | RegInstruction::F32ReinterpretI32(sig)
                | RegInstruction::F32Sqrt(sig)
                | RegInstruction::F32Trunc(sig)
                | RegInstruction::F64Abs(sig)
                | RegInstruction::F64Ceil(sig)
                | RegInstruction::F64ConvertI32S(sig)
                | RegInstruction::F64ConvertI32U(sig)
                | RegInstruction::F64ConvertI64S(sig)
                | RegInstruction::F64ConvertI64U(sig)
                | RegInstruction::F64Floor(sig)
                | RegInstruction::F64Nearest(sig)
                | RegInstruction::F64Neg(sig)
                | RegInstruction::F64PromoteF32(sig)
                | RegInstruction::F64ReinterpretI64(sig)
                | RegInstruction::F64Sqrt(sig)
                | RegInstruction::F64Trunc(sig)
                | RegInstruction::RefIsNull(sig)
                | RegInstruction::MemoryGrow(sig) => {
                    let mut patch_iter = patches.iter();

                    Self::apply_to_signature(
                        &mut patch_iter,
                        sig,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                RegInstruction::I32Add(sig)
                | RegInstruction::I32And(sig)
                | RegInstruction::I32DivS(sig)
                | RegInstruction::I32DivU(sig)
                | RegInstruction::I32Eq(sig)
                | RegInstruction::I32GeS(sig)
                | RegInstruction::I32GeU(sig)
                | RegInstruction::I32GtS(sig)
                | RegInstruction::I32GtU(sig)
                | RegInstruction::I32LeS(sig)
                | RegInstruction::I32LeU(sig)
                | RegInstruction::I32LtS(sig)
                | RegInstruction::I32LtU(sig)
                | RegInstruction::I32Mul(sig)
                | RegInstruction::I32Ne(sig)
                | RegInstruction::I32Or(sig)
                | RegInstruction::I32RemS(sig)
                | RegInstruction::I32RemU(sig)
                | RegInstruction::I32Rotl(sig)
                | RegInstruction::I32Rotr(sig)
                | RegInstruction::I32Shl(sig)
                | RegInstruction::I32ShrS(sig)
                | RegInstruction::I32ShrU(sig)
                | RegInstruction::I32Sub(sig)
                | RegInstruction::I32Xor(sig)
                | RegInstruction::I64Add(sig)
                | RegInstruction::I64And(sig)
                | RegInstruction::I64DivS(sig)
                | RegInstruction::I64DivU(sig)
                | RegInstruction::I64Eq(sig)
                | RegInstruction::I64GeS(sig)
                | RegInstruction::I64GeU(sig)
                | RegInstruction::I64GtS(sig)
                | RegInstruction::I64GtU(sig)
                | RegInstruction::I64LeS(sig)
                | RegInstruction::I64LeU(sig)
                | RegInstruction::I64LtS(sig)
                | RegInstruction::I64LtU(sig)
                | RegInstruction::I64Mul(sig)
                | RegInstruction::I64Ne(sig)
                | RegInstruction::I64Or(sig)
                | RegInstruction::I64RemS(sig)
                | RegInstruction::I64RemU(sig)
                | RegInstruction::I64Rotl(sig)
                | RegInstruction::I64Rotr(sig)
                | RegInstruction::I64Shl(sig)
                | RegInstruction::I64ShrS(sig)
                | RegInstruction::I64ShrU(sig)
                | RegInstruction::I64Sub(sig)
                | RegInstruction::I64Xor(sig)
                | RegInstruction::F32Add(sig)
                | RegInstruction::F32Copysign(sig)
                | RegInstruction::F32Div(sig)
                | RegInstruction::F32Eq(sig)
                | RegInstruction::F32Ge(sig)
                | RegInstruction::F32Gt(sig)
                | RegInstruction::F32Le(sig)
                | RegInstruction::F32Lt(sig)
                | RegInstruction::F32Max(sig)
                | RegInstruction::F32Min(sig)
                | RegInstruction::F32Mul(sig)
                | RegInstruction::F32Ne(sig)
                | RegInstruction::F32Sub(sig)
                | RegInstruction::F64Add(sig)
                | RegInstruction::F64Copysign(sig)
                | RegInstruction::F64Div(sig)
                | RegInstruction::F64Eq(sig)
                | RegInstruction::F64Ge(sig)
                | RegInstruction::F64Gt(sig)
                | RegInstruction::F64Le(sig)
                | RegInstruction::F64Lt(sig)
                | RegInstruction::F64Max(sig)
                | RegInstruction::F64Min(sig)
                | RegInstruction::F64Mul(sig)
                | RegInstruction::F64Ne(sig)
                | RegInstruction::F64Sub(sig) => {
                    let mut patch_iter = patches.iter();

                    Self::apply_to_signature(
                        &mut patch_iter,
                        sig,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                // A store has no destination, so only its operands are resolved.
                RegInstruction::I32Store { input, .. }
                | RegInstruction::I32Store8 { input, .. }
                | RegInstruction::I32Store16 { input, .. }
                | RegInstruction::I64Store { input, .. }
                | RegInstruction::I64Store8 { input, .. }
                | RegInstruction::I64Store16 { input, .. }
                | RegInstruction::I64Store32 { input, .. }
                | RegInstruction::F32Store { input, .. }
                | RegInstruction::F64Store { input, .. } => {
                    let mut patch_iter = patches.iter();

                    Self::apply_to_input_registers(
                        &mut patch_iter,
                        &mut input.registers,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                RegInstruction::MemoryCopy(input) | RegInstruction::MemoryFill(input) => {
                    let mut patch_iter = patches.iter();

                    Self::apply_to_input_registers(
                        &mut patch_iter,
                        &mut input.registers,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                // No operands: the only run it carries is its destination.
                RegInstruction::MemorySize(output) => {
                    Self::apply_to_output_registers(&mut output.start, locals, consts, spills);
                }
                RegInstruction::Move(id) => {
                    let sig = frame_layout.dyn_signatures.get_mut(*id);
                    let mut patch_iter = patches.iter();

                    Self::apply_to_dyn_signature(
                        &mut patch_iter,
                        sig,
                        locals,
                        consts,
                        spills,
                        InstructionSource::Emit,
                    );
                }
                // These carry no operands, so lowering records no patch for them and this
                // loop — which visits only the instructions it has entries for — cannot
                // reach them. Listed rather than covered by a wildcard so that inlining an
                // operand into one of them is a compile error here.
                RegInstruction::LocalSpill { .. }
                | RegInstruction::Loop
                | RegInstruction::Else { .. }
                | RegInstruction::Br { .. }
                | RegInstruction::Return { .. }
                | RegInstruction::Call { .. }
                | RegInstruction::Unreachable
                | RegInstruction::DataDrop(_)
                | RegInstruction::End => {
                    unreachable!("an instruction with no operands has no backpatch entry to apply")
                }
            }
        }
    }
}
