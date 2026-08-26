use crate::instruction::register::{
    Const, RegInstruction, Slot, interner::InternedId, lazy::SpillIndex,
};
use rustc_hash::FxHashMap;

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

    fn absolute_slot_in_frame(self, locals: u16, consts: u16, spills: u16) -> Slot {
        match self {
            BackPatchableSlot::Slot(slot) => slot,
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

#[derive(Clone, Copy)]
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
    pub fn apply(
        &mut self,
        instructions: &mut [RegInstruction],
        locals: u16,
        consts: u16,
        spills: u16,
    ) {
        for (instr_index, patches) in &self.0 {
            let instr = &mut instructions[*instr_index];

            match instr {
                RegInstruction::LocalTee { index: _, input } => {
                    debug_assert!(patches.len() == 1);

                    let (source, patch) = &patches[0];

                    debug_assert!(matches!(source, InstructionSource::Emit));
                    debug_assert!(patch.len() == 1);

                    let patch = patch[0];

                    input.registers[0] = patch.absolute_slot_in_frame(locals, consts, spills)
                }
                // TODO: add all instructions! - most can be handled by macros!
                _ => todo!(),
            }
        }
    }
}
