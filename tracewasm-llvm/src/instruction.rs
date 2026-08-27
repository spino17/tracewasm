pub struct PhiInstruction {}

pub struct PhiInstrId(usize);

impl PhiInstrId {
    pub(crate) fn new(id: usize) -> Self {
        PhiInstrId(id)
    }
}

pub enum Instruction {
    Phi(PhiInstruction),
}

impl Instruction {
    pub fn is_phi(&self) -> bool {
        matches!(self, Instruction::Phi(_))
    }
}
