use thiserror::Error;

#[derive(Error, Debug)]
pub enum BuildError {
    #[error("phi instructions should be added at the start of the basic block")]
    PhiInstructionAddError,
    #[error("phi instructions cannot be added to the first basic block of the function")]
    PhiInstructionCannotBeAddedToEntryBasicBlock,
    #[error("basic block branch already in phi instruction")]
    BasicBlockBranchAlreadyInPhiInstruction,
}
