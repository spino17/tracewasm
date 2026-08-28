use thiserror::Error;
use tracewasm_utils::error::TracewasmUtilsError;

#[derive(Error, Debug)]
pub enum BuildError {
    #[error("phi instructions should be added at the start of the basic block")]
    PhiInstructionAddError,
    #[error("phi instructions cannot be added to the first basic block of the function")]
    PhiInstructionCannotBeAddedToEntryBasicBlock,
    #[error("basic block branch already in phi instruction")]
    BasicBlockBranchAlreadyInPhiInstruction,
    #[error("value of type `{0}` cannot be converted into i1 value")]
    ValueToI1ValueFailed(String),
    #[error("{0}")]
    UtilsError(TracewasmUtilsError),
}

impl From<TracewasmUtilsError> for BuildError {
    fn from(value: TracewasmUtilsError) -> Self {
        BuildError::UtilsError(value)
    }
}
