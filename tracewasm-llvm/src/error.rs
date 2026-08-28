use crate::value::Type;
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
    #[error("constant with type `{0}` failed to be casted as `{1}`")]
    ConstantCastToProvidedTypeFailed(Type, Type),
    /// Two functions cannot share a name: LLVM identifies a definition by it, so
    /// emitting both would produce two `@name` definitions in one module.
    #[error("a function named `{0}` already exists in this module")]
    DuplicateFunctionName(String),
    /// Two blocks in one function cannot share a name, for the same reason: the
    /// label a branch names would be ambiguous.
    #[error("a basic block named `{0}` already exists in this function")]
    DuplicateBasicBlockName(String),
}

impl From<TracewasmUtilsError> for BuildError {
    fn from(value: TracewasmUtilsError) -> Self {
        BuildError::UtilsError(value)
    }
}
