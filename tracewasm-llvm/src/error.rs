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
    /// A phi is typed once and every incoming value has to have that type — the
    /// instruction produces one value, so there is nothing for a second type to be.
    ///
    /// Fields: the type the phi was established with, and the one this branch
    /// brought.
    #[error("phi branch has type `{1}`, but the phi's type is `{0}`")]
    PhiInstructionBranchTypeMismatch(Type, Type),
    /// A phi with no incoming values selects nothing, and its type is whatever its
    /// first branch says — so with none there is no type to give it either.
    #[error("a phi instruction needs at least one branch")]
    PhiInstructionWithNoBranches,
    /// A requested register name is not an LLVM identifier.
    ///
    /// Unquoted locals are `[-a-zA-Z$._][-a-zA-Z$._0-9]*`; anything else would have
    /// to be quoted in the emitted IR. A leading digit is refused for a second
    /// reason: `%0` is the *unnamed* form, so a numeric name would collide with the
    /// numbering rather than merely need quoting.
    #[error(
        "`{0}` is not a valid register name: an LLVM local is \
         `[-a-zA-Z$._][-a-zA-Z$._0-9]*`, and may not begin with a digit"
    )]
    InvalidRegisterName(String),
    /// `load` and `store` address memory through a pointer, so the operand naming
    /// the location has to be one. Reaching memory from an integer needs an
    /// `inttoptr` first.
    #[error("expected a `ptr` operand, but got one of type `{0}`")]
    PointerOperandExpected(Type),
    /// A `load` or `store` moves a value of a known size, so the type has to have
    /// one. `void` and function types do not — everything else does, aggregates
    /// included, so `load {i32, i32}` is fine.
    #[error("a value of type `{0}` cannot be loaded or stored: it has no size")]
    TypeNotLoadable(Type),
    /// An explicit alignment must be a power of two. `0` is not one — leaving the
    /// alignment off is how the ABI default is asked for.
    #[error("alignment must be a power of two, but got `{0}`")]
    AlignmentNotPowerOfTwo(u32),
}

impl From<TracewasmUtilsError> for BuildError {
    fn from(value: TracewasmUtilsError) -> Self {
        BuildError::UtilsError(value)
    }
}
