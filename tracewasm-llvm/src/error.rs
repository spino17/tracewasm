use thiserror::Error;
use tracewasm_utils::error::TracewasmUtilsError;

/// Every variant naming a type carries it **already rendered**, not as a `Type` or a
/// `TyId`. Neither can print itself — an aggregate names its children by id, so
/// spelling one out needs the `TyInterner` that issued them, which an error cannot
/// borrow and outlive. So the failing path renders through
/// [`TyInterner::display`](crate::interner::TyInterner::display) on the way out; a
/// failure is rare enough that the allocation costs nothing that matters.
#[derive(Error, Debug)]
pub enum BuildError {
    #[error("{0}")]
    TypeError(#[from] TypeError),
    #[error("{0}")]
    UtilsError(#[from] TracewasmUtilsError),
    #[error("{0}")]
    InstructionError(#[from] InstructionError),
}

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("value of type `{0}` cannot be converted into i1 value")]
    ValueToI1ValueFailed(String),
    #[error("constant with type `{0}` failed to be casted as `{1}`")]
    ConstantCastToProvidedTypeFailed(String, String),
}

#[derive(Error, Debug)]
pub enum ContextError {
    /// Two functions cannot share a name: LLVM identifies a definition by it, so
    /// emitting both would produce two `@name` definitions in one module.
    #[error("a function named `{0}` already exists in this module")]
    DuplicateFunctionName(String),
    /// Two blocks in one function cannot share a name, for the same reason: the
    /// label a branch names would be ambiguous.
    #[error("a basic block named `{0}` already exists in this function")]
    DuplicateBasicBlockName(String),
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
}

#[derive(Error, Debug)]
pub enum InstructionError {
    /// `load` and `store` address memory through a pointer, so the operand naming
    /// the location has to be one. Reaching memory from an integer needs an
    /// `inttoptr` first.
    #[error("expected a `ptr` operand, but got one of type `{0}`")]
    PointerOperandExpected(String),
    /// A `load` or `store` moves a value of a known size, so the type has to have
    /// one. `void` and function types do not — everything else does, aggregates
    /// included, so `load {i32, i32}` is fine.
    #[error("a value of type `{0}` cannot be loaded or stored: it has no size")]
    TypeNotLoadable(String),
    /// An explicit alignment must be a power of two. `0` is not one — leaving the
    /// alignment off is how the ABI default is asked for.
    #[error("alignment must be a power of two, but got `{0}`")]
    AlignmentNotPowerOfTwo(u32),
    #[error("{0}")]
    Alloca(#[from] AllocaError),
    #[error("{0}")]
    Store(#[from] StoreError),
    #[error("{0}")]
    Phi(#[from] PhiError),
    #[error("{0}")]
    Context(#[from] ContextError),
}

#[derive(Error, Debug)]
pub enum AllocaError {
    #[error("a value of type `{0}` cannot be allocated: `alloca` needs a sized type")]
    TypeNotAllocatable(String),
    #[error("an `alloca` element count must have an integer type, but got `{0}`")]
    AllocaCountNotAnInteger(String),
    #[error("an `alloca` element count of type `{0}` cannot be used as `{1}`")]
    AllocaCountTypeMismatch(String, String),
}

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("a value of type `{0}` cannot be stored as `{1}`")]
    StoredValueTypeMismatch(String, String),
    #[error("a value of type `{0}` cannot be stored through a pointer to `{1}`")]
    StoredValueDoesNotMatchPointee(String, String),
}

#[derive(Error, Debug)]
pub enum PhiError {
    #[error("phi instructions should be added at the start of the basic block")]
    PhiInstructionAddError,
    #[error("phi instructions cannot be added to the first basic block of the function")]
    PhiInstructionCannotBeAddedToEntryBasicBlock,
    #[error("basic block branch already in phi instruction")]
    BasicBlockBranchAlreadyInPhiInstruction,
    /// A phi with no incoming values selects nothing, and its type is whatever its
    /// first branch says — so with none there is no type to give it either.
    #[error("a phi instruction needs at least one branch")]
    PhiInstructionWithNoBranches,
    /// A phi is typed once and every incoming value has to have that type — the
    /// instruction produces one value, so there is nothing for a second type to be.
    ///
    /// Fields: the type the phi was established with, and the one this branch
    /// brought.
    #[error("phi branch has type `{1}`, but the phi's type is `{0}`")]
    PhiInstructionBranchTypeMismatch(String, String),
    #[error("{0}")]
    Context(#[from] ContextError),
}
