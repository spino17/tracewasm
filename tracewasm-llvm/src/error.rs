//! What the builders return when they refuse to build something.
//!
//! The errors are layered so that each builder returns the narrowest type that can
//! describe its failures — [`Cursor::build_alloca`](crate::instruction::cursor::Cursor)
//! yields an [`InstructionError`], not a catch-all — and every layer converts upward
//! through `#[from]`, so `?` composes without hand-written matches.
//!
//! ```text
//! BuildError
//! ├── TypeError
//! ├── TracewasmUtilsError        (interner capacity)
//! └── InstructionError
//!     ├── AllocaError
//!     ├── StoreError
//!     ├── RetError
//!     ├── GepError
//!     ├── PhiError ── ContextError
//!     └── ContextError
//! ```
//!
//! # Types are carried already rendered
//!
//! Every variant naming a type holds a `String`, not a [`Type`](crate::value::Type)
//! or a [`TyId`](crate::interner::TyId). Neither can print itself: an aggregate names
//! its children by id, so spelling one out needs the pool that issued them, and an
//! error cannot borrow the pool and outlive it. The failing path renders through
//! [`Context::display`](crate::cfg::context::Context::display) on the way out, which
//! costs an allocation on a path that is not taken in the common case.

use thiserror::Error;
use tracewasm_utils::error::TracewasmUtilsError;

/// Anything that can go wrong while building a module.
///
/// The top of the hierarchy: every other error in this module converts into it, so a
/// caller that does not care which layer failed can use this one type throughout.
#[derive(Error, Debug)]
pub enum BuildError {
    /// A value could not be given the type it was asked for.
    #[error("{0}")]
    TypeError(#[from] TypeError),
    /// A pool ran out of ids. Only [`try_intern`](tracewasm_utils::interner::Interner::try_intern)
    /// reports this; the pools here use the panicking `intern`, since their contents
    /// come from the compiler driving the builder rather than from untrusted input.
    #[error("{0}")]
    UtilsError(#[from] TracewasmUtilsError),
    /// An instruction could not be built.
    #[error("{0}")]
    InstructionError(#[from] InstructionError),
}

/// A value does not have, and cannot be given, the type a caller asked for.
#[derive(Error, Debug)]
pub enum TypeError {
    /// A conditional branch takes an `i1`, and this value is not one.
    #[error("value of type `{0}` cannot be converted into i1 value")]
    ValueToI1ValueFailed(String),
    /// A constant could not be folded into the requested type. Widths convert freely
    /// among integers and among floats; crossing between them, or reaching a pointer,
    /// needs a real instruction.
    #[error("constant with type `{0}` failed to be casted as `{1}`")]
    ConstantCastToProvidedTypeFailed(String, String),
}

/// Something about the module's own structure is wrong: a name that collides, is not
/// a legal identifier, or a signature that LLVM would not accept.
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
    /// A parameter is passed by value, so it needs a size. `llvm-as` refuses a `void`
    /// one with "void type only allowed for function results" and a function-typed one
    /// with "invalid type for function argument"; aggregates are fine.
    #[error("a function parameter cannot have type `{0}`: it has no size")]
    FunctionParamTypeNotSized(String),
    /// A result may be `void` — that is the one place `void` is allowed — but not a
    /// function type, which `llvm-as` refuses with "invalid function return type".
    #[error("a function cannot return type `{0}`")]
    FunctionResultTypeInvalid(String),
}

/// An instruction could not be built.
///
/// The variants here are the checks shared across instructions; the per-instruction
/// ones live in [`AllocaError`], [`StoreError`], [`RetError`], [`GepError`] and
/// [`PhiError`], each reachable through a `#[from]` arm.
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
    /// The type being loaded disagrees with what the pointer was traced back to.
    ///
    /// **Stricter than LLVM.** Under opaque pointers a `load` reinterprets whatever it
    /// is handed, so `llvm-as` accepts loading an `i32` through an `alloca double`.
    /// This is refused so the mismatch surfaces here rather than as a miscompile.
    #[error("loading `{0}` through a pointer to `{1}`")]
    LoadedTypeDoesNotMatchPointee(String, String),
    /// Nothing said what to load: no type was given and the pointer's pointee could
    /// not be inferred. A pointer that arrived as a function parameter has no
    /// defining instruction, so it always lands here — pass the type explicitly.
    #[error(
        "a `load` needs a type: none was given, and none could be inferred from the \
         pointer operand"
    )]
    LoadedTypeUnknown,
    /// An explicit alignment must be a power of two. `0` is not one — leaving the
    /// alignment off is how the ABI default is asked for.
    #[error("alignment must be a power of two, but got `{0}`")]
    AlignmentNotPowerOfTwo(u32),
    /// The block already ends in a terminator, so nothing may follow. Consuming the
    /// cursor prevents this for the cursor that branched; this catches a *second*
    /// cursor opened at the same block, which the type system cannot see.
    #[error("basic block `{0}` already ends in a branch, so nothing more can be added to it")]
    BasicBlockAlreadyTerminated(String),
    /// See [`AllocaError`].
    #[error("{0}")]
    Alloca(#[from] AllocaError),
    /// See [`StoreError`].
    #[error("{0}")]
    Store(#[from] StoreError),
    /// See [`RetError`].
    #[error("{0}")]
    Ret(#[from] RetError),
    /// See [`CallError`].
    #[error("{0}")]
    Call(#[from] CallError),
    /// See [`PhiError`].
    #[error("{0}")]
    Phi(#[from] PhiError),
    /// See [`GepError`].
    #[error("{0}")]
    Gep(#[from] GepError),
    /// A name could not be issued for the register the instruction defines.
    #[error("{0}")]
    Context(#[from] ContextError),
}

/// An `alloca` could not be built.
#[derive(Error, Debug)]
pub enum AllocaError {
    /// `alloca` reserves room for a value, so the type needs a size. `llvm-as`
    /// refuses `alloca void` and `alloca` of a function type.
    #[error("a value of type `{0}` cannot be allocated: `alloca` needs a sized type")]
    TypeNotAllocatable(String),
    /// The element count is not an integer. `i1` counts — `alloca i32, i1 %c`
    /// assembles — but a float does not.
    #[error("an `alloca` element count must have an integer type, but got `{0}`")]
    AllocaCountNotAnInteger(String),
    /// The count is an integer of the wrong width, and widening a register would need
    /// a `zext`/`sext` the caller did not ask for.
    #[error("an `alloca` element count of type `{0}` cannot be used as `{1}`")]
    AllocaCountTypeMismatch(String, String),
}

/// A `ret` could not be built.
#[derive(Error, Debug)]
pub enum RetError {
    /// `ret void` carries no operand.
    #[error("`ret void` takes no value, but one of type `{0}` was given")]
    ValueGivenForVoid(String),
    /// The value could not be folded into the declared return type.
    #[error("a value of type `{0}` cannot be returned as `{1}`")]
    ReturnedValueTypeMismatch(String, String),
    /// A non-`void` return needs something to return.
    #[error("returning no value needs the `void` type, but `{0}` was given")]
    NonVoidTypeWithoutValue(String),
    /// Neither a type nor a value was given, so there is nothing to return and
    /// nothing to infer a type from.
    #[error("a `ret` needs either a type or a value, and neither was given")]
    TypeAndValueBothAbsent,
    /// The return disagrees with the enclosing function's declared result.
    ///
    /// Checked against the *function*, not just the type/value pair: `ret i64 0` is
    /// internally consistent and still wrong inside an `i32` function, which
    /// `llvm-as` reports as "value doesn't match function result type".
    ///
    /// Fields: the function's name, the result it declares, and what was returned.
    #[error("`{0}` returns `{1}`, so it cannot return `{2}`")]
    DoesNotMatchFunctionResult(String, String, String),
}

/// A `call` could not be built.
#[derive(Error, Debug)]
pub enum CallError {
    /// No function of that name has been added to the module.
    ///
    /// The table holds only what
    /// [`Builder::define_function`](crate::cfg::builder::Builder::define_function) has
    /// registered so far, so this also covers a **forward call** — one to a function
    /// that will be added later — and a host import, which nothing declares.
    #[error("no function named `{0}` has been added to this module")]
    FunctionNotFound(String),
    /// The callee takes a different number of arguments.
    #[error("`{name}` takes `{expected}` argument(s), but `{given}` were given")]
    ParamCountMismatch {
        /// The callee's name.
        name: String,
        /// How many it declares.
        expected: usize,
        /// How many were passed.
        given: usize,
    },
    /// An argument's type differs from the callee's parameter type.
    ///
    /// Fields: the callee, the argument's position, the type declared for it, and
    /// the type actually given.
    #[error("`{0}` expects `{2}` for argument `{1}`, but got `{3}`")]
    ParamTypeMismatch(String, usize, String, String),
    /// An argument could not be folded into the type given for it.
    #[error("argument `{1}` of type `{2}` cannot be passed as `{3}` in a call to `{0}`")]
    ParamCastFailed(String, usize, String, String),
    /// The declared return type differs from the callee's.
    #[error("`{0}` returns `{1}`, so a call to it cannot return `{2}`")]
    ReturnTypeMismatch(String, String, String),
    /// A register name was given for a call that produces nothing. `llvm-as` refuses
    /// `%r = call void @g()` with "instructions returning void cannot have a name".
    #[error("a call to `{0}` returns `void`, so it cannot be assigned to a register")]
    RegisterNameForVoidCall(String),
}

/// A `store` could not be built.
#[derive(Error, Debug)]
pub enum StoreError {
    /// The value could not be folded into the declared type. A constant widens; a
    /// register has to match, since widening one needs a real instruction.
    #[error("a value of type `{0}` cannot be stored as `{1}`")]
    StoredValueTypeMismatch(String, String),
    /// The value disagrees with what the pointer was traced back to. Like the load
    /// check, this is stricter than LLVM on purpose.
    #[error("a value of type `{0}` cannot be stored through a pointer to `{1}`")]
    StoredValueDoesNotMatchPointee(String, String),
}

#[derive(Error, Debug)]
pub enum PhiError {
    /// Phis come first in a block, before any other instruction.
    #[error("phi instructions should be added at the start of the basic block")]
    PhiInstructionAddError,
    /// The block already ends in a terminator. Checked before
    /// [`PhiInstructionAddError`](Self::PhiInstructionAddError), since a terminated
    /// block satisfies both and this is the more useful of the two.
    #[error("basic block `{0}` already ends in a branch, so nothing more can be added to it")]
    BasicBlockAlreadyTerminated(String),
    /// The entry block has no predecessors, so a phi there selects on nothing.
    #[error("phi instructions cannot be added to the first basic block of the function")]
    PhiInstructionCannotBeAddedToEntryBasicBlock,
    /// A phi names one value per predecessor, so the same predecessor cannot appear
    /// twice.
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
    /// A name could not be issued for the register the phi defines.
    #[error("{0}")]
    Context(#[from] ContextError),
}

/// A `getelementptr` could not be built.
///
/// The walk that produces these descends the source type one index at a time. The
/// **first** index is skipped: it steps over the source type as pointer arithmetic
/// rather than into it, so only `indices[1..]` reach the checks below.
#[derive(Error, Debug)]
pub enum GepError {
    /// Every index scales an offset, so every one has to be an integer.
    #[error("a `getelementptr` index must have an integer type, but got `{0}`")]
    IndexNotAnInteger(String),
    /// `llvm-as` refuses an unsized base with "base element of getelementptr must be
    /// sized".
    #[error("a `getelementptr` source type must be sized, but got `{0}`")]
    SourceTypeNotSized(String),
    /// A source type was given *and* the pointer could be traced back, and the two
    /// disagree.
    #[error(
        "the source type `{0}` does not match `{1}`, the pointee type inferred from \
         the pointer operand"
    )]
    SourceTypeDoesNotMatchPointee(String, String),
    /// Nothing said what to index: no source type was given and the pointer's pointee
    /// could not be inferred. A pointer that arrived as a function parameter always
    /// lands here — pass the type explicitly.
    #[error(
        "a `getelementptr` needs a source type: none was given, and none could be \
         inferred from the pointer operand"
    )]
    SourceTypeUnknown,
    /// A struct index names a field, so it must be known now and must be an `i32`
    /// specifically — `llvm-as` refuses an `i64` one with "invalid getelementptr
    /// indices", even though array indices may be any width.
    #[error("an index into a struct must be a constant `i32`, but got `{0}`")]
    StructIndexNotAConstantI32(String),
    /// The field index is past the end, or negative.
    #[error("index `{index}` is out of range for a struct with `{fields}` field(s)")]
    StructIndexOutOfRange { index: i32, fields: usize },
    /// A constant array index is past the end, or negative.
    ///
    /// **Stricter than LLVM.** `getelementptr [4 x i32], ptr %p, i64 0, i64 10`
    /// assembles — out-of-range is runtime UB under `inbounds`, not an assembly
    /// error. Refused here so the mistake surfaces at construction. A non-constant
    /// index is not checked, since its value is not known yet.
    #[error("index `{index}` is out of range for an array of `{size}` element(s)")]
    ArrayIndexOutOfRange { index: u64, size: u64 },
    /// The walk reached a scalar with indices still to consume: only aggregates have
    /// anything to descend into.
    #[error("a value of type `{0}` cannot be indexed into")]
    TypeNotIndexable(String),
}
