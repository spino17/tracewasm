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
//!     ├── CallError
//!     ├── GepError
//!     ├── ICmpError
//!     ├── FCmpError
//!     ├── IArithmeticError
//!     ├── FArithmeticError
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
    /// Two globals cannot share a name. LLVM gives module-level symbols **one**
    /// namespace, so this covers variables, definitions and declarations alike: a
    /// variable may not reuse a function's name, and a function may not be both
    /// declared and defined. Emitting either would produce two `@name`s in one module.
    #[error("a global named `{0}` already exists in this module")]
    DuplicateGlobalName(String),
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
    /// A global variable holds a value, so its type needs a size.
    ///
    /// The same pair as everywhere else is excluded: `llvm-as` refuses
    /// `@g = external global void` with "void type only allowed for function results"
    /// and a function-typed one with "invalid type for global variable". Aggregates
    /// and pointers are fine.
    #[error("a global variable cannot have type `{0}`: it has no size")]
    GlobalVariableTypeNotSized(String),
    /// A global's initializer is not of the type the global was declared with.
    ///
    /// The match has to be **exact**, not merely compatible: `llvm-as` refuses
    /// `@g = global i32 true` with "constant expression type mismatch" even though an
    /// `i1` is an integer, and `@g = global double 0` with "integer constant must
    /// have integer type".
    ///
    /// One mismatch LLVM cannot catch is a width one — an initializer renders bare,
    /// so `@g = global i64 <i32 0>` writes `@g = global i64 0` and the declared type
    /// silently wins. Checking here is what makes that visible.
    ///
    /// Fields: the type the global was declared with, and the initializer's type.
    #[error("a global of type `{0}` cannot be initialised with a value of type `{1}`")]
    GlobalInitializerTypeMismatch(String, String),
    /// A global was declared with neither a type nor an initializer, so there is
    /// nothing to give it and nothing to infer one from.
    #[error("a global variable needs either a type or an initializer, and neither was given")]
    GlobalTypeAndInitializerBothAbsent,
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
    /// See [`ICmpError`].
    #[error("{0}")]
    ICmp(#[from] ICmpError),
    /// See [`FCmpError`].
    #[error("{0}")]
    FCmp(#[from] FCmpError),
    /// See [`IArithmeticError`].
    #[error("{0}")]
    IArithmetic(#[from] IArithmeticError),
    /// See [`FArithmeticError`].
    #[error("{0}")]
    FArithmetic(#[from] FArithmeticError),
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
    /// [`define_function`](crate::cfg::builder::Builder::define_function) and
    /// [`declare_function`](crate::cfg::builder::Builder::declare_function) have
    /// registered *so far*, so this also covers a **forward call** — one to a
    /// function that will be added later — even though LLVM makes every function in
    /// a module mutually visible. A host import is fine once declared.
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
    /// [`build_call`](crate::instruction::cursor::Cursor::build_call) was used for a
    /// callee that returns `void`.
    ///
    /// A `void` call defines no register — `llvm-as` refuses `%r = call void @g()`
    /// with "instructions returning void cannot have a name" — so it cannot yield the
    /// [`Value`](crate::value::Value) this builder promises. Use
    /// [`build_void_call`](crate::instruction::cursor::Cursor::build_void_call), which
    /// takes no register name and returns nothing.
    #[error("`{0}` returns `void`, so it has no value: use `build_void_call`")]
    VoidCalleeNeedsVoidCall(String),
    /// [`build_void_call`](crate::instruction::cursor::Cursor::build_void_call) was
    /// used for a callee that returns a value.
    ///
    /// Discarding it would leave the result unnamed and unreachable. Use
    /// [`build_call`](crate::instruction::cursor::Cursor::build_call), which hands
    /// back the register it defines.
    #[error("`{0}` returns `{1}`, not `void`: use `build_call`")]
    NonVoidCalleeNeedsValueCall(String, String),
}

/// An `icmp` could not be built.
///
/// LLVM requires both operands to have the *same* type — `llvm-as` refuses
/// `icmp slt i64 %a, %b` where `%b` is an `i32` with "'%b' defined with type 'i32' but
/// expected 'i64'". Where the predicate says how to read the operands, a differing-width
/// **constant** is widened to match; everything else is refused here.
#[derive(Error, Debug)]
pub enum ICmpError {
    /// The operands have different types and could not be brought to a common one.
    ///
    /// Either the narrower operand is a register — widening one needs a real
    /// `zext`/`sext`, which this builder will not insert — or it is a constant that
    /// does not fit the common type under the predicate's signedness.
    ///
    /// Fields: the predicate, and the two operand types.
    #[error("`icmp {0}` cannot bring operands of type `{1}` and `{2}` to a common type")]
    OperandsNotCastable(String, String, String),
    /// `eq` and `ne` require operands that *already* share a type.
    ///
    /// Unlike the ordered predicates, these carry no signedness — LLVM has one `eq`,
    /// not a signed and an unsigned one — so there is nothing to say whether a
    /// narrower operand should be zero- or sign-extended. Widening is therefore
    /// refused rather than guessed: `icmp eq i64 %x, -1` and
    /// `icmp eq i64 %x, 4294967295` disagree, and only the caller knows which was
    /// meant.
    ///
    /// Fields: the predicate, and the two operand types.
    #[error("`icmp {0}` needs both operands to have the same type, but got `{1}` and `{2}`")]
    OperandTypesDiffer(String, String, String),
    /// An explicit type was given for an `eq`/`ne` that its operands do not have.
    ///
    /// For these predicates the type argument is a *check*, not a coercion, for the
    /// same reason as [`OperandTypesDiffer`](Self::OperandTypesDiffer).
    ///
    /// Fields: the predicate, the type given, and the type the operands have.
    #[error("`icmp {0}` was given type `{1}`, but its operands have type `{2}`")]
    ProvidedTypeDoesNotMatchOperands(String, String, String),
    /// `icmp` compares integers or pointers, and this is neither.
    ///
    /// Pointers are allowed with every predicate, signed ones included — `llvm-as`
    /// accepts both `icmp ult ptr` and `icmp slt ptr`. Floats are not: it refuses
    /// `icmp eq float` with "icmp requires integer operands". Comparing floats needs
    /// `fcmp`.
    #[error("`icmp` compares integers or pointers, but its operands have type `{0}`")]
    OperandTypeNotComparable(String),
}

/// An `fcmp` could not be built.
///
/// Like `icmp`, LLVM requires both operands to have the same type. Unlike `icmp`,
/// there is no signedness to resolve — a float's sign is part of its format — so a
/// narrower float **constant** widens by `fpext`, which is exact and needs no choice
/// made on the caller's behalf.
#[derive(Error, Debug)]
pub enum FCmpError {
    /// The operands have different types and could not be brought to a common one.
    ///
    /// Either the narrower operand is a register — widening one needs a real `fpext`,
    /// which this builder will not insert — or it is a constant that does not fit the
    /// common type. Integer operands land here too: nothing bridges the integer and
    /// float families without a real `sitofp`.
    ///
    /// Fields: the predicate, and the two operand types.
    #[error("`fcmp {0}` cannot bring operands of type `{1}` and `{2}` to a common type")]
    OperandsNotCastable(String, String, String),
    /// `fcmp` compares floats, and this is not one.
    ///
    /// `half`, `bfloat`, `float` and `double` are all accepted — `llvm-as` assembles
    /// `fcmp oeq` on every one of them. Integers are not: comparing those is `icmp`.
    #[error("`fcmp` compares floating-point values, but its operands have type `{0}`")]
    OperandTypeNotFloat(String),
}

/// An integer arithmetic, bitwise or shift instruction could not be built.
#[derive(Error, Debug)]
pub enum IArithmeticError {
    /// The operands have different types and could not be brought to a common one.
    ///
    /// Reached only by the six operations that carry a signedness. Either the narrower
    /// operand is a register — widening one needs a real `zext`/`sext` — or it is a
    /// constant that does not fit under that reading.
    ///
    /// Fields: the operation, and the two operand types.
    #[error("`{0}` cannot bring operands of type `{1}` and `{2}` to a common type")]
    OperandsNotCastable(String, String, String),
    /// An operation with no signedness was given operands of two types.
    ///
    /// `add`, `sub`, `mul`, `shl`, `and`, `or` and `xor` have a single LLVM opcode
    /// each, so nothing says whether a narrower operand should be zero- or
    /// sign-extended — and the choice changes the result. `add i64 100, -1` is 99,
    /// while the same `i32` constant zero-extended gives 4294967395. Widening is
    /// refused rather than guessed.
    ///
    /// Fields: the operation, and the two operand types.
    #[error("`{0}` needs both operands to have the same type, but got `{1}` and `{2}`")]
    OperandTypesDiffer(String, String, String),
    /// An explicit type was given that the operands do not have.
    ///
    /// For an operation with no signedness the type argument is a *check*, not a
    /// coercion, for the same reason as
    /// [`OperandTypesDiffer`](Self::OperandTypesDiffer).
    ///
    /// Fields: the operation, the type given, and the type the operands have.
    #[error("`{0}` was given type `{1}`, but its operands have type `{2}`")]
    ProvidedTypeDoesNotMatchOperands(String, String, String),
    /// These operations take integers. Floats use the `f`-prefixed instructions.
    #[error("`{0}` takes integer operands, but got ones of type `{1}`")]
    OperandTypeNotInteger(String, String),
}

/// A floating-point arithmetic instruction could not be built.
#[derive(Error, Debug)]
pub enum FArithmeticError {
    /// The operands have different types and could not be brought to a common one.
    ///
    /// Integer operands land here too: nothing bridges the two families without a
    /// real `sitofp`.
    ///
    /// Fields: the operation, and the two operand types.
    #[error("`{0}` cannot bring operands of type `{1}` and `{2}` to a common type")]
    OperandsNotCastable(String, String, String),
    /// These operations take floats. Integers use the unprefixed instructions.
    #[error("`{0}` takes floating-point operands, but got ones of type `{1}`")]
    OperandTypeNotFloat(String, String),
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

/// A phi node could not be built.
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
    StructIndexOutOfRange {
        /// The index that was given.
        index: i32,
        /// How many fields the struct has.
        fields: usize,
    },
    /// A constant array index is past the end, or negative.
    ///
    /// **Stricter than LLVM.** `getelementptr [4 x i32], ptr %p, i64 0, i64 10`
    /// assembles — out-of-range is runtime UB under `inbounds`, not an assembly
    /// error. Refused here so the mistake surfaces at construction. A non-constant
    /// index is not checked, since its value is not known yet.
    #[error("index `{index}` is out of range for an array of `{size}` element(s)")]
    ArrayIndexOutOfRange {
        /// The index that was given.
        index: u64,
        /// How many elements the array has.
        size: u64,
    },
    /// The walk reached a scalar with indices still to consume: only aggregates have
    /// anything to descend into.
    #[error("a value of type `{0}` cannot be indexed into")]
    TypeNotIndexable(String),
}
