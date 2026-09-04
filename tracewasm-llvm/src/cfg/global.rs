use crate::{
    cfg::{context::Context, function::FuncId},
    interner::{StrId, TyId},
    value::{ConstExpr, FuncSignature, Type},
};
use std::{fmt::Display, hash::Hash};

/// How a symbol is linked, and whether it is defined here at all.
///
/// Two of these are *declarations* — [`External`](Self::External) and
/// [`ExternWeak`](Self::ExternWeak) name something defined elsewhere, so they carry no
/// initializer. The rest are definitions and require one; `llvm-as` refuses
/// `@g = internal global i32` with "expected value token" and
/// `@g = external global i32 0` outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Defined elsewhere. Written only on a declaration: a *definition* with external
    /// linkage omits the keyword, since that is the default.
    External,
    /// Local to this module, not visible to the linker.
    Internal,
    /// Like [`Internal`](Self::Internal), but the symbol is renamed to avoid
    /// collisions.
    Private,
    /// May be overridden; not discarded if unused.
    Weak,
    /// May be merged with an identical definition, and discarded if unused.
    Linkonce,
    /// [`Linkonce`](Self::Linkonce), with a guarantee that all definitions agree.
    LinkonceOdr,
    /// [`Weak`](Self::Weak), with a guarantee that all definitions agree.
    WeakOdr,
    /// Tentative definition, in the C sense.
    Common,
    /// Arrays are concatenated at link time rather than merged.
    Appending,
    /// A definition available for inlining, not emitted as a symbol.
    AvailableExternally,
    /// A weak declaration: resolves to null if never defined.
    ExternWeak,
}

impl Linkage {
    /// Whether this linkage means "defined elsewhere", so no initializer is allowed.
    pub(crate) fn is_declaration(&self) -> bool {
        matches!(self, Linkage::External | Linkage::ExternWeak)
    }

    /// Whether this linkage is local to the module.
    ///
    /// LLVM requires a local symbol to have default visibility — `llvm-as` refuses
    /// `@g = internal hidden global i32 0` with "symbol with local linkage must have
    /// default visibility".
    pub(crate) fn is_local(&self) -> bool {
        matches!(self, Linkage::Internal | Linkage::Private)
    }
}

impl Display for Linkage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Linkage::External => "external",
            Linkage::Internal => "internal",
            Linkage::Private => "private",
            Linkage::Weak => "weak",
            Linkage::Linkonce => "linkonce",
            Linkage::LinkonceOdr => "linkonce_odr",
            Linkage::WeakOdr => "weak_odr",
            Linkage::Common => "common",
            Linkage::Appending => "appending",
            Linkage::AvailableExternally => "available_externally",
            Linkage::ExternWeak => "extern_weak",
        })
    }
}

/// Whether a symbol is visible outside the object it is defined in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Visible. The default, and the only one a local symbol may have.
    Default,
    /// Not visible to other objects.
    Hidden,
    /// Visible, but never pre-empted by another definition.
    Protected,
}

impl Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Visibility::Default => "default",
            Visibility::Hidden => "hidden",
            Visibility::Protected => "protected",
        })
    }
}

/// What kind of thing a [`GlobalId`] names.
///
/// Implemented by the three tag types — [`GlobalVar`], [`DefinedFunc`] and
/// [`DeclaredFunc`] — which exist only to distinguish `GlobalId<T>`s at compile time.
/// That is what keeps
/// [`add_basic_block`](crate::cfg::global::GlobalId::add_basic_block) on
/// `GlobalId<DefinedFunc>` alone: adding a block to a declaration or a variable is a
/// type error rather than a runtime check.
pub trait GlobalEntity: Clone + Copy + PartialEq + Eq + Hash {
    /// Erases the tag, giving the untyped [`Global`] the module stores.
    fn to_global(id: GlobalId<Self>) -> Global;
}

/// Tag for a function this module defines, carrying the body's arena id.
///
/// The only tag with a payload: a definition has blocks, and they live in the
/// context's arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefinedFunc(FuncId);

impl GlobalEntity for DefinedFunc {
    fn to_global(id: GlobalId<DefinedFunc>) -> Global {
        Global::DefinedFunc(id)
    }
}

impl DefinedFunc {
    /// Tags a function body. Only
    /// [`Builder::define_function`](crate::cfg::builder::Builder::define_function)
    /// calls this.
    pub(crate) fn new(id: FuncId) -> Self {
        DefinedFunc(id)
    }

    /// The body's arena id, for reaching the [`Function`](crate::cfg::function::Function)
    /// itself.
    pub(crate) fn raw(&self) -> FuncId {
        self.0
    }
}

/// Tag for a function this module only declares.
///
/// Carries nothing: a declaration has no body, so there is no arena id to hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeclaredFunc;

impl GlobalEntity for DeclaredFunc {
    fn to_global(id: GlobalId<DeclaredFunc>) -> Global {
        Global::DeclaredFunc(id)
    }
}

/// Tag for a global variable. Carries nothing; the data lives in [`GlobalData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalVar;

impl GlobalEntity for GlobalVar {
    fn to_global(id: GlobalId<GlobalVar>) -> Global {
        Global::Variable(id)
    }
}

/// Anything a module names with an `@`, with its tag erased.
///
/// This is what a [`Value`](crate::value::Value) holds once a global is used as an
/// operand — by then the distinction no longer matters, since all three are addresses
/// and all three render as `@name`.
#[derive(Debug, Clone, Copy)]
pub enum Global {
    /// A global variable.
    Variable(GlobalId<GlobalVar>),
    /// A function defined in this module.
    DefinedFunc(GlobalId<DefinedFunc>),
    /// A function declared but not defined here.
    DeclaredFunc(GlobalId<DeclaredFunc>),
}

impl Global {
    /// The name it is written under, whichever kind it is.
    pub(crate) fn name(&self) -> StrId {
        match self {
            Global::Variable(var) => var.name,
            Global::DefinedFunc(func) => func.name,
            Global::DeclaredFunc(func) => func.name,
        }
    }
}

/// A handle to something the module names with an `@`, tagged with what it is.
///
/// The tag is a compile-time distinction only — every global is identified by its
/// name, since LLVM gives module-level symbols one namespace. What the tag buys is
/// that operations belonging to one kind cannot be reached on another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalId<T: GlobalEntity> {
    pub(crate) name: StrId,
    pub(crate) tag: T,
}

/// A global variable's own data: what it holds, and what it starts as.
///
/// The two are coupled. An initializer makes this a *definition*, and its type must
/// match `ty` exactly. Without one it is a declaration, which is why
/// [`Linkage::External`] and [`Linkage::ExternWeak`] are the only linkages that
/// accept a missing initializer.
pub struct GlobalVariable {
    pub(crate) ty: TyId,
    pub(crate) initializer: Option<ConstExpr>,
}

/// Which of the two shapes a global takes.
pub enum GlobalKind {
    /// A function, defined or declared — either way only its signature is stored
    /// here; a definition's body lives in the arena.
    Func(FuncSignature),
    /// A variable.
    Variable(GlobalVariable),
}

/// Everything the module records about one global, under its name.
///
/// One shape for all three kinds, which is what lets a
/// [`Value`](crate::value::Value) built from any global resolve its pointee the same
/// way — see [`pointee_ty`](Self::pointee_ty).
pub struct GlobalData {
    pub(crate) linkage: Linkage,
    pub(crate) visibility: Visibility,
    pub(crate) kind: GlobalKind,
}

impl GlobalData {
    /// What a pointer to this global points at.
    ///
    /// A global's *own* type is always `ptr` — `@g` is an address whatever it names.
    /// This is the other half: the type behind that address, which is what lets a
    /// `load` or `store` through a global omit its type.
    ///
    /// For a function that is the function type, so `@f` points at `i32 (i32)`.
    pub fn pointee_ty(&self, ctx: &Context) -> Type {
        match &self.kind {
            GlobalKind::Func(func_sig) => Type::Func(func_sig.clone()),
            GlobalKind::Variable(variable) => ctx.ty_interner.value(variable.ty.raw()).clone(),
        }
    }
}

/// A callable function, however the module came by it.
///
/// A call names its callee with one of these rather than with a string, and the only
/// sources of one are
/// [`define_function`](crate::cfg::builder::Builder::define_function) and
/// [`declare_function`](crate::cfg::builder::Builder::declare_function). So a call to
/// a function the module does not have cannot be written — there is no handle to pass
/// — and the signature behind it is guaranteed to be on record.
///
/// The two are separate at every other point in the API, because only a definition has
/// blocks to add. They come together here because a call does not care: it needs a
/// name and a signature, and both kinds have those. `From` is implemented for each, so
/// a call site can write `f.into()`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum FuncRef {
    /// A function this module defines.
    Defined(GlobalId<DefinedFunc>),
    /// A function this module declares but does not define — a host import, or
    /// anything else resolved at link time.
    Declared(GlobalId<DeclaredFunc>),
}

impl FuncRef {
    pub(crate) fn name(&self) -> StrId {
        match self {
            FuncRef::Defined(func) => func.name,
            FuncRef::Declared(func) => func.name,
        }
    }
}

impl From<GlobalId<DefinedFunc>> for FuncRef {
    fn from(value: GlobalId<DefinedFunc>) -> Self {
        FuncRef::Defined(value)
    }
}

impl From<GlobalId<DeclaredFunc>> for FuncRef {
    fn from(value: GlobalId<DeclaredFunc>) -> Self {
        FuncRef::Declared(value)
    }
}
