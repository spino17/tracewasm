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
pub enum Visiblity {
    /// Visible. The default, and the only one a local symbol may have.
    Default,
    /// Not visible to other objects.
    Hidden,
    /// Visible, but never pre-empted by another definition.
    Protected,
}

impl Display for Visiblity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Visiblity::Default => "default",
            Visiblity::Hidden => "hidden",
            Visiblity::Protected => "protected",
        })
    }
}

pub trait GlobalEntity: Clone + Copy + PartialEq + Eq + Hash {
    fn to_global(id: GlobalId<Self>) -> Global;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefinedFunc(FuncId);

impl GlobalEntity for DefinedFunc {
    fn to_global(id: GlobalId<DefinedFunc>) -> Global {
        Global::DefinedFunc(id)
    }
}

impl DefinedFunc {
    pub(crate) fn new(id: FuncId) -> Self {
        DefinedFunc(id)
    }

    pub(crate) fn raw(&self) -> FuncId {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeclaredFunc;

impl GlobalEntity for DeclaredFunc {
    fn to_global(id: GlobalId<DeclaredFunc>) -> Global {
        Global::DeclaredFunc(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalVar;

impl GlobalEntity for GlobalVar {
    fn to_global(id: GlobalId<GlobalVar>) -> Global {
        Global::Variable(id)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Global {
    Variable(GlobalId<GlobalVar>),
    DefinedFunc(GlobalId<DefinedFunc>),
    DeclaredFunc(GlobalId<DeclaredFunc>),
}

impl Global {
    pub(crate) fn name(&self) -> StrId {
        match self {
            Global::Variable(var) => var.name,
            Global::DefinedFunc(func) => func.name,
            Global::DeclaredFunc(func) => func.name,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalId<T: GlobalEntity> {
    pub(crate) name: StrId,
    pub(crate) tag: T,
}

pub struct GlobalVariable {
    pub(crate) ty: TyId,
    pub(crate) initializer: Option<ConstExpr>,
}

pub enum GlobalKind {
    Func(FuncSignature),
    Variable(GlobalVariable),
}

pub struct GlobalData {
    pub(crate) linkage: Linkage,
    pub(crate) visiblity: Visiblity,
    pub(crate) kind: GlobalKind,
}

impl GlobalData {
    pub fn pointee_ty(&self, ctx: &Context) -> Type {
        match &self.kind {
            GlobalKind::Func(func_sig) => Type::Func(func_sig.clone()),
            GlobalKind::Variable(variable) => ctx.ty_interner.value(variable.ty.raw()).clone(),
        }
    }
}
