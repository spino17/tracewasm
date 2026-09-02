use crate::{
    cfg::{context::Context, function::FuncId},
    interner::{StrId, TyId},
    value::{ConstExpr, FuncSignature, Type},
};
use std::hash::Hash;

pub enum Linkage {
    External,
    Internal,
    Private,
    Weak,
    Linkonce,
    LinkonceOdr,
    WeakOdr,
    Common,
    Appending,
    AvailableExternally,
    ExternWeak,
}

pub enum Visiblity {
    Default,
    Hidden,
    Protected,
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
