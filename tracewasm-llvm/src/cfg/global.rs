use crate::{
    cfg::context::Context,
    interner::TyId,
    value::{ConstExpr, FuncSignature, Type},
};

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

pub struct GlobalVariable {
    pub(crate) ty: TyId,
    pub(crate) initializer: Option<ConstExpr>,
}

pub enum GlobalKind {
    Func(FuncSignature),
    Variable(GlobalVariable),
}

pub struct Global {
    pub(crate) linkage: Linkage,
    pub(crate) visiblity: Visiblity,
    pub(crate) kind: GlobalKind,
}

impl Global {
    pub fn pointee_ty(&self, ctx: &mut Context) -> TyId {
        match &self.kind {
            GlobalKind::Func(func_sig) => {
                ctx.ty_interner.intern(Type::Func(func_sig.clone())).into()
            }
            GlobalKind::Variable(variable) => variable.ty,
        }
    }
}
