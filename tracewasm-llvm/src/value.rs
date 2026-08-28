use crate::{
    error::BuildError,
    interner::{ConstId, StrId},
};

pub enum Type {
    Void,
    I1,
    I8,
    I16,
    I32,
    I64,
    Half,
    Bfloat,
    Float,
    Double,
    Ptr,
    Array { size: u64, element_ty: Box<Type> },
    Struct { fields: Vec<Type>, packed: bool },
}

impl Type {
    fn is_i1(&self) -> bool {
        matches!(self, Type::I1)
    }
}

impl ToString for Type {
    fn to_string(&self) -> String {
        todo!()
    }
}

pub struct Register {
    name: StrId,
}

pub enum ValueKind {
    Reg(Register),
    Const(ConstId),
}

pub struct Value {
    ty: Type,
    kind: ValueKind,
}

impl Value {
    pub fn into_i1(self) -> Result<I1Value, BuildError> {
        if !self.ty.is_i1() {
            return Err(BuildError::ValueToI1ValueFailed(self.ty.to_string()));
        }

        Ok(I1Value { kind: self.kind })
    }
}

pub struct I1Value {
    kind: ValueKind,
}
