use crate::{error::BuildError, interner::StrId};

pub enum Type {
    I1,
}

impl Type {
    fn is_i1(&self) -> bool {
        matches!(self, Type::I1)
    }
}

impl ToString for Type {
    fn to_string(&self) -> String {
        let r = match self {
            Type::I1 => "i1",
        };

        r.to_string()
    }
}

pub struct Register {
    name: StrId,
}

pub enum ValueKind {
    Reg(Register),
    Const, // interned constant it!
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
