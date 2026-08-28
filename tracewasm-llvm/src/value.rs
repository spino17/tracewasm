use std::fmt::{Debug, Display};

use ordered_float::OrderedFloat;

use crate::{
    error::BuildError,
    interner::{ConstId, ConstInterner, StrId},
};

#[derive(PartialEq, Eq)]
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

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Debug for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Type {
    fn is_i1(&self) -> bool {
        matches!(self, Type::I1)
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

    pub fn from_const<C: Const>(
        val: C,
        optional_cast: Option<Type>,
        interner: &mut ConstInterner,
    ) -> Result<Self, BuildError> {
        let ty = C::ty();

        let val = if let Some(ty) = optional_cast {
            let Some(c) = val.try_cast(&ty) else {
                return Err(BuildError::ConstantCastToProvidedTypeFailed(C::ty(), ty));
            };

            c
        } else {
            val.into_const()
        };

        let const_id = interner.intern(val)?;

        Ok(Value {
            ty,
            kind: ValueKind::Const(const_id.into()),
        })
    }
}

pub struct I1Value {
    kind: ValueKind,
}

impl From<I1Value> for Value {
    fn from(value: I1Value) -> Self {
        Value {
            ty: Type::I1,
            kind: value.kind,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstValue {
    I1(i8),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Float(OrderedFloat<f32>),
    Double(OrderedFloat<f64>),
}

pub trait Const {
    fn ty() -> Type;

    fn is_cast_valid(ty: Type) -> bool {
        ty == Self::ty()
    }

    fn into_const(self) -> ConstValue;

    fn try_cast(&self, ty: &Type) -> Option<ConstValue>;
}

impl Const for bool {
    fn ty() -> Type {
        Type::I1
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I1(if self { 1 } else { 0 })
    }

    fn try_cast(&self, ty: &Type) -> Option<ConstValue> {
        if matches!(ty, Type::I1) {
            Some(ConstValue::I1(if *self { 1 } else { 0 }))
        } else {
            None
        }
    }
}

impl Const for i8 {
    fn ty() -> Type {
        Type::I8
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I8(self)
    }

    fn try_cast(&self, ty: &Type) -> Option<ConstValue> {
        let v = match ty {
            Type::I8 => ConstValue::I8(*self),
            Type::I16 => ConstValue::I16(*self as i16),
            Type::I32 => ConstValue::I32(*self as i32),
            Type::I64 => ConstValue::I64(*self as i64),
            _ => return None,
        };

        Some(v)
    }
}

impl Const for i16 {
    fn ty() -> Type {
        Type::I16
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I16(self)
    }

    fn try_cast(&self, ty: &Type) -> Option<ConstValue> {
        let v = match ty {
            Type::I8 => ConstValue::I8(*self as i8),
            Type::I16 => ConstValue::I16(*self),
            Type::I32 => ConstValue::I32(*self as i32),
            Type::I64 => ConstValue::I64(*self as i64),
            _ => return None,
        };

        Some(v)
    }
}

impl Const for i32 {
    fn ty() -> Type {
        Type::I32
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I32(self)
    }

    fn try_cast(&self, ty: &Type) -> Option<ConstValue> {
        let v = match ty {
            Type::I8 => ConstValue::I8(*self as i8),
            Type::I16 => ConstValue::I16(*self as i16),
            Type::I32 => ConstValue::I32(*self),
            Type::I64 => ConstValue::I64(*self as i64),
            _ => return None,
        };

        Some(v)
    }
}

impl Const for i64 {
    fn ty() -> Type {
        Type::I64
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I64(self)
    }

    fn try_cast(&self, ty: &Type) -> Option<ConstValue> {
        let v = match ty {
            Type::I8 => ConstValue::I8(*self as i8),
            Type::I16 => ConstValue::I16(*self as i16),
            Type::I32 => ConstValue::I32(*self as i32),
            Type::I64 => ConstValue::I64(*self),
            _ => return None,
        };

        Some(v)
    }
}

impl Const for f32 {
    fn ty() -> Type {
        Type::Float
    }

    fn into_const(self) -> ConstValue {
        ConstValue::Float(OrderedFloat(self))
    }

    fn try_cast(&self, ty: &Type) -> Option<ConstValue> {
        let v = match ty {
            Type::Float => ConstValue::Float(OrderedFloat(*self)),
            Type::Double => ConstValue::Double(OrderedFloat(*self as f64)),
            _ => return None,
        };

        Some(v)
    }
}

impl Const for f64 {
    fn ty() -> Type {
        Type::Double
    }

    fn into_const(self) -> ConstValue {
        ConstValue::Double(OrderedFloat(self))
    }

    fn try_cast(&self, ty: &Type) -> Option<ConstValue> {
        let v = match ty {
            Type::Float => ConstValue::Float(OrderedFloat(*self as f32)),
            Type::Double => ConstValue::Double(OrderedFloat(*self)),
            _ => return None,
        };

        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_from_constants() {
        let mut interner = ConstInterner::default();

        let a = Value::from_const(32, None, &mut interner);
        let b = Value::from_const(2.3, Some(Type::I32), &mut interner);
        let c = Value::from_const(true, None, &mut interner);
        let d = Value::from_const(true, Some(Type::I1), &mut interner);
        let e = Value::from_const(false, Some(Type::I32), &mut interner);

        assert_eq!(a.is_ok(), true);
        assert_eq!(b.is_err(), true);
        assert_eq!(c.is_ok(), true);
        assert_eq!(d.is_ok(), true);
        assert_eq!(e.is_err(), true);
    }
}
