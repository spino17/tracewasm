use crate::{
    error::BuildError,
    interner::{ConstId, ConstInterner, StrId, StrInterner},
};
use ordered_float::OrderedFloat;
use std::{
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    mem::discriminant,
};

#[derive(PartialEq, Eq, Clone)]
pub struct FuncSignature {
    params: Vec<Type>,
    result: Box<Type>,
}

#[derive(PartialEq, Eq, Clone)]
pub enum Type {
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
    Func(FuncSignature),
    Void,
}

impl Display for Type {
    /// Writes the type as LLVM spells it, so a rendered error reads the same as the
    /// IR it is about.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::I1 => f.write_str("i1"),
            Type::I8 => f.write_str("i8"),
            Type::I16 => f.write_str("i16"),
            Type::I32 => f.write_str("i32"),
            Type::I64 => f.write_str("i64"),
            Type::Half => f.write_str("half"),
            Type::Bfloat => f.write_str("bfloat"),
            Type::Float => f.write_str("float"),
            Type::Double => f.write_str("double"),
            Type::Ptr => f.write_str("ptr"),
            Type::Void => f.write_str("void"),
            Type::Array { size, element_ty } => write!(f, "[{size} x {element_ty}]"),
            Type::Struct { fields, packed } => {
                let (open, close) = if *packed {
                    ("<{ ", " }>")
                } else {
                    ("{ ", " }")
                };

                f.write_str(open)?;

                for (i, field) in fields.iter().enumerate() {
                    if i != 0 {
                        f.write_str(", ")?;
                    }

                    write!(f, "{field}")?;
                }

                f.write_str(close)
            }
            Type::Func(signature) => write!(f, "{signature}"),
        }
    }
}

impl Display for FuncSignature {
    /// `<result> (<params>)`, the order LLVM writes a function type in — result
    /// first, which is the opposite of how the signature reads in source.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (", self.result)?;

        for (i, param) in self.params.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }

            write!(f, "{param}")?;
        }

        f.write_str(")")
    }
}

impl Debug for Type {
    /// The same rendering as [`Display`]. `BuildError` carries a `Type`, so a
    /// `Debug`-formatted error — which is what `unwrap` prints — has to work too.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl Type {
    pub fn is_i1(&self) -> bool {
        matches!(self, Type::I1)
    }

    pub fn is_ptr(&self) -> bool {
        matches!(self, Type::Ptr)
    }

    pub fn is_first_class(&self) -> bool {
        !matches!(self, Type::Void | Type::Func(_))
    }
}

#[derive(Debug, Clone)]
pub enum ValueKind {
    Reg(Register),
    Const(ConstId),
}

#[derive(Debug, Clone)]
pub struct Value {
    ty: Type,
    kind: ValueKind,
}

impl Value {
    pub fn ty(&self) -> &Type {
        &self.ty
    }

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
        let (val, ty) = if let Some(ty) = optional_cast {
            let Some(c) = val.try_cast(&ty) else {
                return Err(BuildError::ConstantCastToProvidedTypeFailed(C::ty(), ty));
            };

            (c, ty)
        } else {
            (val.into_const(), C::ty())
        };

        let const_id = interner.intern(val)?;

        Ok(Value {
            ty,
            kind: ValueKind::Const(const_id.into()),
        })
    }

    pub fn from_register(
        name: String,
        ty: Type,
        interner: &mut StrInterner,
    ) -> Result<Self, BuildError> {
        let reg_id: StrId = interner.intern(name)?.into();

        Ok(Value {
            ty,
            kind: ValueKind::Reg(Register { name: reg_id }),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Register {
    name: StrId,
}

/// A constant the module uses, interned into a per-context pool.
///
/// Identity is by **variant and bit pattern**, not by numeric equality — see the
/// hand-written [`PartialEq`] below for why that is not merely pedantic.
///
/// The float arms hold `OrderedFloat` only because `f32`/`f64` are not `Ord`; its
/// own `Hash` is not used, since it canonicalises `-0.0` to `+0.0` and every NaN
/// alike — see the hand-written [`Hash`] below.
#[derive(Clone, Copy, Debug)]
pub enum ConstValue {
    I1(i8),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Float(OrderedFloat<f32>),
    Double(OrderedFloat<f64>),
    NullPtr,
}

/// Two constants are the same constant only if they are the same *variant* and the
/// same *bits*. Both halves are load-bearing, because this is the interner's dedup
/// key and a merge means two operands sharing one pool entry.
///
/// **Same variant.** LLVM types a constant, so `i8 0` and `i32 0` are different
/// constants even though they are the same number; merging them would emit one where
/// the other was meant.
///
/// **Same bits, not numerically equal.** Numeric equality would merge `+0.0` with
/// `-0.0`, and every NaN with every other. The sign of a zero survives `fdiv`
/// (`1.0 / -0.0` is `-inf`), `copysign` and `llvm.minnum`/`maxnum`, and LLVM prints
/// the two differently — so collapsing them would emit a constant the source never
/// asked for. Comparing bits is also what makes the [`Eq`] impl below sound: a NaN
/// constant has to equal itself, or the pool could neither find nor dedup it.
impl PartialEq for ConstValue {
    fn eq(&self, other: &Self) -> bool {
        match self {
            ConstValue::I1(first) => {
                if let ConstValue::I1(second) = other {
                    first == second
                } else {
                    false
                }
            }
            ConstValue::I8(first) => {
                if let ConstValue::I8(second) = other {
                    first == second
                } else {
                    false
                }
            }
            ConstValue::I16(first) => {
                if let ConstValue::I16(second) = other {
                    first == second
                } else {
                    false
                }
            }
            ConstValue::I32(first) => {
                if let ConstValue::I32(second) = other {
                    first == second
                } else {
                    false
                }
            }
            ConstValue::I64(first) => {
                if let ConstValue::I64(second) = other {
                    first == second
                } else {
                    false
                }
            }
            ConstValue::Float(first) => {
                if let ConstValue::Float(second) = other {
                    first.into_inner().to_bits() == second.into_inner().to_bits()
                } else {
                    false
                }
            }
            ConstValue::Double(first) => {
                if let ConstValue::Double(second) = other {
                    first.into_inner().to_bits() == second.into_inner().to_bits()
                } else {
                    false
                }
            }
            ConstValue::NullPtr => {
                matches!(other, ConstValue::NullPtr)
            }
        }
    }
}

impl Eq for ConstValue {}

/// Hashes the same thing [`PartialEq`] compares: the variant, then the bits.
///
/// Written out because the derive would hash the float arms through
/// `OrderedFloat`, which canonicalises — `-0.0` would land in `+0.0`'s bucket and
/// every NaN in one. That is *sound* against a bit-comparing `PartialEq`, since
/// unequal values may share a hash, but it puts values the pool deliberately keeps
/// apart into the same bucket. Hashing the bits keeps them apart there too.
impl Hash for ConstValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        discriminant(self).hash(state);

        match self {
            ConstValue::I1(v) | ConstValue::I8(v) => v.hash(state),
            ConstValue::I16(v) => v.hash(state),
            ConstValue::I32(v) => v.hash(state),
            ConstValue::I64(v) => v.hash(state),
            ConstValue::Float(v) => v.into_inner().to_bits().hash(state),
            ConstValue::Double(v) => v.into_inner().to_bits().hash(state),
            ConstValue::NullPtr => {}
        }
    }
}

pub trait Const {
    fn ty() -> Type;
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

#[derive(Clone, Copy)]
pub struct NullPtr;

impl Const for NullPtr {
    fn ty() -> Type {
        Type::Ptr
    }

    fn into_const(self) -> ConstValue {
        ConstValue::NullPtr
    }

    fn try_cast(&self, ty: &Type) -> Option<ConstValue> {
        if &NullPtr::ty() != ty {
            return None;
        }

        Some(self.into_const())
    }
}

#[derive(Debug)]
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

        assert!(a.is_ok());
        assert!(b.is_err());
        assert!(c.is_ok());
        assert!(d.is_ok());
        assert!(e.is_err());
    }

    /// The type a value reports is the one it was cast *to*, not the one the Rust
    /// literal had — otherwise a later `into_i1` or a type check reads the wrong
    /// answer.
    #[test]
    fn a_cast_sets_the_values_type_and_its_stored_variant() {
        let mut interner = ConstInterner::default();

        let widened = Value::from_const(7i8, Some(Type::I64), &mut interner).unwrap();

        assert_eq!(widened.ty, Type::I64);
        assert!(
            matches!(widened.kind, ValueKind::Const(_)),
            "a constant value holds a pool id"
        );

        // The stored constant is widened too, not left as the i8 it came from.
        assert_eq!(interner.values(), [ConstValue::I64(7)]);
    }

    /// No cast means the value keeps the source type, which is what `Const::ty`
    /// says it is.
    #[test]
    fn without_a_cast_the_source_type_is_kept() {
        let mut interner = ConstInterner::default();

        for (value, expected) in [
            (Value::from_const(1i8, None, &mut interner), Type::I8),
            (Value::from_const(1i16, None, &mut interner), Type::I16),
            (Value::from_const(1i32, None, &mut interner), Type::I32),
            (Value::from_const(1i64, None, &mut interner), Type::I64),
            (Value::from_const(1.0f32, None, &mut interner), Type::Float),
            (Value::from_const(1.0f64, None, &mut interner), Type::Double),
            (Value::from_const(true, None, &mut interner), Type::I1),
        ] {
            assert_eq!(value.unwrap().ty, expected);
        }
    }

    /// Integer casts go both ways between widths, and a narrowing cast folds the
    /// value the way LLVM's `trunc` would.
    #[test]
    fn integer_casts_widen_and_narrow() {
        let mut interner = ConstInterner::default();

        assert_eq!(
            300i32.try_cast(&Type::I8),
            Some(ConstValue::I8(300i32 as i8)),
            "narrowing truncates rather than refusing"
        );

        assert_eq!(1i8.try_cast(&Type::I32), Some(ConstValue::I32(1)));
        assert_eq!((-1i64).try_cast(&Type::I16), Some(ConstValue::I16(-1)));

        assert!(
            Value::from_const(300i32, Some(Type::I8), &mut interner).is_ok(),
            "so the builder accepts it too"
        );
    }

    /// `null` is a `ptr` constant, which is the type it has to report for a value
    /// built from it to be usable where a pointer is expected.
    #[test]
    fn a_null_pointer_is_typed_ptr() {
        let mut interner = ConstInterner::default();

        assert_eq!(NullPtr::ty(), Type::Ptr);
        assert_eq!(NullPtr.into_const(), ConstValue::NullPtr);

        let value = Value::from_const(NullPtr, None, &mut interner).unwrap();

        assert_eq!(value.ty, Type::Ptr);
        assert_eq!(interner.values(), [ConstValue::NullPtr]);
    }

    /// The only cast a null admits is the one that changes nothing. Anything else
    /// would need a real instruction — `ptrtoint` to reach an integer.
    #[test]
    fn a_null_pointer_casts_only_to_ptr() {
        assert_eq!(NullPtr.try_cast(&Type::Ptr), Some(ConstValue::NullPtr));

        for ty in [
            Type::I1,
            Type::I8,
            Type::I32,
            Type::I64,
            Type::Float,
            Type::Double,
            Type::Void,
            Type::Array {
                size: 1,
                element_ty: Box::new(Type::I8),
            },
        ] {
            assert_eq!(
                NullPtr.try_cast(&ty),
                None,
                "null must not cast to `{ty}` without an instruction"
            );
        }
    }

    /// And nothing casts *to* a pointer either: an integer would need `inttoptr`,
    /// so accepting one here would fold away a conversion that has to be emitted.
    #[test]
    fn nothing_else_casts_to_a_pointer() {
        assert_eq!(0i8.try_cast(&Type::Ptr), None);
        assert_eq!(0i32.try_cast(&Type::Ptr), None);
        assert_eq!(0i64.try_cast(&Type::Ptr), None);
        assert_eq!(0.0f32.try_cast(&Type::Ptr), None);
        assert_eq!(0.0f64.try_cast(&Type::Ptr), None);
        assert_eq!(false.try_cast(&Type::Ptr), None);
    }

    /// A null and an integer zero are different constants — `ptr null` is not
    /// `i64 0`, whatever the target's representation happens to be — so they must
    /// not share a pool entry.
    #[test]
    fn a_null_pointer_is_not_an_integer_zero() {
        let mut interner = ConstInterner::default();

        let null = interner.intern(ConstValue::NullPtr).unwrap();
        let zero_64 = interner.intern(ConstValue::I64(0)).unwrap();
        let zero_32 = interner.intern(ConstValue::I32(0)).unwrap();
        let zero_1 = interner.intern(ConstValue::I1(0)).unwrap();

        assert_ne!(null, zero_64);
        assert_ne!(null, zero_32);
        assert_ne!(null, zero_1);
        assert_eq!(interner.len(), 4, "four distinct constants");
    }

    /// Every null is the same null, so the pool holds one however many times it is
    /// asked for — the unit variant has no payload to tell two apart by.
    #[test]
    fn every_null_pointer_is_the_same_constant() {
        let mut interner = ConstInterner::default();

        let first = Value::from_const(NullPtr, None, &mut interner).unwrap();
        let again = Value::from_const(NullPtr, Some(Type::Ptr), &mut interner).unwrap();

        assert_eq!(first.ty, Type::Ptr);
        assert_eq!(again.ty, Type::Ptr);

        assert_eq!(
            interner.len(),
            1,
            "interning null twice, once through a cast, still costs one entry"
        );

        assert_eq!(
            ConstValue::NullPtr,
            ConstValue::NullPtr,
            "null equals itself"
        );
    }

    /// A refused cast has to say what it refused, and null is the one constant
    /// whose source type has no payload to print.
    #[test]
    fn a_refused_null_cast_names_both_types() {
        let mut interner = ConstInterner::default();

        let err = Value::from_const(NullPtr, Some(Type::I64), &mut interner)
            .expect_err("null does not cast to i64");

        let msg = err.to_string();

        assert!(msg.contains("ptr"), "missing the source type: {msg}");
        assert!(msg.contains("i64"), "missing the target type: {msg}");
        assert_eq!(interner.len(), 0, "a failed cast interns nothing");
    }

    /// Ints and floats are not interchangeable, in either direction — a real
    /// conversion needs an `sitofp`/`fptosi` instruction, not a constant cast.
    #[test]
    fn casts_between_integers_and_floats_are_refused() {
        assert_eq!(1i32.try_cast(&Type::Float), None);
        assert_eq!(1i64.try_cast(&Type::Double), None);
        assert_eq!(1.0f32.try_cast(&Type::I32), None);
        assert_eq!(1.0f64.try_cast(&Type::I64), None);
        assert_eq!(true.try_cast(&Type::I32), None);
    }

    /// Floats cast between the two widths and nowhere else.
    #[test]
    fn float_casts_cover_both_widths_only() {
        assert_eq!(
            1.5f32.try_cast(&Type::Double),
            Some(ConstValue::Double(OrderedFloat(1.5)))
        );

        assert_eq!(
            1.5f64.try_cast(&Type::Float),
            Some(ConstValue::Float(OrderedFloat(1.5)))
        );

        assert_eq!(1.5f32.try_cast(&Type::Half), None);
        assert_eq!(1.5f64.try_cast(&Type::Ptr), None);
    }

    /// A failed cast names both ends, which is the only way to tell which side was
    /// wrong from the message alone.
    #[test]
    fn a_failed_cast_reports_both_types() {
        let mut interner = ConstInterner::default();

        let err = Value::from_const(1.0f64, Some(Type::I32), &mut interner)
            .expect_err("f64 does not cast to i32");

        let msg = err.to_string();

        assert!(msg.contains("double"), "missing the source type: {msg}");
        assert!(msg.contains("i32"), "missing the target type: {msg}");
        assert_eq!(interner.len(), 0, "a failed cast interns nothing");
    }

    /// `into_i1` gates the conditional-branch operand, so it has to reject a
    /// non-`i1` by *returning*, not by panicking while building the message.
    #[test]
    fn into_i1_accepts_only_i1() {
        let mut interner = ConstInterner::default();

        let ok = Value::from_const(true, None, &mut interner).unwrap();

        assert!(ok.into_i1().is_ok());

        let not_i1 = Value::from_const(1i32, None, &mut interner).unwrap();
        let err = not_i1.into_i1().expect_err("i32 is not i1");

        assert!(
            err.to_string().contains("i32"),
            "the message must name the offending type: {err}"
        );
    }

    /// Round-tripping through `I1Value` must come back as an `i1`.
    #[test]
    fn an_i1_value_converts_back_to_a_value() {
        let mut interner = ConstInterner::default();

        let i1 = Value::from_const(true, None, &mut interner)
            .unwrap()
            .into_i1()
            .unwrap();

        assert_eq!(Value::from(i1).ty, Type::I1);
    }

    /// Equal constants share a pool entry; constants that differ in *type* do not,
    /// even when they carry the same number — `i8 0` and `i32 0` are different
    /// constants in the IR.
    #[test]
    fn constants_dedup_by_value_and_type() {
        let mut interner = ConstInterner::default();

        let a = interner.intern(ConstValue::I32(0)).unwrap();
        let b = interner.intern(ConstValue::I32(0)).unwrap();
        let c = interner.intern(ConstValue::I8(0)).unwrap();
        let d = interner.intern(ConstValue::I64(0)).unwrap();

        assert_eq!(a, b, "the same constant interns once");
        assert_ne!(a, c, "i32 0 and i8 0 are distinct");
        assert_ne!(a, d, "i32 0 and i64 0 are distinct");
        assert_eq!(interner.len(), 3);
    }

    /// `-0.0` is not `0.0` in IEEE-754 and LLVM prints them differently: the sign
    /// survives `fdiv`, `copysign` and `minnum`, so collapsing them into one pool
    /// entry would emit the wrong constant.
    #[test]
    fn positive_and_negative_zero_are_distinct_constants() {
        let mut interner = ConstInterner::default();

        let pos = interner
            .intern(ConstValue::Float(OrderedFloat(0.0)))
            .unwrap();
        let neg = interner
            .intern(ConstValue::Float(OrderedFloat(-0.0)))
            .unwrap();

        assert_ne!(pos, neg, "0.0 and -0.0 must not share a pool entry");
        assert_eq!(interner.len(), 2);

        let ConstValue::Float(back) = *interner.value(neg) else {
            panic!("expected a float")
        };

        assert!(
            back.into_inner().is_sign_negative(),
            "the sign must survive interning"
        );
    }

    /// The other half of comparing bits: a NaN has a sign and a payload, and two
    /// NaNs that differ in either are different constants. `OrderedFloat`'s numeric
    /// equality calls every NaN equal, so this is the pair that would merge.
    #[test]
    fn nans_are_distinguished_by_their_bits() {
        let mut interner = ConstInterner::default();

        let quiet = f64::NAN;
        let negative = -f64::NAN;
        let payload = f64::from_bits(f64::NAN.to_bits() | 0x3);

        let a = interner
            .intern(ConstValue::Double(OrderedFloat(quiet)))
            .unwrap();
        let b = interner
            .intern(ConstValue::Double(OrderedFloat(negative)))
            .unwrap();
        let c = interner
            .intern(ConstValue::Double(OrderedFloat(payload)))
            .unwrap();

        assert_ne!(a, b, "a NaN's sign bit is part of its identity");
        assert_ne!(a, c, "so is its payload");
        assert_eq!(interner.len(), 3);

        // And a NaN still equals *itself*, which is what lets the pool find one it
        // has already interned — numeric equality could not do this.
        assert_eq!(
            interner
                .intern(ConstValue::Double(OrderedFloat(quiet)))
                .unwrap(),
            a,
            "re-interning the same NaN must reuse its entry"
        );
        assert_eq!(interner.len(), 3);
    }

    /// A signature, so the function-type tests below read as the types they mean.
    fn func(params: Vec<Type>, result: Type) -> FuncSignature {
        FuncSignature {
            params,
            result: Box::new(result),
        }
    }

    /// LLVM writes a function type **result first**, which is the reverse of how the
    /// signature reads in source — so getting the order wrong would produce IR that
    /// parses as a different type rather than failing.
    #[test]
    fn a_function_type_renders_its_result_before_its_params() {
        let signature = func(vec![Type::I8, Type::Ptr], Type::I32);

        assert_eq!(signature.to_string(), "i32 (i8, ptr)");

        assert_eq!(
            Type::Func(signature).to_string(),
            "i32 (i8, ptr)",
            "the variant renders as its signature, with nothing added"
        );
    }

    /// A function taking nothing still has the parens, and `void` is a result like
    /// any other.
    #[test]
    fn a_function_type_with_no_params_keeps_its_parens() {
        assert_eq!(func(vec![], Type::Void).to_string(), "void ()");
        assert_eq!(func(vec![], Type::I1).to_string(), "i1 ()");
    }

    /// Params may be aggregates, so the renderer has to compose with the array and
    /// struct arms rather than assume a scalar.
    #[test]
    fn a_function_type_composes_with_aggregate_params() {
        let signature = func(
            vec![
                Type::Array {
                    size: 4,
                    element_ty: Box::new(Type::I32),
                },
                Type::Struct {
                    fields: vec![Type::I8, Type::Ptr],
                    packed: false,
                },
            ],
            Type::Void,
        );

        assert_eq!(signature.to_string(), "void ([4 x i32], { i8, ptr })");
    }

    /// The parameter list is part of the type, arity included. Comparing the two
    /// lists positionally would stop at the shorter one and call these the same.
    #[test]
    fn function_types_differ_on_arity() {
        let one = Type::Func(func(vec![Type::I32], Type::Void));
        let two = Type::Func(func(vec![Type::I32, Type::I64], Type::Void));
        let none = Type::Func(func(vec![], Type::Void));

        assert_ne!(one, two, "a prefix of a longer list is a different type");
        assert_ne!(none, one, "and so is the empty list");
    }

    /// The rest of the identity: which params, in which order, and the result.
    #[test]
    fn function_types_differ_on_params_order_and_result() {
        let base = || func(vec![Type::I32, Type::I64], Type::Void);

        assert_eq!(
            Type::Func(base()),
            Type::Func(base()),
            "the same signature twice is the same type"
        );

        assert_ne!(
            Type::Func(base()),
            Type::Func(func(vec![Type::I64, Type::I32], Type::Void)),
            "parameter order matters"
        );

        assert_ne!(
            Type::Func(base()),
            Type::Func(func(vec![Type::I32, Type::I32], Type::Void)),
            "so do the parameter types"
        );

        assert_ne!(
            Type::Func(base()),
            Type::Func(func(vec![Type::I32, Type::I64], Type::I32)),
            "and the result"
        );
    }

    /// `BuildError` renders a `Type` through `Debug`, so a function type has to be
    /// printable on that path too — it is the one variant whose payload is not
    /// itself `Debug`.
    #[test]
    fn a_function_type_is_debug_printable() {
        assert_eq!(
            format!("{:?}", Type::Func(func(vec![Type::Double], Type::I64))),
            "i64 (double)"
        );
    }

    /// What `load` and `store` will accept: everything with a size. LLVM calls
    /// only scalars "first class", but both instructions take aggregates too, so
    /// the set this predicate names is the loadable one — `void` and function
    /// types out, everything else in.
    #[test]
    fn only_void_and_function_types_are_unsized() {
        assert!(!Type::Void.is_first_class());
        assert!(!Type::Func(func(vec![Type::I32], Type::I32)).is_first_class());

        for ty in [
            Type::I1,
            Type::I8,
            Type::I64,
            Type::Half,
            Type::Float,
            Type::Double,
            Type::Ptr,
            Type::Array {
                size: 4,
                element_ty: Box::new(Type::I32),
            },
            Type::Struct {
                fields: vec![Type::I32, Type::Ptr],
                packed: false,
            },
        ] {
            assert!(ty.is_first_class(), "`{ty}` has a size and can be loaded");
        }
    }

    /// Type rendering is what error messages and (later) emitted IR are built from,
    /// so every shape has to spell itself the way LLVM does.
    #[test]
    fn types_render_as_llvm_spells_them() {
        assert_eq!(Type::I1.to_string(), "i1");
        assert_eq!(Type::I64.to_string(), "i64");
        assert_eq!(Type::Half.to_string(), "half");
        assert_eq!(Type::Bfloat.to_string(), "bfloat");
        assert_eq!(Type::Float.to_string(), "float");
        assert_eq!(Type::Double.to_string(), "double");
        assert_eq!(Type::Ptr.to_string(), "ptr");
        assert_eq!(Type::Void.to_string(), "void");

        assert_eq!(
            Type::Array {
                size: 8,
                element_ty: Box::new(Type::I32),
            }
            .to_string(),
            "[8 x i32]"
        );

        assert_eq!(
            Type::Struct {
                fields: vec![Type::I32, Type::Double],
                packed: false,
            }
            .to_string(),
            "{ i32, double }"
        );

        assert_eq!(
            Type::Struct {
                fields: vec![Type::I8],
                packed: true,
            }
            .to_string(),
            "<{ i8 }>"
        );

        assert_eq!(
            Type::Struct {
                fields: vec![],
                packed: false,
            }
            .to_string(),
            "{  }",
            "an empty struct still renders"
        );

        // Nesting has to compose, since a struct field may be any type.
        assert_eq!(
            Type::Array {
                size: 2,
                element_ty: Box::new(Type::Struct {
                    fields: vec![
                        Type::Ptr,
                        Type::Array {
                            size: 3,
                            element_ty: Box::new(Type::I16),
                        },
                    ],
                    packed: true,
                }),
            }
            .to_string(),
            "[2 x <{ ptr, [3 x i16] }>]"
        );
    }

    /// `BuildError` carries a `Type`, so `Debug` — which is what `unwrap` prints —
    /// must render rather than panic.
    #[test]
    fn a_type_is_debug_printable() {
        assert_eq!(format!("{:?}", Type::I32), "i32");

        assert_eq!(
            format!(
                "{:?}",
                Type::Array {
                    size: 1,
                    element_ty: Box::new(Type::Float)
                }
            ),
            "[1 x float]"
        );
    }
}
