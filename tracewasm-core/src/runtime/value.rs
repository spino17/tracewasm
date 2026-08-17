//! The interpreter's value representations, and the table/element/data stores
//! that hold them.
//!
//! Two representations, deliberately:
//!
//! * [`Val`] is **tagged** — it knows its own wasm type. It is what crosses the
//!   API boundary, where a host hands over arguments or reads a global back and
//!   the type has to be checked at runtime.
//! * [`Value`] is **untagged**: eight bytes, no discriminant. It is what a frame
//!   holds during execution, where validation has already established every
//!   operand's type and carrying a tag would cost space and a branch on every
//!   access.
//!
//! Converting between them therefore goes one way for free (`Val -> Value`
//! discards the tag) and needs the expected type supplied the other way
//! (`Value::into_val`), which is why so many signatures thread a [`ValType`]
//! through.
//!
//! Carved out of the old `vm::stack` module, which held the stack and these
//! types together.

use crate::{
    error::TraceWasmError,
    module::{FuncIndex, ValType},
    tracewasm_unreachable,
};

/// A concrete runtime value on the operand stack or in a local slot.
///
/// One variant per supported WebAssembly value type. `Ref` holds an optional
/// function index — `None` is a null reference.
///
/// `V128` (SIMD) is intentionally absent, so there is no `Val` that can carry
/// one. Where a [`ValType`] has to be turned into a value the type is rejected
/// instead: [`Val::has_ty`] reports it as unsupported, and the interpreter's own
/// frame setup treats it as unreachable, which is sound because
/// `Module::compile` refuses such a module up front.
///
/// This is the tagged form, used at the API boundary and for globals. The
/// interpreter's operand stack holds an untagged eight-byte slot instead, since
/// there the type of every slot is already fixed by the instruction reading it.
#[derive(Debug, Copy, Clone)]
pub enum Val {
    /// A 32-bit integer value.
    I32(i32),
    /// A 64-bit integer value.
    I64(i64),
    /// A 32-bit float value.
    F32(f32),
    /// A 64-bit float value.
    F64(f64),
    /// A nullable function reference (`None` is a null reference).
    Ref(Option<FuncIndex>),
}

/// Reports an operand whose variant is not the one the instruction expected.
///
/// The accessors below are called once or more per interpreted instruction, so
/// their failure path is outlined rather than written inline. `panic!` in the body
/// would put a call there, and anything the accessor holds across it would have to
/// occupy a callee-saved register — whose save and restore is emitted at the
/// function's entry and exit, and so is paid on every call.
///
/// One helper per type, each taking no arguments, so there is nothing to keep
/// live. They diverge, which lets the compiler reach them with a plain branch
/// instead of a call.
mod wrong_ty {
    #[inline(never)]
    pub fn i32() -> ! {
        panic!("value is not i32")
    }
    #[inline(never)]
    pub fn i64() -> ! {
        panic!("value is not i64")
    }
    #[inline(never)]
    pub fn f32() -> ! {
        panic!("value is not f32")
    }
    #[inline(never)]
    pub fn f64() -> ! {
        panic!("value is not f64")
    }
    #[inline(never)]
    pub fn reference() -> ! {
        panic!("value is not ref")
    }
}

impl Val {
    /// The default `i32` value (`0`).
    pub fn i32_zero() -> Self {
        Val::I32(0)
    }

    /// The default `i64` value (`0`).
    pub fn i64_zero() -> Self {
        Val::I64(0)
    }

    /// The default `f32` value (`+0.0`).
    pub fn f32_zero() -> Self {
        Val::F32(0.0)
    }

    /// The default `f64` value (`+0.0`).
    pub fn f64_zero() -> Self {
        Val::F64(0.0)
    }

    /// The default reference value (a null reference).
    pub fn ref_zero() -> Self {
        Val::Ref(None)
    }

    /// Unwraps an `i32` value. Panics if this value is not an `I32`; callers rely
    /// on validation having already type-checked the operand.
    pub fn as_i32(&self) -> i32 {
        let Val::I32(val) = self else { wrong_ty::i32() };

        *val
    }

    /// Unwraps an `i64` value. Panics if this value is not an `I64`.
    pub fn as_i64(&self) -> i64 {
        let Val::I64(val) = self else { wrong_ty::i64() };

        *val
    }

    /// Unwraps an `f32` value. Panics if this value is not an `F32`.
    pub fn as_f32(&self) -> f32 {
        let Val::F32(val) = self else { wrong_ty::f32() };

        *val
    }

    /// Unwraps an `f64` value. Panics if this value is not an `F64`.
    pub fn as_f64(&self) -> f64 {
        let Val::F64(val) = self else { wrong_ty::f64() };

        *val
    }

    /// Unwraps a reference value. Panics if this value is not a `Ref`.
    pub fn as_ref(&self) -> Option<FuncIndex> {
        let Val::Ref(val) = self else {
            wrong_ty::reference()
        };

        *val
    }

    /// Returns the zero/default value for `ty`, as used to initialize declared
    /// locals per the WebAssembly spec.
    ///
    /// # Panics
    ///
    /// Panics on `V128`, which the VM does not model. Infallible in practice:
    /// `Module::compile` rejects a `v128` local, so no compiled module can reach
    /// this with one.
    pub fn zero_of_ty(ty: ValType) -> Self {
        match ty {
            ValType::I32 => Self::i32_zero(),
            ValType::I64 => Self::i64_zero(),
            ValType::F32 => Self::f32_zero(),
            ValType::F64 => Self::f64_zero(),
            ValType::Ref(_) => Self::ref_zero(),
            ValType::V128 => unreachable!(
                "hitting this means the validation in `compile` method in module/mod.rs is incorrect"
            ),
        }
    }

    /// Whether this value's variant matches the WebAssembly type `ty`.
    ///
    /// Used in debug assertions to confirm supplied arguments match a function's
    /// declared parameter types.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::Unsupported`] for `V128`.
    pub fn has_ty(&self, ty: ValType) -> Result<bool, TraceWasmError> {
        let val = match ty {
            ValType::I32 => matches!(self, Val::I32(_)),
            ValType::I64 => matches!(self, Val::I64(_)),
            ValType::F32 => matches!(self, Val::F32(_)),
            ValType::F64 => matches!(self, Val::F64(_)),
            ValType::Ref(_) => matches!(self, Val::Ref(_)),
            ValType::V128 => return Err(TraceWasmError::Unsupported("v128 type".to_string())),
        };

        Ok(val)
    }
}

/// Drops the tag, keeping the bits: the inverse of [`Value::into_val`].
///
/// Used where a tagged value crosses into the interpreter — call arguments, host
/// results, global reads.
impl From<&Val> for Value {
    #[inline(always)]
    fn from(value: &Val) -> Self {
        match value {
            Val::I32(val) => Value::from_i32(*val),
            Val::I64(val) => Value::from_i64(*val),
            Val::F32(val) => Value::from_f32(*val),
            Val::F64(val) => Value::from_f64(*val),
            Val::Ref(func_ref) => Value::from_ref(*func_ref),
        }
    }
}

/// By-value form of [`From<&Val>`](Value); [`Val`] is `Copy`, so the two differ
/// only in what the caller happens to hold.
impl From<Val> for Value {
    #[inline(always)]
    fn from(value: Val) -> Self {
        match value {
            Val::I32(val) => Value::from_i32(val),
            Val::I64(val) => Value::from_i64(val),
            Val::F32(val) => Value::from_f32(val),
            Val::F64(val) => Value::from_f64(val),
            Val::Ref(func_ref) => Value::from_ref(func_ref),
        }
    }
}

/// Where [`Value`] keeps the "this reference is non-null" bit.
///
/// A `funcref` payload is a [`FuncIndex`], which is a `u32`, so the whole upper
/// half of the word is free; 56 is simply a bit in that half.
const TAG_SHIFT: u32 = 56;
/// The bit pattern [`Value::from_ref`] sets for a non-null reference. A null one
/// is all zeroes, which is what makes the tag necessary: without it a null
/// reference and `Some(FuncIndex(0))` would be the same word.
const TAG_SOME: u64 = 1 << TAG_SHIFT;

/// One operand-stack slot: eight bytes, no type tag.
///
/// Every wasm value the interpreter holds fits in a `u64`, and the *type* of a
/// given slot is already fixed by the validated instruction stream — an
/// `i64.add` can only ever find two `i64`s beneath it. Carrying a discriminant
/// alongside the bits would therefore pay, on every push and pop, for
/// information the opcode already has. The tagged [`Val`] is used at the API
/// boundary instead, where the type genuinely is dynamic.
///
/// # Invariant
///
/// **A slot must be read back with the same type it was written as.** The
/// accessors reinterpret bits and cannot detect a mismatch: `from_i32(-1)`
/// stores `0x0000_0000_FFFF_FFFF`, so reading it with [`Self::as_i64`] yields
/// `4294967295` rather than `-1`. Validation is what upholds this, so a bug in
/// lowering surfaces as a wrong answer rather than a panic.
///
/// Floats are stored as raw bits rather than converted, so NaN payloads and
/// signed zeroes survive a round trip unchanged, as wasm requires.
#[derive(Default, Clone, Copy)]
pub(crate) struct Value(u64);

impl Value {
    /// Stores an `i32` in the low half, zero-extended.
    ///
    /// Zero- rather than sign-extending keeps the upper half at zero, so the
    /// 32-bit operations that read the slot back never have to mask.
    #[inline(always)]
    pub fn from_i32(val: i32) -> Self {
        Value(val as u32 as u64)
    }

    /// Stores an `i64`, which occupies the whole slot.
    #[inline(always)]
    pub fn from_i64(val: i64) -> Self {
        Value(val as u64)
    }

    /// Stores an `f32` as its raw bits in the low half, preserving NaN payloads.
    #[inline(always)]
    pub fn from_f32(val: f32) -> Self {
        Value(val.to_bits() as u64)
    }

    /// Stores an `f64` as its raw bits, preserving NaN payloads.
    #[inline(always)]
    pub fn from_f64(val: f64) -> Self {
        Value(val.to_bits())
    }

    /// Stores a `funcref`: the index in the low 32 bits, [`TAG_SOME`] set when
    /// the reference is non-null. A null reference is the all-zero word.
    #[inline(always)]
    pub fn from_ref(func_ref: Option<FuncIndex>) -> Self {
        let x = match func_ref {
            Some(x) => x.0,
            None => 0,
        } as u64;

        let tag = if func_ref.is_some() { TAG_SOME } else { 0 };

        Value(tag | x)
    }

    /// Reads the low half as an `i32`, discarding the upper half.
    ///
    /// Correct only for a slot written by [`Self::from_i32`]; see the type's
    /// invariant.
    #[inline(always)]
    pub fn as_i32(&self) -> i32 {
        self.0 as u32 as i32
    }

    /// Reads the whole slot as an `i64`. See the type's invariant.
    #[inline(always)]
    pub fn as_i64(&self) -> i64 {
        self.0 as i64
    }

    /// Reinterprets the low half as an `f32`. See the type's invariant.
    #[inline(always)]
    pub fn as_f32(&self) -> f32 {
        f32::from_bits(self.0 as u32)
    }

    /// Reinterprets the whole slot as an `f64`. See the type's invariant.
    #[inline(always)]
    pub fn as_f64(&self) -> f64 {
        f64::from_bits(self.0)
    }

    /// Reads a `funcref`, `None` for a null one.
    ///
    /// Any bit set at or above [`TAG_SHIFT`] counts as non-null, not just the one
    /// [`Self::from_ref`] writes, so this stays correct if the upper half is ever
    /// used for more than the one flag.
    #[inline(always)]
    pub fn as_ref(&self) -> Option<FuncIndex> {
        let is_some = (self.0 >> TAG_SHIFT) != 0;
        let val = self.0 as u32;
        if is_some { Some(FuncIndex(val)) } else { None }
    }

    /// The initial value wasm gives a declared local of type `ty` — zero for the
    /// numeric types, a null reference for `funcref`.
    ///
    /// Every arm produces the all-zero word, so this compiles away to a store of
    /// zero; it is written out per type to stay correct if a future
    /// representation stops sharing that encoding.
    ///
    /// # Panics
    ///
    /// On [`ValType::V128`], which `Module::compile` rejects, so a frame setup
    /// can never reach it.
    #[inline(always)]
    pub fn zero_of_ty(ty: ValType) -> Self {
        match ty {
            ValType::I32 => Value::from_i32(0),
            ValType::I64 => Value::from_i64(0),
            ValType::F32 => Value::from_f32(0.0),
            ValType::F64 => Value::from_f64(0.0),
            ValType::Ref(_) => Value::from_i32(0),
            ValType::V128 => tracewasm_unreachable::unreachable(),
        }
    }

    /// Re-attaches the type that the slot lost, producing the tagged [`Val`] the
    /// public API and the host boundary deal in.
    ///
    /// `ty` must be the type the slot was written as — it is taken from the
    /// function's declared signature, which is what makes that so.
    ///
    /// # Panics
    ///
    /// On [`ValType::V128`], as for [`Self::zero_of_ty`].
    #[inline(always)]
    pub fn into_val(self, ty: &ValType) -> Val {
        match ty {
            ValType::I32 => Val::I32(self.as_i32()),
            ValType::I64 => Val::I64(self.as_i64()),
            ValType::F32 => Val::F32(self.as_f32()),
            ValType::F64 => Val::F64(self.as_f64()),
            ValType::Ref(_) => Val::Ref(self.as_ref()),
            ValType::V128 => tracewasm_unreachable::unreachable(),
        }
    }
}

/// A materialized table instance: its function-reference slots and the maximum
/// number of elements it may grow to.
pub(crate) struct TableVal {
    /// The table's slots, each a nullable function reference.
    pub table: Vec<Option<FuncIndex>>,
    /// The maximum element count the table may grow to.
    pub maximum: u32,
}

/// A passive element segment's runtime state: its remaining function references,
/// or dropped once consumed.
pub(crate) enum ElementVal {
    /// The segment has been dropped (via `elem.drop` or an active init).
    Dropped,
    /// A still-live passive segment holding nullable function references.
    Passive(Box<[Option<FuncIndex>]>),
}

/// A passive data segment's runtime state: its remaining bytes, or dropped once
/// consumed.
pub(crate) enum DataVal {
    /// The segment has been dropped (via `data.drop` or an active init).
    Dropped,
    /// A still-live passive segment holding its raw byte blob.
    Passive(Box<[u8]>), // data blob
}

#[cfg(test)]
mod tests {
    // ------------------------------------------------------------------
    // Val helpers used during locals init / type checks
    // ------------------------------------------------------------------

    use crate::{module::ValType, runtime::value::Val};

    #[test]
    fn zero_of_ty_produces_typed_zeroes() {
        assert!(matches!(Val::zero_of_ty(ValType::I32), Val::I32(0)));
        assert!(matches!(Val::zero_of_ty(ValType::I64), Val::I64(0)));
        assert!(matches!(Val::zero_of_ty(ValType::F32), Val::F32(x) if x == 0.0));
        assert!(matches!(Val::zero_of_ty(ValType::F64), Val::F64(x) if x == 0.0));
        assert!(matches!(Val::zero_of_ty(ValType::FUNCREF), Val::Ref(None)));
    }

    // `v128` locals are rejected by `Module::compile`, so reaching here is a bug
    // in that validation rather than a supported input — hence a panic, not an
    // error.
    #[test]
    #[should_panic(expected = "module/mod.rs")]
    fn zero_of_ty_panics_on_v128() {
        Val::zero_of_ty(ValType::V128);
    }

    #[test]
    fn is_ty_matches_and_rejects() {
        assert!(Val::I32(1).has_ty(ValType::I32).unwrap());
        assert!(!Val::I32(1).has_ty(ValType::I64).unwrap());
        assert!(!Val::I32(1).has_ty(ValType::F32).unwrap());
        assert!(Val::F64(1.0).has_ty(ValType::F64).unwrap());
        assert!(!Val::F64(1.0).has_ty(ValType::I32).unwrap());
        assert!(Val::Ref(None).has_ty(ValType::FUNCREF).unwrap());
        assert!(!Val::Ref(None).has_ty(ValType::I32).unwrap());
    }

    #[test]
    fn is_ty_rejects_v128() {
        assert!(Val::I32(1).has_ty(ValType::V128).is_err());
    }
}
