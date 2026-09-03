//! Types and the values that carry them.
//!
//! A [`Value`] is what an instruction operates on: a type plus how the value is
//! obtained — a register, a pooled constant, or a constant expression. Both the type
//! and the constant are ids, so a `Value` is small and cheap to clone.
//!
//! [`Type`] describes one node of a type; its children are [`TyId`]s, so the whole
//! type graph lives in the pool rather than in nested boxes. Nearly everything a
//! caller needs is on [`TyId`] rather than on `Type`, because answering "is this a
//! pointer?" or "how does this spell itself?" needs the pool.

use crate::{
    cfg::{
        basic_block::BasicBlockId,
        context::Context,
        global::{Global, GlobalEntity, GlobalId},
    },
    error::{GepError, TypeError},
    instruction::{AllocaOperands, GetElementPtrOperands, InstructionKind},
    interner::{ConstId, StrId, TyId},
};
use ordered_float::OrderedFloat;
use std::{
    fmt::Display,
    hash::{Hash, Hasher},
    mem::discriminant,
};

/// The parameter types and result of a function type.
///
/// Note that LLVM writes a function type **result first** — `i32 (i8, ptr)` — which is
/// the reverse of how a signature reads in source. Getting it backwards produces IR
/// that parses as a *different* type rather than failing.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct FuncSignature {
    /// Parameter types, in declaration order.
    pub params: Vec<TyId>,
    /// The result type, `void` included.
    pub result: TyId,
}

impl FuncSignature {
    /// Builds a signature from already-interned parameter and result types.
    pub(crate) fn new(params: Vec<TyId>, result: TyId) -> Self {
        FuncSignature { params, result }
    }
}

/// One node of a type. An aggregate arm names its children by [`TyId`] rather than
/// holding them, so a type is flat and `Copy`-cheap to compare — two structurally
/// equal types are one pool entry, hence one id.
///
/// The cost is that a type no longer knows how to print itself; see [`TypeDisplay`].
///
/// A `Type` is normally built inline and interned immediately; the predicates and the
/// renderer live on [`TyId`], and [`Context`] has shorthands
/// ([`i32_ty`](Context::i32_ty), [`ptr_ty`](Context::ptr_ty), …) for the scalars.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Type {
    /// The type of a comparison result and a conditional branch's condition.
    I1,
    /// 8-bit integer.
    I8,
    /// 16-bit integer.
    I16,
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
    /// 16-bit IEEE-754 float.
    Half,
    /// 16-bit brain float.
    Bfloat,
    /// 32-bit IEEE-754 float.
    Float,
    /// 64-bit IEEE-754 float.
    Double,
    /// An opaque pointer.
    ///
    /// Since LLVM 15 a pointer carries no pointee: `ptr` is the whole type. What is
    /// pointed at belongs to the *instruction* — `alloca`'s allocated type, a
    /// `getelementptr`'s source type, a `load`'s result — which is what the pointee
    /// inference in this module reconstructs.
    Ptr,
    /// A fixed-length array.
    ///
    /// `size` is a literal because LLVM has no variable-length array type — the
    /// length is part of this type's identity, so it has to be something two
    /// spellings of `[4 x i32]` agree on. A runtime count belongs to `alloca`'s
    /// element-count operand instead.
    Array {
        /// Number of elements.
        size: u64,
        /// The element type.
        element_ty: TyId,
    },
    /// A literal struct, packed or not.
    Struct {
        /// Field types, in declaration order.
        fields: Box<[TyId]>,
        /// Whether the struct is packed — `<{ i8 }>` rather than `{ i8 }`.
        packed: bool,
    },
    /// A function type. Not first class: it cannot be loaded, stored or allocated.
    Func(FuncSignature),
    /// The absence of a value. Legal only as a function result.
    Void,
}

impl TyId {
    /// Borrows this type together with `ctx` so it can be printed.
    ///
    /// Rendering needs the pool: an aggregate names its children by id, and spelling
    /// one out means resolving every level. The ids inside have to have come from
    /// `ctx`, which holds for any type built against one context.
    ///
    /// The result renders the way LLVM spells the type — `[2 x <{ ptr, [3 x i16] }>]`,
    /// `i32 (i8, ptr)` — so it can be dropped straight into an error or into emitted
    /// IR.
    pub fn display<'a>(self, ctx: &'a Context) -> TypeDisplay<'a> {
        TypeDisplay { ty: self, ctx }
    }

    /// Whether this is `i1`, the type a conditional branch's condition must have.
    pub fn is_i1(&self, ctx: &Context) -> bool {
        let ty_obj = ctx.ty_interner.value(self.raw());

        matches!(ty_obj, Type::I1)
    }

    /// Whether this is `ptr`.
    pub fn is_ptr(&self, ctx: &Context) -> bool {
        let ty_obj = ctx.ty_interner.value(self.raw());

        matches!(ty_obj, Type::Ptr)
    }

    /// Whether this type has a size — everything except `void` and function types.
    ///
    /// This is the predicate that gates `load`, `store`, `alloca`, function
    /// parameters and a `getelementptr` source type. LLVM reserves "first class" for
    /// scalars only, but `load {i32, i32}` and `alloca [4 x i32]` both assemble, so
    /// the set named here is the loadable one: aggregates in, `void` and functions
    /// out.
    pub fn is_first_class(&self, ctx: &Context) -> bool {
        let ty_obj = ctx.ty_interner.value(self.raw());

        !matches!(ty_obj, Type::Void | Type::Func(_))
    }

    /// Whether this is an integer of any width.
    ///
    /// `i1` counts, matching LLVM: `alloca i32, i1 %c` assembles.
    pub fn is_integer(&self, ctx: &Context) -> bool {
        let ty_obj = ctx.ty_interner.value(self.raw());

        matches!(
            ty_obj,
            Type::I1 | Type::I8 | Type::I16 | Type::I32 | Type::I64
        )
    }

    /// Whether this is `i32` specifically.
    ///
    /// Narrower than [`is_integer`](Self::is_integer) because a `getelementptr` index
    /// into a *struct* must be an `i32` — `llvm-as` refuses an `i64` one with
    /// "invalid getelementptr indices", even though array indices may be any width.
    pub fn is_i32(&self, ctx: &Context) -> bool {
        let ty_obj = ctx.ty_interner.value(self.raw());

        matches!(ty_obj, Type::I32)
    }

    /// Whether this is `void`.
    pub fn is_void(&self, ctx: &Context) -> bool {
        let ty_obj = ctx.ty_interner.value(self.raw());

        matches!(ty_obj, Type::Void)
    }

    pub fn width(&self, ctx: &Context) -> Option<u8> {
        let ty_obj = ctx.ty_interner.value(self.raw());

        let width = match ty_obj {
            Type::I1 => 1,
            Type::I8 => 8,
            Type::I16 => 16,
            Type::I32 => 32,
            Type::I64 => 64,
            Type::Half => 16,
            Type::Bfloat => 16,
            Type::Float => 32,
            Type::Double => 64,
            Type::Ptr | Type::Array { .. } | Type::Struct { .. } | Type::Func(_) | Type::Void => {
                return None;
            }
        };

        Some(width)
    }

    /// Descends this type by `indices`, returning what the walk lands on.
    ///
    /// Used by `getelementptr` to type-check its indices and to work out what the
    /// resulting pointer points at. **Callers pass `indices[1..]`**: the first index
    /// steps over the source type as pointer arithmetic rather than into it, so a
    /// `getelementptr` with one index or none points at its source type unchanged.
    pub(crate) fn walk_pointee_ty_in_gep(
        &self,
        indices: &[Value],
        ctx: &Context,
    ) -> Result<TyId, GepError> {
        if indices.is_empty() {
            return Ok(*self);
        }

        let index = &indices[0];
        let ty_obj = ctx.ty_interner.value(self.raw());

        match ty_obj {
            Type::Struct { fields, packed: _ } => {
                let total_fields = fields.len();

                // A struct index must be a constant `i32` — not merely a constant
                // integer. `llvm-as` refuses an `i64` one with "invalid getelementptr
                // indices", even though array indices may be any width, because the
                // index names a field rather than scaling an offset.
                if !index.ty().is_i32(ctx) {
                    return Err(GepError::StructIndexNotAConstantI32(
                        ctx.display(index.ty()).to_string(),
                    ));
                }

                let Some(const_val) = index.try_const(ctx) else {
                    return Err(GepError::StructIndexNotAConstantI32(
                        ctx.display(index.ty()).to_string(),
                    ));
                };

                let Some(field_index) = const_val.try_integer() else {
                    return Err(GepError::StructIndexNotAConstantI32(
                        ctx.display(index.ty()).to_string(),
                    ));
                };

                // Checked as a signed value: a negative index is out of range rather
                // than a huge one, which is what `as u32` would turn it into.
                if field_index < 0 || field_index as usize >= total_fields {
                    return Err(GepError::StructIndexOutOfRange {
                        index: field_index,
                        fields: total_fields,
                    });
                }

                let field_ty = fields[field_index as usize];

                field_ty.walk_pointee_ty_in_gep(&indices[1..], ctx)
            }
            Type::Array { size, element_ty } => {
                let size = *size;
                let element_ty = *element_ty;

                if let Some(const_val) = index.try_const(ctx)
                    && let Some(element_index) = const_val.try_integer()
                    && (element_index < 0 || element_index as u64 >= size)
                {
                    return Err(GepError::ArrayIndexOutOfRange {
                        index: element_index as u64,
                        size,
                    });
                }

                element_ty.walk_pointee_ty_in_gep(&indices[1..], ctx)
            }
            _ => Err(GepError::TypeNotIndexable(ctx.display(*self).to_string())),
        }
    }
}

/// A type paired with the pool its children live in, so that it can be rendered.
///
/// [`Type`] cannot implement [`Display`] by itself: an aggregate arm holds [`TyId`]s
/// rather than whole types, and turning one back into something printable needs the
/// interner that issued it. Rendering is therefore a borrow of both — obtained from
/// [`TyId::display`], never constructed directly.
pub struct TypeDisplay<'a> {
    ty: TyId,
    ctx: &'a Context,
}

impl Display for TypeDisplay<'_> {
    /// Writes the type as LLVM spells it, so a rendered error reads the same as the
    /// IR it is about.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Each child is resolved through the same pool, which is what lets a nested
        // aggregate render in full rather than as the id it is stored as.
        let ty_obj = self.ctx.ty_interner.value(self.ty.raw());
        let nested = |id: TyId| self.ctx.display(id);

        match ty_obj {
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
            Type::Array { size, element_ty } => {
                write!(f, "[{size} x {}]", nested(*element_ty))
            }
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

                    write!(f, "{}", nested(*field))?;
                }

                f.write_str(close)
            }
            // `<result> (<params>)`, the order LLVM writes a function type in —
            // result first, which is the opposite of how the signature reads in
            // source.
            Type::Func(signature) => {
                write!(f, "{} (", nested(signature.result))?;

                for (i, param) in signature.params.iter().enumerate() {
                    if i != 0 {
                        f.write_str(", ")?;
                    }

                    write!(f, "{}", nested(*param))?;
                }

                f.write_str(")")
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum Signedness {
    Unsigned, // zext
    Signed,   // sext
}

/// How a [`Value`] is obtained — the three forms an LLVM operand can take.
#[derive(Debug, Clone)]
pub enum ValueKind {
    /// A named register, `%x`, defined by some instruction or a parameter.
    Reg(Register),
    /// A constant folded from other constants, written inline in the IR rather than
    /// computed by an instruction. See [`ConstExpr`].
    ConstExpr(ConstExpr),
    /// A module-level symbol, `@g` — a variable or a function. Its type is always
    /// `ptr`, since what a global *is* as an operand is its address.
    Global(Global),
}

/// An operand: a type, and where the value comes from.
///
/// Both halves are ids, so a `Value` is cheap to clone and cheap to compare. The type
/// is the value's *own* type — for a pointer that means `ptr`, not what it points at.
#[derive(Debug, Clone)]
pub struct Value {
    ty: TyId,
    kind: ValueKind,
}

impl Value {
    /// Builds a value from an already-interned type and a kind.
    pub(crate) fn new(ty: TyId, kind: ValueKind) -> Self {
        Value { ty, kind }
    }

    /// Interns a Rust literal as an LLVM constant.
    ///
    /// With `optional_cast` as `None` the value keeps the type [`Const::ty`] gives it
    /// — `7i32` becomes `i32 7`. With a type given, the constant is folded into it
    /// *now*: `Value::from_const(7i8, Some(i64_ty), ctx)` interns `i64 7`, not an
    /// `i8` to be widened later.
    ///
    /// Widths convert freely among integers, and between `float` and `double`.
    /// Crossing between integers and floats, or reaching a pointer, is refused —
    /// those need a real `sitofp`/`inttoptr` instruction, and folding them here would
    /// silently drop it.
    ///
    /// # Errors
    ///
    /// [`TypeError::ConstantCastToProvidedTypeFailed`] if the literal does not fold
    /// into the requested type.
    pub fn from_const<C: Const>(
        val: C,
        optional_cast: Option<TyId>,
        ctx: &mut Context,
    ) -> Result<Self, TypeError> {
        let (val, ty) = if let Some(ty) = optional_cast {
            let Some(c) = val.try_cast(ty, Signedness::Signed, ctx) else {
                return Err(TypeError::ConstantCastToProvidedTypeFailed(
                    C::ty(ctx).display(ctx).to_string(),
                    ty.display(ctx).to_string(),
                ));
            };

            (c, ty)
        } else {
            (val.into_const(), C::ty(ctx))
        };

        let const_id = ctx.const_interner.intern(val);

        Ok(Value {
            ty,
            kind: ValueKind::ConstExpr(ConstExpr::Const(const_id.into())),
        })
    }

    /// Interns `name` and builds a register of the given type.
    ///
    /// Crate-private because a register has to be *defined* by something: the
    /// builders call this after `name_for_reg` has issued a unique name, and record
    /// the definition so the pointee of a pointer can later be traced back.
    /// Constructing one freely would produce a `%name` that no instruction defines.
    pub(crate) fn from_register(name: String, ty: TyId, ctx: &mut Context) -> Self {
        let reg_id: StrId = ctx.str_interner.intern(name).into();

        Value {
            ty,
            kind: ValueKind::Reg(Register { name: reg_id }),
        }
    }

    /// Wraps a constant expression as an operand, taking its type from the
    /// expression.
    pub fn from_const_expr(expr: ConstExpr, ctx: &mut Context) -> Self {
        Value {
            ty: expr.ty(ctx),
            kind: ValueKind::ConstExpr(expr),
        }
    }

    /// Takes a global's address as an operand.
    ///
    /// The result is a `ptr` whatever the global names — a variable, a defined
    /// function, a declaration — which is exactly how LLVM types `@g`. What it points
    /// at is recoverable separately, so a `load` or `store` through one needs no
    /// explicit type.
    ///
    /// The tag is erased here: by the time a global is an operand, all three kinds
    /// behave alike.
    pub fn from_global<T: GlobalEntity>(global: GlobalId<T>, ctx: &mut Context) -> Self {
        let global = GlobalEntity::to_global(global);

        Value {
            ty: ctx.ptr_ty(),
            kind: ValueKind::Global(global),
        }
    }

    /// The value's own type. For a pointer this is `ptr`, not the pointee.
    pub fn ty(&self) -> TyId {
        self.ty
    }

    /// Where the value comes from.
    pub fn kind(&self) -> &ValueKind {
        &self.kind
    }

    /// Whether this value is a pointer.
    pub fn is_ptr(&self, ctx: &Context) -> bool {
        self.ty().is_ptr(ctx)
    }

    /// The pooled constant behind this value, or `None` if it is a register or a
    /// constant expression.
    ///
    /// Used where a value has to be known *now* rather than at run time — a
    /// `getelementptr` struct index, for instance.
    pub fn try_const<'a>(&self, ctx: &'a Context) -> Option<&'a ConstValue> {
        if let ValueKind::ConstExpr(ConstExpr::Const(const_val)) = self.kind() {
            let const_val = ctx.const_interner.value(const_val.raw());

            Some(const_val)
        } else {
            None
        }
    }

    /// Narrows to an [`I1Value`], the operand a conditional branch takes.
    ///
    /// Checking once here means [`Cursor::build_conditional_br`](crate::instruction::cursor::Cursor::build_conditional_br)
    /// cannot be handed anything else.
    ///
    /// # Errors
    ///
    /// [`TypeError::ValueToI1ValueFailed`] if the value is not an `i1`.
    pub fn into_i1(self, ctx: &Context) -> Result<I1Value, TypeError> {
        if !self.ty().is_i1(ctx) {
            return Err(TypeError::ValueToI1ValueFailed(
                self.ty().display(ctx).to_string(),
            ));
        }

        // The check above is what makes carrying the id sound: it is the pool's `i1`,
        // so converting back needs no interner and cannot fail.
        Ok(I1Value {
            ty: self.ty,
            kind: self.kind,
        })
    }

    /// Whether this value's type is an integer.
    pub fn is_integer(&self, ctx: &Context) -> bool {
        self.ty().is_integer(ctx)
    }

    /// Gives this value the type `ty`, if it can have it.
    ///
    /// The two kinds behave differently, and deliberately:
    ///
    /// - A **constant** is folded into the new type and re-interned, so the result is
    ///   a genuinely different constant — `i32 7` cast to `i64` becomes `i64 7`.
    /// - A **register** or constant expression is only *checked*. Nothing converts it,
    ///   because widening a register needs a real `zext`/`sext`, and folding one here
    ///   would emit IR that skips the instruction the caller never asked for.
    ///
    /// `None` covers all of: an unsized target type, a constant that does not fold,
    /// and a register whose type does not already match.
    pub fn try_cast(&self, ty: TyId, signedness: Signedness, ctx: &mut Context) -> Option<Self> {
        if !ty.is_first_class(ctx) {
            return None;
        }

        let val_ty = self.ty();

        let final_value = match self.kind() {
            ValueKind::ConstExpr(ConstExpr::Const(const_id)) => {
                let const_val = *ctx.const_interner.value(const_id.raw());
                let casted_const_val = const_val.try_cast(ty, signedness, ctx)?;
                let casted_const_id = ctx.const_interner.intern(casted_const_val).into();

                Value::new(ty, ValueKind::ConstExpr(ConstExpr::Const(casted_const_id)))
            }
            ValueKind::ConstExpr(_) | ValueKind::Reg(_) | ValueKind::Global(_) => {
                if val_ty != ty {
                    return None;
                }

                self.clone()
            }
        };

        Some(final_value)
    }

    pub fn try_cast_two(
        a: Value,
        b: Value,
        ty: Option<TyId>,
        signedness: Signedness,
        ctx: &mut Context,
    ) -> Option<(Value, Value)> {
        if let Some(ty) = ty {
            let Some(casted_a) = a.try_cast(ty, signedness, ctx) else {
                return None;
            };

            let Some(casted_b) = b.try_cast(ty, signedness, ctx) else {
                return None;
            };

            return Some((casted_a, casted_b));
        }

        if a.ty() == b.ty() {
            return Some((a, b));
        }

        let Some(a_width) = a.ty().width(ctx) else {
            return None;
        };

        let Some(b_width) = b.ty().width(ctx) else {
            return None;
        };

        if a_width >= b_width {
            let ref_ty = a.ty();

            let Some(casted_b) = b.try_cast(ref_ty, signedness, ctx) else {
                return None;
            };

            Some((a, casted_b))
        } else {
            let ref_ty = b.ty();

            let Some(casted_a) = a.try_cast(ref_ty, signedness, ctx) else {
                return None;
            };

            Some((casted_a, b))
        }
    }

    /// Works out what this pointer points at, by walking back to the instruction
    /// that produced it.
    ///
    /// Opaque pointers carry no pointee, so the only way to recover one is to find
    /// where the pointer came from: an `alloca` knows its allocated type, a
    /// `getelementptr` knows what its indices landed on. That is what lets a `load`
    /// or `store` omit its type and be checked against the slot it addresses.
    ///
    /// **Best-effort — `None` is a normal answer, not a failure.** It covers a
    /// pointer produced by an instruction that records no pointee, a constant `null`,
    /// and a **function parameter**, which nothing in this function defined. When
    /// inference declines, the caller's explicit type stands.
    pub(crate) fn try_inferring_pointee_ty(
        &self,
        block: BasicBlockId,
        ctx: &mut Context,
    ) -> Option<PointeeTy> {
        if !self.ty().is_ptr(ctx) {
            return None;
        }

        let pointee_ty = match &self.kind {
            ValueKind::Reg(reg) => {
                let func_id = ctx.get_block(block).func_id;
                let ptr_reg_name_id = reg.name;

                let Some(def) = ctx.register_defs(func_id).get(&ptr_reg_name_id) else {
                    // params would hit this!
                    return None;
                };

                let ptr_instr = &ctx.get_block(def.block).instructions[def.instr_index];

                match &ptr_instr.kind {
                    InstructionKind::Alloca(AllocaOperands {
                        ty,
                        count,
                        align: _,
                    }) => PointeeTy {
                        ty: *ty,
                        count: count.clone(),
                    },
                    InstructionKind::GetElementPtr(operands) => PointeeTy {
                        ty: operands.result_pointee_ty(ctx)?,
                        count: None,
                    },
                    _ => return None,
                }
            }
            ValueKind::ConstExpr(const_expr) => match const_expr {
                ConstExpr::GetElementPtr(operands) => PointeeTy {
                    ty: operands.result_pointee_ty(ctx)?,
                    count: None,
                },
                ConstExpr::IntToPtr { .. } => return None,
                _ => return None,
            },
            ValueKind::Global(global) => {
                let name = global.name();

                let global = &ctx
                    .module
                    .globals
                    .get(&name)
                    .expect("hitting this means logic for tracking global names is incorrect");

                let ty_obj = global.pointee_ty(ctx);
                let ty = ctx.ty_interner.intern(ty_obj).into();

                PointeeTy { ty, count: None }
            }
        };

        Some(pointee_ty)
    }
}

/// What a pointer was traced back to.
pub(crate) struct PointeeTy {
    /// The type pointed at.
    pub ty: TyId,
    /// How many of them, when the pointer came from an `alloca` with an element
    /// count. `None` for a single element and for pointers from other instructions.
    pub count: Option<Value>,
}

#[derive(Debug, Clone)]
/// A constant folded from other constants, written inline in the IR.
///
/// LLVM allows a limited set of operations to appear where a constant is expected,
/// computed at link time rather than by an instruction. They are useful for global
/// initialisers and for addresses known statically.
///
/// Build one with [`Value::from_const_expr`] and it can be used wherever a constant
/// can: as a `store` value, a `load` address, a call argument.
///
/// Two variants are complete. [`Const`](Self::Const) is a plain literal, and
/// [`GetElementPtr`](Self::GetElementPtr) carries its operands, renders, and has a
/// recoverable pointee — so a `load` or `store` through one is type-checked like any
/// other.
///
/// The other four carry **no operands**: [`PtrToInt`](Self::PtrToInt) and
/// [`IntToPtr`](Self::IntToPtr) have no fields at all, and [`BitCast`](Self::BitCast)
/// and [`Trunc`](Self::Trunc) name only a target type with no value to convert. They
/// can be constructed, but there is nothing to emit, so the emitter refuses them
/// rather than writing IR it cannot spell.
pub enum ConstExpr {
    /// A `getelementptr` over constant operands. Its pointee is recoverable, so it
    /// participates in pointee inference like the instruction does.
    GetElementPtr(Box<GetElementPtrOperands>),
    /// `ptrtoint`. Incomplete: no operand field, so nothing to convert.
    PtrToInt {},
    /// `inttoptr`. Incomplete: no operand field, so nothing to convert.
    IntToPtr {},
    /// `bitcast`. Incomplete: names the target type but not the value being cast.
    BitCast {
        /// The type being cast to.
        ty: TyId,
    },
    /// `trunc`. Incomplete: names the target type but not the value being truncated.
    Trunc {
        /// The narrower type being truncated to.
        target_ty: TyId,
    },
    /// A plain literal — `7`, `null`, `true`.
    ///
    /// Not an operation at all, but it belongs here because LLVM makes no distinction
    /// where a constant is expected: a global initializer or a `store` value takes
    /// either. Folding it in means one kind of operand instead of two.
    Const(ConstId),
}

impl ConstExpr {
    /// The type the expression evaluates to.
    pub fn ty(&self, ctx: &mut Context) -> TyId {
        match self {
            ConstExpr::GetElementPtr(_) => ctx.ptr_ty(),
            ConstExpr::PtrToInt { .. } => ctx.i32_ty(),
            ConstExpr::IntToPtr { .. } => ctx.ptr_ty(),
            ConstExpr::BitCast { ty } => *ty,
            ConstExpr::Trunc { target_ty } => *target_ty,
            ConstExpr::Const(const_val) => {
                // Copied out rather than borrowed: `ConstValue::ty` interns, which
                // needs `&mut ctx` and so cannot run while the pool is borrowed.
                let const_val = *ctx.const_interner.value(const_val.raw());

                const_val.ty(ctx)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// A named local, `%x`.
///
/// The name alone is not an identity: names are interned per *context*, so `%sum` in
/// two functions is one [`StrId`]. What makes a register unique is that name together
/// with the function it belongs to.
pub struct Register {
    /// The interned name, without the leading `%`.
    pub(crate) name: StrId,
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
    /// `i1 true` / `i1 false`, stored as 0 or 1.
    I1(i8),
    /// `i8`.
    I8(i8),
    /// `i16`.
    I16(i16),
    /// `i32`.
    I32(i32),
    /// `i64`.
    I64(i64),
    /// `float`. Wrapped only because `f32` is not `Ord`; the wrapper's own `Hash` is
    /// deliberately not used — see the hand-written [`Hash`] impl below.
    Float(OrderedFloat<f32>),
    /// `double`, with the same caveat as [`Float`](Self::Float).
    Double(OrderedFloat<f64>),
    /// `null`.
    NullPtr,
}

impl ConstValue {
    /// The LLVM type this constant has.
    pub fn ty(&self, ctx: &mut Context) -> TyId {
        ctx.ty_interner
            .intern(match self {
                ConstValue::I1(_) => Type::I1,
                ConstValue::I8(_) => Type::I8,
                ConstValue::I16(_) => Type::I16,
                ConstValue::I32(_) => Type::I32,
                ConstValue::I64(_) => Type::I64,
                ConstValue::Float(_) => Type::Float,
                ConstValue::Double(_) => Type::Double,
                ConstValue::NullPtr => Type::Ptr,
            })
            .into()
    }

    /// The value as an `i32`, or `None` if this is not an integer.
    ///
    /// Narrower widths widen and `i64` truncates, so a caller that needs the exact
    /// value must check the type too — a `getelementptr` struct index does, since
    /// LLVM requires that one to be an `i32` specifically.
    pub fn try_integer(&self) -> Option<i32> {
        let val = match self {
            ConstValue::I1(val) => *val as i32,
            ConstValue::I8(val) => *val as i32,
            ConstValue::I16(val) => *val as i32,
            ConstValue::I32(val) => *val,
            ConstValue::I64(val) => *val as i32,
            _ => return None,
        };

        Some(val)
    }

    /// Whether the value is positive. A null pointer is not.
    ///
    /// For floats this is the *sign bit*, so `-0.0` is negative — the distinction
    /// this type is careful to preserve elsewhere.
    pub fn is_sign_positive(&self) -> bool {
        match self {
            ConstValue::I1(val) => val.is_positive(),
            ConstValue::I8(val) => val.is_positive(),
            ConstValue::I16(val) => val.is_positive(),
            ConstValue::I32(val) => val.is_positive(),
            ConstValue::I64(val) => val.is_positive(),
            ConstValue::Float(val) => val.is_sign_positive(),
            ConstValue::Double(val) => val.is_sign_positive(),
            ConstValue::NullPtr => false,
        }
    }

    /// Folds this constant into `ty`, or `None` if it does not belong there.
    ///
    /// Dispatches to the [`Const`] impl for whichever literal it holds, so the rules
    /// are the same ones described there: widths convert among integers and among
    /// floats, and nothing crosses between them or reaches a pointer.
    pub fn try_cast(
        &self,
        ty: TyId,
        signedness: Signedness,
        ctx: &mut Context,
    ) -> Option<ConstValue> {
        match self {
            ConstValue::I1(val) => val.try_cast(ty, signedness, ctx),
            ConstValue::I8(val) => val.try_cast(ty, signedness, ctx),
            ConstValue::I16(val) => val.try_cast(ty, signedness, ctx),
            ConstValue::I32(val) => val.try_cast(ty, signedness, ctx),
            ConstValue::I64(val) => val.try_cast(ty, signedness, ctx),
            ConstValue::Float(val) => val.try_cast(ty, signedness, ctx),
            ConstValue::Double(val) => val.try_cast(ty, signedness, ctx),
            ConstValue::NullPtr => {
                if ty.is_ptr(ctx) {
                    Some(ConstValue::NullPtr)
                } else {
                    None
                }
            }
        }
    }
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

/// A Rust literal that can be used as an LLVM constant.
///
/// Implemented for `bool`, the signed integers, `f32`/`f64` and [`NullPtr`], which is
/// what lets [`Value::from_const`] be called with a plain literal.
pub trait Const {
    /// The LLVM type this literal has by default: `i32` for `i32`, `double` for `f64`.
    fn ty(ctx: &mut Context) -> TyId;

    /// Wraps the literal as a pool value at its default type.
    fn into_const(self) -> ConstValue;

    /// Folds the literal into `ty`, or `None` if it does not belong there.
    ///
    /// Integers narrow and widen among themselves, truncating the way LLVM's `trunc`
    /// would; floats convert between `float` and `double`. Nothing crosses between
    /// integers and floats, and nothing reaches a pointer — those need a real
    /// conversion instruction.
    fn try_cast(&self, ty: TyId, signedness: Signedness, ctx: &mut Context) -> Option<ConstValue>;
}

impl Const for bool {
    fn ty(ctx: &mut Context) -> TyId {
        ctx.i1_ty()
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I1(if self { 1 } else { 0 })
    }

    fn try_cast(&self, ty: TyId, _signedness: Signedness, ctx: &mut Context) -> Option<ConstValue> {
        if ty.is_i1(ctx) {
            Some(ConstValue::I1(if *self { 1 } else { 0 }))
        } else {
            None
        }
    }
}

impl Const for i8 {
    fn ty(ctx: &mut Context) -> TyId {
        ctx.i8_ty()
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I8(self)
    }

    fn try_cast(&self, ty: TyId, signedness: Signedness, ctx: &mut Context) -> Option<ConstValue> {
        let ty_obj = ctx.ty_interner.value(ty.raw());

        let v = match signedness {
            Signedness::Signed => match ty_obj {
                Type::I8 => ConstValue::I8(*self),
                Type::I16 => ConstValue::I16(*self as i16),
                Type::I32 => ConstValue::I32(*self as i32),
                Type::I64 => ConstValue::I64(*self as i64),
                _ => return None,
            },
            Signedness::Unsigned => match ty_obj {
                Type::I8 => ConstValue::I8(*self),
                Type::I16 => ConstValue::I16(*self as u8 as u16 as i16),
                Type::I32 => ConstValue::I32(*self as u8 as u32 as i32),
                Type::I64 => ConstValue::I64(*self as u8 as u64 as i64),
                _ => return None,
            },
        };

        Some(v)
    }
}

impl Const for i16 {
    fn ty(ctx: &mut Context) -> TyId {
        ctx.i16_ty()
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I16(self)
    }

    fn try_cast(&self, ty: TyId, signedness: Signedness, ctx: &mut Context) -> Option<ConstValue> {
        let ty_obj = ctx.ty_interner.value(ty.raw());

        let v = match signedness {
            Signedness::Signed => match ty_obj {
                Type::I8 => {
                    if *self > i8::MAX as i16 || *self < i8::MIN as i16 {
                        return None;
                    }

                    ConstValue::I8(*self as i8)
                }
                Type::I16 => ConstValue::I16(*self),
                Type::I32 => ConstValue::I32(*self as i32),
                Type::I64 => ConstValue::I64(*self as i64),
                _ => return None,
            },
            Signedness::Unsigned => match ty_obj {
                Type::I8 => {
                    let val = *self as u16;

                    if val > u8::MAX as u16 {
                        return None;
                    }

                    ConstValue::I8(val as u8 as i8)
                }
                Type::I16 => ConstValue::I16(*self),
                Type::I32 => ConstValue::I32(*self as u16 as u32 as i32),
                Type::I64 => ConstValue::I64(*self as u16 as u64 as i64),
                _ => return None,
            },
        };

        Some(v)
    }
}

impl Const for i32 {
    fn ty(ctx: &mut Context) -> TyId {
        ctx.i32_ty()
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I32(self)
    }

    fn try_cast(&self, ty: TyId, signedness: Signedness, ctx: &mut Context) -> Option<ConstValue> {
        let ty_obj = ctx.ty_interner.value(ty.raw());

        let v = match signedness {
            Signedness::Signed => {
                let val = *self;

                match ty_obj {
                    Type::I8 => {
                        if val > i8::MAX as i32 || val < i8::MIN as i32 {
                            return None;
                        }

                        ConstValue::I8(*self as i8)
                    }
                    Type::I16 => {
                        if val > i16::MAX as i32 || val < i16::MIN as i32 {
                            return None;
                        }

                        ConstValue::I16(*self as i16)
                    }
                    Type::I32 => ConstValue::I32(*self),
                    Type::I64 => ConstValue::I64(*self as i64),
                    _ => return None,
                }
            }
            Signedness::Unsigned => {
                let val = *self as u32;

                match ty_obj {
                    Type::I8 => {
                        if val > u8::MAX as u32 {
                            return None;
                        }

                        ConstValue::I8(val as u8 as i8)
                    }
                    Type::I16 => {
                        if val > u16::MAX as u32 {
                            return None;
                        }

                        ConstValue::I16(val as u16 as i16)
                    }
                    Type::I32 => ConstValue::I32(*self),
                    Type::I64 => ConstValue::I64(*self as u32 as u64 as i64),
                    _ => return None,
                }
            }
        };

        Some(v)
    }
}

impl Const for i64 {
    fn ty(ctx: &mut Context) -> TyId {
        ctx.i64_ty()
    }

    fn into_const(self) -> ConstValue {
        ConstValue::I64(self)
    }

    fn try_cast(&self, ty: TyId, signedness: Signedness, ctx: &mut Context) -> Option<ConstValue> {
        let ty_obj = ctx.ty_interner.value(ty.raw());

        let v = match signedness {
            Signedness::Signed => {
                let val = *self;

                match ty_obj {
                    Type::I8 => {
                        if val > i8::MAX as i64 || val < i8::MIN as i64 {
                            return None;
                        }

                        ConstValue::I8(*self as i8)
                    }
                    Type::I16 => {
                        if val > i16::MAX as i64 || val < i16::MIN as i64 {
                            return None;
                        }

                        ConstValue::I16(*self as i16)
                    }
                    Type::I32 => {
                        if val > i32::MAX as i64 || val < i32::MIN as i64 {
                            return None;
                        }

                        ConstValue::I32(*self as i32)
                    }
                    Type::I64 => ConstValue::I64(*self),
                    _ => return None,
                }
            }
            Signedness::Unsigned => {
                let val = *self as u64;

                match ty_obj {
                    Type::I8 => {
                        if val > u8::MAX as u64 {
                            return None;
                        }

                        ConstValue::I8(val as u8 as i8)
                    }
                    Type::I16 => {
                        if val > u16::MAX as u64 {
                            return None;
                        }

                        ConstValue::I16(val as u16 as i16)
                    }
                    Type::I32 => {
                        if val > u32::MAX as u64 {
                            return None;
                        }

                        ConstValue::I32(val as u32 as i32)
                    }
                    Type::I64 => ConstValue::I64(*self),
                    _ => return None,
                }
            }
        };

        Some(v)
    }
}

impl Const for f32 {
    fn ty(ctx: &mut Context) -> TyId {
        ctx.f32_ty()
    }

    fn into_const(self) -> ConstValue {
        ConstValue::Float(OrderedFloat(self))
    }

    fn try_cast(&self, ty: TyId, _signedness: Signedness, ctx: &mut Context) -> Option<ConstValue> {
        let ty_obj = ctx.ty_interner.value(ty.raw());

        let v = match ty_obj {
            Type::Float => ConstValue::Float(OrderedFloat(*self)),
            Type::Double => ConstValue::Double(OrderedFloat(*self as f64)),
            _ => return None,
        };

        Some(v)
    }
}

impl Const for f64 {
    fn ty(ctx: &mut Context) -> TyId {
        ctx.f64_ty()
    }

    fn into_const(self) -> ConstValue {
        ConstValue::Double(OrderedFloat(self))
    }

    fn try_cast(&self, ty: TyId, _signedness: Signedness, ctx: &mut Context) -> Option<ConstValue> {
        let ty_obj = ctx.ty_interner.value(ty.raw());

        let v = match ty_obj {
            Type::Float => ConstValue::Float(OrderedFloat(*self as f32)),
            Type::Double => ConstValue::Double(OrderedFloat(*self)),
            _ => return None,
        };

        Some(v)
    }
}

#[derive(Clone, Copy)]
/// The `null` pointer constant, as something [`Value::from_const`] can take.
///
/// A unit type because `null` has no payload: every null is the same null, so the
/// pool holds one however many times it is asked for.
pub struct NullPtr;

impl Const for NullPtr {
    fn ty(ctx: &mut Context) -> TyId {
        ctx.ptr_ty()
    }

    fn into_const(self) -> ConstValue {
        ConstValue::NullPtr
    }

    fn try_cast(&self, ty: TyId, _signedness: Signedness, ctx: &mut Context) -> Option<ConstValue> {
        if NullPtr::ty(ctx) != ty {
            return None;
        }

        Some(self.into_const())
    }
}

/// A value already known to be an `i1`, so a conditional branch cannot be handed
/// anything else.
///
/// It keeps the pool id it was narrowed from rather than re-deriving `i1` on the way
/// back: [`Value::into_i1`] has already resolved and checked that id, so converting
/// back needs neither the interner nor a second chance to fail.
#[derive(Debug)]
pub struct I1Value {
    pub(crate) ty: TyId,
    pub(crate) kind: ValueKind,
}

impl From<I1Value> for Value {
    fn from(value: I1Value) -> Self {
        Value {
            ty: value.ty,
            kind: value.kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interner::ConstInterner;

    /// Interns `ty` and hands back its id, so a test can spell the shape it means
    /// instead of the pool bookkeeping that shape now costs.
    fn intern(ty: Type, ctx: &mut Context) -> TyId {
        ctx.ty_interner.intern(ty).into()
    }

    /// The type a value reports, resolved back out of the pool. A value holds an id,
    /// so every assertion about "what type is this" goes through here.
    fn ty_of(value: &Value, ctx: &Context) -> Type {
        ctx.ty_interner.value(value.ty().raw()).clone()
    }

    /// How an interned type spells itself against `ctx`'s pool.
    fn rendered(ty: TyId, ctx: &Context) -> String {
        ty.display(ctx).to_string()
    }

    /// Interns a shape written inline and renders it, since rendering now needs the
    /// type to be in the pool first.
    fn intern_rendered(ty: Type, ctx: &mut Context) -> String {
        let id = intern(ty, ctx);

        rendered(id, ctx)
    }

    #[test]
    fn value_from_constants() {
        let mut ctx = crate::test_support::ctx();
        let i32_ty = ctx.i32_ty();
        let i1_ty = ctx.i1_ty();

        let a = Value::from_const(32, None, &mut ctx);
        let b = Value::from_const(2.3, Some(i32_ty), &mut ctx);
        let c = Value::from_const(true, None, &mut ctx);
        let d = Value::from_const(true, Some(i1_ty), &mut ctx);
        let e = Value::from_const(false, Some(i32_ty), &mut ctx);

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
        let mut ctx = crate::test_support::ctx();
        let i64_ty = ctx.i64_ty();

        let widened = Value::from_const(7i8, Some(i64_ty), &mut ctx).unwrap();

        assert_eq!(ty_of(&widened, &ctx), Type::I64);

        assert!(
            matches!(widened.kind, ValueKind::ConstExpr(ConstExpr::Const(_))),
            "a constant value holds a pool id"
        );

        // The stored constant is widened too, not left as the i8 it came from.
        assert_eq!(ctx.const_interner.values(), [ConstValue::I64(7)]);
    }

    /// No cast means the value keeps the source type, which is what `Const::ty`
    /// says it is.
    #[test]
    fn without_a_cast_the_source_type_is_kept() {
        let mut ctx = crate::test_support::ctx();

        // Built first, then checked: resolving an id back needs a shared borrow of
        // the context the values were built through.
        let cases = [
            (Value::from_const(1i8, None, &mut ctx), Type::I8),
            (Value::from_const(1i16, None, &mut ctx), Type::I16),
            (Value::from_const(1i32, None, &mut ctx), Type::I32),
            (Value::from_const(1i64, None, &mut ctx), Type::I64),
            (Value::from_const(1.0f32, None, &mut ctx), Type::Float),
            (Value::from_const(1.0f64, None, &mut ctx), Type::Double),
            (Value::from_const(true, None, &mut ctx), Type::I1),
        ];

        for (value, expected) in cases {
            assert_eq!(ty_of(&value.unwrap(), &ctx), expected);
        }
    }

    /// Widening is where [`Signedness`] earns its place: the same source bits mean
    /// two different numbers depending on how the high bits are filled, and only the
    /// caller knows which was meant.
    ///
    /// Every row here is a case where dropping the flag would silently produce the
    /// other answer.
    #[test]
    fn widening_fills_the_high_bits_the_way_the_flag_says() {
        let mut ctx = crate::test_support::ctx();
        let i16_ty = ctx.i16_ty();
        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();

        // -1 is all-ones at every width, so it is the sharpest probe: sign-extending
        // keeps it -1, zero-extending turns it into that width's unsigned maximum.
        assert_eq!(
            (-1i8).try_cast(i64_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::I64(-1)),
        );

        assert_eq!(
            (-1i8).try_cast(i64_ty, Signedness::Unsigned, &mut ctx),
            Some(ConstValue::I64(255)),
            "zext of an all-ones i8 is u8::MAX, not -1",
        );

        assert_eq!(
            (-1i16).try_cast(i32_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::I32(-1)),
        );

        assert_eq!(
            (-1i16).try_cast(i32_ty, Signedness::Unsigned, &mut ctx),
            Some(ConstValue::I32(65535)),
        );

        // The case that produced a wrong `icmp ult` before the flag existed: as an
        // unsigned 32-bit quantity `-1i32` is 4294967295, and comparing against it
        // sign-extended gives the opposite answer for any operand above 2^32.
        assert_eq!(
            (-1i32).try_cast(i64_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::I64(-1)),
        );

        assert_eq!(
            (-1i32).try_cast(i64_ty, Signedness::Unsigned, &mut ctx),
            Some(ConstValue::I64(4294967295)),
        );

        // The extremes, where the two readings are furthest apart.
        assert_eq!(
            i32::MIN.try_cast(i64_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::I64(-2147483648)),
        );

        assert_eq!(
            i32::MIN.try_cast(i64_ty, Signedness::Unsigned, &mut ctx),
            Some(ConstValue::I64(2147483648)),
        );

        // A non-negative source has the same value under both readings, which is why
        // the bug only ever showed up for high-bit-set constants.
        for signedness in [Signedness::Signed, Signedness::Unsigned] {
            assert_eq!(
                5i8.try_cast(i16_ty, signedness, &mut ctx),
                Some(ConstValue::I16(5)),
            );

            assert_eq!(
                5i32.try_cast(i64_ty, signedness, &mut ctx),
                Some(ConstValue::I64(5)),
            );
        }
    }

    /// Narrowing needs no signedness to *compute* — LLVM has one `trunc`, not a
    /// signed and an unsigned one — but it needs one to decide whether the value
    /// **fits**. `255` fits a byte read as unsigned and does not read as signed, and
    /// both readings produce the identical bits `0xFF`.
    #[test]
    fn narrowing_is_range_checked_against_the_reading_asked_for() {
        let mut ctx = crate::test_support::ctx();
        let i8_ty = ctx.i8_ty();
        let i32_ty = ctx.i32_ty();

        // Signed: the target's own range, both ends inclusive.
        assert_eq!(
            127i32.try_cast(i8_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::I8(127)),
            "i8::MAX itself must be accepted",
        );

        assert_eq!(
            (-128i32).try_cast(i8_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::I8(-128)),
            "i8::MIN itself must be accepted",
        );

        assert_eq!(128i32.try_cast(i8_ty, Signedness::Signed, &mut ctx), None);

        assert_eq!(
            (-129i32).try_cast(i8_ty, Signedness::Signed, &mut ctx),
            None,
            "the lower bound is checked too, not just the upper",
        );

        assert_eq!(
            300i32.try_cast(i8_ty, Signedness::Signed, &mut ctx),
            None,
            "refused rather than folded to 44",
        );

        // Unsigned: the bit pattern read as unsigned, so 0..=255 for a byte. The
        // result is still stored signed, since `ConstValue::I8` holds an `i8`.
        assert_eq!(
            255i32.try_cast(i8_ty, Signedness::Unsigned, &mut ctx),
            Some(ConstValue::I8(-1)),
            "0xFF is a valid unsigned byte, stored as the i8 -1",
        );

        assert_eq!(
            200i32.try_cast(i8_ty, Signedness::Unsigned, &mut ctx),
            Some(ConstValue::I8(-56)),
            "0xC8",
        );

        assert_eq!(256i32.try_cast(i8_ty, Signedness::Unsigned, &mut ctx), None);

        assert_eq!(
            (-1i32).try_cast(i8_ty, Signedness::Unsigned, &mut ctx),
            None,
            "read unsigned, -1i32 is 4294967295 and does not fit a byte",
        );

        // The two readings genuinely disagree at the same input, in both directions.
        assert_eq!(255i32.try_cast(i8_ty, Signedness::Signed, &mut ctx), None);

        assert_eq!(
            (-1i32).try_cast(i8_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::I8(-1)),
        );

        // i64 narrows the same way, across the 32-bit boundary.
        assert_eq!(
            4294967295i64.try_cast(i32_ty, Signedness::Unsigned, &mut ctx),
            Some(ConstValue::I32(-1)),
            "u32::MAX fits an unsigned 32-bit slot",
        );

        assert_eq!(
            4294967295i64.try_cast(i32_ty, Signedness::Signed, &mut ctx),
            None,
            "but it is past i32::MAX read signed",
        );

        assert_eq!(
            4294967296i64.try_cast(i32_ty, Signedness::Unsigned, &mut ctx),
            None,
            "one past u32::MAX",
        );

        assert_eq!(
            (i32::MIN as i64).try_cast(i32_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::I32(i32::MIN)),
        );

        assert_eq!(
            (i32::MIN as i64 - 1).try_cast(i32_ty, Signedness::Signed, &mut ctx),
            None,
        );
    }

    /// A cast to the width a value already has is the identity, whichever reading is
    /// asked for — there are no bits to add or drop, so the flag cannot apply.
    #[test]
    fn a_same_width_cast_changes_nothing() {
        let mut ctx = crate::test_support::ctx();
        let i8_ty = ctx.i8_ty();
        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();

        for signedness in [Signedness::Signed, Signedness::Unsigned] {
            assert_eq!(
                (-1i8).try_cast(i8_ty, signedness, &mut ctx),
                Some(ConstValue::I8(-1)),
            );

            assert_eq!(
                i32::MIN.try_cast(i32_ty, signedness, &mut ctx),
                Some(ConstValue::I32(i32::MIN)),
            );

            assert_eq!(
                i64::MIN.try_cast(i64_ty, signedness, &mut ctx),
                Some(ConstValue::I64(i64::MIN)),
            );
        }
    }

    /// `f32` -> `f64` is `fpext`, and it is exact: every `f32` is representable as an
    /// `f64`, so nothing is rounded and the flag is irrelevant.
    #[test]
    fn widening_a_float_is_exact_and_ignores_signedness() {
        let mut ctx = crate::test_support::ctx();
        let f64_ty = ctx.f64_ty();

        for signedness in [Signedness::Signed, Signedness::Unsigned] {
            assert_eq!(
                0.5f32.try_cast(f64_ty, signedness, &mut ctx),
                Some(ConstValue::Double(OrderedFloat(0.5f64))),
                "a value with an exact binary form survives untouched",
            );

            // 0.1 is not exact in either format, but widening still adds no error:
            // the result is exactly the f32 it came from, not the f64 nearest 0.1.
            assert_eq!(
                0.1f32.try_cast(f64_ty, signedness, &mut ctx),
                Some(ConstValue::Double(OrderedFloat(0.1f32 as f64))),
            );

            assert_ne!(
                0.1f32.try_cast(f64_ty, signedness, &mut ctx),
                Some(ConstValue::Double(OrderedFloat(0.1f64))),
                "widening must not silently re-round to the nearest f64",
            );
        }
    }

    /// `f64` -> `f32` is `fptrunc`. Unlike integer narrowing this is **not** range
    /// checked: out-of-range values become infinity and tiny ones become zero, rather
    /// than being refused.
    ///
    /// Asserted as it behaves today so that changing it is a deliberate act — the
    /// integer path refuses `300i32 -> i8`, and this path does not refuse the
    /// equivalent.
    #[test]
    fn narrowing_a_float_is_lossy_and_unchecked() {
        let mut ctx = crate::test_support::ctx();
        let f32_ty = ctx.f32_ty();

        assert_eq!(
            0.1f64.try_cast(f32_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::Float(OrderedFloat(0.1f64 as f32))),
            "ordinary precision loss",
        );

        assert_eq!(
            1e300f64.try_cast(f32_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::Float(OrderedFloat(f32::INFINITY))),
            "overflow becomes infinity rather than being refused",
        );

        assert_eq!(
            1e-300f64.try_cast(f32_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::Float(OrderedFloat(0.0f32))),
            "underflow becomes zero rather than being refused",
        );

        // The sign of a zero survives, which the constant pool depends on: `+0.0` and
        // `-0.0` are different constants and must not merge.
        let negative_zero = (-0.0f64)
            .try_cast(f32_ty, Signedness::Signed, &mut ctx)
            .expect("-0.0 is representable as an f32");

        assert_eq!(
            negative_zero,
            ConstValue::Float(OrderedFloat(-0.0f32)),
            "-0.0 must not come back as +0.0",
        );

        assert!(!negative_zero.is_sign_positive());
    }

    /// Nothing crosses between the integer and float families, whichever reading is
    /// asked for. Those conversions are `sitofp`/`fptosi` and have to be emitted as
    /// instructions, so folding one here would erase a step the caller owes.
    #[test]
    fn integers_and_floats_do_not_cast_into_each_other() {
        let mut ctx = crate::test_support::ctx();
        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();
        let f32_ty = ctx.f32_ty();
        let f64_ty = ctx.f64_ty();

        for signedness in [Signedness::Signed, Signedness::Unsigned] {
            assert_eq!(1i32.try_cast(f32_ty, signedness, &mut ctx), None);
            assert_eq!(1i32.try_cast(f64_ty, signedness, &mut ctx), None);
            assert_eq!(1i64.try_cast(f64_ty, signedness, &mut ctx), None);
            assert_eq!(1.0f32.try_cast(i32_ty, signedness, &mut ctx), None);
            assert_eq!(1.0f64.try_cast(i64_ty, signedness, &mut ctx), None);
            assert_eq!(true.try_cast(i32_ty, signedness, &mut ctx), None);
        }
    }

    /// `half` and `bfloat` are types this crate can *name* but has no constant for,
    /// so no cast reaches them. Worth pinning: it is a gap in the value
    /// representation rather than a rule, and a test says so out loud.
    #[test]
    fn there_is_no_constant_of_half_or_bfloat_type() {
        let mut ctx = crate::test_support::ctx();

        for ty in [Type::Half, Type::Bfloat] {
            let id = intern(ty, &mut ctx);
            let spelled = rendered(id, &ctx);

            for signedness in [Signedness::Signed, Signedness::Unsigned] {
                assert_eq!(
                    1.0f32.try_cast(id, signedness, &mut ctx),
                    None,
                    "no f32 constant reaches `{spelled}`",
                );

                assert_eq!(
                    1.0f64.try_cast(id, signedness, &mut ctx),
                    None,
                    "and no f64 constant does either",
                );
            }
        }
    }

    /// A cast that refuses must leave the pool untouched. The interner is keyed on
    /// the constant, so a value folded and then rejected would still occupy an entry
    /// and could be handed to a later lookup.
    #[test]
    fn a_refused_cast_interns_nothing() {
        let mut ctx = crate::test_support::ctx();
        let i8_ty = ctx.i8_ty();

        let before = ctx.const_interner.len();

        assert!(Value::from_const(300i32, Some(i8_ty), &mut ctx).is_err());
        assert!(Value::from_const(-1i32, Some(i8_ty), &mut ctx).is_ok());

        assert_eq!(
            ctx.const_interner.len(),
            before + 1,
            "only the accepted cast may reach the pool",
        );
    }

    /// `null` is a `ptr` constant, which is the type it has to report for a value
    /// built from it to be usable where a pointer is expected.
    #[test]
    fn a_null_pointer_is_typed_ptr() {
        let mut ctx = crate::test_support::ctx();

        assert_eq!(rendered(NullPtr::ty(&mut ctx), &ctx), "ptr");
        assert_eq!(NullPtr.into_const(), ConstValue::NullPtr);

        let value = Value::from_const(NullPtr, None, &mut ctx).unwrap();

        assert_eq!(ty_of(&value, &ctx), Type::Ptr);
        assert_eq!(ctx.const_interner.values(), [ConstValue::NullPtr]);
    }

    /// The only cast a null admits is the one that changes nothing. Anything else
    /// would need a real instruction — `ptrtoint` to reach an integer.
    #[test]
    fn a_null_pointer_casts_only_to_ptr() {
        let mut ctx = crate::test_support::ctx();
        let i8_ty = ctx.i8_ty();
        let ptr_ty = ctx.ptr_ty();

        assert_eq!(
            NullPtr.try_cast(ptr_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::NullPtr)
        );

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
                element_ty: i8_ty,
            },
        ] {
            let id = intern(ty, &mut ctx);
            let spelled = rendered(id, &ctx);

            assert_eq!(
                NullPtr.try_cast(id, Signedness::Signed, &mut ctx),
                None,
                "null must not cast to `{spelled}` without an instruction"
            );
        }
    }

    /// And nothing casts *to* a pointer either: an integer would need `inttoptr`,
    /// so accepting one here would fold away a conversion that has to be emitted.
    #[test]
    fn nothing_else_casts_to_a_pointer() {
        let mut ctx = crate::test_support::ctx();
        let ptr_ty = ctx.ptr_ty();

        assert_eq!(0i8.try_cast(ptr_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(0i32.try_cast(ptr_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(0i64.try_cast(ptr_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(0.0f32.try_cast(ptr_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(0.0f64.try_cast(ptr_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(false.try_cast(ptr_ty, Signedness::Signed, &mut ctx), None);
    }

    /// A null and an integer zero are different constants — `ptr null` is not
    /// `i64 0`, whatever the target's representation happens to be — so they must
    /// not share a pool entry.
    #[test]
    fn a_null_pointer_is_not_an_integer_zero() {
        let mut interner = ConstInterner::default();

        let null = interner.intern(ConstValue::NullPtr);
        let zero_64 = interner.intern(ConstValue::I64(0));
        let zero_32 = interner.intern(ConstValue::I32(0));
        let zero_1 = interner.intern(ConstValue::I1(0));

        assert_ne!(null, zero_64);
        assert_ne!(null, zero_32);
        assert_ne!(null, zero_1);
        assert_eq!(interner.len(), 4, "four distinct constants");
    }

    /// Every null is the same null, so the pool holds one however many times it is
    /// asked for — the unit variant has no payload to tell two apart by.
    #[test]
    fn every_null_pointer_is_the_same_constant() {
        let mut ctx = crate::test_support::ctx();

        let ptr_ty = ctx.ptr_ty();

        let first = Value::from_const(NullPtr, None, &mut ctx).unwrap();
        let again = Value::from_const(NullPtr, Some(ptr_ty), &mut ctx).unwrap();

        assert_eq!(ty_of(&first, &ctx), Type::Ptr);
        assert_eq!(ty_of(&again, &ctx), Type::Ptr);

        assert_eq!(
            first.ty(),
            again.ty(),
            "and one `ptr` entry in the type pool, so the two ids are the same id"
        );

        assert_eq!(
            ctx.const_interner.len(),
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
        let mut ctx = crate::test_support::ctx();

        let i64_ty = ctx.i64_ty();

        let err = Value::from_const(NullPtr, Some(i64_ty), &mut ctx)
            .expect_err("null does not cast to i64");

        let msg = err.to_string();

        assert!(msg.contains("ptr"), "missing the source type: {msg}");
        assert!(msg.contains("i64"), "missing the target type: {msg}");
        assert_eq!(ctx.const_interner.len(), 0, "a failed cast interns nothing");
    }

    /// Ints and floats are not interchangeable, in either direction — a real
    /// conversion needs an `sitofp`/`fptosi` instruction, not a constant cast.
    #[test]
    fn casts_between_integers_and_floats_are_refused() {
        let mut ctx = crate::test_support::ctx();
        let f32_ty = ctx.f32_ty();
        let f64_ty = ctx.f64_ty();
        let i32_ty = ctx.i32_ty();
        let i64_ty = ctx.i64_ty();

        assert_eq!(1i32.try_cast(f32_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(1i64.try_cast(f64_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(1.0f32.try_cast(i32_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(1.0f64.try_cast(i64_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(true.try_cast(i32_ty, Signedness::Signed, &mut ctx), None);
    }

    /// Floats cast between the two widths and nowhere else.
    #[test]
    fn float_casts_cover_both_widths_only() {
        let mut ctx = crate::test_support::ctx();
        let f32_ty = ctx.f32_ty();
        let f64_ty = ctx.f64_ty();
        let f16_ty = ctx.f16_ty();
        let ptr_ty = ctx.ptr_ty();

        assert_eq!(
            1.5f32.try_cast(f64_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::Double(OrderedFloat(1.5)))
        );

        assert_eq!(
            1.5f64.try_cast(f32_ty, Signedness::Signed, &mut ctx),
            Some(ConstValue::Float(OrderedFloat(1.5)))
        );

        assert_eq!(1.5f32.try_cast(f16_ty, Signedness::Signed, &mut ctx), None);
        assert_eq!(1.5f64.try_cast(ptr_ty, Signedness::Signed, &mut ctx), None);
    }

    /// A failed cast names both ends, which is the only way to tell which side was
    /// wrong from the message alone.
    #[test]
    fn a_failed_cast_reports_both_types() {
        let mut ctx = crate::test_support::ctx();

        let i32_ty = ctx.i32_ty();

        let err = Value::from_const(1.0f64, Some(i32_ty), &mut ctx)
            .expect_err("f64 does not cast to i32");

        let msg = err.to_string();

        assert!(msg.contains("double"), "missing the source type: {msg}");
        assert!(msg.contains("i32"), "missing the target type: {msg}");
        assert_eq!(ctx.const_interner.len(), 0, "a failed cast interns nothing");
    }

    /// The rendering an error carries has to resolve the ids inside an aggregate, not
    /// print them. This is the property the old `Debug`-renders-like-`Display` impl
    /// on `Type` protected; `BuildError` now carries the rendering rather than the
    /// type, so it is the failing path that has to reach the pool.
    #[test]
    fn a_refused_cast_spells_an_aggregate_target_in_full() {
        let mut ctx = crate::test_support::ctx();
        let i32_ty = intern(Type::I32, &mut ctx);

        let array = intern(
            Type::Array {
                size: 4,
                element_ty: i32_ty,
            },
            &mut ctx,
        );

        let err = Value::from_const(1i32, Some(array), &mut ctx)
            .expect_err("an i32 does not cast to an array");

        assert!(
            err.to_string().contains("[4 x i32]"),
            "the message must spell the aggregate out, got: {err}"
        );
    }

    /// `into_i1` gates the conditional-branch operand, so it has to reject a
    /// non-`i1` by *returning*, not by panicking while building the message.
    #[test]
    fn into_i1_accepts_only_i1() {
        let mut ctx = crate::test_support::ctx();

        let ok = Value::from_const(true, None, &mut ctx).unwrap();
        let not_i1 = Value::from_const(1i32, None, &mut ctx).unwrap();

        assert!(ok.into_i1(&ctx).is_ok());

        let err = not_i1.into_i1(&ctx).expect_err("i32 is not i1");

        assert!(
            err.to_string().contains("i32"),
            "the message must name the offending type: {err}"
        );
    }

    /// Round-tripping through `I1Value` must come back as an `i1` — and as the *same*
    /// `i1`, since the id is carried across rather than re-derived.
    #[test]
    fn an_i1_value_converts_back_to_a_value() {
        let mut ctx = crate::test_support::ctx();

        let value = Value::from_const(true, None, &mut ctx).unwrap();
        let ty = value.ty();
        let i1 = value.into_i1(&ctx).unwrap();
        let back = Value::from(i1);

        assert_eq!(ty_of(&back, &ctx), Type::I1);
        assert_eq!(back.ty(), ty, "the pool id survives the round trip");
    }

    /// Equal constants share a pool entry; constants that differ in *type* do not,
    /// even when they carry the same number — `i8 0` and `i32 0` are different
    /// constants in the IR.
    #[test]
    fn constants_dedup_by_value_and_type() {
        let mut interner = ConstInterner::default();

        let a = interner.intern(ConstValue::I32(0));
        let b = interner.intern(ConstValue::I32(0));
        let c = interner.intern(ConstValue::I8(0));
        let d = interner.intern(ConstValue::I64(0));

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

        let pos = interner.intern(ConstValue::Float(OrderedFloat(0.0)));
        let neg = interner.intern(ConstValue::Float(OrderedFloat(-0.0)));

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

        let a = interner.intern(ConstValue::Double(OrderedFloat(quiet)));
        let b = interner.intern(ConstValue::Double(OrderedFloat(negative)));
        let c = interner.intern(ConstValue::Double(OrderedFloat(payload)));

        assert_ne!(a, b, "a NaN's sign bit is part of its identity");
        assert_ne!(a, c, "so is its payload");
        assert_eq!(interner.len(), 3);

        // And a NaN still equals *itself*, which is what lets the pool find one it
        // has already interned — numeric equality could not do this.
        assert_eq!(
            interner.intern(ConstValue::Double(OrderedFloat(quiet))),
            a,
            "re-interning the same NaN must reuse its entry"
        );

        assert_eq!(interner.len(), 3);
    }

    /// A signature, so the function-type tests below read as the types they mean
    /// rather than as the ids those types are stored under.
    fn func(params: Vec<Type>, result: Type, ctx: &mut Context) -> FuncSignature {
        FuncSignature {
            params: params.into_iter().map(|p| intern(p, ctx)).collect(),
            result: intern(result, ctx),
        }
    }

    /// LLVM writes a function type **result first**, which is the reverse of how the
    /// signature reads in source — so getting the order wrong would produce IR that
    /// parses as a different type rather than failing.
    #[test]
    fn a_function_type_renders_its_result_before_its_params() {
        let mut ctx = crate::test_support::ctx();
        let signature = func(vec![Type::I8, Type::Ptr], Type::I32, &mut ctx);

        assert_eq!(
            intern_rendered(Type::Func(signature), &mut ctx),
            "i32 (i8, ptr)",
            "the variant renders as its signature, with nothing added"
        );
    }

    /// A function taking nothing still has the parens, and `void` is a result like
    /// any other.
    #[test]
    fn a_function_type_with_no_params_keeps_its_parens() {
        let mut ctx = crate::test_support::ctx();

        let void = Type::Func(func(vec![], Type::Void, &mut ctx));
        let i1 = Type::Func(func(vec![], Type::I1, &mut ctx));

        assert_eq!(intern_rendered(void, &mut ctx), "void ()");
        assert_eq!(intern_rendered(i1, &mut ctx), "i1 ()");
    }

    /// Params may be aggregates, so the renderer has to compose with the array and
    /// struct arms rather than assume a scalar.
    #[test]
    fn a_function_type_composes_with_aggregate_params() {
        let mut ctx = crate::test_support::ctx();

        let i32_ty = intern(Type::I32, &mut ctx);
        let i8_ty = intern(Type::I8, &mut ctx);
        let ptr_ty = intern(Type::Ptr, &mut ctx);

        let signature = func(
            vec![
                Type::Array {
                    size: 4,
                    element_ty: i32_ty,
                },
                Type::Struct {
                    fields: Box::new([i8_ty, ptr_ty]),
                    packed: false,
                },
            ],
            Type::Void,
            &mut ctx,
        );

        assert_eq!(
            intern_rendered(Type::Func(signature), &mut ctx),
            "void ([4 x i32], { i8, ptr })"
        );
    }

    /// The parameter list is part of the type, arity included. Comparing the two
    /// lists positionally would stop at the shorter one and call these the same.
    #[test]
    fn function_types_differ_on_arity() {
        let mut ctx = crate::test_support::ctx();

        let one = Type::Func(func(vec![Type::I32], Type::Void, &mut ctx));
        let two = Type::Func(func(vec![Type::I32, Type::I64], Type::Void, &mut ctx));
        let none = Type::Func(func(vec![], Type::Void, &mut ctx));

        assert_ne!(one, two, "a prefix of a longer list is a different type");
        assert_ne!(none, one, "and so is the empty list");
    }

    /// The rest of the identity: which params, in which order, and the result.
    ///
    /// Comparing whole `Type`s here is comparing lists of ids, which is exactly the
    /// property interning has to preserve — equal component types are equal ids, so
    /// two spellings of one signature must still compare equal.
    #[test]
    fn function_types_differ_on_params_order_and_result() {
        let mut ctx = crate::test_support::ctx();

        let base = |ctx: &mut Context| func(vec![Type::I32, Type::I64], Type::Void, ctx);

        let first = Type::Func(base(&mut ctx));
        let second = Type::Func(base(&mut ctx));

        assert_eq!(first, second, "the same signature twice is the same type");

        assert_ne!(
            first,
            Type::Func(func(vec![Type::I64, Type::I32], Type::Void, &mut ctx)),
            "parameter order matters"
        );

        assert_ne!(
            first,
            Type::Func(func(vec![Type::I32, Type::I32], Type::Void, &mut ctx)),
            "so do the parameter types"
        );

        assert_ne!(
            first,
            Type::Func(func(vec![Type::I32, Type::I64], Type::I32, &mut ctx)),
            "and the result"
        );
    }

    /// Structurally equal types are one pool entry, so a second spelling of a type
    /// costs nothing and — the part the `==`s above depend on — comes back as the
    /// very same id.
    #[test]
    fn an_equal_type_interns_to_the_same_id() {
        let mut ctx = crate::test_support::ctx();

        let i32_ty = intern(Type::I32, &mut ctx);

        let first = intern(
            Type::Array {
                size: 4,
                element_ty: i32_ty,
            },
            &mut ctx,
        );

        let again = intern(
            Type::Array {
                size: 4,
                element_ty: i32_ty,
            },
            &mut ctx,
        );

        let longer = intern(
            Type::Array {
                size: 8,
                element_ty: i32_ty,
            },
            &mut ctx,
        );

        assert_eq!(
            first, again,
            "`[4 x i32]` is one type however often it is built"
        );
        assert_ne!(first, longer, "but `[8 x i32]` is a different one");
        assert_eq!(ctx.ty_interner.len(), 3, "i32, [4 x i32], [8 x i32]");
    }

    /// An array's length is part of its type's identity, so it is stored as a plain
    /// number — the one representation on which two spellings of `[4 x i32]` agree.
    ///
    /// A `Value` there would carry its own type id and constant id, so a length that
    /// arrived as an `i32 4` and one that arrived as an `i64 4` would be two pool
    /// entries for the type LLVM writes one way. Since a `TyId` comparison is how
    /// every downstream check now tests types, that would make them *reject valid
    /// IR* rather than fail loudly — see the phi in `instruction.rs`.
    ///
    /// It is also the only thing LLVM can parse: there is no variable-length array
    /// type, and `llvm-as` refuses `[%n x i32]` in the lexer. A runtime count is
    /// `alloca`'s `num_elements` operand, which is a `Value`.
    #[test]
    fn an_array_length_is_a_number_so_one_length_is_one_type() {
        let mut ctx = crate::test_support::ctx();
        let i32_ty = intern(Type::I32, &mut ctx);

        // The same length by four routes — a literal, a widened narrower integer, a
        // computed one, and one that came from the width an `i32` constant has.
        for size in [4u64, u64::from(4u8), 2 + 2, 4i32 as u64] {
            let id = intern(
                Type::Array {
                    size,
                    element_ty: i32_ty,
                },
                &mut ctx,
            );

            assert_eq!(
                rendered(id, &ctx),
                "[4 x i32]",
                "every route to the length spells the same type"
            );
        }

        assert_eq!(
            ctx.ty_interner.len(),
            2,
            "i32 and one `[4 x i32]`: four spellings of the length, one pool entry"
        );
    }

    /// What `load` and `store` will accept: everything with a size. LLVM calls
    /// only scalars "first class", but both instructions take aggregates too, so
    /// the set this predicate names is the loadable one — `void` and function
    /// types out, everything else in.
    #[test]
    fn only_void_and_function_types_are_unsized() {
        let mut ctx = crate::test_support::ctx();

        let i32_ty = intern(Type::I32, &mut ctx);
        let ptr_ty = intern(Type::Ptr, &mut ctx);

        let void = intern(Type::Void, &mut ctx);
        let signature = Type::Func(func(vec![Type::I32], Type::I32, &mut ctx));
        let signature = intern(signature, &mut ctx);

        assert!(!void.is_first_class(&ctx));
        assert!(!signature.is_first_class(&ctx));

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
                element_ty: i32_ty,
            },
            Type::Struct {
                fields: Box::new([i32_ty, ptr_ty]),
                packed: false,
            },
        ] {
            let id = intern(ty, &mut ctx);

            assert!(
                id.is_first_class(&ctx),
                "`{}` has a size and can be loaded",
                rendered(id, &ctx)
            );
        }
    }

    /// Type rendering is what error messages and (later) emitted IR are built from,
    /// so every shape has to spell itself the way LLVM does.
    #[test]
    fn types_render_as_llvm_spells_them() {
        let mut ctx = crate::test_support::ctx();

        for (ty, expected) in [
            (Type::I1, "i1"),
            (Type::I64, "i64"),
            (Type::Half, "half"),
            (Type::Bfloat, "bfloat"),
            (Type::Float, "float"),
            (Type::Double, "double"),
            (Type::Ptr, "ptr"),
            (Type::Void, "void"),
        ] {
            assert_eq!(intern_rendered(ty, &mut ctx), expected);
        }

        let i32_ty = intern(Type::I32, &mut ctx);
        let i8_ty = intern(Type::I8, &mut ctx);
        let double_ty = intern(Type::Double, &mut ctx);

        assert_eq!(
            intern_rendered(
                Type::Array {
                    size: 8,
                    element_ty: i32_ty,
                },
                &mut ctx
            ),
            "[8 x i32]"
        );

        assert_eq!(
            intern_rendered(
                Type::Struct {
                    fields: Box::new([i32_ty, double_ty]),
                    packed: false,
                },
                &mut ctx
            ),
            "{ i32, double }"
        );

        assert_eq!(
            intern_rendered(
                Type::Struct {
                    fields: Box::new([i8_ty]),
                    packed: true,
                },
                &mut ctx
            ),
            "<{ i8 }>"
        );

        assert_eq!(
            intern_rendered(
                Type::Struct {
                    fields: Box::new([]),
                    packed: false,
                },
                &mut ctx
            ),
            "{  }",
            "an empty struct still renders"
        );
    }

    /// Nesting has to compose: a child is an id, so every level of an aggregate is
    /// one more resolution through the pool. Rendering only the outermost node would
    /// pass every assertion above and fail here.
    #[test]
    fn a_nested_aggregate_renders_through_every_level() {
        let mut ctx = crate::test_support::ctx();

        let i16_ty = intern(Type::I16, &mut ctx);
        let ptr_ty = intern(Type::Ptr, &mut ctx);

        let inner = intern(
            Type::Array {
                size: 3,
                element_ty: i16_ty,
            },
            &mut ctx,
        );

        let fields = intern(
            Type::Struct {
                fields: Box::new([ptr_ty, inner]),
                packed: true,
            },
            &mut ctx,
        );

        assert_eq!(
            intern_rendered(
                Type::Array {
                    size: 2,
                    element_ty: fields,
                },
                &mut ctx
            ),
            "[2 x <{ ptr, [3 x i16] }>]"
        );
    }
}
