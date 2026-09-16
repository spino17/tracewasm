//! The three pools a [`Context`](crate::cfg::context::Context) holds, and the ids
//! they hand out.
//!
//! Names, constants and types are each interned once per context and referred to
//! afterwards by a `Copy` id. Two things follow, and both are load-bearing:
//!
//! - **Equal values share an id.** Comparing two types, names or constants is
//!   comparing two integers, which is what lets the builders type-check by `==`
//!   rather than by structural walk.
//! - **An id is a position in one pool.** Resolving an id against a different
//!   [`Context`](crate::cfg::context::Context) reads whatever happens to sit at that
//!   position, silently — the id carries no provenance. One context per module.
//!
//! Each pool is a newtype over [`Interner`] that derefs to it, so the pool's own API
//! (`intern`, `value`, `len`) is available directly.

use crate::value::{ConstValue, Type};
use std::ops::{Deref, DerefMut};
use tracewasm_utils::interner::{InternedId, Interner};

/// The pool of names: function names, block labels and register names all live here.
///
/// Names are interned per *context*, not per function, so `%sum` in two different
/// functions is one entry and one [`StrId`]. That makes a [`StrId`] a poor identity
/// for a register on its own: what identifies one is that name together with the
/// function it belongs to.
#[derive(Default)]
pub struct StrInterner(Interner<String, u32>);

impl Deref for StrInterner {
    type Target = Interner<String, u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for StrInterner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A name in the [`StrInterner`].
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct StrId(pub InternedId<String, u32>);

impl From<InternedId<String, u32>> for StrId {
    fn from(value: InternedId<String, u32>) -> Self {
        StrId(value)
    }
}

/// The pool of constants.
///
/// Identity is by variant *and* bit pattern, so `i8 0` and `i32 0` are separate
/// entries and so are `0.0` and `-0.0` — see [`ConstValue`] for why that matters.
#[derive(Default)]
pub struct ConstInterner(Interner<ConstValue, u32>);

impl Deref for ConstInterner {
    type Target = Interner<ConstValue, u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ConstInterner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A constant in the [`ConstInterner`].
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct ConstId(InternedId<ConstValue, u32>);

impl ConstId {
    /// The underlying pool id, for looking the constant back up.
    pub(crate) fn raw(&self) -> InternedId<ConstValue, u32> {
        self.0
    }
}

impl From<InternedId<ConstValue, u32>> for ConstId {
    fn from(value: InternedId<ConstValue, u32>) -> Self {
        ConstId(value)
    }
}

/// The pool of types.
///
/// A [`Type`]'s aggregate arms name their children by [`TyId`], so every type in the
/// pool is flat and structurally equal types collapse to one entry: `[4 x i32]` built
/// twice is one id, whichever way it was spelled. That is what makes `==` on a
/// [`TyId`] a valid type check.
#[derive(Default)]
pub struct TyInterner(Interner<Type, u32>);

impl Deref for TyInterner {
    type Target = Interner<Type, u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TyInterner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A type in the [`TyInterner`], and the currency the whole builder API deals in.
///
/// Two ids are equal exactly when the types are structurally equal, so a type check
/// is an integer comparison. The predicates and the renderer live on this type rather
/// than on [`Type`], because both need the pool to resolve children — see
/// [`TyId::display`](TyId::display).
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct TyId(InternedId<Type, u32>);

impl TyId {
    /// The underlying pool id, for looking the type back up.
    pub(crate) fn raw(&self) -> InternedId<Type, u32> {
        self.0
    }
}

impl From<InternedId<Type, u32>> for TyId {
    fn from(value: InternedId<Type, u32>) -> Self {
        TyId(value)
    }
}
