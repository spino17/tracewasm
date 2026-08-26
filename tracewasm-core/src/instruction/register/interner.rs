//! A dedup-on-insert pool, for the two things an instruction names by 16-bit id.
//!
//! Both users exist because the value is too wide to sit in an eight-byte
//! instruction: a [`Const`](super::Const) is a tagged 64-bit value, and a memory
//! offset is a `u32`. Naming them by [`InternedId`] costs two bytes each.
//!
//! Interning rather than appending is what makes the 16-bit id safe. Real bodies
//! repeat both heavily — every bare pointer deref rustc emits loads through offset
//! `0`, and `i32.const 1` appears everywhere — so a pool that appended would grow
//! once per *instruction* and the id would be the tightest limit in the lowering. It
//! grows once per distinct value instead.
//!
//! Dedup is by `Eq`/`Hash` on `T`, so what counts as the same value is the value
//! type's business. `Const` defines both by *bits*, which is what keeps `+0.0` apart
//! from `-0.0` and one NaN payload apart from another.

use crate::error::TraceWasmError;
use rustc_hash::FxHashMap;
use std::{fmt::Debug, hash::Hash, marker::PhantomData};

/// A pool entry, by position. Two bytes, and typed so an id cannot be read against
/// the wrong pool.
#[derive(Debug)]
pub(crate) struct InternedId<T>(u16, PhantomData<T>);

impl<T> InternedId<T> {
    /// The raw position, for the arithmetic that places a constant in the frame.
    pub fn raw(&self) -> u16 {
        self.0
    }
}

impl<T> Clone for InternedId<T> {
    fn clone(&self) -> Self {
        InternedId(self.0, PhantomData)
    }
}

impl<T> Copy for InternedId<T> {}

/// What to call a pool of `T` in an error.
///
/// Per type rather than per pool, so a `T` interned for a second purpose would
/// inherit the first one's noun — which is why the offset pool holds a
/// [`MemoryOffset`](super::MemoryOffset) newtype and not a bare `u32`.
pub(crate) trait TyToString {
    fn to_string() -> String;
}

/// The pool: values in id order, plus the reverse index that makes `intern` dedup.
///
/// `values` is what ships with the body; `reverse_map` is lowering-time only.
#[derive(Default)]
pub(crate) struct Interner<T> {
    /// The distinct values, in id order — `values[id]` is what `id` names.
    values: Vec<T>,
    /// Value back to id, so a repeat costs a lookup rather than an entry. Dropped
    /// with the pool; only `values` ships.
    reverse_map: FxHashMap<T, InternedId<T>>,
    /// Most distinct values this pool may hold, i.e. what its id width can name.
    max_size: u16,
}

impl<T: Clone + PartialEq + Eq + Hash + TyToString> Interner<T> {
    /// An empty pool that will not grow past `max_size` distinct values.
    ///
    /// `max_size` is a cap on what an [`InternedId`] can *name*, so it belongs at
    /// `u16::MAX` unless the caller has a tighter reason.
    pub fn new(max_size: u16) -> Self {
        Interner {
            values: Vec::default(),
            reverse_map: FxHashMap::default(),
            max_size,
        }
    }

    /// The id for `val`, adding it only if the pool does not hold it already.
    ///
    /// # Errors
    ///
    /// [`TraceWasmError::ToManyUniqueValues`] when a *new* value would take the pool
    /// past `max_size`. A repeat of a value already interned never fails, however
    /// full the pool is.
    pub fn intern(&mut self, val: T) -> Result<InternedId<T>, TraceWasmError> {
        if let Some(id) = self.reverse_map.get(&val) {
            Ok(*id)
        } else {
            if self.values.len() >= self.max_size as usize {
                return Err(TraceWasmError::ToManyUniqueValues {
                    what: T::to_string(),
                    needed: self.values.len() as u32 + 1,
                    limit: self.max_size as u32,
                });
            }

            let id = InternedId(self.values.len() as u16, PhantomData);

            self.values.push(val.clone());
            self.reverse_map.insert(val, id);

            Ok(id)
        }
    }

    /// The value `id` names. Panics on an id from a different pool.
    pub fn value(&self, id: InternedId<T>) -> &T {
        &self.values[id.0 as usize]
    }

    /// Distinct values interned. For the constant pool this is also the width of the
    /// frame's constant region, and one of the two terms every register index shifts
    /// by.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// The values in id order, taking the pool apart — the form that ships with a
    /// lowered body, leaving the reverse index behind.
    pub fn into_values(self) -> Vec<T> {
        self.values
    }

    /// [`Self::into_values`] without consuming, for a pool still being built.
    pub fn values(&self) -> &[T] {
        &self.values
    }
}
