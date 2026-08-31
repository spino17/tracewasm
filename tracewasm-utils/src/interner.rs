use crate::error::TracewasmUtilsError;
use rustc_hash::FxHashMap;
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

#[derive(Debug)]
pub struct InternedId<T, C>(C, PhantomData<T>);

impl<T, C: Capacity> InternedId<T, C> {
    pub fn raw(&self) -> C {
        self.0
    }
}

// All four of these are written out rather than derived, because a derive would
// bound them on `T` — the marker type, which is never inspected and need not be
// `PartialEq` or `Hash` itself.

impl<T, C: Capacity> PartialEq for InternedId<T, C> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T, C: Capacity> Eq for InternedId<T, C> {}

/// Hashes the position alone, matching what [`PartialEq`] compares. Derived, the
/// two would still agree — `PhantomData` hashes nothing — but only by accident of
/// the field order.
impl<T, C: Capacity> Hash for InternedId<T, C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T, C: Capacity> Clone for InternedId<T, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, C: Capacity> Copy for InternedId<T, C> {}

pub trait Capacity: Clone + Copy + PartialEq + Eq + Hash {
    fn val() -> u64;

    fn from_usize(val: usize) -> Self;

    fn to_usize(&self) -> usize;
}

impl Capacity for u16 {
    fn val() -> u64 {
        u16::MAX as u64
    }

    fn from_usize(val: usize) -> Self {
        val as u16
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl Capacity for u32 {
    fn val() -> u64 {
        u32::MAX as u64
    }

    fn from_usize(val: usize) -> Self {
        val as u32
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl Capacity for u64 {
    fn val() -> u64 {
        u64::MAX
    }

    fn from_usize(val: usize) -> Self {
        val as u64
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

pub struct Interner<T, C> {
    values: Vec<T>,
    reverse_map: FxHashMap<T, InternedId<T, C>>,
    phantom: PhantomData<C>,
}

impl<T, C> Default for Interner<T, C> {
    fn default() -> Self {
        Interner {
            values: vec![],
            reverse_map: FxHashMap::default(),
            phantom: PhantomData,
        }
    }
}

impl<T: Clone + PartialEq + Eq + Hash, C: Capacity> Interner<T, C> {
    pub fn new() -> Self {
        Interner {
            values: Vec::default(),
            reverse_map: FxHashMap::default(),
            phantom: PhantomData,
        }
    }

    /// Interns `val`, reporting the cap rather than exceeding it.
    ///
    /// This is the one to use when the values come from **outside** — a module being
    /// compiled fills a pool from its own body, so overflow there is an input the
    /// caller has to answer for, not a bug. `tracewasm-core`'s register machine
    /// degrades to the stack machine on this error rather than failing the compile.
    ///
    /// Where the caller supplies the values itself, [`intern`](Self::intern) says the
    /// same thing without the `?`.
    pub fn try_intern(&mut self, val: T) -> Result<InternedId<T, C>, TracewasmUtilsError> {
        // The lookup comes first so that a repeat still succeeds once the pool is
        // full: it costs no entry, so refusing it would reject a program that fits.
        if let Some(id) = self.reverse_map.get(&val) {
            Ok(*id)
        } else {
            if self.values.len() >= C::val() as usize {
                return Err(TracewasmUtilsError::ToManyUniqueValues {
                    needed: self.values.len() as u32 + 1,
                    limit: C::val(),
                });
            }

            let id = InternedId(C::from_usize(self.values.len()), PhantomData);

            self.values.push(val.clone());
            self.reverse_map.insert(val, id);

            Ok(id)
        }
    }

    /// Interns `val`, panicking if the pool is full.
    ///
    /// For callers that supply the values themselves — a lowering pass interning the
    /// types and register names it just built — where overflow means a bug in this
    /// crate rather than a file someone handed us, and threading a `Result` through
    /// every call site buys nothing. Use [`try_intern`](Self::try_intern) for values
    /// that came from outside.
    pub fn intern(&mut self, val: T) -> InternedId<T, C> {
        self.try_intern(val).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn value(&self, id: InternedId<T, C>) -> &T {
        &self.values[id.0.to_usize()]
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether anything has been interned yet.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn into_values(self) -> Vec<T> {
        self.values
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use crate::error::TracewasmUtilsError;

    use super::*;

    /// Interning is only worth doing if a repeat is free, so this is the property
    /// everything else rests on.
    #[test]
    fn a_repeat_returns_the_first_id_and_adds_nothing() {
        let mut i: Interner<String, u16> = Interner::new();

        let a = i.intern("x".to_string());
        let b = i.intern("y".to_string());
        let a_again = i.intern("x".to_string());

        assert_eq!(a, a_again, "the same value must get the same id");
        assert_ne!(a, b, "different values must get different ids");
        assert_eq!(i.len(), 2, "a repeat must not grow the pool");
    }

    /// Ids are positions, so they have to be handed out in order and index back to
    /// the value that produced them.
    #[test]
    fn ids_are_positions_and_round_trip() {
        let mut i: Interner<String, u16> = Interner::new();

        let ids: Vec<u16> = ["a", "b", "c"]
            .iter()
            .map(|s| i.intern(s.to_string()).raw())
            .collect();

        assert_eq!(ids, vec![0, 1, 2]);

        for (n, name) in ["a", "b", "c"].iter().enumerate() {
            let id = i.intern(name.to_string());

            assert_eq!(id.raw() as usize, n);
            assert_eq!(i.value(id), name);
        }

        assert_eq!(i.values(), ["a", "b", "c"], "`values` is in id order");
    }

    /// The pool ships as its values in id order, so a consumer can index it with a
    /// raw id after the interner is gone.
    #[test]
    fn into_values_keeps_id_order() {
        let mut i: Interner<String, u16> = Interner::new();

        for s in ["first", "second", "third"] {
            i.intern(s.to_string());
        }

        assert_eq!(i.into_values(), vec!["first", "second", "third"]);
    }

    /// The cap is reported rather than silently truncating the id, which is the
    /// whole reason [`Interner::try_intern`] exists alongside the panicking
    /// [`Interner::intern`]: a caller whose values came from outside needs to answer
    /// for an overrun, not die on it.
    #[test]
    fn exceeding_the_capacity_is_an_error() {
        let mut i: Interner<u32, u16> = Interner::new();

        // `Capacity::val` for `u16` is `u16::MAX`, so that many entries fit.
        for n in 0..u16::MAX as u32 {
            i.try_intern(n)
                .unwrap_or_else(|e| panic!("entry {n} should fit: {e}"));
        }

        let err = i
            .try_intern(u32::MAX)
            .expect_err("one past the cap must fail");

        assert!(
            matches!(
                err,
                TracewasmUtilsError::ToManyUniqueValues { limit, .. }
                    if limit == u16::MAX as u64
            ),
            "the error must name the capacity it hit, got: {err}"
        );

        assert_eq!(
            i.len(),
            u16::MAX as usize,
            "the failed intern added nothing"
        );
    }

    /// The other half of the pair: [`Interner::intern`] refuses the same value, by
    /// panicking rather than returning. A caller reaches for it when the values are
    /// its own, so an overrun there is a bug in the caller and there is nobody to
    /// report it to.
    #[test]
    #[should_panic(expected = "too many unique interned values")]
    fn exceeding_the_capacity_panics_on_the_infallible_intern() {
        let mut i: Interner<u32, u16> = Interner::new();

        for n in 0..u16::MAX as u32 {
            i.intern(n);
        }

        i.intern(u32::MAX);
    }

    /// A repeat must still succeed once the pool is full: it costs no entry, so
    /// refusing it would reject a program that fits.
    ///
    /// Asserted on **both** halves of the API, because this is what an
    /// implementation that checked the capacity before the lookup would get wrong —
    /// and under `intern` that mistake is not a rejection but a panic.
    #[test]
    fn a_repeat_still_succeeds_at_capacity() {
        let mut i: Interner<u32, u16> = Interner::new();

        for n in 0..u16::MAX as u32 {
            i.intern(n);
        }

        assert!(i.try_intern(0).is_ok(), "0 is already interned");
        assert!(i.try_intern(u32::MAX).is_err(), "but a new value is not");

        assert_eq!(
            i.intern(0).raw(),
            0,
            "and `intern` returns it rather than panicking on a full pool"
        );

        assert_eq!(i.len(), u16::MAX as usize, "neither call grew the pool");
    }

    /// The id width follows `C`, so a wider capacity really can name more entries.
    /// A 16-bit id would wrap here and hand two values the same id.
    #[test]
    fn a_u32_capacity_names_more_than_a_u16_id_could() {
        let mut i: Interner<u32, u32> = Interner::new();
        let first = i.intern(0);

        for n in 1..=(u16::MAX as u32 + 1) {
            i.intern(n);
        }

        let past_u16 = i.intern(u16::MAX as u32 + 1);

        assert_eq!(past_u16.raw(), u16::MAX as u32 + 1);
        assert_ne!(first, past_u16, "ids must not wrap");
        assert_eq!(i.value(first), &0);
        assert_eq!(i.value(past_u16), &(u16::MAX as u32 + 1));
    }

    /// `from_usize`/`to_usize` are how a position becomes an id and back, so they
    /// have to be lossless over the range each capacity admits.
    #[test]
    fn capacity_conversions_round_trip_at_their_edges() {
        assert_eq!(u16::from_usize(0).to_usize(), 0);

        assert_eq!(
            u16::from_usize(u16::MAX as usize - 1).to_usize(),
            u16::MAX as usize - 1
        );

        assert_eq!(
            u32::from_usize(u16::MAX as usize + 1).to_usize(),
            u16::MAX as usize + 1
        );

        assert_eq!(
            u64::from_usize(u32::MAX as usize + 1).to_usize(),
            u32::MAX as usize + 1
        );

        assert_eq!(u16::val(), u16::MAX as u64);
        assert_eq!(u32::val(), u32::MAX as u64);
        assert_eq!(u64::val(), u64::MAX);
    }
}
