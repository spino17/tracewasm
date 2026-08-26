//! Side storage for the operand shapes that do not fit an eight-byte instruction.
//!
//! Seven variants use one: `if`, `br_if`, `br_table`, `call_indirect`, `select`,
//! `memory.init`, and [`Move`](super::RegInstruction::Move). The first six carry more
//! operands than six bytes hold; `Move`'s arity is not fixed by its opcode at all.
//! Each keeps a four-byte [`Id`] instead, and the arenas ship inside
//! [`RegFrameLayout`](super::RegFrameLayout) because every id in a body points into
//! one.
//!
//! There is no dedup and no removal — one entry per instruction that needs one, in
//! emission order. Unlike [`interner`](super::interner), which exists to make a
//! 16-bit id safe by collapsing repeats, an entry here is one instruction's own
//! operands and has nothing to share with anything else.

use std::marker::PhantomData;

/// Entries in allocation order, indexed by [`Id`].
pub(crate) struct Arena<T>(Vec<T>);

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Arena(vec![])
    }
}

/// One entry, by position. Four bytes, and typed so an id cannot be read against the
/// wrong arena.
#[derive(Debug)]
pub(crate) struct Id<T>(u32, PhantomData<T>);

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Id(self.0, PhantomData)
    }
}

impl<T> Copy for Id<T> {}

impl<T> Arena<T> {
    /// Appends `entry` and returns the id naming it.
    ///
    /// Unbounded, unlike an interner's cap: an [`Id`] is 32 bits, and a body cannot
    /// hold more instructions than that.
    pub fn alloc(&mut self, entry: T) -> Id<T> {
        let id = self.0.len() as u32;

        self.0.push(entry);

        Id(id, PhantomData)
    }

    /// The entry `id` names. Panics on an id from a different arena.
    pub fn get(&self, id: Id<T>) -> &T {
        &self.0[id.0 as usize]
    }

    /// [`Self::get`] for the end-of-body passes, which resolve operands and jump
    /// targets in place — an entry is written after allocation, never before.
    pub fn get_mut(&mut self, id: Id<T>) -> &mut T {
        &mut self.0[id.0 as usize]
    }
}
