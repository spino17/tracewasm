//! Lazy forwarding of locals and globals, and the spill slots that rescue them.
//!
//! ## Lazy operands
//!
//! `local.get n` moves nothing. The register pass records the operand as "read
//! local `n`" and lets the instruction that eventually consumes it read the local
//! directly, so the overwhelmingly common `local.get; local.get; i32.add` costs no
//! copies at all. Globals work the same way.
//!
//! A forwarded operand is valid only while its origin still holds the value the
//! `local.get` observed. A later `local.set n`, `local.tee n`, or a call that writes
//! a global ends that. Every operand still forwarding to the origin has to be
//! *materialized* first — copied into a frame spill slot the write cannot reach —
//! and the copy is emitted immediately before the write.
//!
//! ## One entry, many borrows
//!
//! A [`LazyEntry`] represents *the value* one local or global held at one point,
//! and every stack slot reading that value holds a [`LazySlot`] handle to the same
//! entry. Two properties fall out of that sharing, and both are needed:
//!
//! * **A local can be borrowed several times at once.** `local.get 0` twice leaves
//!   two operands reading local 0; they share one entry, so
//!   [`LazySlot::spill`] rewrites a single [`LazyLocation`] and both observe the
//!   spill slot. There is no list of borrows to walk and none to miss.
//! * **A borrow ends when its operand is consumed.** `ref_count` counts the stack
//!   slots holding the entry, so the origin table and the spill slot are released
//!   exactly when the last one is popped.
//!
//! This is the shape Lua uses for upvalues: an open upvalue points into the stack,
//! closing it copies the value out and repoints the shared `UpVal`, and every
//! closure sharing it follows. The trigger differs — Lua closes on scope exit, where
//! the storage disappears; here the storage persists but its value is about to
//! change.
//!
//! ## Two levels of indirection
//!
//! [`LazyArena::origin`] maps an origin index to the entry currently forwarding it,
//! if any. That is what lets a second `local.get n` join the existing borrow rather
//! than starting an independent one.
//!
//! Spilling clears the `origin` slot, so a later `local.get n` allocates a fresh
//! entry and the two coexist: the older stack slots read the spill slot holding the
//! old value, the newer ones forward to the local's new value. Two independent
//! snapshots of the same local, live simultaneously, distinguished without any
//! extra bookkeeping.
//!
//! ## Invariants
//!
//! * An entry reachable through `origin` is always [`LazyLocation::Original`];
//!   spilling is what removes it from `origin`, and only freshly allocated entries
//!   are stored back. [`LazySlot::spill`] asserts this.
//! * The location transition is one-way, `Original -> Spilled`.
//! * `ref_count` counts live stack slots only. Reaching zero clears `origin` for an
//!   `Original` entry, or returns the spill slot to the free list for a `Spilled`
//!   one.
//!
//! ## Lifetime
//!
//! Entries are never removed from the arena: [`Arena`] has no removal, and the
//! reference count governs the origin table and the spill slots rather than arena
//! memory. The arena grows with the number of distinct borrows a body opens and is
//! dropped whole when that body finishes lowering, so it is bounded per function.
//! A `Dropped` result does not mean memory was reclaimed.

use id_arena::{Arena, Id};
use std::{fmt, marker::PhantomData};

/// Where the value behind a [`LazyEntry`] currently lives.
///
/// Rewriting this one field is what redirects every stack slot sharing the entry at
/// once.
#[derive(Clone, Copy)]
pub(crate) enum LazyLocation {
    /// Read straight from its origin — the local or global at this index. Valid
    /// until something writes that origin.
    Original(u32),
    /// Materialized into this frame spill slot by a
    /// [`RegInstruction::LocalSpill`](super::RegInstruction::LocalSpill) or
    /// [`GlobalSpill`](super::RegInstruction::GlobalSpill) emitted just before the
    /// write that ended the forwarding.
    Spilled(SpillIndex),
}

/// Whether [`LazySlot::decrease_ref_count`] released the last borrow.
pub(crate) enum LazyEntryDropResult {
    /// The last stack slot holding this entry is gone; the caller releases its
    /// resources — clearing the `origin` slot, or freeing the spill slot.
    Dropped,
    /// Other stack slots still share this entry; its resources stay live.
    StillAlive,
}

/// One lazily forwarded value, shared by every stack slot reading it.
///
/// Not a local or a global as such, but *the value one of them held at one point*.
/// Two entries for the same origin are alive at once when a write separates them.
pub(crate) struct LazyEntry<T> {
    location: LazyLocation,
    /// Live stack slots holding this entry. See [`LazyEntryDropResult`].
    ref_count: u32,
    /// Distinguishes `LazyEntry<Local>` from `LazyEntry<Global>` so a handle into
    /// one arena cannot be used against the other.
    phantom: PhantomData<T>,
}

/// The lazy entries for one origin space — all locals, or all globals, of a
/// function body — plus the index that finds the live entry for a given origin.
///
/// `T` is [`Local`] or [`Global`]; it carries no data and exists so the two arenas
/// and their handles are distinct types.
pub(crate) struct LazyArena<T> {
    arena: Arena<LazyEntry<T>>,
    /// `origin[i]` is the entry currently forwarding local/global `i`, or `None` if
    /// nothing borrows it.
    ///
    /// Public because lowering both reads it — to decide whether a write needs a
    /// spill — and clears it, when a borrow is released or spilled. An entry stored
    /// here is always [`LazyLocation::Original`].
    pub origin: Vec<Option<LazySlot<T>>>,
}

impl<T> LazyArena<T> {
    /// Creates an empty arena whose origin table covers `origin_count` indices: the
    /// function's locals count, or the module's globals count.
    ///
    /// Indexing `origin` is unchecked, so `origin_count` must cover every index the
    /// body can name — for locals, params plus declared locals.
    pub fn new(origin_count: u32) -> Self {
        LazyArena {
            arena: Arena::new(),
            origin: vec![None; origin_count as usize],
        }
    }

    /// Allocates a fresh entry forwarding origin `location`, with one reference.
    ///
    /// Recording it in [`Self::origin`] is the caller's job; the two happen together
    /// only on the first borrow of an origin.
    pub fn allocate(&mut self, location: u32) -> LazySlot<T> {
        LazySlot(self.arena.alloc(LazyEntry {
            location: LazyLocation::Original(location),
            ref_count: 1,
            phantom: PhantomData,
        }))
    }

    /// Resolves a handle. Panics if it did not come from this arena, which would
    /// mean the two origin spaces have been crossed.
    pub fn get_entry(&self, id: LazySlot<T>) -> &LazyEntry<T> {
        let Some(entry_ref) = self.arena.get(id.0) else {
            unreachable!("hitting this means the `LazyEntry` allocation logic is incorrect")
        };

        entry_ref
    }

    /// Mutable [`Self::get_entry`].
    pub fn get_mut_entry(&mut self, id: LazySlot<T>) -> &mut LazyEntry<T> {
        let Some(entry_ref) = self.arena.get_mut(id.0) else {
            unreachable!("hitting this means the `LazyEntry` allocation logic is incorrect")
        };

        entry_ref
    }
}

/// A handle to a [`LazyEntry`], held by a slot on the simulated operand stack.
///
/// `Copy`, and deliberately no more than an index: several stack slots hold the same
/// handle and must observe the same entry, so the state they share lives in the
/// arena rather than in the handle.
pub(crate) struct LazySlot<T>(Id<LazyEntry<T>>);

impl<T> Clone for LazySlot<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for LazySlot<T> {}

impl<T> LazySlot<T> {
    /// Where this entry's value currently lives.
    ///
    /// Read fresh on every use rather than cached: a spill between the borrow and
    /// the read is exactly what this indirection exists to communicate.
    pub fn location(&self, arena: &LazyArena<T>) -> LazyLocation {
        let entry = arena.get_entry(*self);

        entry.location
    }

    /// Records one more stack slot borrowing this entry.
    ///
    /// Called when a second `local.get`/`global.get` of an origin joins the borrow
    /// already recorded in [`LazyArena::origin`].
    pub fn advanced_ref_count(&self, arena: &mut LazyArena<T>) {
        let entry = arena.get_mut_entry(*self);

        entry.ref_count += 1;
    }

    /// Releases one stack slot's hold on this entry.
    ///
    /// [`LazyEntryDropResult::Dropped`] means the caller now owns the cleanup: clear
    /// the entry's `origin` slot if it is still `Original`, or free its spill slot if
    /// it is `Spilled`. Nothing here does that, because which one applies depends on
    /// the location read *before* the decrement.
    pub fn decrease_ref_count(&self, arena: &mut LazyArena<T>) -> LazyEntryDropResult {
        let entry = arena.get_mut_entry(*self);

        entry.ref_count -= 1;

        if entry.ref_count == 0 {
            LazyEntryDropResult::Dropped
        } else {
            LazyEntryDropResult::StillAlive
        }
    }

    /// Redirects this entry — and therefore *every* stack slot sharing it — from
    /// the origin it was lazily reading to `spill_index`.
    ///
    /// An entry reachable through [`LazyArena::origin`] is always
    /// [`LazyLocation::Original`], because the spill that moves it here also
    /// clears its `origin` entry, and only a freshly allocated (hence `Original`)
    /// entry is ever stored back. Spilling twice would therefore mean that
    /// tracking has broken, and silently ignoring it would leak the reserved slot
    /// and emit a spill instruction nothing reads.
    pub fn spill(&self, spill_index: SpillIndex, arena: &mut LazyArena<T>) {
        let entry = arena.get_mut_entry(*self);

        let LazyLocation::Original(_) = entry.location else {
            unreachable!("hitting this means the `LazyArena` origin tracking is incorrect")
        };

        entry.location = LazyLocation::Spilled(spill_index);
    }
}

/// Marker for the globals origin space. Never instantiated.
pub(crate) struct Global;
/// Marker for the locals origin space. Never instantiated.
pub(crate) struct Local;

/// A borrow of a local's value.
pub(crate) type LocalSlot = LazySlot<Local>;
/// A borrow of a global's value.
pub(crate) type GlobalSlot = LazySlot<Global>;

/// The frame's spill area, allocated as a stack of interchangeable slots.
///
/// A spill slot holds one materialized local or global for as long as some stack
/// slot still reads it. Slots carry no identity beyond their index, so a freed one
/// is handed straight back out.
#[derive(Default)]
pub(crate) struct SpillArena {
    allocation_len: u32,
    free_slots: Vec<u32>,
}

impl SpillArena {
    /// Reserves a slot for a value about to be materialized, reusing a freed one
    /// when possible so [`Self::allocation_len`] tracks peak live usage.
    pub fn reserve_slot(&mut self) -> SpillIndex {
        if !self.free_slots.is_empty() {
            return SpillIndex(self.free_slots.pop().unwrap());
        }

        let slot = self.allocation_len;
        self.allocation_len += 1;

        SpillIndex(slot)
    }

    /// Returns a slot to the pool once the last stack slot reading it is popped.
    ///
    /// Safe to reuse immediately: lowering order is execution order, so any later
    /// reservation belongs to an instruction that runs after the last reader.
    pub fn free_slot(&mut self, slot: SpillIndex) {
        self.free_slots.push(slot.0);
    }

    /// Number of spill slots a frame must reserve.
    ///
    /// This is the high-water mark of *simultaneously live* slots, not the number
    /// of spills emitted: [`Self::reserve_slot`] reuses a freed slot before
    /// growing, so the count only advances when every existing slot is live at
    /// once — which is exactly what the frame has to hold.
    pub fn allocation_len(&self) -> u32 {
        self.allocation_len
    }
}

/// A slot in the frame's spill area, holding one materialized local or global.
///
/// The inner index is private and only `SpillArena::reserve_slot` can mint one,
/// so an index that exists is one the arena handed out. There is no `PartialEq`
/// either: two live borrows never share a slot, and comparing them would only be
/// asking which of two distinct slots came first.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpillIndex(u32);

impl SpillIndex {
    #[inline(always)]
    pub fn raw_value(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for SpillIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
