use id_arena::{Arena, Id};
use std::marker::PhantomData;

#[derive(Clone, Copy)]
pub enum LazyLocation {
    Original(u32),
    Spilled(u32),
}

pub enum LazyEntryDropResult {
    Dropped,
    StillAlive,
}

pub struct LazyEntry<T> {
    location: LazyLocation,
    ref_count: u32,
    phantom: PhantomData<T>,
}

pub struct LazyArena<T> {
    arena: Arena<LazyEntry<T>>,
    pub origin: Vec<Option<LazySlot<T>>>,
}

impl<T> LazyArena<T> {
    pub fn new(origin_count: u32) -> Self {
        LazyArena {
            arena: Arena::new(),
            origin: vec![None; origin_count as usize],
        }
    }

    pub fn allocate(&mut self, location: u32) -> LazySlot<T> {
        LazySlot(self.arena.alloc(LazyEntry {
            location: LazyLocation::Original(location),
            ref_count: 1,
            phantom: PhantomData,
        }))
    }

    pub fn get_entry(&self, id: LazySlot<T>) -> &LazyEntry<T> {
        let Some(entry_ref) = self.arena.get(id.0) else {
            unreachable!("hitting this means the `LazyEntry` allocation logic is incorrect")
        };

        entry_ref
    }

    pub fn get_mut_entry(&mut self, id: LazySlot<T>) -> &mut LazyEntry<T> {
        let Some(entry_ref) = self.arena.get_mut(id.0) else {
            unreachable!("hitting this means the `LazyEntry` allocation logic is incorrect")
        };

        entry_ref
    }
}

pub struct LazySlot<T>(Id<LazyEntry<T>>);

impl<T> Clone for LazySlot<T> {
    fn clone(&self) -> Self {
        LazySlot(self.0)
    }
}

impl<T> Copy for LazySlot<T> {}

impl<T> LazySlot<T> {
    pub fn location(&self, arena: &LazyArena<T>) -> LazyLocation {
        let entry = arena.get_entry(*self);

        entry.location
    }

    pub fn advanced_ref_count(&self, arena: &mut LazyArena<T>) {
        let entry = arena.get_mut_entry(*self);

        entry.ref_count += 1;
    }

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
    pub fn spill(&self, spill_index: u32, arena: &mut LazyArena<T>) {
        let entry = arena.get_mut_entry(*self);

        let LazyLocation::Original(_) = entry.location else {
            unreachable!("hitting this means the `LazyArena` origin tracking is incorrect")
        };

        entry.location = LazyLocation::Spilled(spill_index);
    }
}

pub struct Global;
pub struct Local;

pub type LocalSlot = LazySlot<Local>;
pub type GlobalSlot = LazySlot<Global>;

#[derive(Default)]
pub(crate) struct SpillArena {
    allocation_len: u32,
    free_slots: Vec<u32>,
}

impl SpillArena {
    pub fn reserve_slot(&mut self) -> u32 {
        if !self.free_slots.is_empty() {
            return self.free_slots.pop().unwrap();
        }

        let slot = self.allocation_len;
        self.allocation_len += 1;

        slot
    }

    pub fn free_slot(&mut self, slot: u32) {
        self.free_slots.push(slot);
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
