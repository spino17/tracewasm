use id_arena::{Arena, Id};
use std::marker::PhantomData;

#[derive(Clone, Copy)]
pub(crate) enum LazyLocation {
    Original(u32),
    Spilled(u32),
}

pub(crate) enum LazyEntryDropResult {
    Dropped,
    StillAlive,
}

pub(crate) struct LazyEntry<T> {
    location: LazyLocation,
    ref_count: u32,
    phantom: PhantomData<T>,
}

pub(crate) struct LazyArena<T> {
    arena: Arena<LazyEntry<T>>,
}

impl<T> LazyArena<T> {
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

pub(crate) struct LazySlot<T>(Id<LazyEntry<T>>);

impl<T> Clone for LazySlot<T> {
    fn clone(&self) -> Self {
        LazySlot(self.0)
    }
}

impl<T> Copy for LazySlot<T> {}

impl<T> LazySlot<T> {
    pub fn advanced_ref_count(&self, arena: &mut LazyArena<T>) {
        let entry = arena.get_mut_entry(*self);

        entry.ref_count += 1;
    }

    pub fn decrease_ref_count(&mut self, arena: &mut LazyArena<T>) -> LazyEntryDropResult {
        let entry = arena.get_mut_entry(*self);

        entry.ref_count -= 1;

        if entry.ref_count == 0 {
            LazyEntryDropResult::Dropped
        } else {
            LazyEntryDropResult::StillAlive
        }
    }

    pub fn spill(&mut self, spill_index: u32, arena: &mut LazyArena<T>) {
        let entry = arena.get_mut_entry(*self);

        debug_assert!(matches!(entry.location, LazyLocation::Original(_)));

        if let LazyLocation::Original(_) = entry.location {
            entry.location = LazyLocation::Spilled(spill_index);
        }
    }
}

pub(crate) struct Global;
pub(crate) struct Local;

pub type LocalSlot = LazySlot<Local>;
pub type GlobalSlot = LazySlot<Global>;

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
}
