use crate::error::TraceWasmError;
use std::marker::PhantomData;

pub(crate) struct Arena<T>(Vec<T>);

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Arena(vec![])
    }
}

#[derive(Debug)]
pub(crate) struct Id<T>(u32, PhantomData<T>);

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Id(self.0, PhantomData)
    }
}

impl<T> Copy for Id<T> {}

impl<T> Arena<T> {
    pub fn alloc(&mut self, entry: T) -> Id<T> {
        let id = self.0.len() as u32;

        self.0.push(entry);

        Id(id, PhantomData)
    }

    pub fn get(&self, id: Id<T>) -> &T {
        &self.0[id.0 as usize]
    }

    pub fn get_mut(&mut self, id: Id<T>) -> &mut T {
        &mut self.0[id.0 as usize]
    }
}

pub(crate) struct SmolArena<T> {
    arena: Vec<T>,
    max_size: u16,
}

#[derive(Debug)]
pub(crate) struct SmolId<T>(u16, PhantomData<T>);

impl<T> SmolId<T> {
    pub fn raw(&self) -> u16 {
        self.0
    }
}

impl<T> Clone for SmolId<T> {
    fn clone(&self) -> Self {
        SmolId(self.0, PhantomData)
    }
}

impl<T> Copy for SmolId<T> {}

impl<T> SmolArena<T> {
    pub fn new(max_size: u16) -> Self {
        SmolArena {
            arena: vec![],
            max_size,
        }
    }

    /// # Errors
    ///
    /// [`TraceWasmError::RegisterFrameTooLarge`] once the arena would outgrow a
    /// 16-bit id.
    pub fn alloc(&mut self, entry: T) -> Result<SmolId<T>, TraceWasmError> {
        let id = self.arena.len();

        if id > self.max_size as usize {
            return Err(TraceWasmError::RegisterFrameTooLarge {
                what: "memory offsets",
                needed: id as u32 + 1,
                limit: self.max_size as u32,
            });
        }

        self.arena.push(entry);

        Ok(SmolId(id as u16, PhantomData))
    }

    pub fn get(&self, id: SmolId<T>) -> &T {
        &self.arena[id.0 as usize]
    }

    pub fn get_mut(&mut self, id: SmolId<T>) -> &mut T {
        &mut self.arena[id.0 as usize]
    }
}
