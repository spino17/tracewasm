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
