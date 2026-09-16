use tracewasm_core::{
    error::MemoryError,
    memory::{Memory, MemoryView, linear::LinearMemory},
};

/// Counts every guest read, and delegates the rest.
pub struct CountingMemory {
    inner: LinearMemory,
    pub reads: std::cell::Cell<u64>,
}

impl MemoryView for CountingMemory {
    fn size_in_bytes(&self) -> usize {
        self.inner.size_in_bytes()
    }

    fn read(&self, offset: usize, data: &mut [u8]) -> Result<(), MemoryError> {
        self.reads.set(self.reads.get() + 1);
        self.inner.read(offset, data)
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), MemoryError> {
        self.inner.write(offset, data)
    }

    fn copy_within(&mut self, dest: usize, src: usize, len: usize) -> Result<(), MemoryError> {
        self.inner.copy_within(dest, src, len)
    }
}

impl Memory for CountingMemory {
    fn allocate_initial_memory(size_in_pages: u32) -> Self {
        CountingMemory {
            inner: LinearMemory::allocate_initial_memory(size_in_pages),
            reads: std::cell::Cell::new(0),
        }
    }

    fn grow(&mut self, delta: u32, max_pages: u32) -> Result<u32, MemoryError> {
        self.inner.grow(delta, max_pages)
    }
}
