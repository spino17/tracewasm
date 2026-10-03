use crate::{
    error::MemoryError,
    memory::{Memory, MemoryView},
};

#[repr(C)]
pub struct MmapMemory {}

impl Memory for MmapMemory {
    fn allocate_initial_memory(size_in_pages: u32) -> Self {
        todo!()
    }

    fn grow(&mut self, delta_in_pages: u32, max_size_in_pages: u32) -> Result<u32, MemoryError> {
        todo!()
    }
}

impl MemoryView for MmapMemory {
    fn size_in_bytes(&self) -> usize {
        todo!()
    }

    fn copy_within(&mut self, dest: usize, src: usize, len: usize) -> Result<(), MemoryError> {
        todo!()
    }

    fn fill(&mut self, dest: usize, val: u32, len: usize) -> Result<(), MemoryError> {
        todo!()
    }

    fn read(&self, offset: usize, data: &mut [u8]) -> Result<(), MemoryError> {
        todo!()
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), MemoryError> {
        todo!()
    }
}
