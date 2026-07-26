//! The linear-memory abstraction the VM runs against.

use crate::error::MemoryError;
pub mod linear;

/// A module's linear memory, supplied by the embedder.
///
/// The interpreter is generic over this trait so the backing store (a plain
/// `Vec<u8>`, an mmap, a guarded region, …) is the embedder's choice.
pub trait Memory {
    /// Creates a memory pre-allocated to `size` bytes.
    /// Per the WebAssembly spec, this should be completely zeroed.
    fn allocate_initial_memory(size: usize) -> Self;

    /// Returns the size of the memory.
    fn size(&self) -> usize;

    /// Reads `data.len()` bytes starting from the `offset`.
    fn read(&self, offset: usize, data: &mut [u8]) -> Result<(), MemoryError>;

    /// Writes `data.len()` bytes from `data` to the memory at `offset`.
    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), MemoryError>;

    /// Reads a `u8` at `offset`. Errors if the access is out of bounds.
    fn read_u8(&self, offset: usize) -> Result<u8, MemoryError> {
        let mut buf: [u8; 1] = [0; 1];
        self.read(offset, &mut buf)?;
        Ok(u8::from_le_bytes(buf))
    }

    /// Writes a `u8` at `offset`. Errors if the access is out of bounds.
    fn write_u8(&mut self, offset: usize, data: u8) -> Result<(), MemoryError> {
        let buf: [u8; 1] = [data; 1];
        self.write(offset, &buf)?;

        Ok(())
    }

    /// Reads a little-endian `u16` at `offset`. Errors if the access is out of bounds.
    fn read_u16(&self, offset: usize) -> Result<u16, MemoryError> {
        let mut buf: [u8; 2] = [0; 2];
        self.read(offset, &mut buf)?;

        Ok(u16::from_le_bytes(buf))
    }

    /// Writes a little-endian `u16` at `offset`. Errors if the access is out of bounds.
    fn write_u16(&mut self, offset: usize, data: u16) -> Result<(), MemoryError> {
        let buf: [u8; 2] = data.to_le_bytes();
        self.write(offset, &buf)?;

        Ok(())
    }

    /// Reads a little-endian `u32` at `offset`. Errors if the access is out of bounds.
    fn read_u32(&self, offset: usize) -> Result<u32, MemoryError> {
        let mut buf: [u8; 4] = [0; 4];
        self.read(offset, &mut buf)?;

        Ok(u32::from_le_bytes(buf))
    }

    /// Writes a little-endian `u32` at `offset`. Errors if the access is out of bounds.
    fn write_u32(&mut self, offset: usize, data: u32) -> Result<(), MemoryError> {
        let buf: [u8; 4] = data.to_le_bytes();
        self.write(offset, &buf)?;

        Ok(())
    }

    /// Reads a little-endian `u64` at `offset`. Errors if the access is out of bounds.
    fn read_u64(&self, offset: usize) -> Result<u64, MemoryError> {
        let mut buf: [u8; 8] = [0; 8];
        self.read(offset, &mut buf)?;

        Ok(u64::from_le_bytes(buf))
    }

    /// Writes a little-endian `u64` at `offset`. Errors if the access is out of bounds.
    fn write_u64(&mut self, offset: usize, data: u64) -> Result<(), MemoryError> {
        let buf: [u8; 8] = data.to_le_bytes();
        self.write(offset, &buf)?;

        Ok(())
    }

    /// Reads a little-endian `f32` at `offset`, preserving the exact bit pattern
    /// (no NaN canonicalization). Errors if the access is out of bounds.
    fn read_f32(&self, offset: usize) -> Result<f32, MemoryError> {
        let mut buf: [u8; 4] = [0; 4];
        self.read(offset, &mut buf)?;

        Ok(f32::from_bits(u32::from_le_bytes(buf)))
    }

    /// Writes a little-endian `f32` at `offset`, preserving the exact bit pattern
    /// (no NaN canonicalization). Errors if the access is out of bounds.
    fn write_f32(&mut self, offset: usize, data: f32) -> Result<(), MemoryError> {
        let buf = data.to_bits().to_le_bytes();
        self.write(offset, &buf)?;

        Ok(())
    }

    /// Reads a little-endian `f64` at `offset`, preserving the exact bit pattern
    /// (no NaN canonicalization). Errors if the access is out of bounds.
    fn read_f64(&self, offset: usize) -> Result<f64, MemoryError> {
        let mut buf: [u8; 8] = [0; 8];
        self.read(offset, &mut buf)?;

        Ok(f64::from_bits(u64::from_le_bytes(buf)))
    }

    /// Writes a little-endian `f64` at `offset`, preserving the exact bit pattern
    /// (no NaN canonicalization). Errors if the access is out of bounds.
    fn write_f64(&mut self, offset: usize, data: f64) -> Result<(), MemoryError> {
        let buf = data.to_bits().to_le_bytes();
        self.write(offset, &buf)?;

        Ok(())
    }
}
