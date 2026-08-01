//! The linear-memory abstraction the VM runs against.

use crate::{
    error::{MemoryAccessKind, MemoryError},
    module::WASM_MEMORY_PAGE_SIZE,
};
pub mod linear;

/// Read/write access to a linear memory, without the ability to resize it.
///
/// This is the capability handed to host functions: every operation here is
/// bounds-checked and leaves the memory's size unchanged, so a host cannot grow
/// past the module's declared maximum or the instance
/// [`Config`](crate::instance::config::Config) cap — those limits live on the
/// [`Instance`](crate::instance::Instance), which a host function cannot see.
/// Resizing is reserved to [`Memory`], which only the interpreter holds.
pub trait MemoryView {
    /// Returns the size of the memory in bytes.
    ///
    /// This is the authoritative size: bounds checks and
    /// [`Self::size_in_pages`] are both derived from it.
    fn size_in_bytes(&self) -> usize;

    /// Returns the size of the memory in whole WASM pages, rounding down.
    ///
    /// A backing store is normally a whole number of pages, so this is exact;
    /// it rounds down only for a memory built at byte granularity.
    fn size_in_pages(&self) -> u64 {
        self.size_in_bytes() as u64 / WASM_MEMORY_PAGE_SIZE
    }

    /// Copies `len` bytes within this memory from `src` to `dest` (backs
    /// `memory.copy`).
    ///
    /// The two ranges **may overlap**; an implementation must behave like
    /// `memmove` — copying as if through a temporary — not a naive forward byte
    /// loop, which would corrupt the tail when `dest` is just above `src`.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::OutOfBoundsAccess`] if either range runs past the
    /// end of the memory. Both ranges must be checked *before* any byte moves, so
    /// a failed copy leaves the memory unchanged.
    fn copy_within(&mut self, dest: usize, src: usize, len: usize) -> Result<(), MemoryError>;

    /// Fills `len` bytes starting at `dest` with the low byte of `val` (backs
    /// `memory.fill`, whose value operand is an `i32` of which only the low byte
    /// is used).
    ///
    /// The default implementation writes through [`Self::write`] in fixed-size
    /// chunks, so it allocates nothing regardless of `len`. A backend with direct
    /// access to its buffer should override this with an in-place fill.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::OutOfBoundsAccess`] if the range runs past the end
    /// of the memory. The bounds are checked up front, so a failed fill leaves the
    /// memory unchanged.
    fn fill(&mut self, dest: usize, val: u32, len: usize) -> Result<(), MemoryError> {
        let mem_len = self.size_in_bytes();

        // Checked up front rather than relying on `write`: the chunked loop below
        // would otherwise leave the earlier chunks applied before trapping.
        let end = dest.checked_add(len).ok_or(MemoryError::OutOfBoundsAccess(
            MemoryAccessKind::Write,
            dest,
            mem_len,
        ))?;

        if end > mem_len {
            return Err(MemoryError::OutOfBoundsAccess(
                MemoryAccessKind::Write,
                dest,
                mem_len,
            ));
        }

        // Bounded so a multi-gigabyte `memory.fill` doesn't allocate a buffer of
        // the same size; this one lives on the stack.
        const CHUNK_LEN: usize = 4096;
        let chunk = [val as u8; CHUNK_LEN];

        let mut written = 0;

        while written < len {
            let n = (len - written).min(CHUNK_LEN);

            // `dest + written <= dest + len <= mem_len` by the check above, so this
            // cannot overflow and each chunk is in bounds.
            self.write(dest + written, &chunk[..n])?;

            written += n;
        }

        Ok(())
    }

    /// Reads `data.len()` bytes starting from the `offset`.
    ///
    /// # Errors
    ///
    /// [`MemoryError::OutOfBoundsAccess`] with [`MemoryAccessKind::Read`] if the
    /// access is not wholly inside the memory. All-or-nothing: on error `data` is
    /// left untouched. A zero-length read at `offset == size_in_bytes()` succeeds;
    /// any `offset` beyond that traps.
    fn read(&self, offset: usize, data: &mut [u8]) -> Result<(), MemoryError>;

    /// Writes `data.len()` bytes from `data` to the memory at `offset`.
    ///
    /// # Errors
    ///
    /// [`MemoryError::OutOfBoundsAccess`] with [`MemoryAccessKind::Write`] if the
    /// access is not wholly inside the memory. All-or-nothing: on error no byte is
    /// modified. Same zero-length boundary rule as [`Self::read`].
    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), MemoryError>;

    // The sized accessors below default to marshalling through a stack buffer and
    // deferring to `read`/`write`. A backend with direct access to its buffer is
    // expected to override them and decode in place — that is the interpreter's
    // hottest memory path, and the round-trip is pure overhead there.
    //
    // An override must reproduce two things, neither of which the type system
    // enforces:
    //
    // * the same `OutOfBoundsAccess` shape as the default, carrying the requested
    //   `offset` and the current length;
    // * an `offset.checked_add(width)` bound. Written as `offset + width` it wraps
    //   for an offset near `usize::MAX`, the comparison then passes, and the slice
    //   range comes out reversed — a panic instead of a trap. That offset is
    //   reachable: host functions receive a `MemoryView` and a negative guest `i32`
    //   sign-extends into exactly that range.

    /// Reads a `u8` at `offset`. Errors if the access is out of bounds.
    fn read_u8(&self, offset: usize) -> Result<u8, MemoryError> {
        let mut buf: [u8; 1] = [0; 1];
        self.read(offset, &mut buf)?;
        Ok(u8::from_le_bytes(buf))
    }

    /// Reads an `i8` at `offset`: the same byte as [`Self::read_u8`],
    /// reinterpreted as signed (the `load8_s` variants sign-extend from this).
    /// Errors if the access is out of bounds.
    fn read_i8(&self, offset: usize) -> Result<i8, MemoryError> {
        Ok(self.read_u8(offset)? as i8)
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

    /// Reads a little-endian `i16` at `offset`: the same bytes as
    /// [`Self::read_u16`], reinterpreted as signed (the `load16_s` variants
    /// sign-extend from this). Errors if the access is out of bounds.
    fn read_i16(&self, offset: usize) -> Result<i16, MemoryError> {
        Ok(self.read_u16(offset)? as i16)
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

    /// Reads a little-endian `i32` at `offset`: the same bytes as
    /// [`Self::read_u32`], reinterpreted as signed (backs `i32.load`, and
    /// `i64.load32_s` sign-extends from this). Errors if the access is out of
    /// bounds.
    fn read_i32(&self, offset: usize) -> Result<i32, MemoryError> {
        Ok(self.read_u32(offset)? as i32)
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

    /// Reads a little-endian `i64` at `offset`: the same bytes as
    /// [`Self::read_u64`], reinterpreted as signed (backs `i64.load`). Errors if
    /// the access is out of bounds.
    fn read_i64(&self, offset: usize) -> Result<i64, MemoryError> {
        Ok(self.read_u64(offset)? as i64)
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

/// A module's linear memory, supplied by the embedder.
///
/// The interpreter is generic over this trait so the backing store (a plain
/// `Vec<u8>`, an mmap, a guarded region, …) is the embedder's choice.
///
/// Extends [`MemoryView`] with the operations that change the memory's size.
/// Host functions receive only the [`MemoryView`] half, so growing stays with the
/// interpreter, which is the only party that knows the module's declared maximum
/// and the instance's configured cap.
pub trait Memory: MemoryView {
    /// Creates a memory pre-allocated to `size` in WASM pages.
    /// Per the WebAssembly spec, this should be completely zeroed.
    fn allocate_initial_memory(size_in_pages: u64) -> Self;

    /// Grows the memory by `delta_in_pages`, returning the size in pages *before*
    /// the growth. New pages are zeroed, per the spec.
    ///
    /// `max_size_in_pages` is the ceiling the caller allows: the module's declared
    /// maximum, capped by the instance
    /// [`Config`](crate::instance::config::Config).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::GrowFailed`] if the request cannot be satisfied
    /// (past `max_size_in_pages`, or the page count overflows). This is **not** a
    /// trap: `memory.grow` reports failure by pushing `-1`, so a caller
    /// implementing that instruction must map the error to `-1` and continue,
    /// rather than propagating it.
    fn grow(&mut self, delta_in_pages: u64, max_size_in_pages: u64) -> Result<u64, MemoryError>;
}
