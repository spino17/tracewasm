//! A simple `Vec<u8>`-backed [`Memory`] implementation.

use crate::{
    error::{MemoryAccessKind, MemoryError},
    memory::{Memory, MemoryView},
    module::WASM_MEMORY_PAGE_SIZE,
};

/// A contiguous, heap-allocated linear memory backed by a `Vec<u8>`.
///
/// This is the default [`Memory`] backing store: a flat byte buffer with
/// bounds-checked accesses, grown in place by reallocating the `Vec` (see
/// [`Memory::grow`]).
pub struct LinearMemory {
    inner: Vec<u8>,
}

impl LinearMemory {
    /// Creates a memory of `size_in_pages` WASM pages, zero-initialized (as
    /// required for a fresh WebAssembly memory).
    ///
    /// The caller is responsible for capping `size_in_pages` to something
    /// allocatable — [`Module::instantiate`](crate::module::Module::instantiate)
    /// clamps it against the instance
    /// [`Config`](crate::instance::config::Config) — since the page count is
    /// converted to a byte length with no failure path.
    pub fn new(size_in_pages: u64) -> Self {
        Self::with_byte_len((size_in_pages * WASM_MEMORY_PAGE_SIZE) as usize)
    }

    /// Creates a memory of exactly `len` bytes, zero-initialized.
    ///
    /// Byte-granular, unlike the page-granular public API: a length that is not a
    /// whole number of pages rounds down in [`Memory::size_in_pages`]. Crate-internal
    /// so the bounds-checking tests can work at byte granularity.
    pub(crate) fn with_byte_len(len: usize) -> Self {
        LinearMemory {
            inner: vec![0; len], // Per the WebAssembly spec, its completely zeroed
        }
    }
}

impl Memory for LinearMemory {
    fn allocate_initial_memory(size_in_pages: u64) -> Self {
        Self::new(size_in_pages)
    }

    fn grow(&mut self, delta_in_pages: u64, max_size_in_pages: u64) -> Result<u64, MemoryError> {
        let old_size = self.size_in_pages();

        // A `u64` overflow here is only reachable for an absurd delta, and is a
        // failed grow just like exceeding the cap — report it the same way rather
        // than wrapping to a small (and wrongly acceptable) new size.
        let new_size = delta_in_pages
            .checked_add(old_size)
            .ok_or(MemoryError::GrowFailed(
                max_size_in_pages,
                delta_in_pages,
                old_size,
            ))?;

        if new_size > max_size_in_pages {
            return Err(MemoryError::GrowFailed(
                max_size_in_pages,
                delta_in_pages,
                old_size,
            ));
        }

        self.inner
            .resize((new_size * WASM_MEMORY_PAGE_SIZE) as usize, 0);

        Ok(old_size)
    }
}

impl MemoryView for LinearMemory {
    fn size_in_bytes(&self) -> usize {
        self.inner.len()
    }

    fn copy_within(&mut self, dest: usize, src: usize, len: usize) -> Result<(), MemoryError> {
        let mem_len = self.inner.len();

        // Both ranges are validated before anything moves, so a trap leaves the
        // memory untouched. `checked_add` keeps a huge offset from wrapping past
        // the comparison (see `read`).
        let oob = |kind, offset| MemoryError::OutOfBoundsAccess(kind, offset, mem_len);

        let src_end = src
            .checked_add(len)
            .ok_or_else(|| oob(MemoryAccessKind::Read, src))?;

        if src_end > mem_len {
            return Err(oob(MemoryAccessKind::Read, src));
        }

        let dest_end = dest
            .checked_add(len)
            .ok_or_else(|| oob(MemoryAccessKind::Write, dest))?;

        if dest_end > mem_len {
            return Err(oob(MemoryAccessKind::Write, dest));
        }

        // `slice::copy_within` is a memmove, so overlapping ranges are handled
        // correctly, as the trait requires.
        self.inner.copy_within(src..src_end, dest);

        Ok(())
    }

    fn read(&self, offset: usize, data: &mut [u8]) -> Result<(), MemoryError> {
        let len = data.len();
        let mem_len = self.inner.len();

        // `offset + len` could overflow `usize` for a maliciously large `offset`;
        // computing the end with `checked_add` traps instead of wrapping, so the
        // bounds comparison below can't be fooled into passing on wraparound.
        let end = offset
            .checked_add(len)
            .ok_or(MemoryError::OutOfBoundsAccess(
                MemoryAccessKind::Read,
                offset,
                mem_len,
            ))?;

        // A valid access needs `offset + len <= mem_len`; anything past the end traps.
        if mem_len < end {
            return Err(MemoryError::OutOfBoundsAccess(
                MemoryAccessKind::Read,
                offset,
                mem_len,
            ));
        }

        // `offset..end` is now guaranteed in-bounds (no overflow, `end <= mem_len`)
        // and exactly `len` bytes long, so neither the index nor the equal-length
        // `copy_from_slice` can panic.
        data.copy_from_slice(&self.inner[offset..end]);

        Ok(())
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), MemoryError> {
        let len = data.len();
        let mem_len = self.inner.len();

        // See `read`: `checked_add` prevents a `usize` overflow from wrapping past
        // the bounds check.
        let end = offset
            .checked_add(len)
            .ok_or(MemoryError::OutOfBoundsAccess(
                MemoryAccessKind::Write,
                offset,
                mem_len,
            ))?;

        if mem_len < end {
            return Err(MemoryError::OutOfBoundsAccess(
                MemoryAccessKind::Write,
                offset,
                mem_len,
            ));
        }

        // `offset..end` is verified in-bounds and `len`-long, so this cannot panic.
        self.inner[offset..end].copy_from_slice(data);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MemoryError;
    use crate::memory::Memory;

    /// True iff the result is an out-of-bounds trap reporting `expected_offset`
    /// and `expected_mem_len`.
    fn is_oob<T: std::fmt::Debug>(
        res: &Result<T, MemoryError>,
        expected_offset: usize,
        expected_mem_len: usize,
    ) -> bool {
        matches!(
            res,
            Err(MemoryError::OutOfBoundsAccess(_, off, len))
                if *off == expected_offset && *len == expected_mem_len
        )
    }

    // ------------------------------------------------------------------
    // construction
    // ------------------------------------------------------------------

    #[test]
    fn new_is_zeroed_and_correct_size() {
        let m = LinearMemory::with_byte_len(8);
        assert_eq!(m.inner, vec![0u8; 8]);
    }

    #[test]
    fn allocate_initial_memory_is_page_granular() {
        // The trait API is in pages, so 4 means 4 pages of zeroed bytes.
        let m = <LinearMemory as Memory>::allocate_initial_memory(4);
        assert_eq!(m.inner.len(), 4 * WASM_MEMORY_PAGE_SIZE as usize);
        assert!(m.inner.iter().all(|&b| b == 0));
        assert_eq!(m.size_in_pages(), 4);
    }

    #[test]
    fn size_in_pages_rounds_down_for_partial_pages() {
        assert_eq!(LinearMemory::with_byte_len(0).size_in_pages(), 0);
        // a partial page reports 0 whole pages
        assert_eq!(LinearMemory::with_byte_len(8).size_in_pages(), 0);
        assert_eq!(
            LinearMemory::with_byte_len(WASM_MEMORY_PAGE_SIZE as usize + 1).size_in_pages(),
            1
        );
    }

    // ------------------------------------------------------------------
    // fill
    // ------------------------------------------------------------------

    #[test]
    fn fill_writes_only_its_range_with_the_low_byte() {
        let mut m = LinearMemory::with_byte_len(6);
        // 0x1FF truncates to 0xFF: `memory.fill` uses only the low byte.
        m.fill(1, 0x1FF, 3).unwrap();
        assert_eq!(m.inner, vec![0, 0xFF, 0xFF, 0xFF, 0, 0]);
    }

    #[test]
    fn fill_spanning_multiple_chunks() {
        // Longer than the default impl's 4096-byte chunk, so it exercises the loop.
        let len = 4096 * 2 + 17;
        let mut m = LinearMemory::with_byte_len(len + 8);

        m.fill(8, 0xAB, len).unwrap();

        assert!(m.inner[..8].iter().all(|&b| b == 0), "prefix untouched");
        assert!(
            m.inner[8..].iter().all(|&b| b == 0xAB),
            "whole range filled"
        );
    }

    #[test]
    fn fill_to_the_exact_end_is_allowed() {
        let mut m = LinearMemory::with_byte_len(4);
        m.fill(0, 1, 4).unwrap();
        assert_eq!(m.inner, vec![1, 1, 1, 1]);
    }

    #[test]
    fn fill_zero_length_is_a_noop() {
        let mut m = LinearMemory::with_byte_len(4);
        m.fill(4, 9, 0).unwrap(); // dest == mem_len with len 0 does not trap
        assert_eq!(m.inner, vec![0, 0, 0, 0]);
    }

    #[test]
    fn fill_past_end_traps_without_writing_anything() {
        let mut m = LinearMemory::with_byte_len(4);
        let res = m.fill(2, 7, 3); // 2 + 3 = 5 > 4
        assert!(is_oob(&res, 2, 4));
        assert_eq!(m.inner, vec![0, 0, 0, 0], "must be all-or-nothing");
    }

    #[test]
    fn fill_length_overflow_traps_instead_of_wrapping() {
        let mut m = LinearMemory::with_byte_len(4);
        let res = m.fill(8, 7, usize::MAX); // 8 + usize::MAX would wrap
        assert!(is_oob(&res, 8, 4));
        assert_eq!(m.inner, vec![0, 0, 0, 0]);
    }

    // ------------------------------------------------------------------
    // copy_within
    // ------------------------------------------------------------------

    #[test]
    fn copy_within_copies_disjoint_range() {
        let mut m = LinearMemory::with_byte_len(6);
        m.write(0, &[1, 2, 3]).unwrap();

        m.copy_within(3, 0, 3).unwrap(); // dest = 3, src = 0

        assert_eq!(m.inner, vec![1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn copy_within_argument_order_is_dest_then_src() {
        // Regression guard: the trait declares (dest, src, len). Swapping them
        // would copy the wrong direction and yield [1,2,3,1,2,3] here.
        let mut m = LinearMemory::with_byte_len(6);
        m.write(3, &[7, 8, 9]).unwrap(); // [0,0,0,7,8,9]

        m.copy_within(0, 3, 3).unwrap(); // copy src=3..6 into dest=0

        assert_eq!(m.inner, vec![7, 8, 9, 7, 8, 9]);
    }

    #[test]
    fn copy_within_handles_overlap_forward() {
        // dest just above src: a naive forward byte loop would smear the first
        // byte across the range instead of memmove semantics.
        let mut m = LinearMemory::with_byte_len(5);
        m.write(0, &[1, 2, 3, 4, 5]).unwrap();

        m.copy_within(1, 0, 4).unwrap(); // [1,1,2,3,4]

        assert_eq!(m.inner, vec![1, 1, 2, 3, 4]);
    }

    #[test]
    fn copy_within_handles_overlap_backward() {
        let mut m = LinearMemory::with_byte_len(5);
        m.write(0, &[1, 2, 3, 4, 5]).unwrap();

        m.copy_within(0, 1, 4).unwrap(); // [2,3,4,5,5]

        assert_eq!(m.inner, vec![2, 3, 4, 5, 5]);
    }

    #[test]
    fn copy_within_zero_length_is_a_noop() {
        let mut m = LinearMemory::with_byte_len(4);
        m.write(0, &[1, 2, 3, 4]).unwrap();

        m.copy_within(4, 4, 0).unwrap(); // both ends at mem_len, len 0

        assert_eq!(m.inner, vec![1, 2, 3, 4]);
    }

    #[test]
    fn copy_within_out_of_bounds_src_traps_and_changes_nothing() {
        let mut m = LinearMemory::with_byte_len(4);
        m.write(0, &[1, 2, 3, 4]).unwrap();

        let res = m.copy_within(0, 2, 3); // src 2 + 3 = 5 > 4
        assert!(is_oob(&res, 2, 4));
        assert_eq!(m.inner, vec![1, 2, 3, 4], "must be all-or-nothing");
    }

    #[test]
    fn copy_within_out_of_bounds_dest_traps_and_changes_nothing() {
        let mut m = LinearMemory::with_byte_len(4);
        m.write(0, &[1, 2, 3, 4]).unwrap();

        let res = m.copy_within(2, 0, 3); // dest 2 + 3 = 5 > 4
        assert!(is_oob(&res, 2, 4));
        assert_eq!(m.inner, vec![1, 2, 3, 4]);
    }

    #[test]
    fn copy_within_length_overflow_traps_instead_of_wrapping() {
        let mut m = LinearMemory::with_byte_len(4);
        let res = m.copy_within(0, 8, usize::MAX);
        assert!(is_oob(&res, 8, 4));
    }

    // ------------------------------------------------------------------
    // grow
    // ------------------------------------------------------------------

    #[test]
    fn grow_returns_previous_size_and_zeroes_new_pages() {
        let mut m = LinearMemory::new(1);
        m.write(0, &[7, 7]).unwrap();

        assert_eq!(m.grow(2, 10).unwrap(), 1, "returns the pre-growth size");
        assert_eq!(m.size_in_pages(), 3);

        // existing bytes survive, freshly added pages are zeroed
        let mut buf = [0u8; 2];
        m.read(0, &mut buf).unwrap();
        assert_eq!(buf, [7, 7]);
        assert_eq!(m.read_u8(WASM_MEMORY_PAGE_SIZE as usize).unwrap(), 0);
    }

    #[test]
    fn grow_past_max_fails_and_leaves_memory_untouched() {
        let mut m = LinearMemory::new(2);

        let res = m.grow(3, 4); // 2 + 3 = 5 > 4
        assert!(matches!(res, Err(MemoryError::GrowFailed(4, 3, 2))));
        assert_eq!(m.size_in_pages(), 2, "a failed grow must not resize");
    }

    #[test]
    fn grow_to_exactly_max_is_allowed() {
        let mut m = LinearMemory::new(2);
        assert_eq!(m.grow(2, 4).unwrap(), 2);
        assert_eq!(m.size_in_pages(), 4);
    }

    #[test]
    fn grow_by_zero_is_a_noop_reporting_current_size() {
        let mut m = LinearMemory::new(3);
        assert_eq!(m.grow(0, 10).unwrap(), 3);
        assert_eq!(m.size_in_pages(), 3);
    }

    #[test]
    fn grow_page_overflow_fails_instead_of_wrapping() {
        let mut m = LinearMemory::new(1);
        // 1 + u64::MAX would wrap to 0 and wrongly look acceptable.
        let res = m.grow(u64::MAX, u64::MAX);
        assert!(matches!(res, Err(MemoryError::GrowFailed(..))));
        assert_eq!(m.size_in_pages(), 1);
    }

    #[test]
    fn zero_sized_memory_is_empty() {
        let m = LinearMemory::with_byte_len(0);
        assert_eq!(m.inner.len(), 0);
    }

    // ------------------------------------------------------------------
    // raw read / write round-trips
    // ------------------------------------------------------------------

    #[test]
    fn write_then_read_roundtrip() {
        let mut m = LinearMemory::with_byte_len(8);
        m.write(2, &[0xAA, 0xBB, 0xCC]).unwrap();

        let mut buf = [0u8; 3];
        m.read(2, &mut buf).unwrap();
        assert_eq!(buf, [0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn write_only_touches_its_range() {
        let mut m = LinearMemory::with_byte_len(5);
        m.write(1, &[9, 9]).unwrap();
        // bytes 0, 3, 4 stay zero; only 1 and 2 change.
        assert_eq!(m.inner, vec![0, 9, 9, 0, 0]);
    }

    #[test]
    fn access_exactly_to_the_end_is_allowed() {
        let mut m = LinearMemory::with_byte_len(4);
        m.write(0, &[1, 2, 3, 4]).unwrap(); // offset + len == mem_len
        let mut buf = [0u8; 4];
        m.read(0, &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    // ------------------------------------------------------------------
    // out-of-bounds / overflow traps
    // ------------------------------------------------------------------

    #[test]
    fn read_past_end_traps() {
        let m = LinearMemory::with_byte_len(4);
        let mut buf = [0u8; 2];
        let res = m.read(3, &mut buf); // 3 + 2 = 5 > 4
        assert!(is_oob(&res, 3, 4));
    }

    #[test]
    fn write_past_end_traps() {
        let mut m = LinearMemory::with_byte_len(4);
        let res = m.write(3, &[1, 2]); // 3 + 2 = 5 > 4
        assert!(is_oob(&res, 3, 4));
    }

    #[test]
    fn write_reports_write_not_read_in_error() {
        let mut m = LinearMemory::with_byte_len(1);
        let err = m.write(0, &[1, 2]).unwrap_err();
        // regression guard for the copy-paste bug where write said "read".
        assert_eq!(
            err.to_string(),
            "out of bounds access: Write at 0 on memory with length 1"
        );
    }

    #[test]
    fn offset_addition_overflow_traps_instead_of_panicking() {
        let m = LinearMemory::with_byte_len(4);
        let mut buf = [0u8; 1];
        // usize::MAX + 1 would wrap; checked_add must turn this into a trap.
        let res = m.read(usize::MAX, &mut buf);
        assert!(is_oob(&res, usize::MAX, 4));
    }

    // ------------------------------------------------------------------
    // zero-length accesses (boundary semantics)
    // ------------------------------------------------------------------

    #[test]
    fn zero_length_access_at_end_is_ok() {
        let mut m = LinearMemory::with_byte_len(4);
        assert!(m.read(4, &mut []).is_ok()); // offset == size, len 0
        assert!(m.write(4, &[]).is_ok());
    }

    #[test]
    fn zero_length_access_past_end_traps() {
        let m = LinearMemory::with_byte_len(4);
        assert!(is_oob(&m.read(5, &mut []), 5, 4));
    }

    #[test]
    fn zero_sized_memory_allows_only_empty_access_at_zero() {
        let mut m = LinearMemory::with_byte_len(0);
        assert!(m.read(0, &mut []).is_ok());
        assert!(m.write(0, &[]).is_ok());
        assert!(is_oob(&m.read(0, &mut [0u8; 1]), 0, 0));
    }

    // ------------------------------------------------------------------
    // width helpers: little-endian layout + round-trips
    // ------------------------------------------------------------------

    #[test]
    fn write_u32_is_little_endian() {
        let mut m = LinearMemory::with_byte_len(4);
        m.write_u32(0, 0x0102_0304).unwrap();
        assert_eq!(m.inner, vec![0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn integer_width_roundtrips() {
        let mut m = LinearMemory::with_byte_len(8);

        m.write_u8(0, 0xAB).unwrap();
        assert_eq!(m.read_u8(0).unwrap(), 0xAB);

        m.write_u16(0, 0xABCD).unwrap();
        assert_eq!(m.read_u16(0).unwrap(), 0xABCD);

        m.write_u32(0, 0xDEAD_BEEF).unwrap();
        assert_eq!(m.read_u32(0).unwrap(), 0xDEAD_BEEF);

        m.write_u64(0, 0x0123_4567_89AB_CDEF).unwrap();
        assert_eq!(m.read_u64(0).unwrap(), 0x0123_4567_89AB_CDEF);
    }

    #[test]
    fn width_helpers_trap_near_the_end() {
        let mut m = LinearMemory::with_byte_len(4);
        assert!(matches!(
            m.read_u64(0),
            Err(MemoryError::OutOfBoundsAccess(..))
        )); // needs 8, only 4
        assert!(matches!(
            m.write_u32(1, 1),
            Err(MemoryError::OutOfBoundsAccess(..))
        )); // 1 + 4 = 5 > 4
    }

    // ------------------------------------------------------------------
    // float helpers: round-trip + exact bit preservation
    // ------------------------------------------------------------------

    #[test]
    fn float_roundtrips() {
        let mut m = LinearMemory::with_byte_len(8);

        m.write_f32(0, 3.5f32).unwrap();
        assert_eq!(m.read_f32(0).unwrap(), 3.5f32);

        m.write_f64(0, -2.25f64).unwrap();
        assert_eq!(m.read_f64(0).unwrap(), -2.25f64);
    }

    #[test]
    fn float_write_preserves_exact_nan_bits() {
        let mut m = LinearMemory::with_byte_len(8);

        // a non-canonical NaN payload must survive the round-trip bit-for-bit.
        let nan = f32::from_bits(0x7FC0_0001);
        m.write_f32(0, nan).unwrap();
        assert_eq!(m.read_f32(0).unwrap().to_bits(), 0x7FC0_0001);

        let nan64 = f64::from_bits(0x7FF8_0000_0000_0001);
        m.write_f64(0, nan64).unwrap();
        assert_eq!(m.read_f64(0).unwrap().to_bits(), 0x7FF8_0000_0000_0001);
    }

    #[test]
    fn failed_read_leaves_destination_untouched() {
        let m = LinearMemory::with_byte_len(2);
        let mut buf = [7u8; 4];
        assert!(m.read(0, &mut buf).is_err()); // needs 4, only 2
        assert_eq!(buf, [7, 7, 7, 7]); // copy happens only after bounds checks pass
    }
}
