//! A simple `Vec<u8>`-backed [`Memory`] implementation.

use crate::{error::TraceWasmError, memory::Memory};

/// A contiguous, heap-allocated linear memory backed by a `Vec<u8>`.
///
/// This is the default [`Memory`] backing store: a flat byte buffer with
/// bounds-checked accesses. It is fixed-size — the buffer is allocated once at
/// construction and never grown (memory-growth support is a future addition).
pub struct LinearMemory {
    inner: Vec<u8>,
}

impl LinearMemory {
    /// Creates a memory of exactly `size` bytes, zero-initialized (as required
    /// for a fresh WebAssembly memory).
    pub fn new(size: usize) -> Self {
        LinearMemory {
            inner: vec![0; size],
        }
    }
}

impl Memory for LinearMemory {
    fn allocate_initial_memory(size: usize) -> Self {
        Self::new(size)
    }

    fn read(&self, offset: usize, data: &mut [u8]) -> Result<(), TraceWasmError> {
        let len = data.len();
        let mem_len = self.inner.len();

        // `offset + len` could overflow `usize` for a maliciously large `offset`;
        // computing the end with `checked_add` traps instead of wrapping, so the
        // bounds comparison below can't be fooled into passing on wraparound.
        let end = offset.checked_add(len).ok_or_else(|| {
            TraceWasmError::OutOfBoundMemoryAccess("read".to_string(), offset, mem_len)
        })?;

        // A valid access needs `offset + len <= mem_len`; anything past the end traps.
        if mem_len < end {
            return Err(TraceWasmError::OutOfBoundMemoryAccess(
                "read".to_string(),
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

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), TraceWasmError> {
        let len = data.len();
        let mem_len = self.inner.len();

        // See `read`: `checked_add` prevents a `usize` overflow from wrapping past
        // the bounds check.
        let end = offset.checked_add(len).ok_or_else(|| {
            TraceWasmError::OutOfBoundMemoryAccess("write".to_string(), offset, mem_len)
        })?;

        if mem_len < end {
            return Err(TraceWasmError::OutOfBoundMemoryAccess(
                "write".to_string(),
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
    use crate::error::TraceWasmError;
    use crate::memory::Memory;

    /// True iff the result is an out-of-bounds trap reporting `expected_offset`
    /// and `expected_mem_len`.
    fn is_oob<T: std::fmt::Debug>(
        res: &Result<T, TraceWasmError>,
        expected_offset: usize,
        expected_mem_len: usize,
    ) -> bool {
        matches!(
            res,
            Err(TraceWasmError::OutOfBoundMemoryAccess(_, off, len))
                if *off == expected_offset && *len == expected_mem_len
        )
    }

    // ------------------------------------------------------------------
    // construction
    // ------------------------------------------------------------------

    #[test]
    fn new_is_zeroed_and_correct_size() {
        let m = LinearMemory::new(8);
        assert_eq!(m.inner, vec![0u8; 8]);
    }

    #[test]
    fn allocate_initial_memory_matches_new() {
        let m = <LinearMemory as Memory>::allocate_initial_memory(4);
        assert_eq!(m.inner, vec![0u8; 4]);
    }

    #[test]
    fn zero_sized_memory_is_empty() {
        let m = LinearMemory::new(0);
        assert_eq!(m.inner.len(), 0);
    }

    // ------------------------------------------------------------------
    // raw read / write round-trips
    // ------------------------------------------------------------------

    #[test]
    fn write_then_read_roundtrip() {
        let mut m = LinearMemory::new(8);
        m.write(2, &[0xAA, 0xBB, 0xCC]).unwrap();

        let mut buf = [0u8; 3];
        m.read(2, &mut buf).unwrap();
        assert_eq!(buf, [0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn write_only_touches_its_range() {
        let mut m = LinearMemory::new(5);
        m.write(1, &[9, 9]).unwrap();
        // bytes 0, 3, 4 stay zero; only 1 and 2 change.
        assert_eq!(m.inner, vec![0, 9, 9, 0, 0]);
    }

    #[test]
    fn access_exactly_to_the_end_is_allowed() {
        let mut m = LinearMemory::new(4);
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
        let m = LinearMemory::new(4);
        let mut buf = [0u8; 2];
        let res = m.read(3, &mut buf); // 3 + 2 = 5 > 4
        assert!(is_oob(&res, 3, 4));
    }

    #[test]
    fn write_past_end_traps() {
        let mut m = LinearMemory::new(4);
        let res = m.write(3, &[1, 2]); // 3 + 2 = 5 > 4
        assert!(is_oob(&res, 3, 4));
    }

    #[test]
    fn write_reports_write_not_read_in_error() {
        let mut m = LinearMemory::new(1);
        let err = m.write(0, &[1, 2]).unwrap_err();
        // regression guard for the copy-paste bug where write said "read".
        assert_eq!(
            err.to_string(),
            "out of bound memory access: write at offset `0` on memory with len `1`"
        );
    }

    #[test]
    fn offset_addition_overflow_traps_instead_of_panicking() {
        let m = LinearMemory::new(4);
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
        let mut m = LinearMemory::new(4);
        assert!(m.read(4, &mut []).is_ok()); // offset == size, len 0
        assert!(m.write(4, &[]).is_ok());
    }

    #[test]
    fn zero_length_access_past_end_traps() {
        let m = LinearMemory::new(4);
        assert!(is_oob(&m.read(5, &mut []), 5, 4));
    }

    #[test]
    fn zero_sized_memory_allows_only_empty_access_at_zero() {
        let mut m = LinearMemory::new(0);
        assert!(m.read(0, &mut []).is_ok());
        assert!(m.write(0, &[]).is_ok());
        assert!(is_oob(&m.read(0, &mut [0u8; 1]), 0, 0));
    }

    // ------------------------------------------------------------------
    // width helpers: little-endian layout + round-trips
    // ------------------------------------------------------------------

    #[test]
    fn write_u32_is_little_endian() {
        let mut m = LinearMemory::new(4);
        m.write_u32(0, 0x0102_0304).unwrap();
        assert_eq!(m.inner, vec![0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn integer_width_roundtrips() {
        let mut m = LinearMemory::new(8);

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
        let mut m = LinearMemory::new(4);
        assert!(matches!(
            m.read_u64(0),
            Err(TraceWasmError::OutOfBoundMemoryAccess(..))
        )); // needs 8, only 4
        assert!(matches!(
            m.write_u32(1, 1),
            Err(TraceWasmError::OutOfBoundMemoryAccess(..))
        )); // 1 + 4 = 5 > 4
    }

    // ------------------------------------------------------------------
    // float helpers: round-trip + exact bit preservation
    // ------------------------------------------------------------------

    #[test]
    fn float_roundtrips() {
        let mut m = LinearMemory::new(8);

        m.write_f32(0, 3.5f32).unwrap();
        assert_eq!(m.read_f32(0).unwrap(), 3.5f32);

        m.write_f64(0, -2.25f64).unwrap();
        assert_eq!(m.read_f64(0).unwrap(), -2.25f64);
    }

    #[test]
    fn float_write_preserves_exact_nan_bits() {
        let mut m = LinearMemory::new(8);

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
        let m = LinearMemory::new(2);
        let mut buf = [7u8; 4];
        assert!(m.read(0, &mut buf).is_err()); // needs 4, only 2
        assert_eq!(buf, [7, 7, 7, 7]); // copy happens only after bounds checks pass
    }
}
