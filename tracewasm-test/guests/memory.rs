//! Linear-memory access: loads, stores, bulk operations, and the boundary
//! conditions around them.
//!
//! This guest exists to exercise the paths in `memory/linear.rs` and the
//! effective-address computation in the interpreter — narrow loads and their
//! sign/zero extension, unaligned access (legal in wasm, and the alignment hint
//! is advisory only), overlapping `memory.copy`, and `memory.fill`.
//!
//! Everything stays in bounds. The out-of-bounds cases are traps, so they live in
//! `tests/traps.rs` against the deliberate-trap exports at the bottom of this
//! file.
//!
//! Export names are prefixed `mem_`.

#![allow(dead_code)]

use std::hint::black_box;

/// Narrow loads and stores with sign vs zero extension. `i32.load8_s` of `0xFF`
/// is `-1`, `load8_u` is `255` — the classic place to get extension backwards.
#[unsafe(no_mangle)]
pub extern "C" fn mem_narrow_load_store(_: i32) -> i64 {
    let mut buf = [0u8; 64];

    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(37) | 0x80; // high bit set, so extension matters
    }

    let mut acc = 0i64;

    for i in 0..16usize {
        let i = black_box(i);
        acc = acc.wrapping_mul(31).wrapping_add(buf[i] as i64);
        acc = acc.wrapping_mul(31).wrapping_add(buf[i] as i8 as i64);

        let u16v = u16::from_le_bytes([buf[i], buf[i + 1]]);
        acc = acc.wrapping_mul(31).wrapping_add(u16v as i64);
        acc = acc.wrapping_mul(31).wrapping_add(u16v as i16 as i64);

        let u32v = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
        acc = acc.wrapping_mul(31).wrapping_add(u32v as i64);
        acc = acc.wrapping_mul(31).wrapping_add(u32v as i32 as i64);
    }

    // narrow stores keep only the low bytes of the value
    let mut out = [0u8; 8];
    out[0] = black_box(0x1234u32) as u8;
    out[1..3].copy_from_slice(&(black_box(0x1234_5678u32) as u16).to_le_bytes());
    out[4..8].copy_from_slice(&(black_box(0x9abc_def0_1234_5678u64) as u32).to_le_bytes());

    acc.wrapping_mul(31)
        .wrapping_add(u64::from_le_bytes(out) as i64)
}

/// Little-endian byte order, checked by writing a value and reading its bytes
/// back individually.
#[unsafe(no_mangle)]
pub extern "C" fn mem_endianness(_: i32) -> i64 {
    let mut acc = 0i64;

    let v = black_box(0x0123_4567_89ab_cdefu64);
    let bytes = v.to_le_bytes();

    for b in bytes {
        acc = acc.wrapping_mul(257).wrapping_add(b as i64);
    }

    // and the other direction
    let be = v.to_be_bytes();
    for b in be {
        acc = acc.wrapping_mul(257).wrapping_add(b as i64);
    }

    acc = acc
        .wrapping_mul(31)
        .wrapping_add(u64::from_le_bytes(bytes) as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(u64::from_be_bytes(be) as i64);
    acc = acc.wrapping_mul(31).wrapping_add(v.swap_bytes() as i64);
    acc = acc.wrapping_mul(31).wrapping_add(v.rotate_left(8) as i64);

    // floats through their bit patterns, preserving the exact representation
    let f = black_box(-1.5f64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(i64::from_le_bytes(f.to_le_bytes()));

    acc
}

/// Unaligned access. Wasm permits it, and the alignment hint in a memarg is
/// advisory — an interpreter that honours it as a requirement breaks here.
#[unsafe(no_mangle)]
pub extern "C" fn mem_unaligned(_: i32) -> i64 {
    let mut buf = [0u8; 128];

    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(11);
    }

    let mut acc = 0i64;

    // every odd offset is deliberately misaligned for the width being read
    for off in [1usize, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23] {
        let off = black_box(off);

        let u16v = u16::from_le_bytes(buf[off..off + 2].try_into().unwrap());
        let u32v = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let u64v = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());

        acc = acc.wrapping_mul(31).wrapping_add(u16v as i64);
        acc = acc.wrapping_mul(31).wrapping_add(u32v as i64);
        acc = acc.wrapping_mul(31).wrapping_add(u64v as i64);

        // unaligned float loads must preserve the bit pattern exactly
        let f = f64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        acc = acc.wrapping_mul(31).wrapping_add(f.to_bits() as i64);
    }

    acc
}

/// `memory.copy` with overlapping ranges in both directions — `memmove`
/// semantics, not `memcpy`. A naive forward loop corrupts one of these.
#[unsafe(no_mangle)]
pub extern "C" fn mem_overlapping_copy(_: i32) -> i64 {
    let mut buf = [0u8; 256];

    for (i, b) in buf.iter_mut().enumerate() {
        *b = i as u8;
    }

    // forward overlap: dest > src
    buf.copy_within(0..128, 8);

    let a = buf
        .iter()
        .fold(0i64, |acc, b| acc.wrapping_mul(31).wrapping_add(*b as i64));

    // backward overlap: dest < src
    buf.copy_within(64..200, 16);

    let b = buf
        .iter()
        .fold(0i64, |acc, x| acc.wrapping_mul(31).wrapping_add(*x as i64));

    // exact self-copy, which must be a no-op
    buf.copy_within(0..256, 0);

    let c = buf
        .iter()
        .fold(0i64, |acc, x| acc.wrapping_mul(31).wrapping_add(*x as i64));

    // zero-length copy at the very end of the buffer is legal
    buf.copy_within(256..256, 0);

    a.wrapping_mul(7)
        .wrapping_add(b)
        .wrapping_mul(7)
        .wrapping_add(c)
}

/// `memory.fill`, including the zero-length case and a fill that ends exactly at
/// the buffer's end.
#[unsafe(no_mangle)]
pub extern "C" fn mem_fill(_: i32) -> i64 {
    let mut buf = [0u8; 256];

    buf[0..64].fill(0xab);
    buf[64..64].fill(0xcd); // zero length, must not write
    buf[192..256].fill(0xef); // ends exactly at the boundary
    buf[100..120].fill(0);

    let mut acc = 0i64;

    for b in buf {
        acc = acc.wrapping_mul(17).wrapping_add(b as i64);
    }

    // a large heap-backed fill, which becomes a real memory.fill rather than
    // being unrolled
    let mut v = vec![0u8; 8192];
    v.fill(0x5a);
    v[4096..].fill(0xa5);

    acc.wrapping_mul(31)
        .wrapping_add(v.iter().map(|b| *b as i64).sum::<i64>())
}

/// A large buffer walked with a stride, so the access pattern is not sequential
/// and the effective-address computation is exercised over a wide range.
#[unsafe(no_mangle)]
pub extern "C" fn mem_strided_walk(n: i32) -> i64 {
    let len = 64 * 1024usize;
    let mut buf = vec![0u32; len / 4];
    let n = n.clamp(1, 200_000) as usize;
    let mask = buf.len() - 1;

    for i in 0..n {
        let idx = (i * 7) & mask;
        buf[idx] = buf[idx].wrapping_add(i as u32) ^ (buf[(idx * 5) & mask] >> 3);
    }

    buf.iter().fold(0i64, |a, x| a.wrapping_add(*x as i64))
}

/// Growth: `memory.grow` through the allocator, pushing the guest well past its
/// initial page count.
#[unsafe(no_mangle)]
pub extern "C" fn mem_growth(n: i32) -> i64 {
    let n = n.clamp(1, 64) as usize;
    let mut chunks: Vec<Vec<u8>> = Vec::new();

    // each chunk is ~1 page, so this forces repeated memory.grow
    for i in 0..n {
        let mut c = vec![0u8; 65_536];
        c[0] = i as u8;
        c[65_535] = (i * 3) as u8;
        chunks.push(c);
    }

    let mut acc = 0i64;

    for (i, c) in chunks.iter().enumerate() {
        acc = acc
            .wrapping_mul(31)
            .wrapping_add(c[0] as i64)
            .wrapping_add(c[65_535] as i64)
            .wrapping_add(i as i64);
    }

    // drop half and reallocate, so the allocator reuses rather than growing
    chunks.truncate(n / 2);

    for i in 0..n / 2 {
        chunks.push(vec![i as u8; 32_768]);
    }

    acc.wrapping_mul(31).wrapping_add(chunks.len() as i64)
}

/// Slice operations that lower to bulk memory: `copy_from_slice`,
/// `clone_from_slice`, `rotate`, `swap`, and `split_at_mut`.
#[unsafe(no_mangle)]
pub extern "C" fn mem_slice_ops(_: i32) -> i64 {
    let mut a = [0u8; 128];
    let mut b = [0u8; 128];

    for i in 0..128 {
        a[i] = i as u8;
        b[i] = (255 - i) as u8;
    }

    a[0..64].copy_from_slice(&b[64..128]);
    b[32..96].clone_from_slice(&a[0..64]);

    a.rotate_left(13);
    b.rotate_right(29);

    a.swap(0, 127);
    b.swap(5, 100);

    {
        let (left, right) = a.split_at_mut(64);
        left.copy_from_slice(right);
    }

    b.reverse();

    let fold_a = a
        .iter()
        .fold(0i64, |acc, x| acc.wrapping_mul(31).wrapping_add(*x as i64));
    let fold_b = b
        .iter()
        .fold(0i64, |acc, x| acc.wrapping_mul(31).wrapping_add(*x as i64));

    // comparison and search, which lower to loops over memory
    let eq = (a == b) as i64;
    let pos = a.iter().position(|x| *x == 42).unwrap_or(999) as i64;
    let max = a.iter().copied().max().unwrap_or(0) as i64;

    fold_a
        .wrapping_mul(7)
        .wrapping_add(fold_b)
        .wrapping_mul(7)
        .wrapping_add(eq)
        .wrapping_mul(7)
        .wrapping_add(pos)
        .wrapping_mul(7)
        .wrapping_add(max)
}

/// A memory-heavy mixed workload for the throughput table.
#[unsafe(no_mangle)]
pub extern "C" fn mem_mixed_workload(n: i32) -> i64 {
    let mut v = vec![0u32; 4_096];
    let n = n.clamp(1, 400_000) as usize;
    let mask = v.len() - 1;

    for i in 0..n {
        let k = i & mask;
        v[k] = v[k].wrapping_add(i as u32) ^ (v[(k * 7) & mask] >> 3);
    }

    v.iter().fold(0i64, |a, x| a.wrapping_add(*x as i64))
}

// ---------------------------------------------------------------------------
// Deliberate traps, called by tests/traps.rs
// ---------------------------------------------------------------------------

/// Reads past the end of a heap buffer. The index comes through `black_box` so
/// the bounds check cannot be folded away — this panics, which lowers to
/// `unreachable`.
#[unsafe(no_mangle)]
pub extern "C" fn mem_trap_oob_read(_: i32) -> i64 {
    let v = vec![1u8, 2, 3, 4];
    let idx = black_box(1_000_000usize);

    v[idx] as i64
}

/// Integer division by zero, with the divisor hidden from the optimiser.
#[unsafe(no_mangle)]
pub extern "C" fn mem_trap_div_zero(_: i32) -> i64 {
    let a = black_box(1234i64);
    let b = black_box(0i64);

    a / b
}

/// `i64::MIN / -1` is not representable and must trap on overflow rather than
/// wrapping.
#[unsafe(no_mangle)]
pub extern "C" fn mem_trap_div_overflow(_: i32) -> i64 {
    let a = black_box(i64::MIN);
    let b = black_box(-1i64);

    a / b
}

/// An explicit `unreachable`, reached via a panic.
#[unsafe(no_mangle)]
pub extern "C" fn mem_trap_explicit_panic(_: i32) -> i64 {
    if black_box(true) {
        panic!("deliberate panic from the guest");
    }

    0
}
