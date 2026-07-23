//! A deliberately broad Rust module for inspecting the WASM it compiles to.
//!
//! Every function is `#[no_mangle] extern "C"` so it survives as an export and
//! keeps a stable, readable signature in the disassembly. `black_box` is used
//! liberally to stop LLVM from constant-folding whole functions away.
//!
//! The module also declares seven host imports under the "env" namespace, so
//! the function index space is offset: local functions start at global index 7.
//! Nothing resolves them — the module is never instantiated by a real host, only
//! parsed and interpreted.
//!
//! Build:
//!   rustup target add wasm32-unknown-unknown
//!   rustc --target wasm32-unknown-unknown --crate-type cdylib -O wasm_torture.rs
//!   wasm2wat wasm_torture.wasm -o wasm_torture.wat
//!
//! For a version with more locals, less inlining, and DWARF line info:
//!   rustc --target wasm32-unknown-unknown --crate-type cdylib -C debuginfo=2 wasm_torture.rs

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// 0. Host imports
//    → populates the Import section, which shifts the whole function index
//      space: local function 0 becomes global index 7. Every `call` operand
//      in the module moves accordingly.
//
//    Signature variety is deliberate — zero-param, multi-param, i64 results,
//    and void returns all exercise different frame-setup paths.
// ---------------------------------------------------------------------------

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// (i32 i32) -> ()      two params, no result
    fn host_log(ptr: *const u8, len: usize);

    /// () -> i64            zero params, wide result
    fn host_now_ms() -> i64;

    /// () -> i32            zero params — catches "popped from empty stack"
    fn host_rand() -> i32;

    /// (i32 i32) -> i32     the allocator surface you plan to intercept
    fn host_alloc(size: i32, align: i32) -> i32;

    /// (i32 i32 i32) -> ()
    fn host_dealloc(ptr: i32, size: i32, align: i32);

    /// (i32 i32 i32 i32) -> i32
    fn host_realloc(ptr: i32, old_size: i32, align: i32, new_size: i32) -> i32;

    /// (f64) -> f64         float across the host boundary
    fn host_sin(x: f64) -> f64;
}

#[unsafe(no_mangle)]
pub extern "C" fn bench_imports(x: i32) -> i64 {
    let x = black_box(x);
    let msg = "hello from wasm";

    unsafe {
        host_log(msg.as_ptr(), msg.len());

        let t = host_now_ms();
        let r = host_rand();

        // round-trip through the host allocator
        let p = host_alloc(64, 8);
        let p2 = if p != 0 {
            let grown = host_realloc(p, 64, 8, 256);
            if grown != 0 {
                host_dealloc(grown, 256, 8);
            }
            grown
        } else {
            0
        };

        let s = host_sin(x as f64);

        t.wrapping_add((r ^ x ^ p2) as i64).wrapping_add(s as i64)
    }
}

/// Calls an import from inside a loop and a branch, so the `call` operand
/// appears in more than one control-flow context.
#[unsafe(no_mangle)]
pub extern "C" fn bench_imports_in_loop(n: u32) -> i64 {
    let n = black_box(n) % 32;
    let mut acc = 0i64;

    for i in 0..n {
        unsafe {
            if i % 3 == 0 {
                acc = acc.wrapping_add(host_now_ms());
            } else if i % 3 == 1 {
                acc ^= host_rand() as i64;
            } else {
                let label = "tick";
                host_log(label.as_ptr(), label.len());
            }
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// 1. Narrow integer types, struct layout, padding
//    → i32.load8_s/8_u/16_s/16_u, i32.store8/16, i64.load/store,
//      i32.extend8_s/16_s, field offsets in `offset=` immediates
// ---------------------------------------------------------------------------

#[repr(C)]
struct Packed {
    a: u8,  // offset 0
    b: i8,  // offset 1
    c: u16, // offset 2
    d: i16, // offset 4
    // 2 bytes padding
    e: u32, // offset 8
    f: i64, // offset 16 (8-byte aligned → 4 bytes padding before it)
}

#[repr(C)]
struct Nested {
    head: u16,
    inner: Packed,
    tail: u8,
}

#[unsafe(no_mangle)]
pub extern "C" fn bench_narrow_fields(x: i32) -> i64 {
    let x = black_box(x);
    let mut n = Nested {
        head: x as u16,
        inner: Packed {
            a: x as u8,
            b: (x >> 1) as i8,
            c: (x >> 2) as u16,
            d: (x >> 3) as i16,
            e: x as u32,
            f: (x as i64) << 20,
        },
        tail: (x >> 24) as u8,
    };
    n.inner.a = n.inner.a.wrapping_add(1);
    n.inner.d = n.inner.d.wrapping_neg();

    let n = black_box(&n);
    (n.head as i64)
        + (n.inner.a as i64)   // load8_u
        + (n.inner.b as i64)   // load8_s
        + (n.inner.c as i64)   // load16_u
        + (n.inner.d as i64)   // load16_s
        + (n.inner.e as i64)   // load  (zero-extended)
        + n.inner.f            // i64.load
        + (n.tail as i64)
}

// ---------------------------------------------------------------------------
// 2. Dense integer match
//    → br_table with one target per arm plus a default
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_br_table(op: u32, a: i32, b: i32) -> i32 {
    match black_box(op) {
        0 => a.wrapping_add(b),
        1 => a.wrapping_sub(b),
        2 => a.wrapping_mul(b),
        3 => {
            if b != 0 {
                a / b
            } else {
                -1
            }
        } // div_s + zero guard
        4 => {
            if b != 0 {
                a % b
            } else {
                -2
            }
        } // rem_s
        5 => a & b,
        6 => a | b,
        7 => a ^ b,
        8 => a << (b & 31),                           // shl (masked)
        9 => a >> (b & 31),                           // shr_s
        10 => ((a as u32) >> (b as u32 & 31)) as i32, // shr_u
        11 => a.rotate_left(b as u32 & 31),
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// 3. Sparse / non-contiguous match
//    → chain of br_if instead of br_table; good contrast with the above
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_sparse_match(code: i32) -> i32 {
    match black_box(code) {
        -1000 => 1,
        7 => 2,
        4242 => 3,
        1_000_000 => 4,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// 4. Nested loops, labelled break, continue
//    → block/loop nesting, br_if to several different depths
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_nested_loops(n: i32) -> i32 {
    let n = black_box(n);
    let mut total: i32 = 0;

    'outer: for i in 0..n {
        let mut j = 0i32;
        while j < n {
            if (i.wrapping_mul(j)) % 7 == 3 {
                j += 2;
                continue; // br to the inner loop
            }
            if i + j > 100 {
                break 'outer; // br out through two frames
            }
            total = total.wrapping_add(i ^ j);
            j += 1;
        }
        if total > 10_000 {
            break; // br out of the outer loop
        }
    }

    // loop with a mid-body exit, and a do-while shape
    let mut k = 0i32;
    loop {
        k = k.wrapping_mul(31).wrapping_add(7);
        if k % 1009 == 0 {
            break;
        }
        if k > 1 << 20 {
            k = 0;
            break;
        }
    }

    total ^ k
}

// ---------------------------------------------------------------------------
// 5. Heap: Vec growth, reallocation, bulk copies
//    → __rust_alloc / __rust_realloc / __rust_dealloc, memory.grow, memory.copy
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_vec_growth(n: u32) -> u32 {
    let n = black_box(n);

    // repeated push → grow_amortized → realloc chain
    let mut v: Vec<u32> = Vec::new();
    for i in 0..n {
        v.push(i.wrapping_mul(2654435761));
    }

    // bulk append → memory.copy
    let chunk: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut bytes: Vec<u8> = Vec::with_capacity(16);
    for _ in 0..n {
        bytes.extend_from_slice(&chunk);
    }

    // clone + copy_from_slice → more memory.copy
    let mut mirror = vec![0u8; bytes.len()];
    mirror.copy_from_slice(&bytes);
    mirror.reverse();

    // drain / truncate / shrink → dealloc paths
    v.truncate((n / 2) as usize);
    v.shrink_to_fit();

    let mut acc = 0u32;
    for (i, &b) in mirror.iter().enumerate() {
        acc = acc.wrapping_add((b as u32).wrapping_mul(i as u32 | 1));
    }
    for &x in v.iter() {
        acc ^= x;
    }
    acc
}

// ---------------------------------------------------------------------------
// 6. Strings: UTF-8 encode/decode, formatting
//    → narrow loads, memory.copy, the whole core::fmt machinery, call_indirect
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_strings(n: u32) -> usize {
    let n = black_box(n);
    let mut s = String::new();

    for i in 0..n {
        s.push_str("item-");
        s.push((b'0' + (i % 10) as u8) as char); // 1-byte encode
        s.push('\u{00e9}'); // 2-byte encode
        s.push('\u{4e2d}'); // 3-byte encode
        s.push('\u{1f600}'); // 4-byte encode
        s.push(',');
    }

    // char iteration → UTF-8 decode, lots of branching on lead bytes
    let mut digits = 0usize;
    let mut wide = 0usize;
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            digits += 1;
        } else if ch as u32 > 0x7F {
            wide += 1;
        }
    }

    // formatting machinery → Display/Debug vtables, call_indirect
    let report = format!("{digits}:{wide}:{}", s.len());
    let parsed: usize = report
        .split(':')
        .filter_map(|p| p.parse::<usize>().ok())
        .sum();

    // slicing and searching
    let found = s.find("item-9").unwrap_or(0);
    let upper = s.to_uppercase();

    digits + wide + parsed + found + upper.len()
}

// ---------------------------------------------------------------------------
// 7. Trait objects and closures
//    → call_indirect through the funcref table, Box allocations, vtables
// ---------------------------------------------------------------------------

trait Transform {
    fn apply(&self, x: i32) -> i32;
    fn name(&self) -> &'static str;
}

struct AddN(i32);
struct MulN(i32);
struct Clamp {
    lo: i32,
    hi: i32,
}

impl Transform for AddN {
    fn apply(&self, x: i32) -> i32 {
        x.wrapping_add(self.0)
    }
    fn name(&self) -> &'static str {
        "add"
    }
}
impl Transform for MulN {
    fn apply(&self, x: i32) -> i32 {
        x.wrapping_mul(self.0)
    }
    fn name(&self) -> &'static str {
        "mul"
    }
}
impl Transform for Clamp {
    fn apply(&self, x: i32) -> i32 {
        x.max(self.lo).min(self.hi) // likely `select`
    }
    fn name(&self) -> &'static str {
        "clamp"
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn bench_dyn_dispatch(x: i32, rounds: u32) -> i32 {
    let rounds = black_box(rounds);
    let pipeline: Vec<Box<dyn Transform>> = vec![
        Box::new(AddN(7)),
        Box::new(MulN(3)),
        Box::new(Clamp {
            lo: -100_000,
            hi: 100_000,
        }),
    ];

    let mut v = black_box(x);
    let mut name_len = 0usize;
    for _ in 0..rounds {
        for t in &pipeline {
            v = t.apply(v); // call_indirect
            name_len += t.name().len(); // call_indirect
        }
    }
    v.wrapping_add(name_len as i32)
}

#[unsafe(no_mangle)]
pub extern "C" fn bench_closures(x: i32) -> i32 {
    let x = black_box(x);
    let ops: Vec<Box<dyn Fn(i32) -> i32>> = vec![
        Box::new(|v| v.wrapping_add(1)),
        Box::new(move |v| v.wrapping_mul(x | 1)),
        Box::new(|v| v.rotate_left(3)),
        Box::new(|v| if v < 0 { -v } else { v }),
    ];
    ops.iter().fold(x, |acc, f| f(acc))
}

// ---------------------------------------------------------------------------
// 8. Enums with payloads, Option/Result, exhaustive matching
//    → discriminant loads, nested branching, sret out-pointers
// ---------------------------------------------------------------------------

enum Shape {
    Point,
    Circle(f64),
    Rect { w: f64, h: f64 },
    Poly(Vec<(f64, f64)>),
}

fn area(s: &Shape) -> f64 {
    match s {
        Shape::Point => 0.0,
        Shape::Circle(r) => std::f64::consts::PI * r * r,
        Shape::Rect { w, h } => w * h,
        Shape::Poly(pts) => {
            // shoelace formula → indexed loop over a Vec of tuples
            let mut acc = 0.0f64;
            for i in 0..pts.len() {
                let (x0, y0) = pts[i];
                let (x1, y1) = pts[(i + 1) % pts.len()];
                acc += x0 * y1 - x1 * y0;
            }
            acc.abs() * 0.5
        }
    }
}

fn parse_shape(kind: u32, a: f64, b: f64) -> Result<Shape, &'static str> {
    match kind {
        0 => Ok(Shape::Point),
        1 if a >= 0.0 => Ok(Shape::Circle(a)),
        1 => Err("negative radius"),
        2 => Ok(Shape::Rect { w: a, h: b }),
        3 => {
            let n = (a as usize).clamp(3, 64);
            let mut pts = Vec::with_capacity(n);
            for i in 0..n {
                let t = (i as f64) * std::f64::consts::TAU / (n as f64);
                pts.push((b * t.cos(), b * t.sin()));
            }
            Ok(Shape::Poly(pts))
        }
        _ => Err("unknown kind"),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn bench_enums(kind: u32, a: f64, b: f64) -> f64 {
    match parse_shape(black_box(kind), a, b) {
        Ok(s) => area(&s),
        Err(msg) => -(msg.len() as f64),
    }
}

// ---------------------------------------------------------------------------
// 9. Floating point and cast semantics
//    → f64.sqrt/floor/min/max, trunc_sat_*, convert_*, promote/demote
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_floats(x: f64, y: f32) -> f64 {
    let x = black_box(x);
    let y = black_box(y);

    let a = x.sqrt() + x.floor() + x.abs() + x.ceil() + x.trunc();
    let b = f64::from(y) * 1.5; // f64.promote_f32
    let c = a.min(b).max(-1.0e9); // f64.min / f64.max (NaN-propagating!)
    let d = (c as f32) as f64; // demote then promote

    let i = c as i64; // i64.trunc_sat_f64_s
    let u = c as u32; // i32.trunc_sat_f64_u
    let back = (i as f64) + (u as f64) + d; // f64.convert_i64_s / _i32_u

    let bits = c.to_bits(); // i64.reinterpret_f64
    let round_trip = f64::from_bits(bits); // f64.reinterpret_i64

    back / (round_trip.abs() + 1.0)
}

// ---------------------------------------------------------------------------
// 10. Bit twiddling and mixed-width integer arithmetic
//     → clz/ctz/popcnt/rotl/rotr, i64 ops, extend/wrap, checked-overflow
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_bits(x: u32, y: u64) -> u64 {
    let x = black_box(x);
    let y = black_box(y);

    let a = x.leading_zeros() + x.trailing_zeros() + x.count_ones();
    let b = y.rotate_left(17) ^ y.rotate_right(5);
    let c = (x as u64).wrapping_mul(y) >> 32; // i64 widening
    let d = if x == 0 { 0 } else { y / (x as u64) }; // i64.div_u
    let e = y.checked_mul(x as u64).unwrap_or(u64::MAX); // overflow check
    let f = (y as u32) as u64; // wrap then extend
    let g = x.next_power_of_two() as u64;
    let h = y.swap_bytes();

    (a as u64) ^ b ^ c ^ d ^ e ^ f ^ g ^ h
}

// ---------------------------------------------------------------------------
// 11. Recursion and mutual recursion
//     → deep call chains, shadow-stack pressure
// ---------------------------------------------------------------------------

fn is_even_rec(n: u32) -> bool {
    if n == 0 { true } else { is_odd_rec(n - 1) }
}
fn is_odd_rec(n: u32) -> bool {
    if n == 0 { false } else { is_even_rec(n - 1) }
}

fn collatz_len(n: u64, depth: u32) -> u32 {
    if n <= 1 || depth > 500 {
        return depth;
    }
    if n % 2 == 0 {
        collatz_len(n / 2, depth + 1)
    } else {
        collatz_len(3u64.wrapping_mul(n).wrapping_add(1), depth + 1)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn bench_recursion(n: u32) -> u32 {
    let n = black_box(n) % 1000;
    let a = collatz_len(n as u64 + 1, 0);
    let b = if is_even_rec(n % 64) { 1 } else { 0 };
    a + b
}

// ---------------------------------------------------------------------------
// 12. Ordered map: tree traversal, comparators, sorting
//     → heavy branching, Box-per-node allocation, call_indirect for Ord
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_btree(n: u32) -> usize {
    let n = black_box(n);
    let mut map: BTreeMap<u32, String> = BTreeMap::new();

    for i in 0..n {
        let key = i.wrapping_mul(2654435761) % 4096;
        map.insert(key, format!("v{key}"));
    }

    let mut total = 0usize;
    for (k, v) in map.iter() {
        if k % 3 == 0 {
            total += v.len();
        }
    }

    let mut keys: Vec<u32> = map.keys().copied().collect();
    keys.sort_by(|a, b| b.cmp(a)); // reverse sort, closure comparator
    keys.dedup();

    for w in keys.windows(2) {
        if w[0] < w[1] {
            total += 1;
        }
    }
    total + keys.len()
}

// ---------------------------------------------------------------------------
// 13. Iterator chains
//     → fused loops, closures, Option state machines
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_iterators(n: u32) -> i64 {
    let n = black_box(n);
    (0..n)
        .map(|i| i.wrapping_mul(7) as i64)
        .filter(|v| v % 3 != 0)
        .take_while(|v| *v < 1_000_000)
        .enumerate()
        .fold(0i64, |acc, (i, v)| acc.wrapping_add(v ^ (i as i64)))
}

// ---------------------------------------------------------------------------
// 14. Panic paths
//     → bounds checks, overflow checks, unwrap → unreachable
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_panic_paths(i: u32, d: i32) -> i32 {
    let arr = [10i32, 20, 30, 40];
    let i = black_box(i) as usize;

    // index out of bounds → panic → unreachable
    let a = arr[i % 8];

    // explicit unwrap on None → panic
    let b: Option<i32> = if d > 0 { Some(d) } else { None };
    let b = b.unwrap_or(0);

    // division that must be guarded
    let c = if d == 0 { 0 } else { a / d };

    a.wrapping_add(b).wrapping_add(c)
}

// ---------------------------------------------------------------------------
// 15. Raw pointers and manual memory work
//     → unguarded loads/stores, pointer arithmetic, memory.fill
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_raw_ptr(n: u32) -> u32 {
    let n = (black_box(n) % 1024).max(1) as usize;
    let mut buf: Vec<u8> = vec![0; n];

    unsafe {
        let p = buf.as_mut_ptr();
        std::ptr::write_bytes(p, 0xAB, n); // memory.fill
        for i in 0..n {
            let v = std::ptr::read(p.add(i));
            std::ptr::write(p.add(i), v ^ (i as u8));
        }
        let mid = n / 2;
        std::ptr::copy(p, p.add(mid), n - mid); // memory.copy (overlapping!)
    }

    buf.iter().fold(0u32, |a, &b| a.wrapping_add(b as u32))
}

// ---------------------------------------------------------------------------
// 16. Entry point tying several of the above together
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn bench_everything(seed: i32) -> i64 {
    let s = black_box(seed);
    let mut acc = 0i64;

    acc = acc.wrapping_add(bench_imports(s));
    acc = acc.wrapping_add(bench_imports_in_loop(s as u32));
    acc = acc.wrapping_add(bench_narrow_fields(s));
    acc = acc.wrapping_add(bench_br_table((s as u32) % 13, s, s ^ 0x5A5A) as i64);
    acc = acc.wrapping_add(bench_nested_loops(s % 64) as i64);
    acc = acc.wrapping_add(bench_vec_growth((s as u32) % 128) as i64);
    acc = acc.wrapping_add(bench_strings((s as u32) % 16) as i64);
    acc = acc.wrapping_add(bench_dyn_dispatch(s, (s as u32) % 8) as i64);
    acc = acc.wrapping_add(bench_closures(s) as i64);
    acc = acc.wrapping_add(bench_enums((s as u32) % 5, s as f64, 2.0) as i64);
    acc = acc.wrapping_add(bench_floats(s as f64, s as f32) as i64);
    acc = acc.wrapping_add(bench_bits(s as u32, s as u64) as i64);
    acc = acc.wrapping_add(bench_recursion(s as u32) as i64);
    acc = acc.wrapping_add(bench_btree((s as u32) % 256) as i64);
    acc = acc.wrapping_add(bench_iterators((s as u32) % 512));
    acc = acc.wrapping_add(bench_raw_ptr(s as u32) as i64);

    acc
}
