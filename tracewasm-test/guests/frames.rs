//! Call frames: the shapes that stress frame setup, the locals region, and the
//! call-depth guard.
//!
//! TraceWasm keeps each frame's locals on the shared value stack, immediately
//! below that frame's operands, with two separate base heights. Conflating those
//! two bases corrupts memory in a way that only shows up on functions that have
//! *both* declared locals and branches — which is why this guest deliberately
//! mixes many live locals with nested control flow.
//!
//! Recursion here is either indirect (through a function pointer) or non-tail, so
//! LLVM cannot rewrite it into a loop. Accumulator recursion gets flattened and
//! then tests nothing.
//!
//! Export names are prefixed `fr_`.

// A wide, hand-shaped frame is the subject here: the locals are declared in one
// parallel block and updated with explicit `x = x op y` so the wasm reads as the
// operations under test. Clippy's tighter spellings compile to the same thing and
// lose that.
#![allow(unused_mut, clippy::assign_op_pattern)]
#![allow(dead_code)]

use std::hint::black_box;

/// A function with many simultaneously-live locals, so the locals region is wide
/// and the operand region starts well above the frame base.
#[unsafe(no_mangle)]
pub extern "C" fn fr_many_locals(n: i32) -> i64 {
    let mut a = black_box(1i64);
    let mut b = black_box(2i64);
    let mut c = black_box(3i64);
    let mut d = black_box(4i64);
    let mut e = black_box(5i64);
    let mut f = black_box(6i64);
    let mut g = black_box(7i64);
    let mut h = black_box(8i64);
    let mut i = black_box(9i64);
    let mut j = black_box(10i64);
    let mut k = black_box(11i64);
    let mut l = black_box(12i64);
    let mut m = black_box(13i64);
    let mut o = black_box(14i64);
    let mut p = black_box(15i64);
    let mut q = black_box(16i64);
    let mut r = black_box(17i64);
    let mut s = black_box(18i64);

    for step in 0..n.clamp(1, 2_000) {
        // branch inside the loop, so height bookkeeping interacts with locals
        if step % 2 == 0 {
            a = a.wrapping_add(b);
            c = c ^ d;
            e = e.wrapping_add(f);
            g = g ^ h;
        } else {
            i = i.wrapping_add(j);
            k = k ^ l;
            m = m.wrapping_add(o);
            p = p ^ q;
        }

        r = r.wrapping_add(s).rotate_left(3);
        s = s ^ a ^ i;
    }

    a ^ b ^ c ^ d ^ e ^ f ^ g ^ h ^ i ^ j ^ k ^ l ^ m ^ o ^ p ^ q ^ r ^ s
}

/// Maximum arity the typed API supports on the way in (5 params), plus enough
/// locals that the frame is not trivial.
#[unsafe(no_mangle)]
pub extern "C" fn fr_five_params(a: i32, b: i32, c: i32, d: i32, e: i32) -> i64 {
    let mut acc = 0i64;

    for (i, v) in [a, b, c, d, e].iter().enumerate() {
        acc = acc
            .wrapping_mul(31)
            .wrapping_add((*v as i64).wrapping_mul(i as i64 + 1));
    }

    acc
}

/// A long chain of distinct functions, so each frame has a different signature
/// and local count. Exercises frame setup rather than recursion.
#[unsafe(no_mangle)]
pub extern "C" fn fr_call_chain(n: i32) -> i64 {
    fn l1(x: i64) -> i64 {
        let t = x.wrapping_mul(3);
        l2(t, t ^ 1)
    }

    fn l2(x: i64, y: i64) -> i64 {
        let t = x.wrapping_add(y);
        l3(t, t >> 1, t << 1)
    }

    fn l3(x: i64, y: i64, z: i64) -> i64 {
        let t = x ^ y ^ z;
        l4(t, t.rotate_left(7))
    }

    fn l4(x: i64, y: i64) -> i64 {
        let mut acc = x;
        for i in 0..4 {
            acc = acc.wrapping_add(y >> i);
        }
        l5(acc)
    }

    fn l5(x: i64) -> i64 {
        x.wrapping_mul(2_654_435_761)
    }

    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        acc = acc.wrapping_add(l1(i as i64));
    }

    acc
}

/// Non-tail recursion to a caller-chosen depth. The work happens *after* the
/// recursive call, so this stays a genuine call chain.
#[unsafe(no_mangle)]
pub extern "C" fn fr_recurse_depth(n: i32) -> i64 {
    fn go(k: i32) -> i64 {
        if k <= 0 {
            return 1;
        }

        let below = go(k - 1);

        // post-call work with several live locals across the call boundary
        let a = below.wrapping_mul(3);
        let b = below ^ (k as i64);
        let c = a.wrapping_add(b);

        c.rotate_left((k & 31) as u32)
    }

    go(n.max(0))
}

/// Recursion through a function pointer stored in a static, which LLVM cannot
/// devirtualise — so it cannot flatten the recursion either.
#[unsafe(no_mangle)]
pub extern "C" fn fr_recurse_indirect(n: i32) -> i64 {
    static mut TABLE: [fn(i32) -> i64; 2] = [step_a, step_b];

    fn step_a(k: i32) -> i64 {
        if k <= 0 {
            return 0;
        }

        let f = unsafe { TABLE[(k & 1) as usize] };

        (f(k - 1).wrapping_add(k as i64)) ^ (k as i64)
    }

    fn step_b(k: i32) -> i64 {
        if k <= 0 {
            return 1;
        }

        let f = unsafe { TABLE[((k + 1) & 1) as usize] };

        f(k - 1).wrapping_mul(3).wrapping_add(1)
    }

    step_a(n.max(0))
}

/// Deeply nested closures, each capturing its enclosing environment — a chain of
/// indirect calls with progressively larger captured state.
#[unsafe(no_mangle)]
pub extern "C" fn fr_closure_nest(n: i32) -> i64 {
    let base = black_box(n as i64);

    let f1 = move |x: i64| x.wrapping_add(base);
    let f2 = move |x: i64| f1(x).wrapping_mul(3);
    let f3 = move |x: i64| f2(x) ^ base;
    let f4 = move |x: i64| f3(x).rotate_left(5);

    // through a trait object, so the call is indirect
    let boxed: Box<dyn Fn(i64) -> i64> = Box::new(f4);

    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        acc = acc.wrapping_mul(7).wrapping_add(boxed(i as i64));
    }

    acc
}

/// Dynamic dispatch through a heterogeneous collection of trait objects, so the
/// callee differs per iteration and every call is a `call_indirect`.
#[unsafe(no_mangle)]
pub extern "C" fn fr_dyn_dispatch(n: i32) -> i64 {
    trait Op {
        fn apply(&self, x: i64) -> i64;
    }

    struct Add(i64);
    struct Mul(i64);
    struct Xor(i64);
    struct Rot(u32);

    impl Op for Add {
        fn apply(&self, x: i64) -> i64 {
            x.wrapping_add(self.0)
        }
    }
    impl Op for Mul {
        fn apply(&self, x: i64) -> i64 {
            x.wrapping_mul(self.0)
        }
    }
    impl Op for Xor {
        fn apply(&self, x: i64) -> i64 {
            x ^ self.0
        }
    }
    impl Op for Rot {
        fn apply(&self, x: i64) -> i64 {
            x.rotate_left(self.0)
        }
    }

    let ops: Vec<Box<dyn Op>> = vec![
        Box::new(Add(7)),
        Box::new(Mul(3)),
        Box::new(Xor(0x5555)),
        Box::new(Rot(11)),
        Box::new(Add(-13)),
    ];

    let mut acc = 1i64;

    for i in 0..n.clamp(1, 2_000) {
        acc = ops[(i as usize) % ops.len()].apply(acc);
    }

    acc
}

/// Recursion carrying a large stack-allocated array, so each frame is wide and
/// the interpreter's frame cost per call is visible.
#[unsafe(no_mangle)]
pub extern "C" fn fr_wide_frames(n: i32) -> i64 {
    fn go(k: i32) -> i64 {
        let mut scratch = [0i64; 32];

        for (i, s) in scratch.iter_mut().enumerate() {
            *s = (k as i64).wrapping_mul(i as i64 + 1);
        }

        if k <= 0 {
            return scratch.iter().sum();
        }

        let below = go(k - 1);

        scratch
            .iter()
            .fold(below, |a, b| a.wrapping_mul(3).wrapping_add(*b))
    }

    go(n.clamp(0, 60))
}

/// A generic function instantiated at several types, so rustc emits a separate
/// monomorphised body per instantiation.
#[unsafe(no_mangle)]
pub extern "C" fn fr_monomorphised(n: i32) -> i64 {
    fn sum_of<T: Copy + Into<i64>>(items: &[T]) -> i64 {
        items
            .iter()
            .fold(0i64, |a, b| a.wrapping_mul(31).wrapping_add((*b).into()))
    }

    let a: Vec<i8> = (0..n.clamp(1, 64)).map(|i| i as i8).collect();
    let b: Vec<i16> = (0..n.clamp(1, 64)).map(|i| (i * 3) as i16).collect();
    let c: Vec<i32> = (0..n.clamp(1, 64)).map(|i| i * 7).collect();
    let d: Vec<u32> = (0..n.clamp(1, 64)).map(|i| (i * 11) as u32).collect();

    sum_of(&a)
        .wrapping_mul(31)
        .wrapping_add(sum_of(&b))
        .wrapping_mul(31)
        .wrapping_add(sum_of(&c))
        .wrapping_mul(31)
        .wrapping_add(sum_of(&d))
}

/// A call-heavy workload for the throughput table: one indirect call per
/// iteration and nothing else.
#[unsafe(no_mangle)]
pub extern "C" fn fr_mixed_workload(n: i32) -> i64 {
    static mut TABLE: [fn(i64) -> i64; 4] = [t0, t1, t2, t3];

    fn t0(x: i64) -> i64 {
        x.wrapping_mul(2_654_435_761)
    }

    fn t1(x: i64) -> i64 {
        x ^ 0x5bf0_3635
    }

    fn t2(x: i64) -> i64 {
        x.wrapping_add(0x9e37_79b9)
    }

    fn t3(x: i64) -> i64 {
        !x
    }

    let mut acc = 0i64;
    let mut i = 0;

    while i < n {
        let f = unsafe { TABLE[(acc & 3) as usize] };

        acc = f(acc).wrapping_add(i as i64);
        i += 1;
    }

    acc
}

// ---------------------------------------------------------------------------
// Deliberate trap, called by tests/traps.rs
// ---------------------------------------------------------------------------

/// Unbounded indirect recursion. Used to check that exceeding
/// `Config::max_call_stack_depth` produces a catchable trap rather than
/// overflowing the host stack and aborting the process.
#[unsafe(no_mangle)]
pub extern "C" fn fr_trap_infinite_recursion(n: i32) -> i64 {
    static mut SELF: Option<fn(i32) -> i64> = None;

    fn go(k: i32) -> i64 {
        let f = unsafe { SELF }.unwrap();

        // never terminates: the argument grows, and the result is used after the
        // call so it cannot become a loop
        (f(k + 1).wrapping_add(k as i64)) ^ (k as i64)
    }

    unsafe { SELF = Some(go) };

    go(n)
}
