//! Exotic constructs: the parts of Rust that produce unusual wasm, and the
//! language features an interpreter is most likely to have never been exercised
//! against.
//!
//! Nothing here is contrived for its own sake — every item is something real Rust
//! code does, chosen because it lowers to something structurally interesting:
//! niche-optimised enums, vtables, slice patterns, UTF-8 decoding, saturating
//! casts, `#[repr]` layouts, and const evaluation.
//!
//! Export names are prefixed `ex_`.

// These guests are deliberately awkward programs, shaped to produce wasm that a
// straightforward one would not. Clippy's idioms fight that on purpose:
//
// * the packed struct is `repr(Rust, packed)` — the same layout it always had,
//   now stated explicitly, since `repr(C, packed)` would reorder fields and test
//   something else;
// * a block used directly as a match scrutinee is named as such in the source;
// * `None.unwrap_or_default()` and a redundant cast exercise paths a tidier
//   spelling would optimise away.
#![allow(
    clippy::unnecessary_cast,
    clippy::blocks_in_conditions,
    clippy::unnecessary_literal_unwrap,
    unused_qualifications
)]
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::hint::black_box;

/// Enums with payloads, including the niche optimisation (`Option<&T>` and
/// `Option<NonZero>` occupy no more space than the payload).
#[unsafe(no_mangle)]
pub extern "C" fn ex_enum_payloads(n: i32) -> i64 {
    #[derive(Clone, Copy)]
    enum Shape {
        Unit,
        Int(i64),
        Pair(i32, i32),
        Float(f64),
        Nested(Option<i32>),
    }

    fn eval(s: Shape) -> i64 {
        match s {
            Shape::Unit => 1,
            Shape::Int(v) => v.wrapping_mul(3),
            Shape::Pair(a, b) => (a as i64) ^ (b as i64),
            Shape::Float(f) => f as i64,
            Shape::Nested(Some(v)) => v as i64 + 100,
            Shape::Nested(None) => -1,
        }
    }

    let shapes = [
        Shape::Unit,
        Shape::Int(42),
        Shape::Pair(7, 9),
        Shape::Float(-2.75),
        Shape::Nested(Some(5)),
        Shape::Nested(None),
        Shape::Int(i64::MIN),
    ];

    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        let s = shapes[(i as usize) % shapes.len()];

        acc = acc.wrapping_mul(31).wrapping_add(eval(s));
    }

    // Niche optimisation: these occupy no more space than their payloads.
    //
    // Deliberately no pointer-sized type here (`Option<&T>`, `Option<Box<T>>`):
    // wasm32 is a 32-bit target and the native oracle is 64-bit, so any result
    // derived from pointer or `usize` width differs legitimately between the two
    // and would make this a flaky test rather than a real one.
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(size_of::<Option<std::num::NonZeroU32>>() as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(size_of::<Option<Option<bool>>>() as i64);

    acc
}

/// `Option`/`Result` combinator chains, which become long branch sequences.
#[unsafe(no_mangle)]
pub extern "C" fn ex_option_result_chains(n: i32) -> i64 {
    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        let i = black_box(i);

        let a = Some(i)
            .filter(|x| x % 2 == 0)
            .map(|x| x * 3)
            .and_then(|x| if x > 10 { Some(x) } else { None })
            .or(Some(-1))
            .map(i64::from)
            .unwrap_or(0);

        let b: Result<i32, &str> = if i % 5 == 0 { Err("no") } else { Ok(i) };
        let c = b
            .map(|x| x + 1)
            .map_err(|e| e.len() as i32)
            .and_then(|x| if x % 3 == 0 { Err(0) } else { Ok(x) })
            .unwrap_or_else(|e| e - 1) as i64;

        let d = Some(i)
            .zip(Some(i * 2))
            .map(|(x, y)| (x ^ y) as i64)
            .unwrap_or(0);
        let e = Option::<i32>::None.unwrap_or_default() as i64;
        let f = Some(Some(i)).flatten().unwrap_or(-7) as i64;

        acc = acc
            .wrapping_mul(7)
            .wrapping_add(a ^ c)
            .wrapping_mul(7)
            .wrapping_add(d ^ e ^ f);
    }

    acc
}

/// Slice patterns, including bindings on the middle and destructuring in `match`.
#[unsafe(no_mangle)]
pub extern "C" fn ex_slice_patterns(n: i32) -> i64 {
    fn classify(s: &[i32]) -> i64 {
        match s {
            [] => 0,
            [x] => *x as i64,
            [a, b] => (*a as i64) * 10 + *b as i64,
            [first, .., last] => (*first as i64) ^ (*last as i64),
        }
    }

    fn sum_middle(s: &[i32]) -> i64 {
        match s {
            [_, mid @ .., _] => mid.iter().map(|x| *x as i64).sum(),
            _ => -1,
        }
    }

    let mut acc = 0i64;

    for len in 0..n.clamp(1, 24) as usize {
        let v: Vec<i32> = (0..len as i32).map(|x| x * 3 - 1).collect();

        acc = acc.wrapping_mul(31).wrapping_add(classify(&v));
        acc = acc.wrapping_mul(31).wrapping_add(sum_middle(&v));

        // tuple and struct destructuring with nesting
        let ((p, q), r) = ((len as i64, len as i64 * 2), len as i64 * 3);

        acc = acc.wrapping_mul(31).wrapping_add(p ^ q ^ r);
    }

    acc
}

/// UTF-8: multi-byte encoding and decoding, char boundaries, and the fact that
/// byte length is not char count.
#[unsafe(no_mangle)]
pub extern "C" fn ex_utf8(n: i32) -> i64 {
    // one, two, three and four byte encodings
    let samples = ["a", "é", "€", "𝄞", "aé€𝄞", "日本語", "🦀🦀"];
    let mut acc = 0i64;

    for (i, s) in samples.iter().enumerate() {
        acc = acc.wrapping_mul(31).wrapping_add(s.len() as i64);
        acc = acc.wrapping_mul(31).wrapping_add(s.chars().count() as i64);
        acc = acc
            .wrapping_mul(31)
            .wrapping_add(s.chars().map(|c| c as i64).sum::<i64>());
        acc = acc
            .wrapping_mul(31)
            .wrapping_add(s.char_indices().map(|(idx, _)| idx as i64).sum::<i64>());
        acc = acc.wrapping_mul(31).wrapping_add(i as i64);
    }

    // build a string with a repeating multi-byte pattern, then walk it back
    let built: String = (0..n.clamp(1, 200))
        .map(|i| match i % 4 {
            0 => 'a',
            1 => 'é',
            2 => '€',
            _ => '🦀',
        })
        .collect();

    acc = acc.wrapping_mul(31).wrapping_add(built.len() as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(built.chars().rev().map(|c| c as i64).sum::<i64>());

    // char classification and conversion
    for c in ['a', 'Z', '0', ' ', 'é', '🦀'] {
        acc = acc.wrapping_mul(7).wrapping_add(c.is_alphabetic() as i64);
        acc = acc.wrapping_mul(7).wrapping_add(c.is_numeric() as i64);
        acc = acc.wrapping_mul(7).wrapping_add(c.len_utf8() as i64);
        acc = acc
            .wrapping_mul(7)
            .wrapping_add(c.to_ascii_uppercase() as i64);
    }

    acc
}

/// `as` casts between every integer width and sign, which is where wrapping and
/// the saturating float conversions live.
#[unsafe(no_mangle)]
pub extern "C" fn ex_casts(_: i32) -> i64 {
    let mut acc = 0i64;

    for v in [
        0i64,
        1,
        -1,
        127,
        128,
        255,
        256,
        32_767,
        65_535,
        i64::MIN,
        i64::MAX,
    ] {
        let v = black_box(v);

        acc = acc.wrapping_mul(31).wrapping_add(v as i8 as i64);
        acc = acc.wrapping_mul(31).wrapping_add(v as u8 as i64);
        acc = acc.wrapping_mul(31).wrapping_add(v as i16 as i64);
        acc = acc.wrapping_mul(31).wrapping_add(v as u16 as i64);
        acc = acc.wrapping_mul(31).wrapping_add(v as i32 as i64);
        acc = acc.wrapping_mul(31).wrapping_add(v as u32 as i64);
        // `u64`, not `usize`: `usize` is 32-bit under wasm32 and 64-bit on the
        // native oracle, so a `usize` round-trip truncates on one side only.
        acc = acc.wrapping_mul(31).wrapping_add(v as u64 as i64);
        // saturating float conversion in both directions
        acc = acc.wrapping_mul(31).wrapping_add((v as f32) as i64);
        acc = acc.wrapping_mul(31).wrapping_add((v as f64) as i32 as i64);
        acc = acc.wrapping_mul(31).wrapping_add(v as u8 as char as i64);
    }

    // bool and char casts
    acc = acc.wrapping_mul(31).wrapping_add(black_box(true) as i64);
    acc = acc.wrapping_mul(31).wrapping_add(black_box('A') as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(black_box('🦀') as u32 as i64);

    acc
}

/// `#[repr(C)]` and `#[repr(Rust, packed)]` layouts, whose field offsets the guest can
/// observe — a different memory-access pattern from the default layout.
#[unsafe(no_mangle)]
pub extern "C" fn ex_repr_layouts(_: i32) -> i64 {
    #[repr(C)]
    struct CLayout {
        a: u8,
        b: u32,
        c: u16,
        d: u64,
    }

    #[repr(Rust, packed)]
    struct Packed {
        a: u8,
        b: u32,
        c: u16,
    }

    #[repr(align(32))]
    struct Aligned {
        v: u64,
    }

    let c = CLayout {
        a: black_box(1),
        b: black_box(2),
        c: black_box(3),
        d: black_box(4),
    };
    let p = Packed {
        a: black_box(5),
        b: black_box(6),
        c: black_box(7),
    };
    let al = Aligned { v: black_box(8) };

    let mut acc = 0i64;

    acc = acc
        .wrapping_mul(31)
        .wrapping_add(size_of::<CLayout>() as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(align_of::<CLayout>() as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(size_of::<Packed>() as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(size_of::<Aligned>() as i64);
    acc = acc
        .wrapping_mul(31)
        .wrapping_add(align_of::<Aligned>() as i64);

    acc = acc.wrapping_mul(31).wrapping_add(c.a as i64);
    acc = acc.wrapping_mul(31).wrapping_add(c.b as i64);
    acc = acc.wrapping_mul(31).wrapping_add(c.c as i64);
    acc = acc.wrapping_mul(31).wrapping_add(c.d as i64);

    // packed fields need unaligned access
    let pa = p.a;
    let pb = p.b;
    let pc = p.c;

    acc = acc.wrapping_mul(31).wrapping_add(pa as i64);
    acc = acc.wrapping_mul(31).wrapping_add(pb as i64);
    acc = acc.wrapping_mul(31).wrapping_add(pc as i64);
    acc = acc.wrapping_mul(31).wrapping_add(al.v as i64);

    acc
}

/// Const evaluation and statics: values computed at compile time, plus a static
/// table the guest reads at runtime.
#[unsafe(no_mangle)]
pub extern "C" fn ex_const_and_static(n: i32) -> i64 {
    const fn fib(n: u32) -> u64 {
        match n {
            0 => 0,
            1 => 1,
            _ => fib(n - 1) + fib(n - 2),
        }
    }

    const FIB20: u64 = fib(20);
    const TABLE: [i64; 8] = [1, 1, 2, 6, 24, 120, 720, 5_040];
    static LOOKUP: [u8; 16] = [
        0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225,
    ];

    let mut acc = FIB20 as i64;

    for i in 0..n.clamp(1, 2_000) as usize {
        acc = acc
            .wrapping_mul(31)
            .wrapping_add(TABLE[i % TABLE.len()])
            .wrapping_add(LOOKUP[i % LOOKUP.len()] as i64);
    }

    acc
}

/// Trait objects with supertraits, default methods, and generic dispatch through
/// `impl Trait` — several distinct vtable shapes.
#[unsafe(no_mangle)]
pub extern "C" fn ex_traits(n: i32) -> i64 {
    trait Named {
        fn tag(&self) -> i64;
    }

    trait Scored: Named {
        fn raw(&self) -> i64;

        // default method calling a required one
        fn score(&self) -> i64 {
            self.raw().wrapping_mul(3).wrapping_add(self.tag())
        }
    }

    struct A(i64);
    struct B(i64);

    impl Named for A {
        fn tag(&self) -> i64 {
            1
        }
    }
    impl Named for B {
        fn tag(&self) -> i64 {
            2
        }
    }
    impl Scored for A {
        fn raw(&self) -> i64 {
            self.0
        }
    }
    impl Scored for B {
        fn raw(&self) -> i64 {
            self.0.wrapping_mul(-1)
        }

        // overrides the default
        fn score(&self) -> i64 {
            self.raw() ^ 0xff
        }
    }

    fn via_generic(s: &impl Scored) -> i64 {
        s.score()
    }

    fn via_dyn(s: &dyn Scored) -> i64 {
        s.score()
    }

    let items: Vec<Box<dyn Scored>> = (0..n.clamp(1, 32))
        .map(|i| -> Box<dyn Scored> {
            if i % 2 == 0 {
                Box::new(A(i as i64))
            } else {
                Box::new(B(i as i64))
            }
        })
        .collect();

    let mut acc = 0i64;

    for it in &items {
        acc = acc.wrapping_mul(31).wrapping_add(via_dyn(it.as_ref()));
    }

    acc = acc.wrapping_mul(31).wrapping_add(via_generic(&A(7)));
    acc = acc.wrapping_mul(31).wrapping_add(via_generic(&B(7)));

    acc
}

/// Sorting with a custom `Ord` impl, which drives the comparator through many
/// indirect calls and must be a total order for the result to be deterministic.
#[unsafe(no_mangle)]
pub extern "C" fn ex_custom_ord(n: i32) -> i64 {
    #[derive(PartialEq, Eq)]
    struct Key {
        primary: i32,
        secondary: i32,
    }

    impl Ord for Key {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // deliberately non-trivial: descending primary, ascending secondary
            other
                .primary
                .cmp(&self.primary)
                .then_with(|| self.secondary.cmp(&other.secondary))
        }
    }

    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = n.clamp(1, 2_000);

    let mut keys: Vec<Key> = (0..n)
        .map(|i| Key {
            primary: (i * 7) % 13,
            secondary: (i * 11) % 17,
        })
        .collect();

    // stable sort, so equal keys keep their relative order deterministically
    keys.sort();

    let mut acc = 0i64;

    for k in &keys {
        acc = acc
            .wrapping_mul(31)
            .wrapping_add(k.primary as i64)
            .wrapping_mul(31)
            .wrapping_add(k.secondary as i64);
    }

    let min = keys.iter().min().map(|k| k.primary as i64).unwrap_or(-1);
    let max = keys.iter().max().map(|k| k.primary as i64).unwrap_or(-1);

    acc.wrapping_mul(31)
        .wrapping_add(min)
        .wrapping_mul(31)
        .wrapping_add(max)
}

/// Shadowing, block expressions, and nested scopes producing values — shapes that
/// make the lowering pass's height tracking work.
#[unsafe(no_mangle)]
pub extern "C" fn ex_shadowing_blocks(n: i32) -> i64 {
    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        let x = i as i64;
        let x = x.wrapping_mul(3);
        let x = {
            let x = x.wrapping_add(1);
            let inner = {
                let x = x ^ 0xff;
                x.rotate_left(3)
            };
            x.wrapping_add(inner)
        };

        // a block used directly as a match scrutinee
        let y = match {
            let t = x % 5;
            if t < 0 { -t } else { t }
        } {
            0 => 10i64,
            1 => 20,
            2 => 30,
            3 => 40,
            _ => 50,
        };

        acc = acc.wrapping_mul(7).wrapping_add(x).wrapping_add(y);
    }

    acc
}

/// A realistic mixed program: parse-like tokenisation, a symbol table, and a
/// small evaluation pass. The closest thing here to application code.
#[unsafe(no_mangle)]
pub extern "C" fn ex_realistic_program(n: i32) -> i64 {
    #[derive(Clone, Copy, PartialEq)]
    enum Tok {
        Num(i64),
        Ident(u8),
        Plus,
        Star,
        Open,
        Close,
    }

    let n = n.clamp(1, 600);

    // build a deterministic token stream
    let mut toks: Vec<Tok> = Vec::new();

    for i in 0..n {
        match i % 6 {
            0 => toks.push(Tok::Num(i as i64 % 97)),
            1 => toks.push(Tok::Plus),
            2 => toks.push(Tok::Ident((b'a' + (i % 26) as u8) as u8)),
            3 => toks.push(Tok::Star),
            4 => toks.push(Tok::Open),
            _ => toks.push(Tok::Close),
        }
    }

    // symbol table with ordered iteration
    let mut syms: BTreeMap<u8, i64> = BTreeMap::new();

    for (i, t) in toks.iter().enumerate() {
        if let Tok::Ident(name) = t {
            *syms.entry(*name).or_insert(0) += i as i64;
        }
    }

    // a fold that treats the stream as a very loose expression
    let mut acc = 0i64;
    let mut depth = 0i64;
    let mut pending_mul = false;

    for t in &toks {
        match t {
            Tok::Num(v) => {
                acc = if pending_mul {
                    acc.wrapping_mul(*v)
                } else {
                    acc.wrapping_add(*v)
                };
                pending_mul = false;
            }
            Tok::Ident(name) => {
                let v = syms.get(name).copied().unwrap_or(0);
                acc = acc.wrapping_add(v ^ depth);
            }
            Tok::Plus => pending_mul = false,
            Tok::Star => pending_mul = true,
            Tok::Open => depth += 1,
            Tok::Close => depth -= 1,
        }
    }

    let sym_fold = syms.iter().fold(0i64, |a, (k, v)| {
        a.wrapping_mul(31).wrapping_add(*k as i64 ^ *v)
    });

    acc.wrapping_mul(31)
        .wrapping_add(sym_fold)
        .wrapping_mul(31)
        .wrapping_add(depth)
        .wrapping_mul(31)
        .wrapping_add(toks.len() as i64)
}
