//! Heap allocation: the collections and smart pointers real Rust programs lean
//! on, exercised through the allocator that lives inside the guest's own linear
//! memory.
//!
//! This is the guest that most resembles actual application code, and it is the
//! one that shook out two real interpreter bugs during development — both of
//! which the single-function fixture tests were blind to.
//!
//! **Hash iteration order.** `HashMap`/`HashSet` seed differently under wasm than
//! natively, so every result here is an order-independent reduction (sum, xor,
//! count) or goes through a `BTreeMap`. Never return anything derived from hash
//! iteration order, and never `{:?}` a hash container.
//!
//! Export names are prefixed `heap_`.

// A value bound to a local before being returned is holding a borrow open to a
// precise point — inlining it, as clippy suggests, changes drop order and so
// changes what the guest does.
#![allow(clippy::let_and_return)]
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::hint::black_box;
use std::rc::Rc;

/// `Vec` growth across many reallocations. Each regrow is a fresh allocation plus
/// a `memory.copy` of the old contents.
#[unsafe(no_mangle)]
pub extern "C" fn heap_vec_growth(n: i32) -> i64 {
    let n = n.clamp(1, 20_000);
    let mut v: Vec<i64> = Vec::new();

    for i in 0..n {
        v.push((i as i64).wrapping_mul(2_654_435_761));
    }

    // shrink and regrow to exercise the realloc-down path too
    v.truncate(v.len() / 2);
    v.shrink_to_fit();

    for i in 0..n / 2 {
        v.push(i as i64);
    }

    let sum = v.iter().fold(0i64, |a, b| a.wrapping_add(*b));

    sum.wrapping_mul(31).wrapping_add(v.len() as i64)
}

/// Sorting and deduplication — comparison-heavy, with a lot of element movement
/// through `memory.copy`.
#[unsafe(no_mangle)]
pub extern "C" fn heap_sort_dedup(n: i32) -> i64 {
    let n = n.clamp(1, 8_000);
    let mut v: Vec<i32> = (0..n).map(|i| (i * 7_919) % 1_013).collect();

    v.sort_unstable();
    v.dedup();

    let a = v
        .iter()
        .fold(0i64, |acc, x| acc.wrapping_mul(3).wrapping_add(*x as i64));

    // sort_by with a reversing comparator, then a stable sort by key
    v.sort_by(|x, y| y.cmp(x));
    v.sort_by_key(|x| *x % 17);

    let b = v
        .iter()
        .fold(0i64, |acc, x| acc.wrapping_mul(5).wrapping_add(*x as i64));

    v.retain(|x| x % 2 == 0);
    v.reverse();

    let c = v.len() as i64;

    a.wrapping_mul(31)
        .wrapping_add(b)
        .wrapping_mul(31)
        .wrapping_add(c)
}

/// `String` building: `push_str`, `format!`, and the UTF-8 paths.
#[unsafe(no_mangle)]
pub extern "C" fn heap_strings(n: i32) -> i64 {
    let n = n.clamp(1, 2_000);
    let mut s = String::new();

    for i in 0..n {
        s.push_str(&format!("{i}-"));

        if i % 10 == 0 {
            s.push('é'); // multi-byte, so the length is not the char count
        }
    }

    let bytes = s.len() as i64;
    let chars = s.chars().count() as i64;
    let digits = s.chars().filter(char::is_ascii_digit).count() as i64;
    let upper = s.to_uppercase().len() as i64;
    let parts = s.split('-').count() as i64;

    let checksum = s
        .bytes()
        .fold(0i64, |a, b| a.wrapping_mul(31).wrapping_add(b as i64));

    bytes
        .wrapping_mul(7)
        .wrapping_add(chars)
        .wrapping_mul(7)
        .wrapping_add(digits)
        .wrapping_mul(7)
        .wrapping_add(upper)
        .wrapping_mul(7)
        .wrapping_add(parts)
        .wrapping_mul(7)
        .wrapping_add(checksum)
}

/// `HashMap`, reduced order-independently. Sums and counts only — never anything
/// that depends on iteration order.
#[unsafe(no_mangle)]
pub extern "C" fn heap_hashmap(n: i32) -> i64 {
    let n = n.clamp(1, 4_000);
    let mut m: HashMap<i32, i64> = HashMap::new();

    for i in 0..n {
        *m.entry(i % 257).or_insert(0) += i as i64;
    }

    for i in 0..n / 3 {
        m.remove(&(i % 257));
    }

    // sum and xor are both order-independent
    let sum: i64 = m.values().fold(0i64, |a, b| a.wrapping_add(*b));
    let xor: i64 = m.values().fold(0i64, |a, b| a ^ *b);
    let keys: i64 = m.keys().map(|k| *k as i64).sum();

    let mut set: HashSet<i32> = HashSet::new();

    for i in 0..n {
        set.insert(i % 101);
    }

    sum.wrapping_mul(31)
        .wrapping_add(xor)
        .wrapping_mul(31)
        .wrapping_add(keys)
        .wrapping_mul(31)
        .wrapping_add(m.len() as i64)
        .wrapping_mul(31)
        .wrapping_add(set.len() as i64)
}

/// `BTreeMap`/`BTreeSet`, where iteration order *is* deterministic, so ordered
/// traversal can be part of the result.
#[unsafe(no_mangle)]
pub extern "C" fn heap_btreemap(n: i32) -> i64 {
    let n = n.clamp(1, 4_000);
    let mut m: BTreeMap<i32, i64> = BTreeMap::new();

    for i in 0..n {
        m.insert((i * 37) % 1_021, i as i64);
    }

    // ordered, so this is a stable fold
    let ordered = m.iter().fold(0i64, |a, (k, v)| {
        a.wrapping_mul(3).wrapping_add(*k as i64 ^ *v)
    });

    let first = m.keys().next().copied().unwrap_or(-1) as i64;
    let last = m.keys().next_back().copied().unwrap_or(-1) as i64;
    let range: i64 = m.range(100..500).map(|(_, v)| *v).sum();

    let set: BTreeSet<i32> = (0..n).map(|i| (i * 13) % 97).collect();

    let set_fold = set
        .iter()
        .fold(0i64, |a, x| a.wrapping_mul(3).wrapping_add(*x as i64));

    ordered
        .wrapping_mul(31)
        .wrapping_add(first)
        .wrapping_mul(31)
        .wrapping_add(last)
        .wrapping_mul(31)
        .wrapping_add(range)
        .wrapping_mul(31)
        .wrapping_add(set_fold)
}

/// `VecDeque` and `BinaryHeap` — ring-buffer wraparound and sift operations.
#[unsafe(no_mangle)]
pub extern "C" fn heap_deque_and_binheap(n: i32) -> i64 {
    let n = n.clamp(1, 4_000);
    let mut dq: VecDeque<i32> = VecDeque::new();

    for i in 0..n {
        if i % 2 == 0 {
            dq.push_back(i);
        } else {
            dq.push_front(i);
        }

        if i % 5 == 0 {
            dq.pop_front();
        }

        if i % 7 == 0 {
            dq.pop_back();
        }
    }

    let dq_fold = dq
        .iter()
        .fold(0i64, |a, x| a.wrapping_mul(3).wrapping_add(*x as i64));

    let mut bh: BinaryHeap<i32> = BinaryHeap::new();

    for i in 0..n {
        bh.push((i * 7_919) % 1_013);
    }

    // popping a BinaryHeap is deterministic (descending)
    let mut bh_fold = 0i64;

    while let Some(top) = bh.pop() {
        bh_fold = bh_fold.wrapping_mul(3).wrapping_add(top as i64);
    }

    dq_fold
        .wrapping_mul(31)
        .wrapping_add(dq.len() as i64)
        .wrapping_mul(31)
        .wrapping_add(bh_fold)
}

/// `Box`, `Rc`, and `RefCell` — pointer indirection and interior mutability, plus
/// a reference-counted graph whose refcounts have to be observable.
#[unsafe(no_mangle)]
pub extern "C" fn heap_smart_pointers(n: i32) -> i64 {
    let n = n.clamp(1, 2_000);

    let boxed: Vec<Box<i64>> = (0..n).map(|i| Box::new(i as i64 * 3)).collect();
    let box_sum = boxed.iter().fold(0i64, |a, b| a.wrapping_add(**b));
    let shared = Rc::new(RefCell::new(0i64));
    let clones: Vec<Rc<RefCell<i64>>> = (0..n.min(64)).map(|_| Rc::clone(&shared)).collect();

    for (i, c) in clones.iter().enumerate() {
        *c.borrow_mut() += i as i64;
    }

    let count = Rc::strong_count(&shared) as i64;
    let inner = *shared.borrow();

    drop(clones);

    let after_drop = Rc::strong_count(&shared) as i64;

    // a boxed slice and a boxed trait object
    let bs: Box<[i64]> = (0..n.min(128)).map(i64::from).collect();
    let bs_sum = bs.iter().fold(0i64, |a, b| a.wrapping_add(*b));

    box_sum
        .wrapping_mul(31)
        .wrapping_add(count)
        .wrapping_mul(31)
        .wrapping_add(inner)
        .wrapping_mul(31)
        .wrapping_add(after_drop)
        .wrapping_mul(31)
        .wrapping_add(bs_sum)
}

/// `Drop` order. Values drop in reverse declaration order, and a `Vec` drops its
/// elements front to back — observable through a shared counter.
#[unsafe(no_mangle)]
pub extern "C" fn heap_drop_order(n: i32) -> i64 {
    struct Recorder<'a> {
        id: i64,
        log: &'a RefCell<Vec<i64>>,
    }

    impl Drop for Recorder<'_> {
        fn drop(&mut self) {
            self.log.borrow_mut().push(self.id);
        }
    }

    let log = RefCell::new(Vec::new());

    {
        let _a = Recorder { id: 1, log: &log };
        let _b = Recorder { id: 2, log: &log };
        let _c = Recorder { id: 3, log: &log };

        let v: Vec<Recorder> = (10..10 + n.clamp(1, 8) as i64)
            .map(|id| Recorder { id, log: &log })
            .collect();

        drop(v);

        // explicit early drop, out of declaration order
        let d = Recorder { id: 99, log: &log };

        drop(d);
    }

    // bound to a local so the `Ref` is dropped before `log` is
    let folded = log
        .borrow()
        .iter()
        .fold(0i64, |a, id| a.wrapping_mul(101).wrapping_add(*id));

    folded
}

/// Nested collections: a map of vectors of strings, which chains allocations
/// several levels deep.
#[unsafe(no_mangle)]
pub extern "C" fn heap_nested_collections(n: i32) -> i64 {
    let n = n.clamp(1, 400);
    let mut m: BTreeMap<i32, Vec<String>> = BTreeMap::new();

    for i in 0..n {
        m.entry(i % 16)
            .or_default()
            .push(format!("item-{}-{}", i, i * i));
    }

    let mut acc = 0i64;

    for (k, v) in &m {
        acc = acc.wrapping_mul(7).wrapping_add(*k as i64);
        acc = acc.wrapping_mul(7).wrapping_add(v.len() as i64);

        for s in v {
            acc = acc
                .wrapping_mul(7)
                .wrapping_add(s.bytes().map(i64::from).sum::<i64>());
        }
    }

    // move the whole thing, then consume it
    let flattened: Vec<String> = m.into_values().flatten().collect();

    acc.wrapping_mul(31).wrapping_add(flattened.len() as i64)
}

/// Allocation churn: many short-lived allocations of varying size, which is the
/// pattern that stresses the allocator's free lists rather than raw growth.
#[unsafe(no_mangle)]
pub extern "C" fn heap_churn(n: i32) -> i64 {
    let n = n.clamp(1, 4_000);
    let mut acc = 0i64;

    for i in 0..n {
        let size = ((i % 64) + 1) as usize;
        let v: Vec<u8> = (0..size).map(|x| (x ^ i as usize) as u8).collect();

        acc = acc
            .wrapping_mul(31)
            .wrapping_add(v.iter().map(|b| *b as i64).sum::<i64>());

        // an interleaved larger allocation so the sizes are not monotonic
        if i % 16 == 0 {
            let big: Vec<u64> = vec![i as u64; 256];
            acc = acc.wrapping_add(big[black_box(128)] as i64);
        }
    }

    acc
}

/// A heap-heavy mixed workload for the throughput table.
#[unsafe(no_mangle)]
pub extern "C" fn heap_mixed_workload(n: i32) -> i64 {
    let n = n.clamp(1, 4_000);
    let mut v: Vec<i32> = (0..n).map(|i| (i * 7_919) % 1_013).collect();

    v.sort_unstable();
    v.dedup();

    let mut m: BTreeMap<i32, String> = BTreeMap::new();

    for (i, x) in v.iter().enumerate() {
        m.insert(*x, format!("k{i}"));
    }

    let total: i64 = m.keys().map(|k| *k as i64).sum();
    let bytes: i64 = m.values().map(|s| s.len() as i64).sum();

    total.wrapping_mul(31).wrapping_add(bytes)
}
