//! Control flow: the shapes that stress branch lowering, stack-height
//! bookkeeping, and dead-code handling.
//!
//! These are the constructs where TraceWasm's lowering pass does real work —
//! resolving structured control flow to absolute program counters, recording the
//! stack height each label unwinds to, and tracking which regions are
//! unreachable. A bug there shows up as a wrong answer or a stack underflow, not
//! a compile error, which is exactly why these need a differential oracle.
//!
//! Export names are prefixed `cf_` to keep `#[unsafe(no_mangle)]` symbols unique
//! when every guest is linked into one native test binary.

#![allow(dead_code)]

use std::hint::black_box;

/// Deeply nested loops with a `break` out of the *outer* one from the inner body.
/// Lowering has to unwind several labels at once and land on the right height.
#[unsafe(no_mangle)]
pub extern "C" fn cf_nested_loops(n: i32) -> i64 {
    // clamped hard: this is a triple-nested loop, so the bound is cubed
    let n = n.clamp(1, 12);
    let mut acc = 0i64;

    'outer: for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                acc = acc.wrapping_mul(3).wrapping_add((i ^ j ^ k) as i64);

                if acc % 1_000_003 == 0 {
                    break 'outer;
                }

                if k > 4 {
                    continue;
                }
            }

            if j > 6 {
                break;
            }
        }
    }

    acc
}

/// A dense `match` on a contiguous range, which rustc lowers to `br_table`.
/// Every arm is a different label with a different unwind height.
#[unsafe(no_mangle)]
pub extern "C" fn cf_br_table_dense(n: i32) -> i64 {
    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        let v = match black_box(i % 12) {
            0 => 1i64,
            1 => 10,
            2 => 100,
            3 => 1_000,
            4 => -1,
            5 => -10,
            6 => i as i64 * 2,
            7 => i as i64 * 3,
            8 => (i as i64).wrapping_mul(i as i64),
            9 => !(i as i64),
            10 => i64::from(i).rotate_left(3),
            _ => 0,
        };

        acc = acc.wrapping_mul(31).wrapping_add(v);
    }

    acc
}

/// A sparse `match`, which rustc lowers to a comparison chain rather than a
/// jump table — a different lowering path from `cf_br_table_dense`.
#[unsafe(no_mangle)]
pub extern "C" fn cf_br_table_sparse(n: i32) -> i64 {
    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        let v = match black_box(i) {
            0 => 1i64,
            7 => 2,
            113 => 3,
            1_000 => 4,
            65_535 => 5,
            1_000_000 => 6,
            _ if i % 3 == 0 => 7,
            _ if i % 5 == 0 => 8,
            _ => 9,
        };

        acc = acc.wrapping_mul(7).wrapping_add(v);
    }

    acc
}

/// Early `return` from inside nested blocks. `return` is a branch to the
/// function's outermost label, so it must unwind every intervening label.
#[unsafe(no_mangle)]
pub extern "C" fn cf_early_return(n: i32) -> i64 {
    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        {
            let inner = black_box(i);

            if inner == 13 {
                return acc.wrapping_mul(-1);
            }

            if inner > 40 {
                return acc ^ 0x5555;
            }

            acc = acc.wrapping_add(inner as i64);
        }
    }

    acc
}

/// Code after an unconditional branch is unreachable, and unreachable code is
/// stack-polymorphic. TraceWasm freezes height tracking through such regions;
/// getting that wrong underflows the height counter at lowering time.
#[unsafe(no_mangle)]
pub extern "C" fn cf_unreachable_regions(n: i32) -> i64 {
    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        loop {
            acc = acc.wrapping_add(i as i64);

            if black_box(true) {
                break;
            }

            // dead: after an unconditional break
            acc = acc.wrapping_mul(999);
        }

        if black_box(i) < 0 {
            // dead: the condition is never true, but it still has to lower
            let mut dead = 0i64;
            for j in 0..10 {
                dead = dead.wrapping_add(j);
                if dead > 5 {
                    break;
                }
            }
            return dead;
        }
    }

    acc
}

/// `if`/`else` where both arms produce a value, plus a chain deep enough that the
/// `else_index` backpatching has to nest.
#[unsafe(no_mangle)]
pub extern "C" fn cf_if_else_chain(n: i32) -> i64 {
    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        let v = if black_box(i) % 2 == 0 {
            if i % 4 == 0 {
                if i % 8 == 0 { 1i64 } else { 2 }
            } else if i % 6 == 0 {
                3
            } else {
                4
            }
        } else if i % 3 == 0 {
            if i % 9 == 0 { 5 } else { 6 }
        } else {
            7
        };

        acc = acc.wrapping_mul(11).wrapping_add(v);
    }

    acc
}

/// `while let` and `loop`+`break value`, which produce a value out of a loop —
/// the loop label's arity is non-zero.
#[unsafe(no_mangle)]
pub extern "C" fn cf_loop_with_value(n: i32) -> i64 {
    let mut acc = 0i64;
    let mut stack: Vec<i32> = (0..n.clamp(1, 2_000).min(64)).collect();

    while let Some(top) = stack.pop() {
        acc = acc.wrapping_mul(3).wrapping_add(top as i64);
    }

    let found = loop {
        acc = acc.wrapping_add(1);

        if acc % 7 == 0 {
            break acc;
        }

        if acc > 10_000 {
            break -1;
        }
    };

    acc.wrapping_mul(31).wrapping_add(found)
}

/// Labelled block with `break` carrying a value — lowers to a `block` whose label
/// has a result type.
#[unsafe(no_mangle)]
pub extern "C" fn cf_labelled_block(n: i32) -> i64 {
    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        let v = 'blk: {
            if i % 5 == 0 {
                break 'blk 100i64;
            }

            if i % 3 == 0 {
                break 'blk 200;
            }

            (i as i64) * 3
        };

        acc = acc.wrapping_mul(13).wrapping_add(v);
    }

    acc
}

/// Bounded recursion, direct (self-recursive `call`).
#[unsafe(no_mangle)]
pub extern "C" fn cf_recursion_direct(n: i32) -> i64 {
    fn go(k: i32) -> i64 {
        if k <= 0 {
            return 0;
        }

        // not tail-recursive: the multiply happens after the call returns, so
        // LLVM cannot rewrite this into a loop
        (go(k - 1).wrapping_mul(3) + k as i64) ^ (k as i64)
    }

    go(n.clamp(0, 200))
}

/// Mutual recursion, which LLVM will not flatten.
#[unsafe(no_mangle)]
pub extern "C" fn cf_recursion_mutual(n: i32) -> i64 {
    fn even(k: i32) -> i64 {
        if k == 0 {
            1
        } else {
            odd(k - 1).wrapping_mul(2)
        }
    }

    fn odd(k: i32) -> i64 {
        if k == 0 {
            0
        } else {
            even(k - 1).wrapping_add(3)
        }
    }

    even(n.clamp(0, 200))
}

/// Iterator chains, which compile into a lot of inlined control flow with many
/// intermediate blocks.
#[unsafe(no_mangle)]
pub extern "C" fn cf_iterator_chains(n: i32) -> i64 {
    let n = n.clamp(1, 4096);

    let sum: i64 = (0..n)
        .filter(|x| x % 3 != 0)
        .map(|x| (x as i64).wrapping_mul(7))
        .take_while(|x| *x < 1_000_000)
        .skip(2)
        .step_by(3)
        .fold(0i64, |a, b| a.wrapping_add(b));

    let any = (0..n).any(|x| x == n - 1) as i64;
    let all = (0..n).all(|x| x < n) as i64;
    let pos = (0..n).position(|x| x * x > 1000).unwrap_or(0) as i64;
    let chained: i64 = (0..n)
        .chain(0..n / 2)
        .rev()
        .zip((0..n).map(|x| x % 5))
        .map(|(a, b)| (a ^ b) as i64)
        .sum();

    sum.wrapping_mul(31)
        .wrapping_add(any)
        .wrapping_mul(31)
        .wrapping_add(all)
        .wrapping_mul(31)
        .wrapping_add(pos)
        .wrapping_mul(31)
        .wrapping_add(chained)
}

/// `?` propagation through several layers, which becomes a branch per layer.
#[unsafe(no_mangle)]
pub extern "C" fn cf_question_mark(n: i32) -> i64 {
    fn parse(v: i32) -> Result<i32, ()> {
        if v % 17 == 0 { Err(()) } else { Ok(v * 2) }
    }

    fn stage2(v: i32) -> Result<i32, ()> {
        let a = parse(v)?;
        let b = parse(a + 1)?;

        Ok(a.wrapping_add(b))
    }

    fn stage3(v: i32) -> Option<i64> {
        let x = stage2(v).ok()?;
        let y = stage2(x).ok()?;

        Some(x as i64 + y as i64)
    }

    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        acc = acc
            .wrapping_mul(7)
            .wrapping_add(stage3(black_box(i)).unwrap_or(-1));
    }

    acc
}

/// A `select` (ternary) over both integer and float operands, plus the `eqz`
/// shape that feeds a branch directly.
#[unsafe(no_mangle)]
pub extern "C" fn cf_select_and_eqz(n: i32) -> i64 {
    let mut acc = 0i64;

    for i in 0..n.clamp(1, 2_000) {
        let c = black_box(i) % 3 == 0;
        let a = if c { i as i64 } else { -(i as i64) };
        let f = if c { 1.5f64 } else { -2.5 };

        acc = acc.wrapping_mul(5).wrapping_add(a);
        acc = acc.wrapping_mul(5).wrapping_add(f as i64);
        acc = acc.wrapping_mul(5).wrapping_add((i == 0) as i64);
    }

    acc
}

/// A control-flow-heavy workload for the throughput table.
#[unsafe(no_mangle)]
pub extern "C" fn cf_mixed_workload(n: i32) -> i64 {
    let mut acc = 0i64;
    let mut i = 0;

    while i < n {
        acc = match i % 7 {
            0 => acc.wrapping_add(1),
            1 => acc.wrapping_mul(3),
            2 => acc ^ 0x5555,
            3 => acc.rotate_left(5),
            4 => {
                if acc < 0 {
                    acc.wrapping_neg()
                } else {
                    acc >> 1
                }
            }
            5 => {
                let mut inner = 0i64;
                for j in 0..4 {
                    inner = inner.wrapping_add(j);
                }
                acc.wrapping_add(inner)
            }
            _ => acc.wrapping_sub(7),
        };

        i += 1;
    }

    acc
}
