# VM execute path: performance analysis

Measured on `rustc 1.96 / LLVM 22`, aarch64-apple-darwin, release builds, on wasm compiled
from real Rust (`wasm32-unknown-unknown -O`). Reference engine: `wasmi 2.0.0-beta.9`.

All file references are `tracewasm-core/src/...`.

---

## 1. Baseline

Re-recorded after levers A, C, E, F and G. This is the reference every future optimisation
compares against.

### How to reproduce — read this before comparing

```sh
cargo test --release -p tracewasm-test --test metrics -- --nocapture --test-threads=1
```

`--test-threads=1` is **not optional**. The metric tests are timing loops, and letting
cargo run them concurrently inflates the throughput rows and pushes run-to-run spread from
~3% to ~11% — wide enough to swallow any realistic single optimisation. Numbers taken
without it are not comparable to this table.

Also: the *first* run after a rebuild is unusable (one row was observed at 232 ns/op cold
versus 145 warm on the same binary). Discard it, then take the median of ≥3 runs.

Everything below is the median of 5 warm single-threaded runs, aarch64-apple-darwin, on a
laptop — so treat <5% movement as noise, not signal.

### Throughput by workload category — the primary signal

`ns/op` here is **per guest loop iteration**, not per wasm instruction: one op is one pass
of the guest's `while i < n` body (`WORK = 20_000` iterations, `REPS = 5` invocations),
which is on the order of 10-30 wasm instructions. Do not read these as cycles-per-instruction.

| workload | median ns/op | min | max | spread |
| --- | --- | --- | --- | --- |
| arithmetic (i64 + f64 mix) | **58.39** | 57.85 | 62.25 | 7.5% |
| control flow (match + loops) | **83.39** | 81.90 | 85.16 | 3.9% |
| memory (load/store) | **61.13** | 60.55 | 61.37 | 1.3% |
| calls (one indirect per iter) | **44.51** | 43.84 | 45.03 | 2.7% |
| heap (alloc + collections) | **6.33 ms** | 6.28 | 6.36 | 1.3% |

The four `ns/op` rows are the ones to watch — each isolates a different interpreter cost
(dispatch, control flow, linear memory, indirect calls), which is what localises a
regression that a single blended number would hide.

### Host-boundary and frame cost

| measurement | median | min | max | spread |
| --- | --- | --- | --- | --- |
| trivial call, n=1 | **122.74 ns** | 119.36 | 130.87 | 9.4% |
| 5-param call | **84.16 ns** | 83.06 | 86.28 | 3.8% |
| direct recursion, depth 100 | **53.42 ns/op** | 53.23 | 55.08 | 3.5% |
| direct recursion, depth 1000 | **62.48 ns/op** | 61.36 | 68.79 | 11.9% |
| direct recursion, depth 3000 | **65.70 ns/op** | 62.56 | 66.06 | 5.3% |

Recursion cost is flat in depth (53 → 66 ns/op from depth 100 to 3000), so frame setup is
not super-linear — though these three rows carry the widest spread in the table, so read the
trend rather than the numbers. Frame cost: **1,311 B/frame, 6,396 frames** on an 8 MiB stack.

The two invocation rows are separate exports on separate guests, so the gap between them is
not the cost of extra parameters.

### Compile and instantiate

The guests are ~1.4-1.5 MB each because they link `std`, so this is dominated by lowering
the whole module, not by the small part any one test executes.

| measurement | median | min | max | spread |
| --- | --- | --- | --- | --- |
| compile + instantiate: arithmetic | **0.12 ms** | 0.11 | 0.13 | 9.3% |
| compile + instantiate: exotic | **0.53 ms** | 0.52 | 0.60 | 15.3% |
| compile + instantiate: heap | **1.13 ms** | 1.13 | 1.15 | 1.8% |

Compile is untouched by the execute-path work, as expected — it does not run the interpreter.

### Guest linear-memory growth

| measurement | median | min | max | spread |
| --- | --- | --- | --- | --- |
| allocate ~1 page | **0.00 ms** | 0.00 | 0.00 | 13.6% |
| allocate ~8 pages | **0.02 ms** | 0.02 | 0.03 | 10.6% |
| allocate ~32 pages | **0.09 ms** | 0.09 | 0.09 | 6.8% |
| vec growth to 20k elements | **2.05 ms** | 2.04 | 2.15 | 5.4% |
| 4k short-lived allocations | **8.44 ms** | 8.33 | 8.81 | 5.7% |

These move with the interpreter too: the guest's allocator is guest wasm, so growth work is
interpreted work.

The small-page rows have ~20% spread because they are microseconds — ignore them unless a
change moves them by an order of magnitude.

### Instruction-stream footprint — deterministic, no timing

Instruction count per guest × `size_of::<Instruction>()`. This has no measurement noise, so
any movement is real. Recompute it by summing `func_bodies[i].instructions.len()`.

| guest | instructions | stream @ 16 B | was @ 40 B |
| --- | --- | --- | --- |
| arithmetic | 2,626 | 41 KB | 102 KB |
| control_flow | 7,884 | 123 KB | 307 KB |
| memory | 10,501 | 164 KB | 410 KB |
| frames | 8,230 | 128 KB | 321 KB |
| heap | 39,972 | 624 KB | 1,561 KB |
| exotic | 16,695 | 260 KB | 652 KB |
| **total** | **85,908** | **1,342 KB** | **3,355 KB** |

### What the completed levers bought

Against the figures recorded before this round of work, on the same harness and machine:

| row | before | now | change |
| --- | --- | --- | --- |
| arithmetic | 142.89 | **58.39** | −59% |
| memory (load/store) | 177.33 | **61.13** | −66% |
| calls (one indirect per iter) | 116.57 | **44.51** | −62% |
| heap | 16.01 ms | **6.33** | −60% |
| control flow | 138.55 | **83.39** | −40% |
| trivial call, n=1 | 317.43 ns | **122.74** | −61% |
| 5-param call | 237.69 ns | **84.16** | −65% |

About **2.5x overall.** Attribution, from the interleaved A/Bs that measured each:

| lever | what it moved |
| --- | --- |
| G — inline dispatch | ~40% everywhere; the single largest |
| E — `Val` 16 → 8 B untagged | 7-31%, most on the stack-traffic-heavy rows |
| F — stack on the `Instance` | invocation latency, not throughput |
| §3/§5 outlining | 5-10% on `push`, ~5% on `pop` |
| A — `Instruction` 40 → 16 B | footprint only; no measurable throughput |

Per wasm instruction, dividing by the executed instruction counts in §1's method:

| workload | ns/instr | cycles @ 4.05 GHz |
| --- | --- | --- |
| arithmetic | 1.67 | **6.8** |
| memory | 1.79 | 7.3 |
| calls | 1.98 | 8.0 |
| control flow | 2.81 | 11.4 |

Three of the four are now inside the 4-8 cycle range a well-tuned stack interpreter occupies;
arithmetic is at the top of the band. Control flow is the outlier and the obvious next target.

The remaining structural levers are threaded dispatch (each arm dispatching to the next, so
the branch predictor sees per-opcode sites instead of one shared site — most likely to help
exactly the control-flow row), and operand fusion.

### What the 40 → 16 B work actually bought

**Footprint: 2.5×, measured and exact** — 2 MB across the six guests.

**Throughput: no measurable change.** A/B against `495edde` on the same harness and machine
put every row within ~2%, i.e. inside the noise band, and the sign flipped between median
and minimum on two rows. The instruction stream is evidently not the binding constraint at
these sizes. Recorded so nobody re-runs this expecting a throughput win — the memory win is
the reason to keep it.

The older figures in this section (`loop_arith` at 5.16 ns/instr, `call_heavy`, `mem_heavy`,
`locals_heavy`, `noop`) came from an ad-hoc harness that is not in the repo and cannot be
reproduced; those workload names exist nowhere in the source tree. The
cycles-per-instruction analysis just below, and §2, still derive from them — treat both as
estimates awaiting a re-measurement against a real per-instruction counter.

**~3-5 ns per wasm instruction ≈ 10-15 cycles** at 3 GHz. A well-tuned stack-machine
interpreter runs 4-6 cycles per instruction, so roughly **2-3× is available inside the
current architecture**. The remainder of the gap to wasmi is its register-machine
translation, which is a rewrite rather than an optimisation.

### On comparing against wasmi

wasmi 2.0-beta measures ~0.18 ns/instr on `loop_arith`, which is sub-cycle and therefore
impossible instruction-by-instruction. Its translator must fuse stack sequences into wider
register operations, so the raw ratio is not comparing like with like. **Optimise against
cycles-per-instruction, not against the wasmi ratio.**

For scale, wasmi's own 0.40 → 2.0 rewrite — same team, stack machine to register machine —
bought them 2-6×. That is what the architectural step is worth.

---

## 2. Anatomy of one instruction

Work performed per driver-loop iteration, for something as cheap as `local.get`:

1. `&instructions[pc]` — bounds check, then a **16-byte** load, 4 per cache line and never
   straddling one (was 40 bytes, straddling ~40% of the time; see §3A)
2. call into `TraceVMState::execute` — real call/ret
3. jump-table dispatch over a 192-variant discriminant
4. `get_local` → `stack.inner[base + i]` — `Vec` pointer load, 16-byte load. Unchecked
   (`get_unchecked` behind a `debug_assert!`), so no bounds check in release
5. `push` → `inner.len()` load, compare, branch, bounds check, 16-byte store, `sp`
   increment. The growth path is outlined, so the branch falls through to a store

Roughly **1 bounds check and 5 bookkeeping loads to move one 16-byte value.** The
measurement is consistent with the code; there is no hidden cost elsewhere.

Item 1 is lever A and is done — and notably it did *not* move throughput (§1), which is
evidence the driver loop is bound by items 2-5 rather than by instruction fetch. Items 4 and
5 are levers C and E. The plumbing that *used* to appear here — an `sret` round-trip for the
`Result`, a spilled `pc`, and a per-iteration `len()` reload — has been removed, and removing
it bought almost nothing; see §5.

---

## 3. Levers, ranked

| # | Lever | Expected | Effort | Status |
| --- | --- | --- | --- | --- |
| G | Inline dispatch into the driver loop | **~40%** | small | **done** — see §4 |
| B | Per-function metadata table | high for call-heavy | small | **next** |
| A | Shrink `Instruction` 40 → 16 B | high | mechanical | **done** — see below |
| E | `Val` 16 → 8 B untagged | high | invasive | **done** — see below |
| C | Remove redundant bounds checks | 10-25% | small | **partly done** — locals unchecked |
| F | Reuse the operand stack across calls | latency spikes only | trivial | **done** — see below |
| D | Hoist loop invariants | 5-10% | trivial | **done** — no measurable gain |

B is first because per-call work is where the slack has consistently been: the call-heavy
workload is the one that has responded most, and moving the operand stack off the per-call
path (F) cut invocation latency 12-17%.

### A. `Instruction` was 40 bytes — now 16 B (done)

Starting payload sizes:

| variant | payload |
| --- | --- |
| `CallIndirect { Box<[ValType]>, Box<[ValType]>, TableIndex }` | **40 B** ← set the size |
| `BrTable { Vec<TargetBranch> }` | 24 B |
| `If { Option<usize>, usize }` | 24 B |
| `Load { offset: u64, align: u8 }` | 16 B |
| `Br { usize, u32, u32 }` | 16 B |

The original plan predicted 24 B and called 16 B unreachable. That was wrong on both counts.
It also mis-modelled two things worth recording, since both cost a round of work:

- **`Box<[T]>` is a 16-byte fat pointer, not 8.** `[T]` is unsized, so the length rides in
  the pointer. Boxing the `CallIndirect` payload into `Box<CallIndirectMeta>` would have
  worked (sized → 8 B), but the two boxed *slices* were 16 B each, which is what made the
  variant 40 B.
- **`If` was already the widest variant at 24 B, not `CallIndirect`'s equal.** The enum
  measured 24 B rather than 32 only because `Option<usize>`'s tag has spare values, so
  rustc niche-packed the 192-variant discriminant into it for free. That made `If` look
  cheaper than it was.

What actually landed, in order — note that steps 2-4 each buy **nothing alone**, because the
size is set by whichever variant is widest at the time:

1. `CallIndirect` → `{ ty_index: TyIndex, table_index: TableIndex }` (8 B), resolving the
   signature from `module.types` at execution. **40 → 24 B.**
2. Every backpatched instruction index `usize` → `u32`, and `If`'s `Option<usize>` →
   `Option<u32>`. Bound: a function body's size is a `u32` in the binary and each operator
   is ≥1 byte, so the instruction count cannot reach `u32::MAX`. Br-family 16 → 12 B.
3. `memarg` `offset: u64` → `u32`. Loads/stores 16 → 8 B.
4. `BrTable { Vec<TargetBranch> }` → `{ start_index: u32, len: u32 }` naming a range in a
   flat per-function `Box<[TargetBranch]>` on `FuncBody`. 24 → 8 B, and the last 8-aligned
   payload is gone. **→ 16 B.**

Result: **16 B**, a 2.5× reduction in instruction-stream traffic — 4 instructions per
64-byte cache line instead of 1.6. Enforced by a `const _: () = assert!(size_of::<Instruction>() <= 16)`
next to the enum, so a regression is a compile error.

An `FxHashMap<u32, _>` keyed by `pc` was tried for step 4 first. It reached the same 16 B,
but put a hash on the `br_table` path (`match` in a hot loop is one static site with
unbounded dynamic count), needed a per-execution `.expect()`, and cost 32 inline bytes on
every `FuncBody` including the majority that have no `br_table`. The flat range wins on all
four and is the same amount of code.

16 B is now the floor: `I64Const`/`F64Const` carry 8-byte immediates at align 8, so going
lower needs a constant pool, which is not worth it.

Beyond the enum, `TargetBranch` itself went 16 → 12 B (`usize` → `u32` plus the padding it
was carrying), cutting every `br_table`'s heap array by 25% — and rustc emits wide
`br_table`s for `match`.

### B. Frame setup re-derives everything on every call

`TraceVM::execute` looks up `func_decl`, `ty`, `params_len`, `results_len`, `func_body`,
`instructions`, `instruction_offsets`, `locals_ty` and `locals_len` — a chain of dependent
loads through `Arc<Module>` into several `Box<[…]>` — **before executing a single
instruction**.

Precompute a `Box<[FuncMeta]>` at compile time and index it once per call.

This is the top lever for call-heavy work and it matters
disproportionately for real Rust output, which is full of small functions.

### C. `push` does redundant work

```rust
if self.stack_pointer < self.inner.len() {
    self.inner[self.stack_pointer] = val;   // bounds-checked store
    self.stack_pointer += 1;
} else {
    self.push_grow(val);                     // #[inline(never)]
}
```

A length load, a branch, and a bounds check on the indexed store remain. Filling the backing
`Vec` to capacity once so `len() == capacity` always would remove the branch and one length
load — but measurement says that is worth ~0: the branch is perfectly predicted, and the cost
that outlining removed was the callee-saved spilling forced by a value live across
`Vec::push`'s allocator call, not the compare. See `OUTLINING_COLD_PATHS.md`.

The same treatment is worth trying on `Stack::pop` (242 call sites) and `Val::as_*`, whose
bounds check and `panic!` are the equivalent hidden calls. Untested.

**Locals: done.** `get_local`/`set_local` now use `get_unchecked` behind a `debug_assert!`,
justified by the frame-layout invariants enumerated in `vm/mod.rs`. Worth ~3% on arithmetic,
inside noise elsewhere.

### D. Hoist loop invariants

`instructions.len()` is reloaded on every iteration of the driver loop. Bind it once before
the loop.

### E. `Val` 16 → 8 B untagged (done)

Validation proves the static type of every stack slot, so the tag is redundant *during*
execution — it is only needed at the host boundary. The operand stack now holds `Value`, a
`u64` newtype with `from_i32`/`as_i32`-style accessors that reinterpret the bits, halving the
traffic of every push, pop and local access.

Worth 7-31% depending on workload, most on the rows that move the most values (memory −31%,
calls −27%). Combined with G it took memory from 89.41 to 61.13 ns/op.

Two boundaries keep the tagged `Val`:

- **The host boundary.** `ImportRegistry::execute` still takes `&[Val]`; the interpreter
  re-tags from the callee's declared parameter types via `Value::into_val(ty)`, collected into
  a `ParamVals` so an import of five or fewer parameters costs no allocation.
- **Constant expressions.** `const_expr_evaluator` keeps its own `Stack<Val>`. It runs at
  instantiation, not on the hot path, and its results feed globals and table/data initializers
  that are typed rather than positional.

Note for anyone touching this: `Value` and `Val` both expose `as_i32()`, so mixing them up is
a logic error the type checker will not catch. The differential suite is the net that will.

### F. The 8 MiB operand-stack reservation (done)

`VM_STACK_INITIAL_ALLOCATION_SIZE = 512 * 1024` **elements** × 16 B = **8 MiB**. (The comment
saying `// 512Kib` is wrong by 16×.)

The stack is now a field on `Instance`, allocated once at instantiation and reset at each
top-level entry, rather than constructed per call. That cut per-call latency by 12-17% at
p50 and p99. Resetting on entry rather than on exit is what keeps an instance usable after a
trap, which returns early with values still on the stack.

Two consequences worth noting. The 8 MiB is now held for an instance's whole lifetime, so an
embedder holding many idle instances pays it per instance. And because `Stack::reset` leaves
the backing storage alone, the high-water mark survives across calls — growth happens on the
order of tens of times per instance rather than on every call, which is what makes outlining
it in §3C worthwhile.

Still open: 512 Ki slots is a guess. Growth is cheap and rare now, so a much smaller initial
reservation would cut the per-instance footprint at little cost.

---

## 4. Negative results — do not repeat these

Three `#[inline]` experiments produced **no measurable change**:

| attempt | result |
| --- | --- |
| `#[inline]` on the recursive call-path fns (`call_func`, both `execute`s) | declined by LLVM — recursive edge; frame sizes byte-identical |
| `#[inline]` on the dispatch fn as a plain hint | declined; throughput and frame size unchanged |
| `#[inline]` on `Stack::push`/`pop`/`tee`/`height` | no measurable change |

`#[inline(always)]` on the dispatch function is the exception, and it is now **applied**
(lever G). It is worth **~40% across every workload**, far more than the 27% first seen: the
boundary is crossed once per wasm instruction, and `pc` plus the stack's base and length
cannot stay in registers across a real call.

The frame cost is real and unavoidable: fusing ~200 opcode arms into one function puts every
arm's spill slots in a single frame, ~1,311 → ~5,100 B per nested call, cutting reachable
depth from ~6,400 to ~1,650. Outlining the cold arms does **not** recover it — the two
`format!` calls in the `call_indirect` mismatch path were outlined first and moved the frame
32 bytes, 0.6%. The bloat is spill slots the allocator cannot overlap across arms, not cold
code.

The mitigation is instead on the other side: `Config::max_call_stack_depth` defaults to 1200,
sized so a full chain stays inside a typical 8 MiB stack. The real fix is to stop nesting
native frames per wasm call at all.

A `&mut pc` refactor to remove the `ExecutionResult` round-trip gave **zero throughput
gain** (the enum was already returned in a register), but incidentally shrank the frame
1,423 → 1,311 B. It was later reverted in favour of returning the next `pc` — see §5.

**The lesson: inlining is not the bottleneck.** LLVM is already making reasonable choices.
The cost is memory traffic and redundant work per instruction, which is why A, B, C and E
are all about *doing less* rather than *calling less*.

The one route to the `inline(always)` win without the frame cost: outline the cold arms
(`CallIndirect`, the eight `trunc` variants, bulk memory ops) behind `#[inline(never)]` so
the inlined remainder is slim and the big spill slots live off the recursive path. Unproven.

---

## 5. Completed: the call boundary (boxed error + `pc` by value)

Two changes to the per-instruction boundary, both verified in the disassembly and both
**almost entirely without effect on throughput**. Recorded in full because the reasoning was
sound and the outcome was still negative — that is the useful part.

### What was wrong

`TraceVMState::execute` returned `Result<(), InstructionExecutionError>`. That error is
**56 bytes** (variants carry `String`s and a `Box<TraceWasmError>`), and aarch64 returns
anything over 16 bytes through an `sret` pointer. So *every successful instruction* paid a
56-byte stack write plus a discriminant reload, to report a failure that essentially never
happens. Separately, `pc` was lent out as `&mut usize`, which gives it an address and forces
it to a stack slot reloaded on every iteration.

### What changed

- `execute` returns the next `pc` on success rather than taking `&mut pc`, so the `pc` stays
  in a register instead of being spilled to a stack address.
- `instructions.len()` hoisted out of the loop condition (lever D).
- The error was boxed at the time, making the whole `Result` 16 B and returnable in two
  registers, with `impl From<MemoryError> for Box<InstructionExecutionError>` bridging `?` at
  the ~50 memory-access sites.

  **The box is no longer there** — it was removed in `d8ac131`, a nine-file test-suite commit,
  with no rationale recorded and no measurement either way. So `execute` currently returns a
  56-byte `Result` via `sret`. Re-boxing is an untested lever, not a completed one; the "After"
  disassembly below was captured while the box was in place.

### Verified in the disassembly

Before:

```asm
ldr  x25, [x28, #0x18]   ; instructions.len() reloaded every iteration
add  x0,  sp, #0x48      ; sret pointer — Result returned via MEMORY
add  x4,  sp, #0x40      ; &mut pc — a stack address
bl   ...execute
ldr  x8,  [sp, #0x48]    ; reload Result discriminant from memory
ldr  x26, [sp, #0x40]    ; reload pc from memory
```

After:

```asm
mov  x3, x26             ; pc passed in a register
bl   ...execute
mov  x24, x1             ; next pc returned in x1
tbnz w0, #0x0, ...        ; discriminant tested in w0
```

### Result

| workload | before | after |
| --- | --- | --- |
| `loop_arith` | 12,160 µs | 12,403 |
| `call_heavy` | 31,032 | **29,521** (−5%) |
| `mem_heavy` | 35,524 | 35,522 |
| `locals_heavy` | 24,368 | 24,266 |
| `noop` | 74 ns | 74 ns |

Frame size unchanged (1,311 B / 6,396 frames), all tests passing, differential checks match
native, depth guard still traps.

**Why it did so little:** the `sret` store and its reload target the *same hot stack slot* —
L1-resident, and store-to-load forwarding hides most of the latency. A 56-byte round trip
sounds expensive and is not.

Keep the change: it is strictly less work, it removed a spill, and it cost nothing. But it
is the fourth prediction in a row about the call boundary that did not pay — so **stop
optimising the plumbing around dispatch** and go after the work itself (A, B, E).

### Measurement trap discovered here

The depth rig initially reported a regression to 1,995 frames / 4,204 B. That was the binary
search hitting `Config::max_call_stack_depth` (default 2000), not the native stack. Any
frame-size measurement must raise the guard first, as `tests/metrics.rs` does via
`Config::set_max_call_stack_depth`.

---

## 6. Suggested order

1. **Break the native recursion.** Not a throughput lever — an enabling one. `execute` being
   inlined costs ~5 KB of frame per nested wasm call, which is why
   `Config::max_call_stack_depth` had to drop to 1200. An explicit frame stack decouples
   depth from frame size, and unblocks any further inlining.
2. **Threaded dispatch.** Each arm dispatching to the next gives the branch predictor
   per-opcode sites instead of one shared 192-target site. Most likely to help control flow,
   the one row still above 10 cycles.
3. **B** — per-function metadata table. Contained.
4. **Operand fusion** — `local.get; local.get; i32.add` as one instruction. Cuts executed
   instruction count rather than per-instruction cost, which is the other axis entirely.

A, D, E, F and G are done. D produced no measurable gain; A moved footprint but not
throughput; F moved latency but not throughput; E and G moved throughput. C is half done —
locals are unchecked; re-boxing `execute`'s error (§5) remains untested.

**Measure after each step with `--test metrics` (regression bounds) or `--test perf_report`
(distributions), not just `cargo test`.** The unit fixtures are mostly single-function with
no declared locals and went green through two real bugs in the locals-on-stack work; the
differential suite caught both in one run.

---

## Appendix: the original ad-hoc harness

**This harness is not in the repo and these workloads cannot be run.** The names below appear
nowhere in the source tree; they are kept because §3B, §4 and §5 quote their numbers. Anything
attributed to them is historical and unverifiable — see §1 for the reproducible baseline.

- Workloads: `loop_arith` (pure dispatch), `call_heavy` (`call_indirect` per iteration),
  `mem_heavy` (linear-memory load/store), `locals_heavy` (8 live locals), `noop`
  (per-invocation cost), each `n = 200_000`, 20 iterations, instantiated once outside the
  timing loop.
- ns/instr is derived by counting instructions in the disassembled loop body and accounting
  for LLVM's 4× unroll (visible as the `i32.lt_u 4` guard and the `and 2147483644` mask).
- Two traps to avoid when writing these: LLVM devirtualises a `fn`-pointer call and folds
  the loop away unless the callee index depends on the running value (an early
  `call_heavy` reported 0.00 ms); and accumulator recursion is rewritten into a `loop`, so
  depth tests need an indirect call to stay a genuine call chain.
- Profiling used macOS `sample`. Its output is a cumulative call tree, so leaf attribution
  is unreliable — the structural findings above come from type sizes and disassembly, not
  from profile self-time.
