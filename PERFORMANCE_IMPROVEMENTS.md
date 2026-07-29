# VM execute path: performance analysis

Measured on `rustc 1.96 / LLVM 22`, aarch64-apple-darwin, release builds, on wasm compiled
from real Rust (`wasm32-unknown-unknown -O`). Reference engine: `wasmi 2.0.0-beta.9`.

All file references are `tracewasm-core/src/...`.

---

## 1. Baseline

| workload | tracewasm | ns/instr | what it isolates |
| --- | --- | --- | --- |
| `loop_arith` | 11.9 ms | **5.16** | pure dispatch + i64 arithmetic |
| `call_heavy` | 31.0 ms | — | `call_indirect` per iteration |
| `mem_heavy` | 35.5 ms | — | load/store through linear memory |
| `locals_heavy` | 25.1 ms | **3.14** | 8 live locals, frame-layout bound |
| `noop` | 74 ns | — | per-invocation setup/teardown |

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

1. `&instructions[pc]` — bounds check, then a **40-byte** load (straddles a cache line ~40%
   of the time)
2. call into `TraceVMState::execute` — 5 arguments, real call/ret
3. jump-table dispatch over a 192-variant discriminant
4. `get_local` → `stack.inner[base + i]` — `Vec` pointer load, bounds check, 16-byte load
5. `push` → `inner.len()` load, compare, branch, **second** bounds check, 16-byte store,
   `sp` increment
6. `ExecutionResult` match → `*pc += 1`
7. loop condition → another `instructions.len()` load

Roughly **2 bounds checks and 5 bookkeeping loads to move one 16-byte value.** The
measurement is consistent with the code; there is no hidden cost elsewhere.

---

## 3. Levers, ranked

| # | Lever | Expected | Effort |
| --- | --- | --- | --- |
| A | Shrink `Instruction` 40 → 24 B | high | mechanical |
| B | Per-function metadata table | high for call-heavy | small |
| C | Remove redundant bounds checks | 10-25% | small |
| D | Hoist loop invariants | 5-10% | trivial |
| E | `Val` 16 → 8 B untagged | high | invasive |
| F | Reuse the operand stack across calls | latency spikes only | trivial |

### A. `Instruction` is 40 bytes, and one variant causes it

Measured payload sizes:

| variant | payload |
| --- | --- |
| `CallIndirect { Box<[ValType]>, Box<[ValType]>, TableIndex }` | **40 B** ← sets the size |
| `BrTable { Vec<TargetBranch> }` | 24 B |
| `If { Option<usize>, usize }` | 24 B |
| `Load { offset: u64, align: u8 }` | 16 B |
| `Br { usize, u32, u32 }` | 16 B |

Fixes, in order:

1. Box the `CallIndirect` payload into one `Box<CallIndirectMeta>` → 8 B.
2. `Vec<TargetBranch>` → `Box<[TargetBranch]>` → 16 B.
3. `Option<usize>` → a `usize::MAX` sentinel — the idiom already used for backpatching
   (`instruction.rs` module docs) → 16 B.

Result: **24 B**, a 1.67× reduction in instruction-stream traffic, 2.7 instructions per
cache line instead of 1.6.

16 B is *not* reachable without side tables, because `Br { target_index, arity,
recorded_height }` needs 16 B of payload by itself and the discriminant pushes the enum to
24 B.

Also free while in there: `offset: u64` on every load/store can be `u32`. memory64 is
unsupported, and `pop_effective_address` already traps with `OffsetTooLarge` above `u32`.

### B. Frame setup re-derives everything on every call

`TraceVM::execute` looks up `func_decl`, `ty`, `params_len`, `results_len`, `func_body`,
`instructions`, `instruction_offsets`, `locals_ty` and `locals_len` — a chain of dependent
loads through `Arc<Module>` into several `Box<[…]>` — **before executing a single
instruction**.

Precompute a `Box<[FuncMeta]>` at compile time and index it once per call.

This is the top lever for `call_heavy` (31 ms, the worst absolute number) and it matters
disproportionately for real Rust output, which is full of small functions.

### C. `push` does redundant work

```rust
if self.stack_pointer < self.inner.len() {
    self.inner[self.stack_pointer] = val;   // bounds-checked store
} else {
    self.inner.push(val);                    // capacity check + maybe realloc
}
```

That is a length load, a branch, **and** a bounds check on the indexed store. Filling the
backing `Vec` to capacity once at construction so `len() == capacity` always removes the
branch permanently and one length load with it.

Same for `get_local`/`set_local` (`vm/mod.rs`): validation already guarantees the index is
in range — the invariant already documented for indexing `inner` directly — so
`get_unchecked` behind a `debug_assert!` is defensible.

### D. Hoist loop invariants

`instructions.len()` is reloaded on every iteration of the driver loop. Bind it once before
the loop.

### E. `Val` is 16 bytes because it is tagged

Validation proves the static type of every stack slot, so the tag is redundant *during
execution* — it is only needed at the host boundary. An 8-byte untagged cell halves traffic
on every push, pop and local access.

Biggest remaining win and the most invasive. This is what wasmi does internally.

### F. `Stack::default()` reserves 8 MiB per top-level call

`VM_STACK_INITIAL_ALLOCATION_SIZE = 512 * 1024` **elements** × 16 B = **8 MiB**, allocated
fresh in `TraceVM::run` on every top-level call. (The comment saying `// 512Kib` is wrong by
16×.)

Invisible in these benchmarks because the allocator recycles the block, but it is 8 MiB of
footprint per concurrent execution and a latency cliff whenever pages actually fault. Hang
one stack off the `Instance` and reuse it.

---

## 4. Negative results — do not repeat these

Three `#[inline]` experiments produced **no measurable change**:

| attempt | result |
| --- | --- |
| `#[inline]` on the recursive call-path fns (`call_func`, both `execute`s) | declined by LLVM — recursive edge; frame sizes byte-identical |
| `#[inline]` on the dispatch fn as a plain hint | declined; throughput and frame size unchanged |
| `#[inline]` on `Stack::push`/`pop`/`tee`/`height` | no measurable change |

`#[inline(always)]` on the dispatch function **did** work — 27% off `call_heavy` and
`mem_heavy` — but cost **3.7× in frame size** (1,423 → 5,326 B), dropping max call depth
from 5,895 to 1,575 frames, below the original baseline. Bad trade.

A `&mut pc` refactor to remove the `ExecutionResult` round-trip gave **zero throughput
gain** (the enum was already returned in a register), but incidentally shrank the frame
1,423 → 1,311 B, so it was kept for depth.

**The lesson: inlining is not the bottleneck.** LLVM is already making reasonable choices.
The cost is memory traffic and redundant work per instruction, which is why A, B, C and E
are all about *doing less* rather than *calling less*.

The one route to the `inline(always)` win without the frame cost: outline the cold arms
(`CallIndirect`, the eight `trunc` variants, bulk memory ops) behind `#[inline(never)]` so
the inlined remainder is slim and the big spill slots live off the recursive path. Unproven.

---

## 5. Suggested order

1. **B and D** — small, safe, and B targets the worst workload.
2. **A** — mechanical, but touches lowering plus every arm reading those fields.
3. **C**.
4. **E** — only after measuring the first four; it may look different once the instruction
   stream is 24 B.

---

## Appendix: reproducing the measurements

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
