# Testing plan: TraceWasm for Rust-compiled WebAssembly

Scope: TraceWasm exists to run **wasm compiled from Rust programs**, not to be a
spec-compliance engine. This document establishes empirically what that target actually
requires, and plans the testing work accordingly.

Companion document: [`SPEC_TESTSUITE_PLAN.md`](./SPEC_TESTSUITE_PLAN.md) covers the full
spec-compliance path. This file supersedes it for the Rust-focused goal; §7 lists what
changed and why.

All file references are `tracewasm-core/src/...` unless otherwise noted.

---

## 1. Headline: the engine already runs real Rust wasm

Measured, not assumed. A realistic Rust program — `Vec`, `sort_unstable`, `dedup`,
`HashMap`, `BTreeMap`, `format!`, `&dyn Debug`, `String::join`, `f64::sqrt/sin`, iterators
— compiled with rustc 1.96 / LLVM 22 and run through `Module::compile` → `instantiate` →
`TypedFunc::call`:

| binary | compile | instantiate | `work(50)` |
| --- | --- | --- | --- |
| `wasm32-unknown-unknown`, `-O` | OK | OK | **96** |
| `wasm32-unknown-unknown`, `-Cdebuginfo=2` | OK | OK | **96** |
| `wasm32-wasip1`, `-O` | OK | needs 5 WASI imports | — |
| `wasm32-wasip1`, `-Cdebuginfo=2` | OK | needs 5 WASI imports | — |

Native Rust, same source: `work(50) = 96`. **Exact match.** `div(5,0) = -1` and
`floaty(2.0) = 1414213562` also match native.

A Rust `panic!` (index out of bounds) surfaces correctly as an `unreachable` trap, carrying
a full 10-frame wasm call chain from the panic site back to the entry point — which is the
whole point of this engine.

`wasm32-wasip1` fails only for want of a WASI host module. That is an embedder question,
not an engine gap.

---

## 2. What rustc actually emits

`rustc 1.96 / LLVM 22`, `wasm32-unknown-unknown`, from the `target_features` custom section:

```
+bulk-memory  +bulk-memory-opt  +call-indirect-overlong  +multivalue
+mutable-globals  +nontrapping-fptoint  +reference-types  +sign-ext
```

(`wasm32-wasip1` adds `+extended-const`.)

So **plain MVP is not the target** — rustc has not emitted MVP-only wasm for years. But the
*enabled feature set* overstates the case. Enumerating every opcode actually present in the
emitted code, the post-MVP instructions are only:

| opcode | feature | TraceWasm status |
| --- | --- | --- |
| `i32.extend8_s`, `i32.extend16_s` | sign-ext | **implemented** |
| `i32.trunc_sat_f64_s/u`, `i64.trunc_sat_f64_s` | nontrapping-fptoint | **implemented** |
| `memory.copy` | bulk-memory | **implemented** |
| `memory.fill` | bulk-memory | **implemented** |
| `select` (untyped) | MVP | **implemented** |

That is the entire post-MVP surface. Everything on it already works, which is exactly why
the end-to-end run above succeeds.

What is **enabled but not exercised**:

- `reference-types` — no `ref.*` opcode is emitted, and element segments use the MVP
  `func` item form. Its only real effect is the `call-indirect-overlong` encoding (table
  index as a proper LEB rather than a reserved zero byte), which wasmparser decodes and
  which is proven working: an indirect-call recursion test ran correctly.
- `multivalue` — **0** functions with more than one result in any binary tested. Worth a
  watch item, not work: the interpreter core already handles multi-value blocks and returns;
  only the `TypedFunc` surface caps results at 3.

Module shape, for reference: 1 table (`34 34 funcref`), 1 memory (17 pages), 3 globals,
1 active element segment, 1 data segment, no start section, no imports
(`wasm32-unknown-unknown`).

---

## 3. Therefore: what to drop

The spec-compliance plan called for a `Store` refactor, a linker, `spectest`, externref, 8
table instructions, and an error-classification split. Against this target, **almost all of
it is unnecessary.**

| dropped | why |
| --- | --- |
| SIMD (59 files) | not emitted unless `-Ctarget-feature=+simd128` |
| GC (17), exceptions (4), threads, relaxed-SIMD (7) | not emitted by rustc |
| memory64 / table64 (25) | not a `wasm32-*` target concern |
| multi-memory (41) | rustc emits exactly one memory |
| tail-call | not emitted |
| externref, `select t` | no `ref.*` opcodes emitted |
| `table.get/set/size/grow/fill/copy/init`, `elem.drop` | not emitted; `call_indirect` is the only table op rustc uses, and it works |
| **`Store` + linker + imported tables/memories** | rustc emits no table/memory imports. This was the single most expensive item on the old plan. |
| **`spectest` host module** | only needed by files that import it — all of which are being skipped |
| **`assert_invalid` / `assert_malformed` split** | 4,652 assertions that test *wasmparser*, not TraceWasm. Validation is delegated to `Validator::validate_all` (`module/mod.rs:957`); asserting on it is testing a dependency. |

Skipping the two validation directives is the biggest simplification: it removes the
`TraceWasmError::Parsing` → `{Malformed, Invalid}` refactor entirely.

---

## 4. What is genuinely broken — found by testing, not by reading

### 4.1 Process abort at ~2–4k call frames — **fixed**

Calls are recursive native Rust calls, and originally had no depth counter, no stack probe
and no fuel. Measured with recursion through a function pointer (LLVM cannot flatten an
indirect call, so these are genuine wasm frames), the process aborted with
`fatal runtime error: stack overflow` somewhere between **2,000 and 4,000 wasm frames** on a
default 8 MB main thread — a `SIGABRT`, not a catchable error.

`call_func`'s local-callee branch now checks the running depth against
`Config::max_call_stack_depth` before recursing and returns
`TraceWasmError::CallStackExhausted`. The default is 2,000 frames. Imported callees run on
the host's own stack and are not counted.

What remains: the guard is a fixed frame count, not a measurement of native stack left.
2,000 is sized for a release build on a normal main-thread stack — a debug build costs
roughly 30 KB of native stack per wasm frame, so raising the limit, or running on a small
thread stack, can still overflow underneath it. The recursion tests spawn a large-stack
thread for that reason.

Worth keeping in mind when writing depth tests: the first version of this one appeared to
survive 2M frames because LLVM had turned it into a `loop`. Accumulator recursion gets
flattened; indirect and tree recursion do not.

### 4.2 Calling ergonomics

`TypedFunc` requires static generics with params arity ≤ 5, results arity ≤ 3, and `WasmTy`
only for `i32/i64/f32/f64` (`instance/traits.rs:30-85`). There is no
`invoke(name, &[Val]) -> Vec<Val>` path, and **no way to read an exported global**
(`Export::Global` exists but only `to_func()` has an accessor, `module/mod.rs:780`).

Any test harness needs the dynamic path. So does any embedder that does not know export
signatures at compile time.

### 4.3 Trap messages

Fine for humans, not machine-matchable, and two distinct traps are collapsed:
divide-by-zero and `INT_MIN / -1` overflow are both
`InstructionExecutionError::Division` (`error.rs:293`), separable only by parsing
interpolated operands. Worth splitting on its own merits.

### 4.4 Smaller items

- `Validator::new()` (`module/mod.rs:957`) uses `WasmFeatures::default()`, which enables GC,
  SIMD, memory64, threads, exceptions and the component model — far more than the second
  parse pass can represent. Pin the feature set to what rustc emits, so unsupported input
  is rejected at validation with a clear message instead of surfacing as `Unsupported` from
  a deeper layer.
- `Config::max_memory_size_in_pages` defaults to 1000 (`instance/config.rs:27`) ≈ 64 MB.
  Test binaries start at 17 pages, but any real workload will want more.
- `Config::max_locals_per_func` is never read anywhere in the repo.
- `table64` input **panics** rather than erroring: `.as_i32()` on an `i64` const-expr result
  (`module/mod.rs:1806`, panic at `vm/stack.rs:108`). 64-bit *memory* no longer reaches that
  path — `Module::compile` rejects it as `Unsupported("64-bit memory")` — but tables are not
  guarded. Not reachable from
  `wasm32-*`, but it is a panic on untrusted input.
- `cargo build` fails at the workspace root: `tracewasm-scratch` is a `cdylib` guest crate
  that will not link for the host target. Blocks any CI.

---

## 5. Testing strategy

The spec suite is the *third* most valuable harness for this goal, not the first.

### Tier 1 — Differential testing against native Rust (highest value)

Compile the same Rust source twice — natively and to `wasm32-unknown-unknown` — run both,
compare. This is what proved the engine works above, and it tests exactly the code paths
Rust actually generates, weighted by how often it generates them.

Build a corpus that exercises what real programs do: collections, string formatting,
iterator chains, float math, `dyn` dispatch, panics and their unwind paths, `Drop` order,
integer overflow in debug vs release, allocator churn, deep recursion. Assert on return
values *and* on trap behaviour.

### Tier 2 — Differential against another engine

Run the same `.wasm` through `wasmi` (pure Rust, easy to embed) and compare. Catches cases
where native and wasm legitimately differ, and gives a reference for trap semantics without
having to encode spec strings.

### Tier 3 — A curated spec-suite subset

Systematic coverage of numeric and memory edge cases that a hand-written corpus will not
find: float rounding, conversion boundaries, endianness, address wrapping, trap conditions.

Include (~55 files) — the ops rustc emits, in files that need no imports or linking:

- **Numeric**: `i32 i64 f32 f64 f32_cmp f64_cmp f32_bitwise f64_bitwise conversions
  int_exprs int_literals float_exprs float_literals float_misc const`
- **Memory**: `address align endianness load store memory memory_trap memory_redundancy
  memory_size float_memory memory_copy memory_fill memory_init`
- **Control flow**: `block br br_if br_table call call_indirect if loop labels return
  switch unreachable unwind forward fac stack left-to-right select nop func local_get
  local_set local_tee`
- **Structure**: `start data type id token binary binary-leb128 custom inline-module
  comments names traps skip-stack-guard-page`

Exclude: everything SIMD/GC/threads/exceptions/memory64/multi-memory, plus
`imports exports linking table_* elem bulk ref_* globals` (need linking or table ops).

Caveat worth knowing: the testsuite root is a *flattened* mirror, so numeric-suffixed files
(`load0`, `linking0`, `memory_size0`…) are the **multi-memory** variants, and the only
`memory_grow.wast` in the root is the multi-memory one. Pull single-memory versions from
`WebAssembly/spec` `test/core/` if you want them.

Harness scope: handle only `module`, `invoke`, `assert_return`, `assert_trap`,
`assert_exhaustion`. Skip `assert_invalid`, `assert_malformed`, `assert_unlinkable`,
`register`, `module definition`, `module instance`.

Two implementation details that bite:
- Non-NaN float comparison must be **bitwise** (`to_bits()`), never `==` — the suite
  distinguishes `+0.0` from `-0.0` and asserts exact NaN payloads.
- `nan:canonical` = the canonical quiet NaN payload, sign ignored; `nan:arithmetic` = any
  NaN with the quiet bit set. TraceWasm does no NaN canonicalization, which is correct, and
  its `min`/`max` NaN result is the canonical quiet NaN — both patterns are satisfiable as-is.

---

## 6. Plan

### Phase 1 — Fix the abort, unblock CI (small, do first)

1. Exclude `tracewasm-scratch` from workspace default members (or gate on
   `target_arch = "wasm32"`) so `cargo build`/`cargo test` work at the root.
2. ~~Add `Config::max_call_depth`, a depth counter in the VM call path, and a
   `CallStackExhausted` error.~~ **Done** — shipped as `Config::max_call_stack_depth`;
   see §4.1.
3. Raise `Config::max_memory_size_in_pages`; either enforce or delete
   `max_locals_per_func`.
4. Pin `WasmFeatures` to the set rustc emits.

### Phase 2 — Dynamic API

1. `invoke(name, &[Val]) -> Result<Vec<Val>, Trap>`.
2. Exported-global read (and memory/table accessors while there).
3. Split `Division` into divide-by-zero and overflow; align trap message text.
4. Relax `FuncCallError::new`'s `debug_assert!` that the cause is always
   `InstructionExecution` (`error.rs:206`) — a dynamic path can fail otherwise.

### Phase 3 — Tier 1 differential harness — **built**

Native-vs-wasm corpus. `tracewasm-test/build.rs` compiles each `guests/*.rs` program to
`wasm32-unknown-unknown`, and `tests/differential.rs` `include!`s the same sources natively,
so every guest has a ground truth that cannot drift from the wasm under test — 23 cases
today. Remains the main ongoing investment: the corpus grows with the guests.

### Phase 4 — Tier 3 spec subset

Add `wast` as a dev-dependency, vendor the testsuite as a submodule, run the curated list
with an expectations file so it gates CI from day one.

### Phase 5 — Optional, on demand

Tier 2 differential vs `wasmi`. WASI host module if `wasm32-wasip1` becomes a target.
Multi-value if a future rustc starts emitting it (the interpreter core is already ready;
only `TypedFunc` needs widening).

---

## 7. What changed from the spec-compliance framing

| `SPEC_TESTSUITE_PLAN.md` | this plan |
| --- | --- |
| `Store` refactor + linker + `spectest` | **dropped** — rustc emits no table/memory imports |
| externref, 8 table instructions, `select t` | **dropped** — no `ref.*` opcodes emitted |
| error-classification split for `assert_invalid`/`assert_malformed` | **dropped** — tests wasmparser, not TraceWasm |
| SIMD / GC / memory64 / multi-memory / threads / exceptions | **dropped** |
| 257 spec files | ~55, as third-tier coverage |
| depth guard as "Phase 2, for `assert_exhaustion`" | was **Phase 1, top priority** — a process-abort bug at ~2–4k frames; now fixed, see §4.1 |
| spec suite as primary harness | differential-vs-native as primary |

Net: the expensive architectural work comes off the table, and the one item that was
under-weighted — the stack overflow — moves to the front. Your instinct was right that most
of the suite is unnecessary; the correction is that the target is not MVP but
"MVP + sign-ext + nontrapping-fptoint + bulk-memory subset", all of which already works.
