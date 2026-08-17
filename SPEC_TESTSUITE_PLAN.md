# Running the WebAssembly spec testsuite on TraceWasm

An audit of the current engine against [WebAssembly/testsuite](https://github.com/WebAssembly/testsuite),
and a phased plan for what has to be added.

All file references are `tracewasm-core/src/...` unless otherwise noted.

---

## 1. What exists today

**43k LOC**, 5-crate workspace; `tracewasm-core` is the engine.

Pipeline:

```
bytes
  → Module::compile          wasmparser::Validator::validate_all, then a second owned-parse pass
  → Vec<Instruction>         192 variants; structured control flow resolved to absolute PCs,
                             operand-stack heights precomputed at lowering time
  → Instance<M, I>           compiled module + Memory + ImportRegistry
  → TypedFunc<P, R>::call
```

186 tests pass. Fixtures are checked-in `.wasm` binaries pulled in with `include_bytes!`.
**There is no `.wat`/`.wast` anywhere in the tree and no `wast`/`wat` dependency.**

### Already spec-correct

These were checked against the spec rather than assumed:

- Integer div/rem trap conditions, including the subtle `INT_MIN % -1 == 0` non-trap
  (`runtime/mod.rs`).
- `f32`/`f64` `min`/`max` written longhand with the correct NaN and `±0` tie rules;
  `nearest` uses `round_ties_even`; `copysign` is a real sign transplant
  (`runtime/mod.rs`).
- Non-saturating `trunc` uses a half-open range with `f32→f64` promotion so bounds stay
  exact (`runtime/mod.rs`); `trunc_sat` maps NaN→0.
- Memory bounds checks `checked_add` *before* any byte movement, so a trapping
  `fill`/`copy` leaves memory untouched (`memory/linear.rs:82-171`); `copy_within` gives
  correct overlap semantics for `memory.copy`.
- Multi-value blocks and returns work in the interpreter core; `br_table` correctly carries
  per-arm height/arity so mixed loop/non-loop targets work.
- `start` runs at the end of instantiation; active data/elem segments are bounds-checked
  there.
- No NaN canonicalization anywhere — which is the right default. Host float ops produce
  arithmetic NaNs, and the `min`/`max` NaN result is the canonical positive quiet NaN, so
  both `nan:canonical` and `nan:arithmetic` assertions are satisfiable without extra work.

### Immediate friction

None outstanding. `cargo build` at the workspace root succeeds: `tracewasm-scratch` is a
`cdylib` guest crate that will not link for the host target, and it is not a workspace
member, so nothing tries to.

---

## 2. The gap, in the order it will block you

### 2.1 No harness, and no API a harness could use

`vm` is `pub(crate)`. The only public call path is `TypedFunc<P, R>::call` with **static
generics**: params arity ≤ 5, results arity ≤ 3, and `WasmTy` implemented only for
`i32/i64/f32/f64` (`instance/traits.rs:30-85`). A `.wast` runner needs
`invoke(name, &[Val]) -> Result<Vec<Val>, Trap>`.

There is **no way to read an exported global**. `Export::Global` exists but only
`to_func()` has an accessor (`module/mod.rs:780`), and `Instance` exposes nothing but
`memory_view()`. `exports.wast`, `linking.wast` and `globals.wast` need `(get "g")`.

`Module::compile` fuses decode and validate into a single
`TraceWasmError::Parsing(String)` (`error.rs:112`). `assert_malformed` (1,940 assertions)
and `assert_invalid` (2,712) must be distinguishable, so this has to split.

### 2.2 No linking — the architectural crux

- `ImportRegistry` is **not object-safe**: `execute<V: MemoryView>` is generic
  (`instance/traits.rs:358`). No `Box<dyn ImportRegistry>`, so a registry cannot be
  assembled at runtime from a set of already-instantiated modules.
- **Table and memory imports are rejected outright**:
  `Unsupported("only function or global imports allowed")` (`module/mod.rs:1046-1050`).
- `Instance<M, I>` **owns its memory by value**, and imported globals are pulled as a
  one-time *copy* at instantiation (`module/mod.rs:1650-1678`). Consequences:
  `global.set` on an imported mutable global is invisible to the host, and two instances
  cannot share a memory.

`spectest` requires an imported `memory 1 2` (used 80×) and `table 10 20 funcref` (48×).
The suite has 78 `register` directives, and `linking.wast` reaches *back* to earlier
instances across intervening module definitions, so instances must stay alive rather than
being replaced.

**This needs a `Store`-style design** — memories, tables and globals owned in arenas by the
store, instances holding indices into it. This is how wasmi and wasmtime solve exactly this
problem. It is the largest single change on the list, and `linking.wast`, `imports.wast`,
`exports.wast` and `instance.wast` all depend on it.

### 2.3 Reference types are half-present; tables are read-only

- `Val::Ref(Option<FuncIndex>)` (`runtime/value.rs`) is **funcref-only — there is no
  externref payload**. Worse, `Val::has_ty` matches any `ValType::Ref(_)` against any
  `Val::Ref` (`runtime/value.rs`), so funcref and externref are indistinguishable at
  runtime.
- **Zero table instructions exist.** `table.get`, `table.set`, `table.size`, `table.grow`,
  `table.fill`, `table.copy`, `table.init` and `elem.drop` have no `Instruction` variant
  and no lowering arm — they fall through to the lowering match's catch-all and hit
  `Unsupported` at *compile* time
  (`instruction/mod.rs:2264`), so `table_*.wast`, `elem.wast`, `bulk.wast` and `ref_*.wast`
  never load at all.
- `select t` (`Operator::TypedSelect`) is likewise unsupported, which takes down plain
  `select.wast` too.
- `ElementVal` state is built at instantiation and stored on the `Instance`
  (`instance/mod.rs:35`) but **never passed to the VM** — dead state waiting for
  `table.init` / `elem.drop`.

### 2.4 Call-depth guard — **done**

Calls are still recursive native Rust calls (`runtime/mod.rs`), but the local-callee branch
of `call_func` now tests the running depth against `Config::max_call_stack_depth` before
recursing and returns `TraceWasmError::CallStackExhausted` (`runtime/mod.rs`). The
default limit is **2000** frames (`instance/config.rs:30`), and the error's `Display` text
deliberately contains the substring `call stack exhausted` so `assert_exhaustion` matches
it verbatim (`error.rs:59`). Imported callees run on the host's own stack and are
excluded from the count.

So the 15 `assert_exhaustion` assertions across `call.wast`, `call_indirect.wast`,
`fac.wast` and `skip-stack-guard-page.wast` now produce an ordinary trap instead of
killing the test process. `Config` has four knobs now; `max_locals_per_func` is still
never read anywhere in the repo.

What is left: the guard is a fixed *frame count*, not a measurement of native stack
remaining, and there is still no stack probe and no fuel. 2000 is sized for a release
build on a normal main-thread stack — a debug build costs roughly 30 KB of native stack
per wasm frame, so a runner that raises the limit, or runs on a small thread stack, can
still overflow underneath the guard. The suite's recursion tests spawn a large-stack
thread for exactly this reason.

### 2.5 Trap messages do not match, and two traps are collapsed

The suite matches with `actual.contains(expected)`. Current text is TraceWasm prose:

| current | spec expects |
| --- | --- |
| `reached an \`unreachable\` instruction` | `unreachable` |
| `table slot out of bounds` | `undefined element` |
| `null element in the table slot` | `uninitialized element` |
| `function signature mismatch: ...` | `indirect call type mismatch` |
| `out of bounds access: ...` | `out of bounds memory access` |
| `division failed: {num}/{deno}` | `integer divide by zero` **or** `integer overflow` |
| `float truncation of \`{0}\` to {1} failed` | `invalid conversion to integer` / `integer overflow` |

Critically, divide-by-zero and `INT_MIN / -1` overflow are both
`InstructionExecutionError::Division` (`error.rs:293`), distinguishable only by parsing the
interpolated operands. `Division` needs splitting, and a variant→spec-string mapping layer
is required.

### 2.6 Validator features are unpinned, which mis-classifies errors

`Validator::new()` (`module/mod.rs:957`) means `WasmFeatures::default()`, which enables GC,
SIMD, memory64, threads, exceptions **and the component model**. The second parse pass then
rejects most of that as `Unsupported`.

Net effect for the suite: `assert_invalid` cases can pass *for the wrong reason*, and
modules the suite considers valid come back as `Unsupported` — which the runner must not
score as a legitimate failure. Features have to be pinned explicitly, and `Unsupported`
must stay distinct from `Invalid`.

### 2.7 Limits and smaller items

- `Config::max_memory_size_in_pages` defaults to **1000** (`instance/config.rs:27`);
  `memory_grow.wast` grows toward 65536. The runner needs the full ceiling.
- Multiple memories are rejected in three places (`module/mod.rs:1470`, `module/mod.rs:1857`,
  and `Instruction::check_memory_index`, `instruction/mod.rs:1288`). Note the flattened
  testsuite root contains **41 multi-memory
  files** — the numeric-suffixed `address0`, `load0..2`, `linking0..3` etc. are the
  multi-memory variants of the same-named core tests, not fragments — *including* the only
  `memory_grow.wast`.
- `memory64` / `table64` will **panic, not error**: `.as_i32()` on an `i64` const-expr
  result (`module/mod.rs`, panic at `runtime/stack.rs`).
- `data_count` and `code_sec_count` are recorded but never cross-checked against the
  actual section lengths (`module/mod.rs:446`). wasmparser covers this today, but the
  second pass does not.

---

## 3. Scale of the target

257 `.wast` files in the testsuite root (plus `custom/` 3, `legacy/` 4, `proposals/` 24).

Directive census over the root suite:

| directive | count |
| --- | --- |
| `assert_return` | 52,636 |
| `assert_trap` | 4,977 |
| `assert_invalid` | 2,712 |
| `module` | 2,247 |
| `assert_malformed` | 1,940 |
| `assert_unlinkable` | 200 |
| `register` | 78 |
| `assert_exception` | 18 |
| `assert_exhaustion` | 15 |

Of the 2,247 `module` directives: 1,296 are `(module quote ...)`, 1,069 are
`(module binary ...)`, 17 `module definition`, 3 `module instance`.

Feature split: **59 SIMD**, **41 multi-memory**, **25 memory64/table64**, **17 GC**,
**8 bulk-memory**, **7 relaxed-SIMD**, **4 exceptions**.

**Only ~53 files are pure-MVP.** Reference types, bulk memory, multi-value, sign-extension
and saturating float→int are effectively mandatory for any meaningful fraction of the
modern root suite — the pre-2020 "MVP-only subset" no longer exists.

Two pieces of good news:

- **84% of `assert_invalid` is the single string `type mismatch`** (2,286 of 2,712).
- Most `assert_malformed` assertions are *text*-parser errors (`malformed UTF-8 encoding`
  714×, `unknown operator` 555×, `unexpected token` 146×) that the `wast` crate produces for
  free. Only the binary-decoder family (`magic header not detected`,
  `unknown binary version`, `section size mismatch`,
  `integer representation too long`, `unexpected end of section or function`,
  `function and code section have inconsistent lengths`, `data count section required`,
  `malformed limits flags`, `illegal opcode`) is TraceWasm's responsibility.

`assert_unlinkable` has only **two** distinct strings: `incompatible import type` (184×)
and `unknown import` (16×). `assert_exhaustion` has exactly one: `call stack exhausted`.

### The `spectest` host module

Every export the suite may import, with exact types:

| export | kind | type / value |
| --- | --- | --- |
| `print` | func | `[] -> []` |
| `print_i32` | func | `[i32] -> []` |
| `print_i64` | func | `[i64] -> []` |
| `print_f32` | func | `[f32] -> []` |
| `print_f64` | func | `[f64] -> []` |
| `print_i32_f32` | func | `[i32, f32] -> []` |
| `print_f64_f64` | func | `[f64, f64] -> []` |
| `global_i32` | global | **immutable** `i32` = `666` |
| `global_i64` | global | **immutable** `i64` = `666` |
| `global_f32` | global | **immutable** `f32` = `666.6` |
| `global_f64` | global | **immutable** `f64` = `666.6` |
| `table` | table | `funcref`, i32-indexed, `min 10`, `max 20`, all slots `ref.null func` |
| `table64` | table | `funcref`, i64-indexed, `min 10`, `max 20` |
| `memory` | memory | i32-indexed, `min 1`, `max 2` pages |

Immutability is load-bearing: `global.wast` asserts that `global.set` on an imported
`spectest` global is **invalid** with `immutable global`. Any *other* name must resolve to
nothing, producing `unknown import` — `imports.wast` imports `"spectest" "unknown"` 9× on
purpose.

---

## 4. Plan

### Phase 0 — Build and runner skeleton

- ~~Exclude `tracewasm-scratch` from workspace default members~~ — done; it is not a
  member, and `cargo build`/`cargo test` work at the root.
- Add `wast = { version = "254", default-features = false, features = ["wasm-module"] }`
  as a dev-dependency.
- Vendor `WebAssembly/testsuite` as a git submodule.
- Write a directive-walking runner that emits pass/fail/skip counts plus an
  **expectations file** (a checked-in known-failures list), so it gates CI from day one
  rather than only after everything works.
- Two correctness traps to get right up front:
  - Use `QuoteWat::to_test()` and distinguish `Binary` from `Text`. A `Text` result means
    the *text* parse is what should fail (and may not even be valid UTF-8, per the
    `utf8-*.wast` files). Conflating the two makes ~1,300 quote-module assertions
    unrunnable.
  - A module that fails `assert_malformed`/`assert_invalid`/`assert_unlinkable` must
    **not** disturb the "current instance" pointer.
- Non-NaN float comparison must be **bitwise** (`to_bits()`), never `==`, since the suite
  distinguishes `+0.0` from `-0.0` and asserts exact NaN payloads.

### Phase 1 — Public API

- Make a dynamic call path: `invoke(name, &[Val]) -> Result<Vec<Val>, Trap>`.
- Add exported-global / memory / table accessors on `Instance`.
- Split `Module::compile` into `decode` + `validate`, with distinct
  `TraceWasmError::{Malformed, Invalid}` replacing the fused `Parsing`.
- Pin `WasmFeatures` explicitly; keep `Unsupported` distinct from `Invalid` so the runner
  can score it as *skip*, not *fail*.
- Raise `Config` limits to real spec ceilings.
- Relax `FuncCallError::new`'s `debug_assert!` that the cause is always
  `InstructionExecution` (`error.rs:504`) — a dynamic path can fail for other reasons.

### Phase 2 — Depth guard and trap mapping

Cheap, and it stops the harness from aborting, so do it *before* Phase 3.

- Add `Config::max_call_depth` and a real `CallStackExhausted` trap in the VM's recursive
  call path.
- Split `InstructionExecutionError::Division` into divide-by-zero and overflow.
- Add a variant→spec-string mapping layer, including the fuzzy escape hatches Wasmtime
  needs: `uninitialized element 2` accepting `uninitialized element`, and the GC
  `null array/structure/i31 reference` family accepting plain `null reference`.

### Phase 3 — `Store` and linking

The big one. Unlocks the largest single jump in pass rate.

- Introduce a store owning memories, tables and globals in arenas; instances hold indices.
- Make import resolution object-safe (an `Extern::{Func, Global, Table, Memory}` model
  behind a `Linker`, rather than a generic `ImportRegistry`).
- Accept table, memory and global imports in the parser, with correct index-space merging
  (currently only funcs and globals merge; see `module/mod.rs:1693`).
- Make imported mutable globals genuinely shared rather than copied.
- Then build `spectest`, `register`, named-module lookup, and
  `module definition` / `module instance` (needed by `instance.wast`, which instantiates
  one definition twice and observes independent mutable state).

### Phase 4 — Reference types and tables

- Split `Val::Ref` into funcref/externref with real identity and a working `has_ty`.
- Add the 8 table instructions, `elem.drop`, and `select t`.
- Wire the already-built `ElementVal` state into the VM.

### Phase 5 — Long tail

- **Multi-memory** (41 files) — highest remaining leverage.
- **memory64 / table64** (25 files) — also removes two live panic paths.
- **SIMD** (59 files) — a `v128` value type plus ~400 instructions. Its own project.
- **GC, threads, exceptions, relaxed-SIMD** — skip-list indefinitely.

### Expected trajectory

| after | outcome |
| --- | --- |
| Phases 0–2 | green harness over a small MVP subset; no process aborts; honest numbers |
| Phase 3 | largest single jump — everything import/link-shaped |
| Phase 4 | the reference-types files that currently fail at *compile* time |
| Phase 5 | diminishing returns; multi-memory first |

---

## 5. Open decisions

1. **Runner location** — `tests/spec.rs` inside `tracewasm-core`, or a separate
   `tracewasm-spectest` crate? A separate crate keeps the `wast` dependency out of
   `tracewasm-core`'s dev-deps and lets the runner have its own binary for ad-hoc use.
2. **Phase 3 scope** — take the `Store` refactor as one change, or ship Phases 0–2 first
   and let the expectations file show where the value actually is?
