# Where function boundaries go, and why it is worth 2x

Two changes in this repo, both of them only moving a boundary — no algorithm, data structure
or comparison changed:

* Pushing a rare call **out** of `Stack::push` into its own function: **5-10%** of interpreter
  throughput.
* Pulling the per-instruction dispatch **in** to the driver loop: **~40%**, and about **2x**
  across the whole interpreter when combined with the rest.

They are the same rule applied in opposite directions. This document explains that rule from
first principles, works both cases through the disassembly, and records one remedy that looked
obviously right and measured worthless.

No assembly knowledge assumed.

---

## 1. The one-sentence version

> If a value is needed *after* a rare call, it has to be kept somewhere the call cannot destroy
> — and if that same value is also loaded on the common path, the common path pays to protect
> it, on every single execution. Moving the call into its own function breaks that link.

Everything below is that sentence, unpacked.

**The thing it is easy to get wrong** (and which the first draft of this document got wrong):
this is *not* "a function containing a call is slow." After the change `Stack::push` still
contains a call. It is not a leaf function before or after. What changes is *where the setup
cost lands* — see §2.5 and §3.5.

---

## 2. Background: what a function call actually costs

### 2.1 Registers

A CPU does arithmetic in **registers** — a small, fixed set of named slots inside the
processor itself. On the machine this was measured on (ARM64) there are 31 of them, called
`x0` through `x30`.

Registers are the only place the CPU can compute. Everything else — your `Vec`, your struct
fields, the heap — lives in RAM, and touching RAM is far slower than touching a register.
Fast code keeps what it is working on in registers and goes to RAM as little as possible.

There are only 31. That scarcity is the source of everything in this document.

### 2.2 Two functions, one set of registers

If function `A` calls function `B`, they both want registers, and there is only one set. So
every platform defines a **calling convention** — a contract splitting the registers into two
groups:

| group | rule | analogy |
| --- | --- | --- |
| **caller-saved** | `B` may freely overwrite these. If `A` had something valuable there, `A` must stash it before calling. | A shared desk. Anything you leave on it may be gone when you come back. |
| **callee-saved** | `B` must leave these exactly as it found them. If `B` wants to use one, `B` must save the old value and put it back before returning. | A borrowed workspace. Use it, but restore it before you hand it back. |

Both rules cost memory writes. The only question is who pays.

### 2.3 The return address

There is one more piece. When `A` calls `B`, the CPU has to remember *where to come back to*.
On ARM64 the call instruction (`bl`, "branch with link") writes the return address into
register `x30` automatically.

Now suppose `B` itself calls `C`. That second call overwrites `x30` with a new return
address — and `B`'s own return address is gone. So **any function that makes a call must save
`x30` to memory first**, and restore it before returning.

This is the crux. Hold onto it.

### 2.4 Prologue and epilogue

Putting 2.2 and 2.3 together: a function that needs to preserve registers begins with a
**prologue** that saves them to memory, and ends with an **epilogue** that restores them.
These are memory writes and reads — exactly the slow thing from 2.1.

A function that calls nothing needs neither. It never overwrites `x30`, so it has nothing to
save, and it can restrict itself to caller-saved registers, so it owes nobody a restore. Just
its own work and a return. That is the ideal shape for a small, extremely hot function.

### 2.5 Live ranges, and why a *rare* call can cost the common path

Here is the part that actually matters, and it is subtler than "calls are expensive."

A value is **live across a call** if it is computed before the call and still needed after it.
Such a value cannot sit in a caller-saved register — the callee is allowed to destroy those. It
must go in a **callee-saved** register, and callee-saved registers must be saved in the
prologue.

Now the two facts that combine into the whole problem:

1. **The register allocator assigns one register per live range.** If a value is loaded early
   and used after a call, that *same* register is the one holding it early — including on paths
   that never reach the call.
2. **The prologue that saves it must run before any branching**, because the save has to happen
   before the register is first written.

So:

> If a value used *after a rare call* is also loaded on the *common* path, then the common path
> is charged for saving and restoring the register that holds it — on every single execution.

The fix is not "avoid calls." It is to make sure **nothing on the common path is live across a
call**. When that holds, compilers apply an optimisation called **shrink-wrapping**: the
prologue moves out of the function entry and into only the branches that need it. The common
path then executes zero prologue instructions *even though the function still contains a call.*

That last point is worth stating flatly, because it is easy to get wrong:

> Outlining does **not** make the function a leaf. `Stack::push` contains a call to
> `push_grow` after the change, just as it contained a call to the allocator before. What
> changes is that the frame setup moves *inside the cold branch*, because no value on the fast
> path needs to survive a call any more.

---

## 3. The actual case: `Stack::push`

### 3.1 The code

```rust
pub fn push(&mut self, val: T) {
    if self.stack_pointer < self.inner.len() {
        self.inner[self.stack_pointer] = val;   // common: a slot is already allocated
    } else {
        self.inner.push(val);                   // rare: extend the backing storage
    }
    self.stack_pointer += 1;
}
```

Nothing here looks like a function call, and `Vec::push` being inlined is *not* one — inlining
pastes the code in, it does not generate a call. But the pasted-in body contains a call of its
own: when the vector is full, `Vec::push` calls the allocator via `RawVec::grow_one`.

So after inlining, `Stack::push` contains `bl RawVec::grow_one`. By §2.5 the damage is not the
call's existence — it is that values loaded on the *common* path are still needed after that
call, which forces them into callee-saved registers and forces a prologue at the entry point.
§3.3 shows exactly which values.

### 3.2 How rare is "rare"?

Measured by instrumenting the growth branch, one 20,000-iteration workload:

| guest | pushes | times the growth path ran | share |
| --- | --- | --- | --- |
| arithmetic | 580,021 | **9** | 0.0016% |
| memory | 587,055 | **33** | 0.0056% |
| frames | 360,032 | **10** | 0.0028% |

And **zero** on every call after the first, because the stack lives on the `Instance` and
keeps its high-water mark across invocations.

The growth path runs nine times. The prologue supporting it runs 580,021 times.

### 3.3 What the machine actually does — before

The compiled function. `x20` is the register to watch — follow it down the listing:

```
 1  stp x22, x21, [sp, #-0x30]!   save two registers to memory ┐
 2  stp x20, x19, [sp, #0x10]     save x20 and x19             │ PROLOGUE
 3  stp x29, x30, [sp, #0x20]     save frame pointer + return  │ runs every time
 4  add x29, sp, #0x20            set up the frame pointer     ┘
 5  ldp x20, x8, [x0, #0x10]      x20 = inner.len, x8 = stack_pointer   ← on the FAST path
 6  cmp x8, x20                   is stack_pointer < len?
 7  b.hs <grow>                   no → jump to the growth path
 8  ldr x9, [x0, #0x8]            load the data pointer
 9  ldr q0, [x1]                  load the 16-byte value
10  str q0, [x9, x8, lsl #4]      store it into the slot        ← THE ACTUAL WORK
11  b <merge>                     jump forward to 25
--- growth path, reached 9 times in 580,021 ---
13  cmp x20, x8                   x20 used
15  mov x19, x0                   stash `self`      in a callee-saved register
16  mov x21, x1                   stash the value   in a callee-saved register
17  bl RawVec::grow_one           THE CALL
18  mov x1, x21                   value  needed again AFTER the call
19  mov x0, x19                   `self` needed again AFTER the call
22  str q0, [x8, x20, lsl #4]     x20 needed again AFTER the call
23  add x8, x20, #0x1             x20 needed again AFTER the call
--- merge ---
25  ldr x8, [x0, #0x18]           reload stack_pointer  ┐
26  add x8, x8, #0x1              add one               │ (couldn't stay in a register)
27  str x8, [x0, #0x18]           store it back         ┘
28  ldp x29, x30, [sp, #0x20]     restore registers ┐
29  ldp x20, x19, [sp, #0x10]     restore           │ EPILOGUE
30  ldp x22, x21, [sp], #0x30     restore           ┘ runs every time
31  ret                           return
```

Now the chain from §2.5, concretely:

- `inner.len` is needed at **22 and 23, after the call at 17**. So its live range crosses a
  call, so it must live in a **callee-saved** register — hence `x20`, not a scratch register.
- Callee-saved means it has to be saved before it is first written. It is first written at
  **instruction 5, on the fast path**. So the save must happen earlier still: instruction 2, in
  the prologue, before the branch at 7.
- Same story for `self` and the value pointer (`x19`, `x21`, stashed at 15–16 and reloaded at
  18–19).

**That is why the fast path pays.** Not because a call exists, but because the values the cold
path needs after its call are the same values the fast path loads — so the registers holding
them must be preserved, and the preserving has to happen before anyone knows which path will
be taken.

Reading only the common-case path — 1-11, then 25-31 — that is **18 instructions executed, of
which 7 are pure prologue and epilogue**. Three do the real work: 8, 9, 10.

### 3.4 The change

```rust
pub fn push(&mut self, val: T) {
    if self.stack_pointer < self.inner.len() {
        self.inner[self.stack_pointer] = val;
        self.stack_pointer += 1;
    } else {
        self.push_grow(val);     // ← the call now lives behind a function boundary
    }
}

#[inline(never)]                 // ← and the compiler is told not to paste it back in
fn push_grow(&mut self, val: T) {
    self.inner.push(val);
    self.stack_pointer += 1;
}
```

Same logic. Same branch. Same comparison. The *only* difference is that the call to the
allocator is no longer lexically inside `push`.

### 3.5 What the machine does — after

```
 1  ldp x9, x8, [x0, #0x10]       load inner.len and stack_pointer
 2  cmp x8, x9                    is stack_pointer < len?
 3  b.hs <cold>                   no → jump to the cold path below
 4  ldr x9, [x0, #0x8]            load the data pointer
 5  ldr q0, [x1]                  load the 16-byte value
 6  str q0, [x9, x8, lsl #4]      store it into the slot
 7  add x8, x8, #0x1              stack_pointer += 1
 8  str x8, [x0, #0x18]           store it back
 9  ret                           return          ← no epilogue. nothing to restore.
--- cold path, reached 9 times in 580,021 ---
10  sub sp, sp, #0x20             NOW set up a frame
11  stp x29, x30, [sp, #0x10]     NOW save the return address
...
16  bl push_grow
```

**Nine instructions. No prologue. No epilogue. No stack traffic whatsoever.**

Notice what did *not* happen: `push` still contains a call — `bl push_grow` at instruction 16.
**It is still not a leaf function.** What changed is that the frame setup moved from the entry
point down into the cold branch, at instructions 10–11.

That is possible because `inner.len` now lands in `x9`, a scratch (caller-saved) register, and
is dead by instruction 3. `push_grow` reloads `self` and `inner.len` for itself, so nothing on
the fast path is live across a call. With no callee-saved register in use before the branch,
there is nothing the prologue must preserve, and the compiler is free to **shrink-wrap** the
frame setup into the only branch that needs it.

### 3.6 The scoreboard

| | before | after |
| --- | --- | --- |
| instructions executed, common path | ~18 | **9** |
| memory accesses for register save/restore | 6 | **0** |
| function total size | 31 instructions | 19 |
| callee-saved registers used before the branch | 3 (`x20`, `x19`, `x21`) | **0** |
| frame setup on the common path | yes | **no** (shrink-wrapped into the cold branch) |
| contains a call at all? | yes | **yes** — this was never about leaf-ness |

Roughly half the work in `push` was bookkeeping for a call that essentially never happened.

### 3.7 Measured effect

ns per guest loop iteration, release, interleaved A/B (see §8 on why interleaved):

| workload | `Vec::push` inline | `#[inline(never)]` | change |
| --- | --- | --- | --- |
| arithmetic | 146 | **138** | −5.4% |
| memory | 176 | **158** | −9.9% |
| frames | 116 | **108** | −7.2% |

Every sample of the fast variant beat every sample of the slow variant, in every round.

---

## 4. The principle

> **A value live across a call must occupy a callee-saved register. If that value's live range
> also covers the common path, the common path pays to save and restore it — on every
> execution, because the save has to happen before the branch that would have avoided it.**

Note what this principle is *not*: it is not "a call makes the function slow" and not "the
function stops being a leaf." A function can contain a call and still cost the common path
nothing, provided nothing on the common path needs to survive that call. `push` after the
change is exactly that: it calls `push_grow`, and the fast path pays zero.

The compiler is not being stupid, and it is not helpless either. Given the choice it will
**shrink-wrap** — push the frame setup down into just the branches that need it. What blocks it
is a live range that straddles both paths, and that is precisely what inlining a growth or
panic path creates.

So the fix is to sever the live range, which means moving the call *and the values it
needs* behind a function boundary. Reordering the `if` achieves nothing. `#[inline(never)]` is
needed because without it the compiler will paste the helper straight back in and recreate the
straddling live range.

The standard library does this to itself constantly. `Vec::push` does not contain the allocator
logic — it calls `RawVec::grow_one`, a separate function, for exactly this reason. std outlined
one level. This change outlined the second.

### The four ways cold code taxes hot code

Callee-saved spilling is the one measured here. The others are worth recognising:

| tax | mechanism |
| --- | --- |
| **callee-saved spilling** | a value live across a call forces a save/restore before the branch |
| **register pressure** | more values live across a call → fewer scratch registers for real work |
| **instruction cache** | rarely-used bytes sit between the bytes you actually run |
| **optimization barrier** | a call might modify memory, so cached values must be reloaded after it |

---

## 5. Why `pop` and `as_i32` have the same problem

The same scan that found `push` lists what the interpreter's dispatch loop calls, and whether
each sets up a frame at its entry point:

| function | call sites in `execute` | frame at entry? |
| --- | --- | --- |
| `Stack::pop` | **242** | yes |
| `Val::as_i32` | 81 | yes |
| `Val::as_i64` | 67 | yes |
| `Val::as_f32` | 44 | yes |
| `Val::as_f64` | 44 | yes |
| `pop_effective_address` | 23 | yes |

None of these allocates. None of them looks like it calls anything. But:

```rust
pub fn pop(&mut self) -> T {
    let val = self.inner[self.stack_pointer - 1].clone();   // ← indexing
    self.stack_pointer -= 1;
    val
}
```

Indexing a slice is bounds-checked, and a failed bounds check **calls**
`core::panicking::panic_bounds_check` to report the error. That is a call on a path that cannot
happen — validation already guaranteed the operand stack is non-empty — and by §4 it can drag a
save/restore onto the common path, 242 call sites over.

```rust
pub fn as_i32(&self) -> i32 {
    let Val::I32(val) = self else {
        panic!("value is not i32")      // ← formats a message: another call
    };
    *val
}
```

Same shape. `panic!` with a message pulls in the formatting machinery. Validation already
type-checked every operand, so the panic is unreachable — and paid for on every arithmetic
instruction the interpreter executes.

**This is the same bug as `push`, wearing a different costume.** In `push` the hidden call was
the allocator. Here it is the panic reporter. The mechanism, and the fix, are identical.

### 5.1 The fix, and why `-> !` matters

```rust
#[inline(never)]
#[cold]
fn not_i32() -> ! {
    panic!("value is not i32")
}

pub fn as_i32(&self) -> i32 {
    match self {
        Val::I32(v) => *v,
        _ => not_i32(),
    }
}
```

Two things are doing work here.

`#[inline(never)]` severs the live range, as in §3.5 — the helper reloads whatever it needs, so
nothing on the common path has to survive the call.

`-> !` ("never returns") adds a second saving, from §2.3: a normal call must preserve the
return address because control comes back, but there is no coming back from a diverging call,
so the compiler can emit a plain jump instead. A plain jump does not touch the return-address
register, so there is nothing to save even in the cold branch.

This is why diverging helpers are the standard idiom for panic paths throughout `core` and
`hashbrown`. It is not stylistic.

### 5.2 What it measured

Both were changed together: `pop` indexes with `.get()` and diverges to a no-argument
`pop_underflow()`, and the five accessors call no-argument helpers in a private `wrong_ty`
module. Interleaved A/B, four rounds:

| workload | before | after | delta | rounds won |
| --- | --- | --- | --- | --- |
| memory | 161.17 | **152.67** | −5.3% | 4/4 |
| frames | 112.98 | **109.38** | −3.2% | 4/4 |
| arithmetic | 136.57 | 136.82 | +0.2% | 2/4 |

The memory row's ranges do not overlap (off 158.1-164.2, on 146.5-154.1), so that one is not
ambiguous.

Note what the numbers say about attribution. `arithmetic` has by far the highest `as_*`
density — 81 + 67 call sites for `as_i32`/`as_i64` — and did not move. `memory` and `frames`
did. Since both changes landed together they cannot be separated without another A/B, but the
pattern points at `pop` (242 sites) carrying the whole win and the accessor change carrying
none.

An unrelated benefit of the `pop` rewrite: the old empty-stack behaviour differed by profile —
debug panicked on the subtraction, release wrapped and hit the bounds check. Both now report
`pop from an empty operand stack`.

## 6. The mirror case: pulling a boundary *in*

Everything so far moves a call **out** of a hot function. The largest win in this repo came
from the opposite move, and it is the same rule.

### 6.1 The boundary

The interpreter's driver loop:

```rust
loop {
    let instr = &instructions[pc];
    pc = state.execute(instr, module, pc, call_stack_depth, br_table_targets)?;
    if pc == instr_count { break; }
}
```

`execute` is a real function, so this is one `bl` and one `ret` **per wasm instruction**. The
call/ret pair is the smaller half of the cost. The larger half is §2.2: `pc`, and the operand
stack's base pointer and length, are loop-carried, and a call may destroy any caller-saved
register — so none of them can stay in a register across it. They are reloaded from memory on
every iteration, forever.

A boundary in the middle of a hot loop is where the register allocator gives up.

### 6.2 What removing it bought

`#[inline(always)]` on `execute`. Interleaved A/B, ns per guest loop iteration:

| workload | before | after | delta |
| --- | --- | --- | --- |
| arithmetic | 134.70, 139.18 | 82.78, 82.44 | **−40%** |
| memory | 150.27, 151.09 | 86.52, 88.00 | **−42%** |
| frames | 107.85, 106.43 | 59.69, 60.40 | **−44%** |

Re-measuring the whole suite afterwards, every row improved 32-46%, about **2x overall**, and
arithmetic went from 16.5 to **8.6 cycles per wasm instruction**. That is larger than every
other change in this document combined.

### 6.3 The price, and it is not small

| variant | driver frame |
| --- | --- |
| baseline | 416 B |
| `#[inline(always)]` | **5,088 B** |

Fusing ~200 opcode arms into one function means every arm's locals and spill slots must fit in
that one function's frame. A frame is allocated **once at entry, sized for the worst path
through the function** (§2.4) — so the driver now reserves what the heaviest arm needs, on
every entry, regardless of which opcode actually runs.

That would be tolerable if the driver were called once. It recurses: a wasm call runs
`driver -> call_func -> driver`. Per nested wasm call the cost goes from ~1,311 B to ~5,100 B,
cutting reachable depth from ~6,400 frames to ~1,650.

### 6.4 The remedy that looked obvious and measured worthless

The natural next step is the technique from §3 and §5: outline the cold arms so the fused
function slims down. The heaviest cold code in dispatch is the `call_indirect`
signature-mismatch trap, which expands `core::fmt` twice via `format!`. Outlining it:

**5,088 B → 5,056 B. Thirty-two bytes. 0.6%.**

The reason is that the two frames have different causes, and they are indistinguishable in the
disassembly:

| | §3's frame | §6's frame |
| --- | --- | --- |
| made of | callee-saved registers holding values live across a call | spill slots for locals the allocator could not keep in registers |
| how many | a handful of registers | hundreds of slots across ~200 arms |
| fixed by | moving the call, so nothing is live across it | nothing cheap — there is no single call to move |
| tell | a small function with a prologue | a huge function with a prologue |

Outlining an arm removes that arm's *code*. It does not help the allocator prove that the
remaining arms' slots are mutually exclusive, and with this much control flow it largely
cannot. **Do not spend time outlining arms to recover this frame** — measure a single one
first, as here, and the answer arrives in one build.

### 6.5 What to do about the frame instead

Since the frame will not shrink, the depth limit moves to match it:
`Config::max_call_stack_depth` defaults to 1200, sized so a full chain stays inside a typical
8 MiB stack. This matters more than it sounds — a limit the native stack cannot hold converts
a clean trap into a process abort, because the overflow arrives before the guard does.

The real fix is to stop nesting a native frame per wasm call at all: an explicit frame stack,
with the driver looping over it instead of recursing. Then frame size stops bounding depth, and
inlining costs nothing but code size.

### 6.6 Both directions, one rule

> A function boundary is a barrier the register allocator cannot see across. Put boundaries
> **between** hot and cold code, and **never inside** a hot loop.

* **Outline** when a rare path drags state into callee-saved registers that the common path
  then pays to preserve. Cost: none. See §3.
* **Inline** when a boundary sits inside a hot loop and forces loop-carried state back to
  memory each iteration. Cost: the frame grows to the union of everything fused in, which
  matters if — and only if — that function recurses.

The two are not in tension. `push` is outlined *and* inlined into the dispatch that was itself
inlined into the loop; each boundary is placed where it belongs.

---

## 7. How to find this yourself

**Step 0 — check whether the function you suspect was even inlined.**

```sh
nm target/release/deps/<test-binary> | grep -i 'Stack.*push'
```

If the symbol **exists**, it was not inlined, and that function is where to look. Skipping
this step cost three wrong answers during this investigation: the change lived entirely
inside `Stack::push`, while measurements were being taken on `execute`, which the change does
not touch at all.

**Step 1 — disassemble it.**

```sh
otool -tV target/release/deps/<bin> > asm.txt     # macOS
objdump -d --disassemble='<symbol>' <bin>          # Linux
cargo asm 'tracewasm_core::vm::stack::Stack::push' # cargo-show-asm, much easier
```

**Step 2 — read the first five instructions.** Highest-signal check available. A small hot
function should start doing work immediately. If it starts saving registers to memory, ask
*what call is in here?*

**Step 3 — find the calls, then trace the registers.** Search for `bl` (ARM) or `call` (x86).
For each call, note which callee-saved registers (`x19`-`x28`) are used *after* it, then look
for where those registers are first written. If that is on the common path, you have found the
problem — that is the §3.3 trace, and it is the step that actually explains the cost. Rare call
→ outline it. Rare **and** diverging → outline with `-> !`.

**Step 4 — list what your hot loop calls, and check each for a frame at entry.** This one table
localised the whole problem and is worth automating.

**Step 5 — measure, interleaved.** See §8.

### Detecting the mirror pattern (§6)

Different question, different check. For your hottest loop:

**Is there a `bl` in the loop body at all?** If dispatch, or anything else the loop does every
iteration, is a real call, that is a barrier. Cheap test: `#[inline(always)]` on it and measure
before deciding whether the frame cost is acceptable.

**Is loop-carried state reloaded every iteration?** Look at the top of the loop body for `ldr`
of the same struct fields each time round — a loop counter, a base pointer, a length. If the
compiler could keep them in registers it would; reloading means something in the body forced
them out, and a call is the usual culprit.

**Then work out which kind of frame you have**, because the two look identical from the
prologue alone:

| symptom | likely cause | remedy |
| --- | --- | --- |
| small function, prologue, one rare call in the body | callee-saved registers held across the call (§2.5) | outline the call — worth 5-10% |
| large function, big frame, no single dominant call | spill slots across many arms (§6.3) | not recoverable by outlining; measure one arm and stop |

The discriminator is size. If the function is a few dozen instructions, §3 applies. If it is
thousands, §6 does, and outlining is a trap.

### Signatures worth memorising

```
stp x29, x30, [sp, ...]     frame set up at entry → something is live across a call
stp x20, x19, [sp, ...]     each extra pair = another value live across a call
sub sp, sp, #0x270          large frame = heavy spilling to memory
bl ...panic_bounds_check    a bounds-check failure path got inlined
bl ...panic_fmt             a panic with a formatted message got inlined
bl ...RawVec...grow         a container's growth path got inlined
bl __rust_alloc             an allocation got inlined
```

A frame set up *after* a forward branch rather than at entry is the shape you want — that is
shrink-wrapping, and it means the common path pays nothing.

The shape you want is `push`'s current form: loads, the work, `ret`, with any `sub sp`
appearing only *after* a forward branch.

### Common Rust constructs that hide a call

- `Vec::push`, `HashMap::insert` → allocator
- slice indexing, `.unwrap()`, `.expect()`, `panic!` → panic reporting and formatting
- `format!` → allocation plus `core::fmt`
- `?` on a large error type → copying the error
- constructing a big `Err` variant inline

---

## 8. What this is *not*, and how we know

Three plausible explanations were tested and are all wrong. Recording them because each one
looked obviously right at the time.

**Not the branch.** The natural first guess is that `if stack_pointer < inner.len()` costs a
comparison and a mispredicted branch. It does not. An earlier attempt pre-filled the backing
store so the condition was *always* true, making the branch perfectly predictable —
throughput did not move at all. The branch is one `cmp`, and the CPU predicts it correctly
every time. It was never the cost.

That attempt also filled 8 MiB on every call, which made per-call latency **363× worse**
(317 ns → 115 µs). The test suite's `MAX_PER_CALL_NS` bound caught it in one run.

**Not code size.** `execute`'s compiled size was measured before and after: identical, to the
byte (624-byte frame, 3,335 instructions, 171 spill operations). The reason is §7 Step 0 —
`Stack::push` is a separate function that was never inlined into `execute`, so `execute`'s
codegen is untouched by this change.

**Not `#[cold]`.** The original version carried `#[cold]` alongside `#[inline(never)]`. A
three-way interleaved test showed `#[inline(never)]` alone captures the entire win, and adding
`#[cold]` is consistently **worse** — 3% on frames, 5% on memory. It was removed. Do not
restore it without measuring.

| variant | arithmetic | memory | frames |
| --- | --- | --- | --- |
| `Vec::push` inline | 146 | 176 | 116 |
| `#[inline(never)]` | **138** | **156** | **108** |
| `#[cold]` + `#[inline(never)]` | 140 | 162 | 111 |

### Measurement discipline

The first A/B run for this change measured all three "outlined" samples, then all three
"inline" samples. It reported a 14% win. That number was partly **thermal drift** — the
machine slowed down over the five minutes the measurements took, and the second variant
measured always looks worse.

**Alternate variants round by round.** A-B-A-B-A-B, not AAA-BBB. The interleaved re-run
confirmed the win was real at 5-10%, but it could just as easily have evaporated.

Also: discard the first run after a rebuild (a cold row was observed at 232 ns/op against 145
warm on the same binary), and use `--test-threads=1`, which narrows run-to-run spread from
~11% to ~3%.

---

## 9. Summary

1. Registers are scarce and fast; RAM is plentiful and slow.
2. Registers split into **caller-saved** (scratch, a callee may destroy them) and
   **callee-saved** (must be restored, so using one costs a save and a restore).
3. A value that is **live across a call** must therefore occupy a callee-saved register.
4. The register allocator uses one register per live range, and the save must precede the first
   write. So if that value is also loaded on the common path, **the common path pays** — before
   any branch has been taken.
5. Therefore a call reached 9 times in 580,021 was charged 580,021 times.
6. Moving the call *and the values it needs* into an `#[inline(never)]` helper severs the live
   range. Nothing on the common path survives a call, so the compiler **shrink-wraps** the frame
   setup into the cold branch. The function still contains a call; the common path just stops
   paying for it. Worth 5-10% here.
6b. The same rule run backwards: a boundary *inside* a hot loop forces loop-carried state back
   to memory every iteration. Removing it — `#[inline(always)]` on the per-instruction
   dispatch — was worth **~40%**, the largest win in this repo. It costs frame size, because
   the fused function reserves what its heaviest path needs on every entry, which bounds
   recursion depth.
6c. Those two frame costs have the same symptom and different cures. Callee-saved registers
   held across a call are removable by outlining; spill slots spread over hundreds of arms are
   not. Outlining the heaviest cold arm of the fused dispatch recovered **0.6%** — measure one
   before investing in the rest.
7. `Stack::pop` and `Val::as_*` look like the same problem, via bounds checks and `panic!`
   rather than the allocator. Unverified: `pop` sets up a frame even though its panic path is
   already diverging, so trace its registers before assuming the `push` result transfers.
8. Verify by disassembling — and check first that the function you are measuring is the one
   the change actually affects.
