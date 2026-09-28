# tracewasm-llvm

Build LLVM IR in memory, render it as textual `.ll` — and, optionally, run it.

The crate has two halves:

- **The IR builder** (always on, pure Rust). Build a control-flow graph of
  functions, basic blocks and instructions, and emit it as text that `llvm-as`
  accepts. Nothing in it parses IR or links against libLLVM.
- **The JIT** (`jit` feature). A safe wrapper over LLVM's ORC JIT that parses
  textual IR — from the builder or anywhere else — links host functions into it,
  optimizes and compiles it, and hands back type-checked functions to call.

It's a construction layer for a compiler backend. It knows nothing about the
language being compiled.

## Features

| Feature | Default | What it adds | Costs |
|---|---|---|---|
| *(none)* | ✓ | `cfg`, `instruction`, `value`, `interner`, `error` — the IR builder and emitter | nothing beyond small pure-Rust deps |
| `jit` | | the `jit` module and the `#[imported]` attribute | links libLLVM through [`llvm-sys`](https://crates.io/crates/llvm-sys), which needs a matching LLVM installed |

```toml
[dependencies]
tracewasm-llvm = { path = "../tracewasm-llvm" }                      # builder only
tracewasm-llvm = { path = "../tracewasm-llvm", features = ["jit"] }  # builder + JIT
```

## Building IR

Three things are threaded through almost every call:

- a **`Context`**, which owns the arenas and the interned types, strings and
  constants. Everything is addressed by id, and **an id only means anything
  against the context that issued it**;
- a **`Builder`**, which owns the context: it adds functions and globals, and
  hands out cursors;
- a **`Cursor`**, which points at one basic block and writes instructions into it.

```rust
use tracewasm_llvm::cfg::{
    context::Context,
    emit::IREmitter,
    module::{DataLayout, Triple},
};
use tracewasm_llvm::instruction::{IBinOp, cursor::OperandTy};

let ctx = Context::new(
    Triple::new("arm64".into(), "apple".into(), "macosx".into(), None),
    DataLayout::default(),
);
let mut builder = ctx.builder();

// define i64 @add(i64 %a, i64 %b)
let i64_ty = builder.i64_ty();
let f = builder.define_function("add", &[(i64_ty, "a".into()), (i64_ty, "b".into())], i64_ty)?;
let entry = f.add_basic_block("entry", &mut builder)?;
let (a, b) = (f.params(&builder)[0], f.params(&builder)[1]);

let mut cursor = builder.cursor_at_block(entry);
let sum = cursor.build_ibinop(IBinOp::Add, OperandTy::Inferred, a, b, "sum".into())?;
cursor.build_ret(Some(sum), i64_ty.into())?; // a terminator: consumes the cursor

let ir = IREmitter::emit(builder.build())?;

assert!(ir.contains("define i64 @add(i64 %a, i64 %b) {"));
assert!(ir.contains("%sum = add i64 %a, %b"));
# Ok::<(), anyhow::Error>(())
```

`Builder::build` numbers unnamed registers (`%0`, `%1`, …) in printed order, the
way LLVM requires, so a frontend can create blocks in whatever order suits it.

### Stricter than LLVM, on purpose

The builders refuse some IR that `llvm-as` would accept — a `getelementptr` with an
out-of-range constant array index, a `load` whose type disagrees with the pointer's
inferred pointee — so a bug in the compiler driving them surfaces at construction
instead of as a miscompile. Where the crate is *looser* than LLVM, that's a bug;
where it's stricter, it's deliberate.

### Types are interned

A `Type` names its children by `TyId` instead of holding them, so structurally
equal types are one pool entry and comparing types is comparing integers. The cost
is that a type can't print itself: rendering goes through `Context::display`.

## Running IR (`jit`)

The JIT follows LLVM's own flow:

1. **`JITHandler::new`** creates an ORC `LLJIT` for the host machine.
2. **`parse_module`** parses textual IR, runs LLVM's verifier, checks the module's
   target against the JIT's, and links every `#[imported]` host function.
3. **`link_host_func`** links more host functions by hand.
4. **`optimize`** runs a standard `default<O…>` pipeline (`OptLevel::O0`–`O3`,
   `Os`, `Oz`).
5. **`compile`** compiles the module into its own JITDylib, so the same IR can be
   compiled any number of times, each copy with its own globals.
6. **`get_func::<P, R>`** looks up an export whose IR signature is exactly `P → R`
   and returns a `Func` whose `call` takes exactly those arguments.

Dropping a compiled instance frees its code, and every `Func` borrows its
instance, so a function can't be called after its code is gone.

### From the builder to a call

Emit the graph and pass the text straight to the JIT. Give the `Context` the
host's triple — here, Apple silicon — and leave the data layout unset so the
JIT supplies its own.

```rust,no_run
use llvm_sys::target_machine::LLVMCodeGenOptLevel;
use tracewasm_llvm::cfg::{
    context::Context,
    emit::IREmitter,
    module::{DataLayout, Triple},
};
use tracewasm_llvm::instruction::{IBinOp, cursor::OperandTy};
use tracewasm_llvm::jit::{JITHandler, OptLevel};

let ctx = Context::new(
    Triple::new("arm64".into(), "apple".into(), "macosx".into(), None),
    DataLayout::default(),
);
let mut builder = ctx.builder();
let i64_ty = builder.i64_ty();
let f = builder.define_function("add", &[(i64_ty, "a".into()), (i64_ty, "b".into())], i64_ty)?;
let entry = f.add_basic_block("entry", &mut builder)?;
let (a, b) = (f.params(&builder)[0], f.params(&builder)[1]);
let mut cursor = builder.cursor_at_block(entry);
let sum = cursor.build_ibinop(IBinOp::Add, OperandTy::Inferred, a, b, "sum".into())?;
cursor.build_ret(Some(sum), i64_ty.into())?;
let ir = IREmitter::emit(builder.build())?;

let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)?;
let mut module = jit.parse_module("add", &ir)?;
module.optimize(OptLevel::O3)?;

// SAFETY: `add` is defined for every input.
let instance = unsafe { module.compile() }?;
let add = instance.get_func::<(i64, i64), i64>("add")?;

assert_eq!(add.call(40, 2), 42);
# Ok::<(), anyhow::Error>(())
```

### Host functions

Put `#[imported]` on a function and every module the JIT parses has it linked
under that name — no `link_host_func` call needed. It needn't be `extern "C"`;
the macro generates the C-ABI shim.

Registration is process-wide: two `#[imported]` functions with the same name make
every `parse_module` fail with `HostFuncAlreadyLinked`. Tests that share a binary
share registrations too, including edition-2024 doctests, which are merged into
one binary unless marked `standalone_crate`.

```rust,standalone_crate
use llvm_sys::target_machine::LLVMCodeGenOptLevel;
use tracewasm_llvm::jit::{JITHandler, imported};

#[imported]
fn double(x: i64) -> i64 {
    x * 2
}

let ir = r#"
declare i64 @double(i64)

define i64 @quadruple(i64 %x) {
  %d = call i64 @double(i64 %x)
  %q = call i64 @double(i64 %d)
  ret i64 %q
}
"#;

let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault)?;
let module = jit.parse_module("m", ir)?;

// SAFETY: `quadruple` is defined for every input.
let instance = unsafe { module.compile() }?;

assert_eq!(instance.get_func::<(i64,), i64>("quadruple")?.call(5), 20);
# Ok::<(), tracewasm_llvm::jit::error::JITError>(())
```

Arguments and results can be `i32`, `u32`, `i64`, `u64`, `f32`, `f64`, `*const T`
or `*mut T`, with `()` for `void`, up to eight parameters. `i8`, `u8`, `i16`, `u16`
and `bool` are left out on purpose: their C ABI needs `zeroext`/`signext`
attributes the IR would have to carry. A pointer to a `#[repr(C)]` struct works —
the JIT lays the IR's struct out by the same data layout.

### The rules the JIT enforces

Each of these is an error, not a silent miscompile:

- **Signatures must match.** A host function against its `declare`, and a
  `get_func::<P, R>` request against the function's definition.
- **Link before you optimize.** LLVM treats a declaration named like a C library
  function (`abs`, `strlen`, …) as that function and folds calls to it. Linking
  marks the declaration `nobuiltin`, which only helps if it happens first, so
  `link_host_func` refuses once `optimize` has run. `#[imported]` functions are
  linked by `parse_module`, so they're always in time.
- **Every declared symbol must be linked**, or be an LLVM intrinsic. Library calls
  the backend adds on its own (`memcpy` for a large `llvm.memcpy`, `fmod` for
  `frem`) resolve against the host process.
- **The target must match.** A module's triple must name the host's architecture
  and OS, and a data layout it states must be exactly the host's.
- **No static constructors.** A non-empty `llvm.global_ctors`/`llvm.global_dtors`
  is refused: ORC's C API can't run them. `optimize` often folds a constructor into
  the globals it initializes, which removes it.

### Why `compile` is `unsafe`

The signature checks make every `call` type-correct, but IR can do anything —
dereference null, read out of bounds — and the type system can't see inside it.
So `compile` is where you vouch for the code: it must have no undefined behaviour
for any arguments it can be called with. Everything else is safe.

### Setting up LLVM

`llvm-sys` 221 builds against **LLVM 22.1**. Point it at an install or a build
tree with:

```sh
export LLVM_SYS_221_PREFIX=/path/to/llvm-22.1   # the directory containing bin/llvm-config
```

Your editor's rust-analyzer only sees the `jit` module with the feature on. In
Zed, the workspace's `.zed/settings.json` does this; elsewhere, set
`rust-analyzer.cargo.features` to `["tracewasm-llvm/jit"]`.

## Layout

| Module | What's in it |
|---|---|
| `cfg` | the graph being built — `context`, `builder`, `module`, `function`, `basic_block`, `global` — plus `walk` (a visitor over a finished graph) and `emit` (the visitor that renders it as `.ll`) |
| `instruction` | instructions and their operands; `instruction::cursor` holds the builders that write them |
| `value` | types, constants and the values instructions operate on |
| `interner` | the type, string and constant pools and their ids |
| `error` | what the builders return when they refuse to build something |
| `jit` | *(feature `jit`)* the ORC JIT wrapper; `jit::func` has the boundary types and `Func`, `jit::error` has `JITError` |

The `#[imported]` attribute lives in the companion crate `tracewasm-llvm-macros`;
use it through `tracewasm_llvm::jit::imported`.

## Testing

```sh
cargo test -p tracewasm-llvm                                   # the builder
cargo test -p tracewasm-llvm --features jit                    # plus the JIT and this README's examples
cargo run  -p tracewasm-llvm --features jit --example jit --release
```

The JIT's tests are `tests/jit_*.rs`; `examples/jit.rs` builds a small module with
an imported `host_print`, optimizes it at `O3`, and runs it.
