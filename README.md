# TraceWasm: Wasm interpreter for sophisticated runtime tracing

TraceWasm is a WebAssembly interpreter written from scratch in Rust, built for a single
purpose: making the runtime behaviour of a program completely observable.

Most tooling watches a program from the outside. Profilers sample it, tracers interrupt
it, instrumentation rewrites it — and all of them observe the machine rather than the
program, at whatever resolution the operating system is willing to give up. TraceWasm
inverts that. If you own the machine the program runs on, every memory access, every
call, every branch and every allocation passes through code you control. Nothing has to
be inferred, sampled, or reconstructed after the fact.

WebAssembly is an unusually good substrate for this. It is small, fully specified,
deterministic, and its memory is an explicit, bounded region rather than an opaque
address space. That makes a complete account of a program's execution genuinely
achievable instead of merely approximate.

## Two interpreters, one module

Tracing and speed pull in opposite directions, so TraceWasm provides both and lets you
choose per run.

A **stack virtual machine** mirrors WebAssembly's own semantics one-to-one. Every operand
push and pop is a real event, which is exactly what you want when the question is *what
did this program actually do*.

A **register virtual machine** lowers the same module ahead of execution, resolving
operands to registers and eliminating the redundant stack traffic the format implies but
the program does not need. Constants and locals become operands read in place rather than
instructions that move them, so `local.get; local.get; i32.add` costs one instruction
instead of three. This is the mode for long workloads and realistic inputs, where a
faithful but slow interpreter would change the very behaviour you are trying to observe.

Measured against the stack machine on the same build, the register machine runs
arithmetic-heavy code **1.62×** faster, memory-heavy code **1.40×**, and call-heavy code
**1.09×**, with a smaller compiled footprint. Neither is a toy: both pass the same
differential suite, and every module is checked against both machines *and* against the
same program compiled natively.

Which machine a module is compiled for is a type parameter, fixed once at compile time —
`Module::<Stack>::compile(&wasm)` or `Module::<Register>` — so the two produce separate
compiled modules from one input and everything downstream is generic over the choice.

## The machine is a component, not a fixture

The design borrows its central idea from **deterministic simulation testing**, the
technique FoundationDB and TigerBeetle use to test distributed systems. The insight there
is that a system is only as testable as its environment is *substitutable*: replace the
network, the clock and the disk with deterministic models you control, and behaviour that
would otherwise take a fleet and a month of bad luck to observe becomes reproducible on
one thread in milliseconds.

TraceWasm applies the same idea one level down. The thing being substituted is not the
network between machines — it is **the machine itself**. The pieces a guest program can
observe are injected rather than fixed internals, so:

- **Memory** is any `Memory` implementation. The default is a flat byte buffer. Swap in a
  cache-hierarchy model and the same unmodified program reports its miss rate, its access
  patterns, its locality, and which lines of your source caused them.
- **Host imports** are any `ImportRegistry`. That is the seam where the guest meets the
  outside world, and it is yours — including the ability to make it fail.
- **The interpreter** is one of the two machines above, chosen by type parameter.
- **The allocator** is next: linking one into the guest turns every allocation and free
  into an imported call the host sees.

Nothing is sampled and nothing is inferred, so a run is reproducible exactly. Two runs of
the same module against the same components produce the same trace, which is what makes a
recorded failure worth keeping.

The effect is that the interpreter becomes an experimental instrument. You are not only
observing how a program behaves — you are asking how it *would* behave on a machine you
have described, without owning that machine and without touching the program.

## Usage

Not published yet, so depend on it by path or git:

```toml
[dependencies]
tracewasm-core = { git = "https://github.com/spino17/tracewasm" }
tracewasm-macros = { git = "https://github.com/spino17/tracewasm" }
```

### Running a module

```rust
use tracewasm_core::{Register, Stack, anyhow, memory::linear::LinearMemory, module::Module};
use tracewasm_macros::imports;

// A module that declares no imports still needs a registry; an empty
// `#[imports]` block generates one.
struct NoImports;

#[imports]
impl NoImports {}

fn main() -> Result<(), anyhow::Error> {
    let wasm = std::fs::read("guest.wasm")?;

    // The machine is a type parameter, fixed when the module is compiled.
    let module = Module::<Register>::compile(&wasm)?;
    let double = module.get_typed_func::<(i32,), (i32,)>("double")?;
    let mut instance = module.instantiate::<LinearMemory, _>(NoImports, None)?;

    assert_eq!(double.call((21,), &mut instance)?, (42,));

    // The same bytes under the tracing machine, unchanged.
    let traced = Module::<Stack>::compile(&wasm)?;
    let double = traced.get_typed_func::<(i32,), (i32,)>("double")?;
    let mut instance = traced.instantiate::<LinearMemory, _>(NoImports, None)?;

    assert_eq!(double.call((21,), &mut instance)?, (42,));

    Ok(())
}
```

### Host imports

`#[imports]` turns an ordinary `impl` block into an `ImportRegistry`, checking each
method's signature against what the guest declares. Ask for a `MemoryView` parameter and
the guest's memory is handed to you; return a `Result` and a failure becomes a trap in the
calling frame rather than a panic.

```rust
struct Host {
    calls: u32,
}

#[imports]
impl Host {
    #[module("env")]
    fn log(&mut self, value: i32) -> (i32,) {
        self.calls += 1;
        (value,)
    }
}

let mut instance = module.instantiate::<LinearMemory, _>(Host { calls: 0 }, None)?;
```

### Substituting the memory

Implement `MemoryView` and `Memory` and the interpreter uses yours instead. The methods
are the only path a guest has to its own address space, so anything they record is a
complete account.

```rust
use tracewasm_core::{
    error::MemoryError,
    memory::{Memory, MemoryView, linear::LinearMemory},
};

/// Counts every guest read, and delegates the rest.
struct CountingMemory {
    inner: LinearMemory,
    reads: std::cell::Cell<u64>,
}

impl MemoryView for CountingMemory {
    fn size_in_bytes(&self) -> usize {
        self.inner.size_in_bytes()
    }

    fn read(&self, offset: usize, data: &mut [u8]) -> Result<(), MemoryError> {
        self.reads.set(self.reads.get() + 1);
        self.inner.read(offset, data)
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), MemoryError> {
        self.inner.write(offset, data)
    }

    fn copy_within(&mut self, dest: usize, src: usize, len: usize) -> Result<(), MemoryError> {
        self.inner.copy_within(dest, src, len)
    }
}

impl Memory for CountingMemory {
    fn allocate_initial_memory(size_in_pages: u32) -> Self {
        CountingMemory {
            inner: LinearMemory::allocate_initial_memory(size_in_pages),
            reads: std::cell::Cell::new(0),
        }
    }

    fn grow(&mut self, delta: u32, max_pages: u32) -> Result<u32, MemoryError> {
        self.inner.grow(delta, max_pages)
    }
}

// Then instantiate against it. Nothing else changes.
let mut instance = module.instantiate::<CountingMemory, _>(NoImports, None)?;
```

The narrower reads and writes — `read_u32`, `write_f64`, `fill` and the rest — have
default implementations in terms of the four above, so overriding those four is enough.

### Source-level traps

```rust
match func.call((10,), &mut instance) {
    Ok(vals) => println!("{}", vals.0),
    Err(err) => {
        let trace = err.stack_trace().to_source_trace()?;

        println!("{}", trace.render());
    }
}
```

## Seeing the heap

In WebAssembly the heap is not a primitive. It is a region of linear memory managed by
whatever allocator the guest program brought with it, which is why conventional tooling
can only report that memory grew, never who asked for it or why.

TraceWasm aims to close that gap by supplying an allocator the guest links against, whose
operations cross back into the host as imported calls. Every allocation, reallocation and
free becomes a first-class observable event — with its size, its alignment, its lifetime,
and the exact call site responsible. Fragmentation, leaks, short-lived churn and
allocation hot spots stop being statistics and become specific lines of code.

## Traps that point at your source

When a WebAssembly module traps, what you are handed is an offset into a binary — true,
and nearly useless.

Programs compiled from Rust carry DWARF debug information describing the original source
they came from. TraceWasm uses it to reconstruct a complete stack trace in terms of that
source: real file names, real line numbers, demangled function names, and the inlined
frames that optimisation folded away. A failure deep inside a dependency reads the way it
would have if the program had never been compiled to WebAssembly at all.

## Rust first, not Rust only

Rust is the primary supported language, and the DWARF and allocator work is tuned for
what the Rust toolchain emits. But nothing in the design is specific to it. Any language
that targets WebAssembly and carries standard debug information can be supported, and the
generic components apply unchanged — the interpreter has no notion of the source language
beyond the debug information it is handed.

## What this makes possible

- Understanding memory behaviour — cache misses, access patterns, locality — for programs
  on hardware you do not have.
- Attributing every heap allocation in a program to the line of source that caused it.
- Getting a source-level stack trace from a trap inside a dependency, with inlined frames
  intact.
- Reproducing and inspecting runtime behaviour deterministically, as many times as needed,
  with no sampling error and nothing lost between runs.
- Testing a program against a machine that does not exist yet, by describing it rather
  than building it.

## Status

TraceWasm is under active development.

Working: both machines, module loading and instantiation, the generic memory and import
interfaces, and source-level stack traces from DWARF. Both machines run the same
differential suite against each other and against natively compiled Rust.

In progress: allocator tracing, the trace recording and query surface, and a JIT tier for
hot functions.

Not supported: SIMD (`v128`), threads and atomics, the component model, 64-bit memories
and tables, and more than one memory per module. A module using any of them is rejected at
compile time with `TraceWasmError::Unsupported` rather than mis-executed.

## License

See `LICENSE`.
