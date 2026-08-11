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
the program does not need. This is the mode for long workloads and realistic inputs,
where a faithful but slow interpreter would change the very behaviour you are trying to
observe.

Both consume the same compiled module, so you can trace exhaustively and then run at
speed without changing anything about the program under test.

## The machine is yours to define

The components of the VM are generic. Rather than exposing hooks into a fixed
implementation, TraceWasm lets you supply your own — and the observations you get are
whatever your implementation chooses to record.

Memory is the clearest example. The default is what you would expect: a flat, linear
byte buffer. But nothing requires it to be flat. Substitute a model of a cache hierarchy
and the same unmodified program now reports its miss rate, its access patterns, its
locality, and where in your source those misses come from. Substitute something else and
you are asking a different question entirely.

This turns the interpreter into an experimental instrument. You are not only observing
how a program behaves — you are asking how it *would* behave on a machine you have
described, without owning that machine, and without touching the program.

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

TraceWasm is under active development. The core interpreter and module loading are
working; the register machine, the generic component interfaces, and the allocator
tracing are in progress.

## License

See `LICENSE`.
