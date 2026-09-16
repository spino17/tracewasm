//! A guest that traps a couple of frames deep, for resolving a trace back to
//! source.
//!
//! Unlike the other guests this one is built **with** debug info and **without**
//! optimisation, because the thing under test is the `.debug_*` sections
//! themselves: no DWARF means nothing to resolve, and inlining would collapse the
//! frames this exists to tell apart.
//!
//! The trap is an out-of-bounds load rather than a `panic!`. A panic would work,
//! but it drags in the whole `core::panicking` chain and buries the guest's own
//! frames under a dozen std ones whose line numbers move between compiler
//! versions. A bad load traps in the guest itself, so frame `#0` is a line in
//! this file.
//!
//! The two `// MARKER:` comments are load-bearing. `tests/traps.rs` finds them by
//! searching this file and asserts the resolved trace lands on those exact lines,
//! so the expectations cannot drift when a line is added above them.
//!
//! Export names are prefixed `st_`.

#![allow(dead_code)]

/// Reads through a pointer that is far outside linear memory, which traps.
///
/// `#[inline(never)]` so it stays a frame of its own: the point of the test is
/// that the trapping frame and its caller resolve to *different* lines.
#[inline(never)]
fn st_read_through(addr: i32) -> i32 {
    unsafe { *(addr as *const i32) } // MARKER: trapping-load
}

/// Calls the trapping function, so the trace has a caller frame to resolve too.
#[unsafe(no_mangle)]
pub extern "C" fn st_trap_in_callee(addr: i32) -> i32 {
    st_read_through(addr) // MARKER: call-site
}
