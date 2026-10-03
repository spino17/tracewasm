//! Small pieces `tracewasm-core` and `tracewasm-llvm` both need.
//!
//! Only [`interner`] so far: a deduplicating pool that hands out small typed ids.
//! It lives here rather than in either crate because both intern, and neither
//! should depend on the other to do it.

pub mod error;
pub mod interner;
