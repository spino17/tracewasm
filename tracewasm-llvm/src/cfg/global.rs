//! Module-level global variables.

/// A global variable.
///
/// Not yet modelled. [`Module`](crate::cfg::module::Module) carries a list of these
/// so the shape is in place, but nothing constructs one and the emitter writes no
/// global definitions.
pub struct Global {}
