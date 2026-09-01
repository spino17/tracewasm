//! The module: its target settings and the functions it defines.

use crate::{
    cfg::{function::FuncId, global::Global},
    interner::StrId,
};
use rustc_hash::FxHashSet;
use std::fmt::Display;

/// A target triple, such as `arm64-apple-macosx`.
///
/// Not yet modelled — [`Module`] stores the triple as a plain `String`, and
/// [`Builder::new`](crate::cfg::builder::Builder::new) takes one directly.
pub struct Triple {}

// `Display` rather than `ToString` directly: the blanket impl gives `to_string` for
// free, and implementing it by hand opts out of every formatting context.
impl Display for Triple {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

/// A data layout string, describing sizes, alignments and endianness.
///
/// Not yet modelled, like [`Triple`].
pub struct DataLayout {}

impl Display for DataLayout {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

/// One LLVM module: target settings, globals and functions.
///
/// Owned by [`Builder`](crate::cfg::builder::Builder) while building and by
/// [`ControlFlowGraph`](crate::cfg::ControlFlowGraph) afterwards. Functions are held
/// as ids into the context's arena; `func_names` is what makes a duplicate `@name`
/// a build error rather than something `llvm-as` discovers later.
///
/// An empty `triple` or `data_layout` means "unset" — the emitter omits the line
/// rather than writing `target triple = ""`.
pub struct Module {
    pub(crate) triple: String,
    pub(crate) data_layout: String,
    pub(crate) globals: Vec<Global>,
    pub(crate) functions: Vec<FuncId>,
    pub(crate) func_names: FxHashSet<StrId>,
}

impl Module {
    /// An empty module for the given target.
    pub fn new(triple: Triple, data_layout: DataLayout) -> Self {
        Module {
            triple: triple.to_string(),
            data_layout: data_layout.to_string(),
            globals: vec![],
            functions: vec![],
            func_names: FxHashSet::default(),
        }
    }
}
