use crate::{
    cfg::{function::FuncId, global::Global},
    interner::StrId,
};
use rustc_hash::FxHashSet;
use std::fmt::Display;

pub struct Triple {}

// `Display` rather than `ToString` directly: the blanket impl gives `to_string` for
// free, and implementing it by hand opts out of every formatting context.
impl Display for Triple {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

pub struct DataLayout {}

impl Display for DataLayout {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

pub struct Module {
    pub(crate) triple: String,
    pub(crate) data_layout: String,
    pub(crate) globals: Vec<Global>,
    pub(crate) functions: Vec<FuncId>,
    pub(crate) func_names: FxHashSet<StrId>,
}

impl Module {
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
