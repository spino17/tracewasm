use crate::{
    cfg::{function::FuncId, global::Global},
    interner::StrId,
};
use rustc_hash::FxHashSet;

pub struct Triple {}

impl ToString for Triple {
    fn to_string(&self) -> String {
        todo!()
    }
}

pub struct DataLayout {}

impl ToString for DataLayout {
    fn to_string(&self) -> String {
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
