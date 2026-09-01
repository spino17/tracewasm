use crate::{
    cfg::{function::FuncId, global::Global},
    interner::StrId,
};
use rustc_hash::FxHashSet;

pub struct Module {
    pub(crate) triple: String,
    pub(crate) data_layout: String,
    pub(crate) globals: Vec<Global>,
    pub(crate) functions: Vec<FuncId>,
    pub(crate) func_names: FxHashSet<StrId>,
}
