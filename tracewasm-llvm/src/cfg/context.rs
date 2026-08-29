use crate::{
    cfg::{
        basic_block::BasicBlock,
        function::{FuncId, Function},
    },
    interner::{ConstInterner, StrInterner},
};
use id_arena::Arena;

pub struct Context {
    pub(crate) blocks: Arena<BasicBlock>,
    pub(crate) funcs: Arena<Function>,
    pub(crate) str_interner: StrInterner,
    pub(crate) const_interner: ConstInterner,
}

impl Default for Context {
    fn default() -> Self {
        Context {
            blocks: Arena::default(),
            funcs: Arena::default(),
            str_interner: StrInterner::default(),
            const_interner: ConstInterner::default(),
        }
    }
}

impl Context {
    pub(crate) fn name_for_reg(&mut self, name: Option<&str>, func_id: FuncId) -> String {
        todo!()
    }
}
