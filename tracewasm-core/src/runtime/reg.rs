use crate::runtime::value::Value;

pub(crate) struct RegFrame {
    inner: Vec<Value>,
}

impl Default for RegFrame {
    fn default() -> Self {
        todo!()
    }
}
