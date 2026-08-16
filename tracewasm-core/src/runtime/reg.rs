use crate::runtime::value::Value;

pub struct RegFrame {
    inner: Vec<Value>,
}

impl Default for RegFrame {
    fn default() -> Self {
        todo!()
    }
}
