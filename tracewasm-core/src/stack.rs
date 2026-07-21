pub enum Value {
    I32(i32),
    I64(i64),
}

pub struct Stack {
    inner: Vec<Value>,
    stack_pointer: usize,
}

impl Stack {
    pub fn push(&mut self, val: Value) {
        self.inner.push(val);
    }

    pub fn pop(&mut self) -> Option<Value> {
        self.inner.pop()
    }
}
