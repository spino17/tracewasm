pub enum Value {
    I32(i32),
    I64(i64),
}

pub struct Stack {
    inner: Vec<Value>,
    // points to the top of the stack.
    // NOTE: this may or may not be the last element of the array because in branching we truncate the stack without deallocating anything!
    // so always use `stack_pointer` for the height of the stack and not `.len()`
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
