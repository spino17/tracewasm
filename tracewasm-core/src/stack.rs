//! The runtime operand stack.
//!
//! This pairs with the height model precomputed in [`crate::instruction`]: a
//! branch resets the logical top of the stack to `recorded_height` in O(1) by
//! moving [`Stack::stack_pointer`] rather than popping element by element.
//!
//! Note: this module is not yet wired into `lib.rs`, so it is not compiled as
//! part of the crate.

/// A value on the operand stack.
///
/// Only the integer types are modelled so far.
pub enum Value {
    I32(i32),
    I64(i64),
}

/// A WebAssembly operand stack with a logical top decoupled from the backing
/// vector's length.
pub struct Stack {
    inner: Vec<Value>,
    /// Logical height of the stack (index one past the top value).
    ///
    /// Invariant: this may be **less** than `inner.len()`. Branching truncates
    /// the stack by lowering `stack_pointer` without deallocating the popped
    /// slots, so always use `stack_pointer` — never `inner.len()` — for the
    /// current height.
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
