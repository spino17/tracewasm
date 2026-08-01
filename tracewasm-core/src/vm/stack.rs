//! The VM's operand stack, its value representation (`Val`), and function
//! locals.
//!
//! ## Operand stack design
//!
//! `Stack<T>` is a growable stack whose logical top is tracked by an explicit
//! `stack_pointer` that is **decoupled from the backing `Vec`'s length**. This
//! is the key to matching the height model precomputed in
//! [`crate::instruction`]: a branch resets the stack to a known height in O(1)
//! by moving the pointer, never by freeing elements.
//!
//! The invariant is `stack_pointer <= inner.len()`. Popping and truncating only
//! lower `stack_pointer`, leaving the popped slots allocated; a later push
//! overwrites a stale slot in place instead of reallocating. Consequently the
//! live region is always `inner[..stack_pointer]`, and `inner.len()` reflects
//! the high-water mark, not the current height. The backing storage is reserved
//! once (`VM_STACK_INITIAL_ALLOCATION_SIZE`) so steady-state execution does not
//! reallocate.
//!
//! One `Stack` is held per [`Instance`](crate::instance::Instance) and shared by
//! every frame of a call tree; [`Stack::reset`] empties it at each top-level entry
//! without releasing the storage. Locals live below the operand region — see the
//! frame layout in [`crate::vm`].
//!
//! ## Ordering conventions
//!
//! The stack grows upward: `inner[stack_pointer - 1]` is the top. Bulk pops come
//! in two flavors that differ only in the order of the returned `Vec`:
//! `pops` returns top-first (the former top at `v[0]`), `pops_and_reverse`
//! returns push order (deepest-first). Both are currently test-only utilities;
//! execution instead uses `truncate_by_preserving_arity` for branch unwinding
//! and `pop_params`/`pop_results` for call arguments and results.
//!
//! ## Preconditions
//!
//! The stack methods are infallible in signature: they assume the caller respects
//! the operand-stack discipline that WebAssembly validation already guarantees
//! (never pop below zero, never preserve more values than are present, only
//! truncate downward). Violations panic via index or underflow rather than
//! returning an error, so the bounds check remains as a backstop.
//!
//! [`Instance::stack`](crate::instance::Instance)'s backing storage is also read
//! directly, bypassing these methods, by the locals accessors in [`crate::vm`]:
//! locals sit below `stack_pointer` for the whole life of a frame, so the operand
//! discipline never touches them. Those accessors index unchecked and carry the
//! safety argument for it.

use crate::{
    error::TraceWasmError,
    instance::traits::{ParamVals, ResultVals},
    module::{FuncIndex, ValType},
};
use smallvec::smallvec;

/// Elements of backing storage reserved for a fresh operand stack, sized so a
/// normal function's execution never has to reallocate mid-run.
pub const VM_STACK_INITIAL_ALLOCATION_SIZE: usize = 512 * 1024; // 512Kib

/// A concrete runtime value on the operand stack or in a local slot.
///
/// One variant per supported WebAssembly value type. `V128` (SIMD) is
/// intentionally absent and rejected at the `Val` constructors below. `Ref`
/// holds an optional function index — `None` is a null reference.
#[derive(Debug, Copy, Clone)]
pub enum Val {
    /// A 32-bit integer value.
    I32(i32),
    /// A 64-bit integer value.
    I64(i64),
    /// A 32-bit float value.
    F32(f32),
    /// A 64-bit float value.
    F64(f64),
    /// A nullable function reference (`None` is a null reference).
    Ref(Option<FuncIndex>),
}

/// Reports an operand whose variant is not the one the instruction expected.
///
/// The accessors below are called once or more per interpreted instruction, so
/// their failure path is outlined rather than written inline. `panic!` in the body
/// would put a call there, and anything the accessor holds across it would have to
/// occupy a callee-saved register — whose save and restore is emitted at the
/// function's entry and exit, and so is paid on every call.
///
/// One helper per type, each taking no arguments, so there is nothing to keep
/// live. They diverge, which lets the compiler reach them with a plain branch
/// instead of a call.
mod wrong_ty {
    #[inline(never)]
    pub fn i32() -> ! {
        panic!("value is not i32")
    }
    #[inline(never)]
    pub fn i64() -> ! {
        panic!("value is not i64")
    }
    #[inline(never)]
    pub fn f32() -> ! {
        panic!("value is not f32")
    }
    #[inline(never)]
    pub fn f64() -> ! {
        panic!("value is not f64")
    }
    #[inline(never)]
    pub fn reference() -> ! {
        panic!("value is not ref")
    }
}

mod tracewasm_unreachable {
    #[inline(never)]
    pub fn unreachable() -> ! {
        unreachable!()
    }
}

impl Val {
    /// The default `i32` value (`0`).
    pub fn i32_zero() -> Self {
        Val::I32(0)
    }

    /// The default `i64` value (`0`).
    pub fn i64_zero() -> Self {
        Val::I64(0)
    }

    /// The default `f32` value (`+0.0`).
    pub fn f32_zero() -> Self {
        Val::F32(0.0)
    }

    /// The default `f64` value (`+0.0`).
    pub fn f64_zero() -> Self {
        Val::F64(0.0)
    }

    /// The default reference value (a null reference).
    pub fn ref_zero() -> Self {
        Val::Ref(None)
    }

    /// Unwraps an `i32` value. Panics if this value is not an `I32`; callers rely
    /// on validation having already type-checked the operand.
    pub fn as_i32(&self) -> i32 {
        let Val::I32(val) = self else { wrong_ty::i32() };

        *val
    }

    /// Unwraps an `i64` value. Panics if this value is not an `I64`.
    pub fn as_i64(&self) -> i64 {
        let Val::I64(val) = self else { wrong_ty::i64() };

        *val
    }

    /// Unwraps an `f32` value. Panics if this value is not an `F32`.
    pub fn as_f32(&self) -> f32 {
        let Val::F32(val) = self else { wrong_ty::f32() };

        *val
    }

    /// Unwraps an `f64` value. Panics if this value is not an `F64`.
    pub fn as_f64(&self) -> f64 {
        let Val::F64(val) = self else { wrong_ty::f64() };

        *val
    }

    /// Unwraps a reference value. Panics if this value is not a `Ref`.
    pub fn as_ref(&self) -> Option<FuncIndex> {
        let Val::Ref(val) = self else {
            wrong_ty::reference()
        };

        *val
    }

    /// Returns the zero/default value for `ty`, as used to initialize declared
    /// locals per the WebAssembly spec.
    ///
    /// # Panics
    ///
    /// Panics on `V128`, which the VM does not model. Infallible in practice:
    /// `Module::compile` rejects a `v128` local, so no compiled module can reach
    /// this with one.
    pub fn zero_of_ty(ty: ValType) -> Self {
        match ty {
            ValType::I32 => Self::i32_zero(),
            ValType::I64 => Self::i64_zero(),
            ValType::F32 => Self::f32_zero(),
            ValType::F64 => Self::f64_zero(),
            ValType::Ref(_) => Self::ref_zero(),
            ValType::V128 => unreachable!(
                "hitting this means the validation in `compile` method in module/mod.rs is incorrect"
            ),
        }
    }

    /// Whether this value's variant matches the WebAssembly type `ty`.
    ///
    /// Used in debug assertions to confirm supplied arguments match a function's
    /// declared parameter types.
    ///
    /// # Errors
    ///
    /// Returns [`TraceWasmError::Unsupported`] for `V128`.
    pub fn has_ty(&self, ty: ValType) -> Result<bool, TraceWasmError> {
        let val = match ty {
            ValType::I32 => matches!(self, Val::I32(_)),
            ValType::I64 => matches!(self, Val::I64(_)),
            ValType::F32 => matches!(self, Val::F32(_)),
            ValType::F64 => matches!(self, Val::F64(_)),
            ValType::Ref(_) => matches!(self, Val::Ref(_)),
            ValType::V128 => return Err(TraceWasmError::Unsupported("v128 type".to_string())),
        };

        Ok(val)
    }
}

impl From<Val> for Value {
    fn from(value: Val) -> Self {
        match value {
            Val::I32(val) => Value::from_i32(val),
            Val::I64(val) => Value::from_i64(val),
            Val::F32(val) => Value::from_f32(val),
            Val::F64(val) => Value::from_f64(val),
            Val::Ref(func_ref) => Value::from_ref(func_ref),
        }
    }
}

const TAG_SHIFT: u32 = 56;
const TAG_SOME: u64 = 1 << TAG_SHIFT;

#[derive(Clone, Copy)]
pub(crate) struct Value(u64);

impl Value {
    #[inline(always)]
    pub fn from_i32(val: i32) -> Self {
        Value(val as u32 as u64)
    }

    #[inline(always)]
    pub fn from_i64(val: i64) -> Self {
        Value(val as u64)
    }

    #[inline(always)]
    pub fn from_f32(val: f32) -> Self {
        Value(val.to_bits() as u64)
    }

    #[inline(always)]
    pub fn from_f64(val: f64) -> Self {
        Value(val.to_bits())
    }

    #[inline(always)]
    pub fn from_ref(func_ref: Option<FuncIndex>) -> Self {
        let x = match func_ref {
            Some(x) => x.0,
            None => 0,
        } as u64;

        let tag = func_ref.is_some() as u64;

        Value((tag << TAG_SHIFT) | x)
    }

    #[inline(always)]
    pub fn as_i32(&self) -> i32 {
        self.0 as u32 as i32
    }

    #[inline(always)]
    pub fn as_i64(&self) -> i64 {
        self.0 as i64
    }

    #[inline(always)]
    pub fn as_f32(&self) -> f32 {
        f32::from_bits(self.0 as u32)
    }

    #[inline(always)]
    pub fn as_f64(&self) -> f64 {
        f64::from_bits(self.0)
    }

    #[inline(always)]
    pub fn as_ref(&self) -> Option<FuncIndex> {
        let is_some = (self.0 >> TAG_SHIFT) != 0;
        let val = self.0 as u32;
        if is_some { Some(FuncIndex(val)) } else { None }
    }

    #[inline(always)]
    pub fn zero_of_ty(ty: ValType) -> Self {
        match ty {
            ValType::I32 => Value::from_i32(0),
            ValType::I64 => Value::from_i64(0),
            ValType::F32 => Value::from_f32(0.0),
            ValType::F64 => Value::from_f64(0.0),
            ValType::Ref(_) => Value::from_i32(0),
            ValType::V128 => tracewasm_unreachable::unreachable(),
        }
    }
}

/// A materialized table instance: its function-reference slots and the maximum
/// number of elements it may grow to.
pub(crate) struct TableVal {
    /// The table's slots, each a nullable function reference.
    pub table: Vec<Option<FuncIndex>>,
    /// The maximum element count the table may grow to.
    pub maximum: u64,
}

/// A passive element segment's runtime state: its remaining function references,
/// or dropped once consumed.
pub(crate) enum ElementVal {
    /// The segment has been dropped (via `elem.drop` or an active init).
    Dropped,
    /// A still-live passive segment holding nullable function references.
    Passive(Box<[Option<FuncIndex>]>),
}

/// A passive data segment's runtime state: its remaining bytes, or dropped once
/// consumed.
pub(crate) enum DataVal {
    /// The segment has been dropped (via `data.drop` or an active init).
    Dropped,
    /// A still-live passive segment holding its raw byte blob.
    Passive(Box<[u8]>), // data blob
}

/// Reports a pop from an empty stack.
///
/// Not generic and takes no arguments, so one copy is shared by every `Stack<T>`
/// and callers keep nothing live for it.
#[inline(never)]
fn pop_underflow() -> ! {
    panic!("pop from an empty operand stack")
}

/// A LIFO operand stack whose logical height (`stack_pointer`) is tracked
/// independently of the backing vector's length. See the module docs for the
/// invariants and rationale.
///
/// Generic over the element type so the same machinery can hold runtime `Val`s
/// during execution and be unit-tested with simpler types.
pub(crate) struct Stack<T> {
    /// Backing storage. Only `inner[..stack_pointer]` is live; slots at or above
    /// `stack_pointer` are stale leftovers kept to avoid reallocation.
    pub inner: Vec<T>,
    /// Logical height: index one past the top value. The top is
    /// `inner[stack_pointer - 1]`. Always `<= inner.len()`.
    stack_pointer: usize, // points to the top of the stack
}

impl<T> Default for Stack<T> {
    /// Creates an empty stack with `VM_STACK_INITIAL_ALLOCATION_SIZE` elements of
    /// capacity reserved up front, so steady-state pushes never reallocate.
    fn default() -> Self {
        Stack {
            inner: Vec::with_capacity(VM_STACK_INITIAL_ALLOCATION_SIZE),
            stack_pointer: 0,
        }
    }
}

impl<T: Clone> Stack<T> {
    /// Creates an empty stack sized for constant-expression evaluation, which
    /// needs only a handful of slots, avoiding the large `Default` reservation.
    pub(crate) fn for_const_expr_evaluation() -> Self {
        Stack {
            inner: Vec::with_capacity(2), // needs very small stack
            stack_pointer: 0,
        }
    }

    /// Empties the stack in O(1) by moving the pointer back to 0.
    ///
    /// The backing storage is left untouched, so both the allocation and the
    /// high-water mark (`inner.len()`) survive. That is what keeps
    /// [`Self::push_grow`] rare across the life of an instance rather than rare
    /// only within a single call.
    pub fn reset(&mut self) {
        self.stack_pointer = 0;
    }

    /// The current logical height (number of live values).
    pub fn height(&self) -> u32 {
        self.stack_pointer as u32
    }

    /// Pushes `val` onto the top of the stack.
    ///
    /// Reuses a stale slot when the pointer is below `inner.len()` (i.e. after a
    /// prior pop/truncate), only growing the backing vector when the pointer is
    /// already at the high-water mark. Either way `stack_pointer` advances by one.
    ///
    /// Growth is outlined into [`Self::push_grow`] rather than calling `Vec::push`
    /// here, and that is worth 5-10% of interpreter throughput. `Vec::push`'s body
    /// calls the allocator, and values this function loads before the branch would
    /// then be live across that call — forcing them into callee-saved registers,
    /// whose save and restore is emitted at the function's entry and exit and so is
    /// paid on every push. Keeping the call in a separate function leaves nothing
    /// on the common path live across it, so the frame setup sinks into the cold
    /// branch.
    ///
    /// Do not add `#[cold]` to the helper: it measures slower than `#[inline(never)]`
    /// alone.
    pub fn push(&mut self, val: T) {
        if self.stack_pointer < self.inner.len() {
            self.inner[self.stack_pointer] = val;
            self.stack_pointer += 1;
        } else {
            self.push_grow(val);
        }
    }

    /// The high-water-mark case of [`Self::push`]: extends the backing storage.
    ///
    /// Runs at most once per stack slot over the life of the stack. Since the stack
    /// lives on the [`Instance`](crate::instance::Instance) and [`Self::reset`]
    /// leaves `inner` alone, that is once per slot per instance rather than per
    /// call — on the order of tens of calls against hundreds of thousands of pushes.
    ///
    /// Kept out of line so [`Self::push`] stays cheap; see the note there.
    #[inline(never)]
    fn push_grow(&mut self, val: T) {
        self.inner.push(val);
        self.stack_pointer += 1;
    }

    /// Removes and returns the top value.
    ///
    /// Precondition: the stack is non-empty. Popping an empty stack underflows
    /// `stack_pointer` and panics.
    pub fn pop(&mut self) -> T {
        // Indexing would reach `panic_bounds_check`, which takes the index and the
        // length as arguments — both must be materialised and held to the call, and
        // a value live across a call has to occupy a callee-saved register whose
        // save and restore is then paid on every pop. `pop_underflow` takes nothing.
        let sp = self.stack_pointer.wrapping_sub(1);

        let Some(val) = self.inner.get(sp) else {
            pop_underflow()
        };

        let val = val.clone();
        self.stack_pointer = sp;

        val
    }

    /// Removes the top `num` values and returns them **top-first**: the returned
    /// `v[0]` is the former top, `v[num - 1]` the deepest popped value.
    ///
    /// Precondition: at least `num` values are present.
    pub fn pops(&mut self, num: u32) -> Vec<T> {
        let mut v = Vec::with_capacity(num as usize);

        for i in 0..(num as usize) {
            v.push(self.inner[self.stack_pointer - 1 - i].clone());
        }

        self.stack_pointer -= num as usize;

        v
    }

    /// Removes the top `num` values and returns them in **push order**: `v[0]` is
    /// the deepest popped value, `v[num - 1]` the former top.
    ///
    /// This is the order arguments were pushed, convenient for binding a call's
    /// operands into the callee's locals (arg0..argN-1).
    ///
    /// Precondition: at least `num` values are present.
    pub fn pops_and_reverse(&mut self, num: u32) -> Vec<T> {
        let mut v = Vec::with_capacity(num as usize);

        for i in 0..(num as usize) {
            v.push(self.inner[self.stack_pointer - num as usize + i].clone());
        }

        self.stack_pointer -= num as usize;

        v
    }

    /// Sets the logical height to `new_height`, discarding everything above it in
    /// O(1) (backing storage is retained).
    ///
    /// Precondition: `new_height <= stack_pointer` (downward only). Raising the
    /// height would expose stale slots.
    pub fn truncate(&mut self, new_height: u32) {
        self.stack_pointer = new_height as usize;
    }

    /// Unwinds to `new_height` while preserving the top `arity` values — the
    /// stack shape a taken branch produces.
    ///
    /// The top `arity` values (at `[stack_pointer - arity, stack_pointer)`) are
    /// moved down to start at `new_height`, in order, and the height becomes
    /// `new_height + arity`. Any values between `new_height` and the preserved
    /// block are dropped.
    ///
    /// The copy runs bottom-up (`new_height + i` from `stack_pointer - arity + i`)
    /// so that when the source and destination ranges overlap — the common case,
    /// since a branch usually keeps a few results above a handful of block-local
    /// operands — a destination write never clobbers a not-yet-read source slot.
    ///
    /// Precondition: `new_height + arity <= stack_pointer` (downward move).
    pub fn truncate_by_preserving_arity(&mut self, new_height: u32, arity: u32) {
        let arity = arity as usize;

        for i in 0..arity {
            self.inner[new_height as usize + i] =
                self.inner[self.stack_pointer - arity + i].clone();
        }

        self.stack_pointer = new_height as usize + arity;
    }

    /// Returns a clone of the top value **without** removing it (a peek).
    ///
    /// This backs `local.tee`, which reads the top of the stack and writes it to
    /// a local while leaving it in place — unlike `pop`, `stack_pointer` is
    /// unchanged.
    ///
    /// Precondition: the stack is non-empty. Peeking an empty stack underflows
    /// `stack_pointer` and panics.
    pub fn tee(&self) -> T {
        self.inner[self.stack_pointer - 1].clone()
    }
}

impl Stack<Val> {
    /// Removes the top `num` values and returns them as a callee's parameters, in
    /// push order (`arg0..argN-1`) for binding into the callee's locals.
    ///
    /// Precondition: at least `num` values are present.
    pub fn pop_params(&mut self, num: u32) -> ParamVals {
        let mut s = smallvec![];

        for i in 0..(num as usize) {
            s.push(self.inner[self.stack_pointer - num as usize + i]);
        }

        self.stack_pointer -= num as usize;

        ParamVals::new(s)
    }

    /// Removes the top `num` values and returns them as a function's results, in
    /// push order (`result0..resultN-1`).
    ///
    /// Precondition: at least `num` values are present.
    pub fn pop_results(&mut self, num: u32) -> ResultVals {
        let mut s = smallvec![];

        for i in 0..(num as usize) {
            s.push(self.inner[self.stack_pointer - num as usize + i]);
        }

        self.stack_pointer -= num as usize;

        ResultVals::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an empty stack without the multi-megabyte `Default` reservation, so
    /// the suite stays cheap. Same-module access to the private fields lets us
    /// assert on the pointer/backing-vec invariants directly.
    fn stack<T>() -> Stack<T> {
        Stack {
            inner: Vec::new(),
            stack_pointer: 0,
        }
    }

    /// The logically-live portion of the stack (`inner[..sp]`), bottom-to-top.
    /// Slots above `stack_pointer` are stale and intentionally excluded.
    fn live<T: Clone>(s: &Stack<T>) -> Vec<T> {
        s.inner[..s.stack_pointer].to_vec()
    }

    // ------------------------------------------------------------------
    // construction
    // ------------------------------------------------------------------

    #[test]
    fn default_is_empty_and_reserves_capacity() {
        let s: Stack<i32> = Stack::default();

        assert_eq!(s.stack_pointer, 0);
        assert_eq!(s.inner.len(), 0);
        // the allocation is reserved up front so pushes don't realloc mid-execution
        assert!(s.inner.capacity() >= VM_STACK_INITIAL_ALLOCATION_SIZE);
    }

    // ------------------------------------------------------------------
    // push / pop
    // ------------------------------------------------------------------

    #[test]
    fn push_pop_is_lifo() {
        let mut s = stack::<i32>();
        s.push(10);
        s.push(20);
        s.push(30);

        assert_eq!(s.stack_pointer, 3);
        assert_eq!(live(&s), vec![10, 20, 30]);

        assert_eq!(s.pop(), 30);
        assert_eq!(s.pop(), 20);
        assert_eq!(s.pop(), 10);
        assert_eq!(s.stack_pointer, 0);
    }

    #[test]
    fn push_reuses_slot_after_pop_without_growing() {
        let mut s = stack::<i32>();
        s.push(1);
        s.push(2);
        s.push(3); // inner.len() == 3

        s.pop(); // sp == 2, but backing vec is still length 3
        assert_eq!(s.inner.len(), 3);

        s.push(9); // sp (2) < len (3) => overwrite inner[2], do NOT grow
        assert_eq!(s.stack_pointer, 3);
        assert_eq!(s.inner.len(), 3, "push after pop must reuse the freed slot");
        assert_eq!(live(&s), vec![1, 2, 9]);
    }

    #[test]
    fn push_grows_backing_only_when_pointer_at_top() {
        let mut s = stack::<i32>();
        s.push(1); // sp == len == 1
        assert_eq!(s.inner.len(), 1);
        s.push(2); // sp == len == 2, must grow
        assert_eq!(s.inner.len(), 2);
    }

    #[test]
    #[should_panic]
    fn pop_on_empty_panics() {
        // documents the precondition: callers must never pop below 0 (validation
        // guarantees this for real modules). sp - 1 underflows usize here.
        let mut s = stack::<i32>();
        s.pop();
    }

    #[test]
    fn tee_peeks_top_without_consuming() {
        let mut s = stack::<i32>();
        s.push(10);
        s.push(20);

        // repeated tees are non-destructive and observe the same top
        assert_eq!(s.tee(), 20);
        assert_eq!(s.tee(), 20);
        assert_eq!(s.stack_pointer, 2, "tee must not move the pointer");
        assert_eq!(live(&s), vec![10, 20]);

        // and the peeked value is still poppable afterwards
        assert_eq!(s.pop(), 20);
    }

    #[test]
    #[should_panic]
    fn tee_on_empty_panics() {
        let s = stack::<i32>();
        s.tee();
    }

    // ------------------------------------------------------------------
    // pops (top-first) / pops_and_reverse (push order)
    // ------------------------------------------------------------------

    #[test]
    fn pops_returns_top_first_and_consumes() {
        let mut s = stack::<i32>();
        for v in [10, 20, 30, 40] {
            s.push(v);
        }

        let popped = s.pops(3);
        assert_eq!(popped, vec![40, 30, 20], "pops yields top-first");
        assert_eq!(s.stack_pointer, 1);
        assert_eq!(s.pop(), 10);
    }

    #[test]
    fn pops_and_reverse_returns_push_order_and_consumes() {
        let mut s = stack::<i32>();
        for v in [10, 20, 30, 40] {
            s.push(v);
        }

        let popped = s.pops_and_reverse(3);
        assert_eq!(
            popped,
            vec![20, 30, 40],
            "pops_and_reverse yields deepest-first (push order)"
        );
        assert_eq!(s.stack_pointer, 1);
        assert_eq!(s.pop(), 10);
    }

    #[test]
    fn pops_is_the_reverse_of_pops_and_reverse() {
        let mut a = stack::<i32>();
        let mut b = stack::<i32>();
        for v in [1, 2, 3, 4, 5] {
            a.push(v);
            b.push(v);
        }

        let mut top_first = a.pops(4);
        top_first.reverse();
        assert_eq!(top_first, b.pops_and_reverse(4));
    }

    #[test]
    fn pops_zero_is_noop() {
        let mut s = stack::<i32>();
        s.push(7);
        s.push(8);

        assert_eq!(s.pops(0), Vec::<i32>::new());
        assert_eq!(s.pops_and_reverse(0), Vec::<i32>::new());
        assert_eq!(s.stack_pointer, 2);
        assert_eq!(live(&s), vec![7, 8]);
    }

    #[test]
    fn pops_all_empties_the_stack() {
        let mut s = stack::<i32>();
        s.push(1);
        s.push(2);

        assert_eq!(s.pops(2), vec![2, 1]);
        assert_eq!(s.stack_pointer, 0);
    }

    // ------------------------------------------------------------------
    // truncate
    // ------------------------------------------------------------------

    #[test]
    fn truncate_lowers_pointer_but_keeps_backing() {
        let mut s = stack::<i32>();
        for v in [1, 2, 3, 4, 5] {
            s.push(v);
        }

        s.truncate(2);
        assert_eq!(s.stack_pointer, 2);
        assert_eq!(s.inner.len(), 5, "truncate must not deallocate");
        assert_eq!(live(&s), vec![1, 2]);
    }

    #[test]
    fn truncate_to_same_height_is_noop() {
        let mut s = stack::<i32>();
        for v in [1, 2, 3] {
            s.push(v);
        }
        s.truncate(3);
        assert_eq!(s.stack_pointer, 3);
        assert_eq!(live(&s), vec![1, 2, 3]);
    }

    #[test]
    fn truncate_to_zero_empties() {
        let mut s = stack::<i32>();
        s.push(42);
        s.truncate(0);
        assert_eq!(s.stack_pointer, 0);
    }

    #[test]
    fn push_after_truncate_overwrites_stale_slot() {
        let mut s = stack::<i32>();
        for v in [1, 2, 3, 4, 5] {
            s.push(v);
        }
        s.truncate(2); // sp == 2, inner still [1,2,3,4,5]
        s.push(99); // overwrites inner[2]
        assert_eq!(s.stack_pointer, 3);
        assert_eq!(live(&s), vec![1, 2, 99]);
        assert_eq!(s.pop(), 99);
    }

    // ------------------------------------------------------------------
    // truncate_by_preserving_arity — the branch-unwind primitive
    // ------------------------------------------------------------------

    #[test]
    fn tbpa_arity_zero_behaves_like_truncate() {
        let mut s = stack::<i32>();
        for v in [10, 20, 30] {
            s.push(v);
        }
        s.truncate_by_preserving_arity(1, 0);
        assert_eq!(s.stack_pointer, 1);
        assert_eq!(live(&s), vec![10]);
    }

    #[test]
    fn tbpa_overlapping_ranges_preserve_top_values() {
        // The case that broke before: dest [1,2] overlaps source [2,3].
        let mut s = stack::<i32>();
        for v in [10, 20, 30, 40] {
            s.push(v);
        }
        // keep the top 2 results (30, 40), unwind base to height 1
        s.truncate_by_preserving_arity(1, 2);
        assert_eq!(s.stack_pointer, 3);
        assert_eq!(
            live(&s),
            vec![10, 30, 40],
            "top `arity` values must survive intact"
        );
        assert_eq!(s.pop(), 40);
        assert_eq!(s.pop(), 30);
        assert_eq!(s.pop(), 10);
    }

    #[test]
    fn tbpa_identity_when_source_equals_dest() {
        // new_height + arity == sp => every write is a self-assignment.
        let mut s = stack::<i32>();
        for v in [10, 20, 30] {
            s.push(v);
        }
        s.truncate_by_preserving_arity(1, 2);
        assert_eq!(s.stack_pointer, 3);
        assert_eq!(live(&s), vec![10, 20, 30]);
    }

    #[test]
    fn tbpa_non_overlapping_with_gap() {
        // source [4,5] and dest [0,1] are disjoint.
        let mut s = stack::<i32>();
        for v in [10, 20, 30, 40, 50, 60] {
            s.push(v);
        }
        s.truncate_by_preserving_arity(0, 2);
        assert_eq!(s.stack_pointer, 2);
        assert_eq!(live(&s), vec![50, 60]);
    }

    #[test]
    fn tbpa_full_preserve_is_identity() {
        let mut s = stack::<i32>();
        for v in [10, 20, 30] {
            s.push(v);
        }
        s.truncate_by_preserving_arity(0, 3);
        assert_eq!(s.stack_pointer, 3);
        assert_eq!(live(&s), vec![10, 20, 30]);
    }

    // ------------------------------------------------------------------
    // integrated VM-like scenarios
    // ------------------------------------------------------------------

    #[test]
    fn branch_unwind_scenario() {
        // Mirrors: a function with one param, entering a block whose result
        // arity is 1, pushing internal temps, then `br` back to the block's end
        // (recorded_height = 1) keeping the single result on top.
        let mut s = stack::<i32>();
        s.push(100); // function param — lives below the block (base height 1)

        // block body: temps + the result on top
        s.push(200);
        s.push(300);
        s.push(400); // <- the block result

        // `br`: unwind to recorded_height 1, preserving arity 1
        s.truncate_by_preserving_arity(1, 1);

        assert_eq!(s.stack_pointer, 2);
        assert_eq!(live(&s), vec![100, 400], "block result kept, temps dropped");
        assert_eq!(s.pop(), 400);
        assert_eq!(s.pop(), 100);
    }

    #[test]
    fn call_argument_marshalling_scenario() {
        // A call pops its N args; `pops_and_reverse` hands them back in
        // declaration order (arg0..argN-1) for binding into the callee's locals.
        let mut s = stack::<i32>();
        s.push(7); // unrelated value left on the caller stack
        s.push(11); // arg0
        s.push(22); // arg1
        s.push(33); // arg2

        let args = s.pops_and_reverse(3);
        assert_eq!(args, vec![11, 22, 33], "args in declaration order");
        assert_eq!(s.stack_pointer, 1);
        assert_eq!(s.pop(), 7, "only the args were consumed");
    }

    #[test]
    fn interleaved_operations_scenario() {
        let mut s = stack::<i32>();
        s.push(1);
        s.push(2);
        s.push(3);
        assert_eq!(s.pop(), 3); // consume a temp
        s.push(4);
        s.push(5); // stack: [1, 2, 4, 5]
        assert_eq!(live(&s), vec![1, 2, 4, 5]);

        let top_two = s.pops(2);
        assert_eq!(top_two, vec![5, 4]);

        s.truncate(1); // unwind everything but the base
        s.push(9);
        assert_eq!(live(&s), vec![1, 9]);
    }

    // ------------------------------------------------------------------
    // genericity: Stack<T> must work for any Clone type
    // ------------------------------------------------------------------

    #[test]
    fn works_with_heap_owned_values() {
        let mut s = stack::<String>();
        s.push("a".to_string());
        s.push("b".to_string());
        s.push("c".to_string());

        // exercises the Clone-based move path for a non-Copy type
        s.truncate_by_preserving_arity(0, 1);
        assert_eq!(s.stack_pointer, 1);
        assert_eq!(s.pop(), "c".to_string());
    }

    #[test]
    fn works_with_val_the_real_vm_element() {
        let mut s = stack::<Val>();
        s.push(Val::I32(5));
        s.push(Val::F64(2.5));

        assert!(matches!(s.pop(), Val::F64(x) if x == 2.5));
        assert!(matches!(s.pop(), Val::I32(5)));
    }

    // ------------------------------------------------------------------
    // Val helpers used during locals init / type checks
    // ------------------------------------------------------------------

    #[test]
    fn zero_of_ty_produces_typed_zeroes() {
        assert!(matches!(Val::zero_of_ty(ValType::I32), Val::I32(0)));
        assert!(matches!(Val::zero_of_ty(ValType::I64), Val::I64(0)));
        assert!(matches!(Val::zero_of_ty(ValType::F32), Val::F32(x) if x == 0.0));
        assert!(matches!(Val::zero_of_ty(ValType::F64), Val::F64(x) if x == 0.0));
        assert!(matches!(Val::zero_of_ty(ValType::FUNCREF), Val::Ref(None)));
    }

    // `v128` locals are rejected by `Module::compile`, so reaching here is a bug
    // in that validation rather than a supported input — hence a panic, not an
    // error.
    #[test]
    #[should_panic(expected = "module/mod.rs")]
    fn zero_of_ty_panics_on_v128() {
        Val::zero_of_ty(ValType::V128);
    }

    #[test]
    fn is_ty_matches_and_rejects() {
        assert!(Val::I32(1).has_ty(ValType::I32).unwrap());
        assert!(!Val::I32(1).has_ty(ValType::I64).unwrap());
        assert!(!Val::I32(1).has_ty(ValType::F32).unwrap());

        assert!(Val::F64(1.0).has_ty(ValType::F64).unwrap());
        assert!(!Val::F64(1.0).has_ty(ValType::I32).unwrap());

        assert!(Val::Ref(None).has_ty(ValType::FUNCREF).unwrap());
        assert!(!Val::Ref(None).has_ty(ValType::I32).unwrap());
    }

    #[test]
    fn is_ty_rejects_v128() {
        assert!(Val::I32(1).has_ty(ValType::V128).is_err());
    }
}
