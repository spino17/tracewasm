use crate::{ast::FuncIndex, error::TraceWasmError};
use wasmparser::ValType;

pub const VM_STACK_INITIAL_ALLOCATION_SIZE: usize = 512 * 1024; // 512Kib

#[derive(Debug, Copy, Clone)]
pub(crate) enum Val {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Ref(Option<FuncIndex>),
}

impl Val {
    pub fn i32_zero() -> Self {
        Val::I32(0)
    }

    pub fn i64_zero() -> Self {
        Val::I64(0)
    }

    pub fn f32_zero() -> Self {
        Val::F32(0.0)
    }

    pub fn f64_zero() -> Self {
        Val::F64(0.0)
    }

    pub fn ref_zero() -> Self {
        Val::Ref(None)
    }

    pub fn zero_of_ty(ty: ValType) -> Result<Self, TraceWasmError> {
        let val = match ty {
            ValType::I32 => Self::i32_zero(),
            ValType::I64 => Self::i64_zero(),
            ValType::F32 => Self::f32_zero(),
            ValType::F64 => Self::f64_zero(),
            ValType::Ref(_) => Self::ref_zero(),
            ValType::V128 => return Err(TraceWasmError::Unsupported("v128 type".to_string())),
        };

        Ok(val)
    }

    pub fn is_ty(&self, ty: ValType) -> Result<bool, TraceWasmError> {
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

pub(crate) struct Locals {
    inner: Vec<Val>, // size = params + declared locals
}

impl Locals {
    pub fn new(locals: Vec<Val>) -> Self {
        Locals { inner: locals }
    }

    pub fn set(&mut self, index: usize, val: Val) {
        self.inner[index] = val;
    }

    pub fn get(&self, index: usize) -> Val {
        self.inner[index]
    }
}

pub(crate) struct Stack<T> {
    inner: Vec<T>,
    stack_pointer: usize, // points to the top of the stack
}

impl<T> Default for Stack<T> {
    fn default() -> Self {
        Stack {
            inner: Vec::with_capacity(VM_STACK_INITIAL_ALLOCATION_SIZE),
            stack_pointer: 0,
        }
    }
}

impl<T: Clone> Stack<T> {
    pub fn push(&mut self, val: T) {
        if self.stack_pointer < self.inner.len() {
            self.inner[self.stack_pointer] = val;
        } else {
            self.inner.push(val);
        }

        self.stack_pointer += 1;
    }

    pub fn pop(&mut self) -> T {
        let val = self.inner[self.stack_pointer - 1].clone();
        self.stack_pointer -= 1;

        val
    }

    pub fn pops(&mut self, num: u32) -> Vec<T> {
        let mut v = Vec::with_capacity(num as usize);

        for i in 0..(num as usize) {
            v.push(self.inner[self.stack_pointer - 1 - i].clone());
        }

        self.stack_pointer -= num as usize;

        v
    }

    pub fn pops_and_reverse(&mut self, num: u32) -> Vec<T> {
        let mut v = Vec::with_capacity(num as usize);

        for i in 0..(num as usize) {
            v.push(self.inner[self.stack_pointer - num as usize + i].clone());
        }

        self.stack_pointer -= num as usize;

        v
    }

    pub fn truncate(&mut self, new_height: usize) {
        self.stack_pointer = new_height;
    }

    pub fn truncate_by_preserving_arity(&mut self, new_height: usize, arity: u32) {
        let arity = arity as usize;

        for i in 0..arity {
            self.inner[new_height + i] =
                self.inner[self.stack_pointer as usize - arity + i].clone();
        }

        self.stack_pointer = new_height + arity;
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
        assert!(matches!(Val::zero_of_ty(ValType::I32).unwrap(), Val::I32(0)));
        assert!(matches!(Val::zero_of_ty(ValType::I64).unwrap(), Val::I64(0)));
        assert!(matches!(Val::zero_of_ty(ValType::F32).unwrap(), Val::F32(x) if x == 0.0));
        assert!(matches!(Val::zero_of_ty(ValType::F64).unwrap(), Val::F64(x) if x == 0.0));
        assert!(matches!(
            Val::zero_of_ty(ValType::FUNCREF).unwrap(),
            Val::Ref(None)
        ));
    }

    #[test]
    fn zero_of_ty_rejects_v128() {
        assert!(Val::zero_of_ty(ValType::V128).is_err());
    }

    #[test]
    fn is_ty_matches_and_rejects() {
        assert!(Val::I32(1).is_ty(ValType::I32).unwrap());
        assert!(!Val::I32(1).is_ty(ValType::I64).unwrap());
        assert!(!Val::I32(1).is_ty(ValType::F32).unwrap());

        assert!(Val::F64(1.0).is_ty(ValType::F64).unwrap());
        assert!(!Val::F64(1.0).is_ty(ValType::I32).unwrap());

        assert!(Val::Ref(None).is_ty(ValType::FUNCREF).unwrap());
        assert!(!Val::Ref(None).is_ty(ValType::I32).unwrap());
    }

    #[test]
    fn is_ty_rejects_v128() {
        assert!(Val::I32(1).is_ty(ValType::V128).is_err());
    }
}
