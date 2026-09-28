use llvm_sys::target_machine::LLVMCodeGenOptLevel;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};
use tracewasm_jit::error::JITError;
use tracewasm_jit::func::{Func, LLVMFunc};
use tracewasm_jit::{JITCompiledInstance, JITHandler};

const IR: &str = r#"
@msg = private constant [13 x i8] c"Hello, world!"

declare void @host_print(ptr, i64)
declare i64 @host_double(i64)

define i32 @hello(i32 %x) {
  call void @host_print(ptr @msg, i64 13)
  %r = add i32 %x, 1
  ret i32 %r
}

define i64 @double_plus_one(i64 %x) {
  %d = call i64 @host_double(i64 %x)
  %r = add i64 %d, 1
  ret i64 %r
}

define i64 @sum_to(i64 %n) {
entry:
  %i = alloca i64
  %acc = alloca i64
  store i64 0, ptr %i
  store i64 0, ptr %acc
  br label %loop
loop:
  %iv = load i64, ptr %i
  %done = icmp sge i64 %iv, %n
  br i1 %done, label %exit, label %body
body:
  %a = load i64, ptr %acc
  %a2 = add i64 %a, %iv
  store i64 %a2, ptr %acc
  %iv2 = add i64 %iv, 1
  store i64 %iv2, ptr %i
  br label %loop
exit:
  %r = load i64, ptr %acc
  ret i64 %r
}

define double @mix(double %x, float %y, i32 %z) {
  %yd = fpext float %y to double
  %zd = sitofp i32 %z to double
  %s = fadd double %x, %yd
  %r = fadd double %s, %zd
  ret double %r
}

define void @store(ptr %p, i64 %v) {
  store i64 %v, ptr %p
  ret void
}

define i64 @answer() {
  ret i64 42
}

define internal i64 @hidden() {
  ret i64 0
}
"#;

static PRINTED: Mutex<Vec<String>> = Mutex::new(Vec::new());
static DOUBLED: AtomicI64 = AtomicI64::new(0);

extern "C" fn host_print(ptr: *const u8, len: u64) {
    // SAFETY: the IR passes a pointer to `len` bytes of `@msg`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };

    PRINTED
        .lock()
        .unwrap()
        .push(String::from_utf8_lossy(bytes).into_owned());
}

extern "C" fn host_double(x: i64) -> i64 {
    DOUBLED.fetch_add(1, Ordering::SeqCst);

    x * 2
}

fn jit() -> JITHandler {
    JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap()
}

fn instance(jit: &JITHandler, optimize: bool) -> JITCompiledInstance<'_> {
    let mut module = jit.parse_module("m", IR).unwrap();

    if optimize {
        module
            .optimize(LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)
            .unwrap();
    }

    module
        .link_host_func("host_print", host_print as extern "C" fn(_, _))
        .unwrap();
    module
        .link_host_func("host_double", host_double as extern "C" fn(_) -> _)
        .unwrap();

    // SAFETY: every function in `IR` is defined for the inputs these tests use.
    unsafe { module.compile() }.unwrap()
}

#[test]
fn calls_a_function() {
    let jit = jit();
    let inst = instance(&jit, false);
    let answer = inst.get_func::<(), i64>("answer").unwrap();

    assert_eq!(answer.call(), 42);
}

#[test]
fn func_exposes_params_and_results() {
    fn params_and_results<F: LLVMFunc<Params = (i64,), Results = i64>>(_: &F) {}

    let jit = jit();
    let inst = instance(&jit, false);
    let sum_to: Func<'_, (i64,), i64> = inst.get_func("sum_to").unwrap();

    params_and_results(&sum_to);
}

#[test]
fn jit_code_calls_back_into_the_host() {
    let jit = jit();
    let inst = instance(&jit, false);
    let before = DOUBLED.load(Ordering::SeqCst);
    let f = inst.get_func::<(i64,), i64>("double_plus_one").unwrap();

    assert_eq!(f.call(20), 41);
    assert!(DOUBLED.load(Ordering::SeqCst) > before);
}

#[test]
fn host_receives_pointer_and_length() {
    let jit = jit();
    let inst = instance(&jit, false);
    let hello = inst.get_func::<(i32,), i32>("hello").unwrap();

    assert_eq!(hello.call(41), 42);
    assert!(PRINTED.lock().unwrap().iter().any(|s| s == "Hello, world!"));
}

#[test]
fn optimized_module_still_runs() {
    let jit = jit();
    let inst = instance(&jit, true);
    let sum_to = inst.get_func::<(i64,), i64>("sum_to").unwrap();

    assert_eq!(sum_to.call(10), 45);
    assert_eq!(sum_to.call(1_000_000), 499_999_500_000);
}

#[test]
fn mixed_float_and_int_arguments() {
    let jit = jit();
    let inst = instance(&jit, false);
    let mix = inst.get_func::<(f64, f32, i32), f64>("mix").unwrap();

    assert_eq!(mix.call(1.5, 2.25, 3), 6.75);
}

#[test]
fn void_function_writes_through_a_pointer() {
    let jit = jit();
    let inst = instance(&jit, false);
    let store = inst.get_func::<(*mut i64, i64), ()>("store").unwrap();
    let mut slot = 0_i64;

    store.call(&mut slot, 7);

    assert_eq!(slot, 7);
}

#[test]
fn funcs_are_copy_and_can_be_called_repeatedly() {
    let jit = jit();
    let inst = instance(&jit, false);
    let sum_to = inst.get_func::<(i64,), i64>("sum_to").unwrap();
    let again = sum_to;

    assert_eq!(sum_to.call(4), 6);
    assert_eq!(again.call(5), 10);
}

#[test]
fn wrong_signature_is_rejected() {
    let jit = jit();
    let inst = instance(&jit, false);
    let err = inst.get_func::<(i32,), i32>("sum_to").err().unwrap();

    match err {
        JITError::FuncSignatureMismatch {
            name,
            declared,
            requested,
        } => {
            assert_eq!(name, "sum_to");
            assert_eq!(declared, "i64 (i64)");
            assert_eq!(requested, "i32 (i32)");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn void_vs_value_return_is_rejected() {
    let jit = jit();
    let inst = instance(&jit, false);
    let err = inst.get_func::<(), ()>("answer").err().unwrap();

    assert!(matches!(err, JITError::FuncSignatureMismatch { .. }));
}

#[test]
fn internal_function_is_not_exported() {
    let jit = jit();
    let inst = instance(&jit, false);
    let err = inst.get_func::<(), i64>("hidden").err().unwrap();

    assert!(matches!(err, JITError::FuncNotExported(ref n) if n == "hidden"));
}

#[test]
fn host_declaration_is_not_an_export() {
    let jit = jit();
    let inst = instance(&jit, false);
    let err = inst.get_func::<(i64,), i64>("host_double").err().unwrap();

    assert!(matches!(err, JITError::FuncNotExported(_)));
}

#[test]
fn unknown_name_is_not_exported() {
    let jit = jit();
    let inst = instance(&jit, false);
    let err = inst.get_func::<(), i64>("nowhere").err().unwrap();

    assert!(matches!(err, JITError::FuncNotExported(_)));
}

#[test]
fn unlinked_host_function_fails_at_lookup() {
    let jit = jit();
    let module = jit
        .parse_module(
            "m",
            "declare i64 @missing()\ndefine i64 @f() {\n  %r = call i64 @missing()\n  ret i64 %r\n}",
        )
        .unwrap();

    // SAFETY: `f` never runs; the lookup fails first. (ORC prints the missing
    // symbol's name to stderr; the error itself only names `f`.)
    let inst = unsafe { module.compile() }.unwrap();
    let err = inst.get_func::<(), i64>("f").err().unwrap();

    assert!(
        matches!(err, JITError::LLVMError(ref s) if s.contains("materialize")),
        "{err:?}"
    );
}

#[test]
fn two_modules_in_one_jit() {
    let jit = jit();
    let a = jit
        .parse_module("a", "define i64 @a() {\n  ret i64 1\n}")
        .unwrap();
    let b = jit
        .parse_module("b", "define i64 @b() {\n  ret i64 2\n}")
        .unwrap();

    // SAFETY: both functions are total.
    let (a, b) = unsafe { (a.compile().unwrap(), b.compile().unwrap()) };

    assert_eq!(a.get_func::<(), i64>("a").unwrap().call(), 1);
    assert_eq!(b.get_func::<(), i64>("b").unwrap().call(), 2);
}

#[test]
fn duplicate_export_across_modules_is_an_error() {
    let jit = jit();
    let ir = "define i64 @same() {\n  ret i64 1\n}";
    let a = jit.parse_module("a", ir).unwrap();
    let b = jit.parse_module("b", ir).unwrap();

    // SAFETY: `same` is total.
    unsafe { a.compile() }.unwrap();

    assert!(matches!(
        unsafe { b.compile() },
        Err(JITError::LLVMError(_))
    ));
}
