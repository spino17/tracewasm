// `#[imported]` functions are linked into every module parsed in the process, so
// these tests live in their own binary, away from the ones that link by hand.

use llvm_sys::target_machine::LLVMCodeGenOptLevel;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tracewasm_jit::error::JITError;
use tracewasm_jit::{JITCompiledInstance, JITHandler, OptLevel, imported};

static PRINTED: Mutex<Vec<String>> = Mutex::new(Vec::new());
static TICKS: AtomicU64 = AtomicU64::new(0);

// A plain Rust function: the macro supplies the C ABI.
#[imported]
fn add_one(x: i64) -> i64 {
    x + 1
}

#[imported]
extern "C" fn already_c(x: i32) -> i32 {
    x * 3
}

#[imported]
fn print_bytes(ptr: *const u8, len: u64) {
    // SAFETY: the IR passes a pointer to `len` bytes of a constant.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };

    PRINTED
        .lock()
        .unwrap()
        .push(String::from_utf8_lossy(bytes).into_owned());
}

#[imported(name = "tick")]
fn count_a_tick() {
    TICKS.fetch_add(1, Ordering::SeqCst);
}

#[imported]
fn mix(a: f64, b: f32, c: u32) -> f64 {
    a + b as f64 + c as f64
}

const IR: &str = r#"
@msg = private constant [5 x i8] c"hello"

declare i64 @add_one(i64)
declare i32 @already_c(i32)
declare void @print_bytes(ptr, i64)
declare void @tick()
declare double @mix(double, float, i32)

define i64 @add_two(i64 %x) {
  %a = call i64 @add_one(i64 %x)
  %b = call i64 @add_one(i64 %a)
  ret i64 %b
}

define i32 @triple(i32 %x) {
  %r = call i32 @already_c(i32 %x)
  ret i32 %r
}

define void @greet() {
  call void @print_bytes(ptr @msg, i64 5)
  call void @tick()
  ret void
}

define double @call_mix() {
  %r = call double @mix(double 1.5, float 2.25, i32 3)
  ret double %r
}
"#;

fn jit() -> JITHandler {
    JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap()
}

fn instance(jit: &JITHandler, optimize: bool) -> JITCompiledInstance<'_> {
    // No `link_host_func` anywhere: `parse_module` links every `#[imported]` fn.
    let mut module = jit.parse_module("m", IR).unwrap();

    if optimize {
        module.optimize(OptLevel::O3).unwrap();
    }

    // SAFETY: every function in `IR` is total.
    unsafe { module.compile() }.unwrap()
}

#[test]
fn imported_functions_are_linked_automatically() {
    for optimize in [false, true] {
        let jit = jit();
        let inst = instance(&jit, optimize);

        assert_eq!(
            inst.get_func::<(i64,), i64>("add_two").unwrap().call(40),
            42
        );
        assert_eq!(inst.get_func::<(i32,), i32>("triple").unwrap().call(5), 15);
        assert_eq!(inst.get_func::<(), f64>("call_mix").unwrap().call(), 6.75);
    }
}

#[test]
fn void_and_pointer_imports_run() {
    let jit = jit();
    let inst = instance(&jit, false);
    let before = TICKS.load(Ordering::SeqCst);

    inst.get_func::<(), ()>("greet").unwrap().call();

    assert!(TICKS.load(Ordering::SeqCst) > before);
    assert!(PRINTED.lock().unwrap().iter().any(|s| s == "hello"));
}

#[test]
fn renamed_import_is_linked_under_its_new_name() {
    // `count_a_tick` isn't declared under its own name anywhere; `tick` resolved,
    // or `compile` would have failed with `UnresolvedSymbol`.
    let jit = jit();
    let module = jit
        .parse_module(
            "m",
            "declare void @count_a_tick()\ndefine void @f() {\n  ret void\n}",
        )
        .unwrap();

    // SAFETY: `f` is total.
    let err = unsafe { module.compile() }.err().unwrap();

    assert!(matches!(err, JITError::UnresolvedSymbol(ref n) if n == "count_a_tick"));
}

#[test]
fn modules_that_declare_no_imports_are_unaffected() {
    let jit = jit();
    let module = jit
        .parse_module("m", "define i64 @f() {\n  ret i64 9\n}")
        .unwrap();

    // SAFETY: `f` is total.
    let inst = unsafe { module.compile() }.unwrap();

    assert_eq!(inst.get_func::<(), i64>("f").unwrap().call(), 9);
}

#[test]
fn a_declaration_with_the_wrong_signature_fails_to_parse() {
    let jit = jit();
    let err = jit
        .parse_module("m", "declare i32 @add_one(i32)")
        .err()
        .unwrap();

    match err {
        JITError::HostFuncSignatureMismatch {
            name,
            declared,
            host,
        } => {
            assert_eq!(name, "add_one");
            assert_eq!(declared, "i32 (i32)");
            assert_eq!(host, "i64 (i64)");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn linking_the_same_name_by_hand_to_a_different_function_is_rejected() {
    extern "C" fn other(x: i64) -> i64 {
        x
    }

    let jit = jit();
    let mut module = jit.parse_module("m", IR).unwrap();
    let err = module
        .link_host_func("add_one", other as extern "C" fn(_) -> _)
        .err()
        .unwrap();

    assert!(matches!(err, JITError::HostFuncAlreadyLinked(ref n) if n == "add_one"));
}
