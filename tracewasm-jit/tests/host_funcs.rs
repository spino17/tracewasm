use llvm_sys::target_machine::LLVMCodeGenOptLevel;
use tracewasm_jit::JITHandler;
use tracewasm_jit::error::JITError;

const IR: &str = r#"
declare void @host_print(ptr, i64)
declare i64 @add(i64, i64)
declare i32 @narrow(i32)
declare double @scale(double, float)
declare ptr @identity(ptr)
declare void @tick()
declare i32 @variadic(i32, ...)

define i64 @defined(i64 %x) {
  ret i64 %x
}
"#;

extern "C" fn host_print(ptr: *const u8, len: u64) {
    let _ = (ptr, len);
}

extern "C" fn add(a: i64, b: i64) -> i64 {
    a + b
}

extern "C" fn sub(a: i64, b: i64) -> i64 {
    a - b
}

extern "C" fn narrow(x: u32) -> u32 {
    x
}

extern "C" fn scale(x: f64, by: f32) -> f64 {
    x * by as f64
}

extern "C" fn identity(p: *mut i64) -> *mut i64 {
    p
}

extern "C" fn tick() {}

extern "C" fn one(x: i32) -> i32 {
    x
}

type Add = extern "C" fn(i64, i64) -> i64;

fn jit() -> JITHandler {
    JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap()
}

#[test]
fn links_every_supported_shape() {
    let jit = jit();
    let m = jit.parse_module("m", IR).unwrap();

    m.link_host_func("host_print", host_print as extern "C" fn(_, _))
        .unwrap();
    m.link_host_func("add", add as Add).unwrap();
    m.link_host_func("narrow", narrow as extern "C" fn(_) -> _)
        .unwrap();
    m.link_host_func("scale", scale as extern "C" fn(_, _) -> _)
        .unwrap();
    m.link_host_func("identity", identity as extern "C" fn(_) -> _)
        .unwrap();
    m.link_host_func("tick", tick as extern "C" fn()).unwrap();
}

#[test]
fn signature_mismatch_is_rejected() {
    let jit = jit();
    let m = jit.parse_module("m", IR).unwrap();

    // IR: i32 (i32). Host: i64 (i64, i64).
    let err = m.link_host_func("narrow", add as Add).err().unwrap();

    match err {
        JITError::HostFuncSignatureMismatch {
            name,
            declared,
            host,
        } => {
            assert_eq!(name, "narrow");
            assert_eq!(declared, "i32 (i32)");
            assert_eq!(host, "i64 (i64, i64)");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn void_vs_value_return_is_a_mismatch() {
    let jit = jit();
    let m = jit.parse_module("m", "declare i64 @f()").unwrap();
    let err = m
        .link_host_func("f", tick as extern "C" fn())
        .err()
        .unwrap();

    assert!(matches!(err, JITError::HostFuncSignatureMismatch { .. }));
}

#[test]
fn variadic_declaration_is_a_mismatch() {
    let jit = jit();
    let m = jit.parse_module("m", IR).unwrap();
    let err = m
        .link_host_func("variadic", one as extern "C" fn(_) -> _)
        .err()
        .unwrap();

    assert!(matches!(err, JITError::HostFuncSignatureMismatch { .. }));
}

#[test]
fn undeclared_name_is_rejected() {
    let jit = jit();
    let m = jit.parse_module("m", IR).unwrap();
    let err = m.link_host_func("nowhere", add as Add).err().unwrap();

    assert!(matches!(err, JITError::HostFuncNotDeclared(ref n) if n == "nowhere"));
}

#[test]
fn name_with_a_body_in_the_module_is_rejected() {
    let jit = jit();
    let m = jit.parse_module("m", IR).unwrap();
    let err = m.link_host_func("defined", add as Add).err().unwrap();

    assert!(matches!(err, JITError::HostFuncNotDeclared(ref n) if n == "defined"));
}

#[test]
fn nul_in_func_name_is_an_error() {
    let jit = jit();
    let m = jit.parse_module("m", IR).unwrap();
    let err = m.link_host_func("a\0b", add as Add).err().unwrap();

    assert!(matches!(err, JITError::InvalidFuncName));
}

#[test]
fn relinking_the_same_function_is_a_no_op() {
    let jit = jit();
    let m = jit.parse_module("m", IR).unwrap();

    m.link_host_func("add", add as Add).unwrap();
    m.link_host_func("add", add as Add).unwrap();
}

#[test]
fn two_modules_can_share_a_host_function() {
    let jit = jit();
    let a = jit.parse_module("a", IR).unwrap();
    let b = jit.parse_module("b", IR).unwrap();

    a.link_host_func("add", add as Add).unwrap();
    b.link_host_func("add", add as Add).unwrap();
}

#[test]
fn a_different_function_under_a_linked_name_is_rejected() {
    let jit = jit();
    let a = jit.parse_module("a", IR).unwrap();
    let b = jit.parse_module("b", IR).unwrap();

    a.link_host_func("add", add as Add).unwrap();

    let err = b.link_host_func("add", sub as Add).err().unwrap();

    assert!(matches!(err, JITError::HostFuncAlreadyLinked(ref n) if n == "add"));
}
