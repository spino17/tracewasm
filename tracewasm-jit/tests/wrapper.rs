use llvm_sys::target_machine::LLVMCodeGenOptLevel;
use tracewasm_jit::JITHandler;
use tracewasm_jit::error::JITError;

const IR: &str = r#"
define i64 @sum_to(i64 %n) {
entry:
  %i = alloca i64
  store i64 0, ptr %i
  %v = load i64, ptr %i
  %r = add i64 %v, %n
  ret i64 %r
}
"#;

fn jit() -> JITHandler {
    JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault).unwrap()
}

#[test]
fn parses_and_optimizes() {
    let jit = jit();
    let mut module = jit.parse_module("m", IR).unwrap();

    module
        .optimize(LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)
        .unwrap();
}

#[test]
fn optimize_can_run_twice() {
    let jit = jit();
    let mut module = jit.parse_module("m", IR).unwrap();

    module
        .optimize(LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)
        .unwrap();
    module
        .optimize(LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)
        .unwrap();
}

#[test]
fn several_modules_share_one_jit() {
    let jit = jit();
    let mut a = jit.parse_module("a", IR).unwrap();
    let mut b = jit.parse_module("b", IR).unwrap();

    a.optimize(LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)
        .unwrap();
    b.optimize(LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)
        .unwrap();
}

#[test]
fn bad_ir_is_an_error() {
    let jit = jit();
    let err = jit.parse_module("m", "define i32 @f( {").err().unwrap();

    assert!(matches!(err, JITError::LLVMError(ref s) if s.starts_with("IR parse error")));
}

#[test]
fn nul_in_name_is_an_error_not_a_panic() {
    let jit = jit();
    let err = jit.parse_module("a\0b", IR).err().unwrap();

    assert!(matches!(err, JITError::InvalidModuleName));
}
