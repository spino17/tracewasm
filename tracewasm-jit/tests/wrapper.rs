use llvm_sys::target_machine::LLVMCodeGenOptLevel;
use tracewasm_jit::error::JITError;
use tracewasm_jit::{JITHandler, OptLevel};

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

    module.optimize(OptLevel::O3).unwrap();
}

#[test]
fn optimize_can_run_twice() {
    let jit = jit();
    let mut module = jit.parse_module("m", IR).unwrap();

    module.optimize(OptLevel::O3).unwrap();
    module.optimize(OptLevel::O3).unwrap();
}

#[test]
fn several_modules_share_one_jit() {
    let jit = jit();
    let mut a = jit.parse_module("a", IR).unwrap();
    let mut b = jit.parse_module("b", IR).unwrap();

    a.optimize(OptLevel::O3).unwrap();
    b.optimize(OptLevel::O3).unwrap();
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

#[test]
fn every_opt_level_runs() {
    let jit = jit();
    let mut module = jit.parse_module("m", IR).unwrap();

    for level in [
        OptLevel::O0,
        OptLevel::O1,
        OptLevel::O2,
        OptLevel::O3,
        OptLevel::Os,
        OptLevel::Oz,
    ] {
        module.optimize(level).unwrap();
    }
}

#[test]
fn parseable_but_invalid_ir_fails_verification() {
    let jit = jit();
    // `%a` uses `%b` before it's defined: valid syntax, invalid IR.
    let ir = "define i64 @f() {\n  %a = add i64 %b, 1\n  %b = add i64 1, 1\n  ret i64 %a\n}";
    let err = jit.parse_module("m", ir).err().unwrap();

    assert!(
        matches!(err, JITError::InvalidModule(ref s) if s.contains("dominate")),
        "{err:?}"
    );
}

#[test]
fn foreign_architecture_is_rejected() {
    let jit = jit();
    let triple = if cfg!(target_arch = "x86_64") {
        "aarch64-unknown-linux-gnu"
    } else {
        "x86_64-unknown-linux-gnu"
    };
    let ir = format!("target triple = \"{triple}\"\n{IR}");
    let err = jit.parse_module("m", &ir).err().unwrap();

    assert!(matches!(err, JITError::TripleMismatch { ref module, .. } if module == triple));
}

#[test]
#[cfg(target_arch = "aarch64")]
fn arm64_alias_and_other_os_spelling_are_accepted() {
    let jit = jit();
    let ir = format!("target triple = \"arm64-apple-macosx\"\n{IR}");

    jit.parse_module("m", &ir).unwrap();
}

#[test]
fn foreign_data_layout_is_rejected() {
    let jit = jit();
    // Leaves most fields to LLVM's defaults, which aren't the host's.
    let ir = format!("target datalayout = \"e-p:32:32\"\n{IR}");
    let err = jit.parse_module("m", &ir).err().unwrap();

    assert!(
        matches!(err, JITError::DataLayoutMismatch { ref module, .. } if module == "e-p:32:32")
    );
}

#[test]
fn same_architecture_on_another_os_is_rejected() {
    let jit = jit();
    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "x86_64"
    };
    let other_os = if cfg!(target_os = "linux") {
        "apple-macosx"
    } else {
        "unknown-linux-gnu"
    };
    let triple = format!("{arch}-{other_os}");
    let ir = format!("target triple = \"{triple}\"\n{IR}");
    let err = jit.parse_module("m", &ir).err().unwrap();

    assert!(matches!(err, JITError::TripleMismatch { ref module, .. } if *module == triple));
}
