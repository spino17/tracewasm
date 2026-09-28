use llvm_sys::target_machine::LLVMCodeGenOptLevel;
use tracewasm_jit::error::JITError;
use tracewasm_jit::{JITHandler, OptLevel};

// Two exports and one import.
//
// `sum_to` is written the way a simple front end emits code: every local lives in
// an `alloca`, and every use is a load or store. The optimizer should promote the
// locals to registers and then replace the whole loop with a closed-form formula.
const IR: &str = r#"
@msg = private constant [13 x i8] c"Hello, world!"

declare void @host_print(ptr, i64)

define i32 @hello(i32 %x) {
entry:
  call void @host_print(ptr @msg, i64 13)
  %r = add i32 %x, 1
  ret i32 %r
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
"#;

// The host implementation of the import. `link_host_func` checks its signature
// against the declaration.
extern "C" fn host_print(ptr: *const u8, len: u64) {
    // SAFETY: generated code passes a pointer to `len` valid bytes.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };

    println!("{}", String::from_utf8_lossy(bytes));
}

fn main() -> Result<(), JITError> {
    // 1. Create the JIT, with code generation at the aggressive level.
    let jit = JITHandler::new(LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive)?;

    // 2. Parse the IR, link the import, then run the optimizer for the host CPU.
    //    Linking has to come first; `optimize` would otherwise treat library-named
    //    declarations as the C library's functions.
    let mut module = jit.parse_module("hello_module", IR)?;

    module.link_host_func("host_print", host_print as extern "C" fn(_, _))?;
    module.optimize(OptLevel::O3)?;

    let instance = unsafe { module.compile() }?;

    let hello = instance.get_func::<(i32,), i32>("hello")?;
    let sum_to = instance.get_func::<(i64,), i64>("sum_to")?;

    // 5. Call them.
    println!("===== running =====");
    println!("hello(41) returned {}", hello.call(41));
    println!("sum_to(10) returned {}", sum_to.call(10));
    println!("sum_to(1_000_000) returned {}", sum_to.call(1_000_000));

    Ok(())
}
