//! Shared plumbing for the wasm → LLVM lowering suite.
//!
//! Three independent checks, because each catches what the others cannot:
//!
//! | Check | Catches |
//! |---|---|
//! | [`assert_verifies`] | IR that is malformed — a phi short an edge, a block with no terminator |
//! | [`run`] vs the interpreter | IR that is well-formed and computes the wrong thing |
//! | structural assertions in the test files | what both of the above are blind to |
//!
//! The third is not redundant. A pass can emit verifying IR that returns the right
//! answer and still be wrong in a way that bites later: results paired with the wrong
//! phi when they happen to share a type, dead code kept alive, a block left
//! unreachable. Those are asserted on the text.
//!
//! # The external tools are optional
//!
//! `llvm-as` and `lli` do the first two checks, and a machine without them still
//! runs the suite — those assertions are skipped with a note rather than failing.
//! `TRACEWASM_REQUIRE_LLVM=1` turns a missing toolchain into a failure, which is what
//! CI should set once it has one.

use std::{path::PathBuf, process::Command, sync::Arc};
use tracewasm_core::{Stack, module::Module};
use tracewasm_llvm::cfg::emit::IREmitter;

/// Lowers a wat module to LLVM IR through the pass under test.
///
/// # Panics
///
/// If the wat is not valid wasm — that is a bug in the test, not the pass, and
/// `wat::parse_str` assembles without validating.
pub fn lower(wat: &str) -> String {
    let bytes = wat::parse_str(wat).expect("invalid wat");

    wasmparser::Validator::new()
        .validate_all(&bytes)
        .expect("wat does not validate — the test would prove nothing");

    let module = Arc::new(Module::<Stack>::compile(&bytes).expect("module compiles"));
    let cfg = module.build_cfg().expect("cfg builds");

    IREmitter::emit(cfg).expect("ir emits")
}

// ---------------------------------------------------------------------------
// The LLVM toolchain

fn llvm_tool(name: &str) -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("/opt/homebrew/opt/llvm/bin").join(name),
        PathBuf::from("/usr/local/opt/llvm/bin").join(name),
        PathBuf::from("/usr/bin").join(name),
        PathBuf::from(name),
    ];

    candidates
        .into_iter()
        .find(|p| Command::new(p).arg("--version").output().is_ok())
}

/// Whether a missing toolchain should fail rather than skip.
fn llvm_required() -> bool {
    std::env::var("TRACEWASM_REQUIRE_LLVM").is_ok_and(|v| v == "1")
}

fn skip(what: &str, tool: &str) -> bool {
    assert!(
        !llvm_required(),
        "`{tool}` not found and TRACEWASM_REQUIRE_LLVM=1, so `{what}` cannot be checked"
    );

    eprintln!("note: `{tool}` not found — skipping `{what}`");

    true
}

fn scratch(name: &str, ext: &str, contents: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("tracewasm-llvm-lowering");

    std::fs::create_dir_all(&dir).expect("scratch dir");

    let path = dir.join(format!("{name}.{ext}"));

    std::fs::write(&path, contents).expect("write scratch");

    path
}

// ---------------------------------------------------------------------------
// Check 1: the IR is well-formed

/// Asserts `ir` assembles and passes LLVM's own verifier.
///
/// The verifier is the point, not just the parser: `llvm-as` alone accepts a phi
/// with a missing edge in an unreachable block, and it is `opt -passes=verify` that
/// reports the structural rules — one phi entry per predecessor, every block
/// terminated.
pub fn assert_verifies(name: &str, ir: &str) {
    let Some(llvm_as) = llvm_tool("llvm-as") else {
        skip("IR verification", "llvm-as");
        return;
    };

    let ll = scratch(name, "ll", ir);
    let bc = ll.with_extension("bc");

    let out = Command::new(llvm_as)
        .arg(&ll)
        .arg("-o")
        .arg(&bc)
        .output()
        .expect("run llvm-as");

    assert!(
        out.status.success(),
        "`{name}` does not assemble:\n{}\n--- IR ---\n{ir}",
        String::from_utf8_lossy(&out.stderr)
    );

    let Some(opt) = llvm_tool("opt") else {
        skip("the LLVM verifier", "opt");
        return;
    };

    let out = Command::new(opt)
        .arg("-passes=verify")
        .arg(&bc)
        .arg("-o")
        .arg("/dev/null")
        .output()
        .expect("run opt");

    assert!(
        out.status.success(),
        "`{name}` does not verify:\n{}\n--- IR ---\n{ir}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---------------------------------------------------------------------------
// Check 2: the IR computes what the interpreter computes

/// The `define` line, parsed: the function's name, parameter types and result type.
///
/// Read off the IR rather than assumed, so the driver below adapts if the pass
/// changes how it names or shapes a function instead of silently calling something
/// that no longer exists.
struct Signature {
    name: String,
    params: Vec<String>,
    result: String,
}

fn signature(ir: &str) -> Signature {
    let line = ir
        .lines()
        .find(|l| l.starts_with("define "))
        .expect("the module defines a function");

    let rest = line.strip_prefix("define ").unwrap();
    let at = rest.find('@').expect("a function name");
    let open = rest.find('(').expect("a parameter list");
    let close = rest.rfind(')').expect("a parameter list");

    let params = rest[open + 1..close]
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            // "i32 %param0" — the type is everything before the register.
            p.split_whitespace()
                .next()
                .expect("a parameter type")
                .to_string()
        })
        .collect();

    Signature {
        name: rest[at + 1..open].to_string(),
        params,
        result: rest[..at].trim().to_string(),
    }
}

/// Runs the lowered function under `lli`, once per argument list, and returns what
/// each call produced.
///
/// Every result is widened to `i64` for printing, so one parser handles `i32` and
/// `i64` alike. A multi-value function contributes one entry per field, in
/// declaration order. `None` means the toolchain is absent and the caller should
/// not assert.
pub fn run(name: &str, ir: &str, calls: &[&[i32]]) -> Option<Vec<Vec<i64>>> {
    let (Some(llvm_as), Some(lli)) = (llvm_tool("llvm-as"), llvm_tool("lli")) else {
        skip("execution", "llvm-as/lli");
        return None;
    };

    let sig = signature(ir);
    let mut driver = String::from(ir);

    driver.push_str(
        "\n@.tracewasm_fmt = private unnamed_addr constant [6 x i8] c\"%lld\\0A\\00\"\n\
         declare i32 @printf(ptr, ...)\n\n\
         define i32 @main() {\nentry:\n",
    );

    // The trailing parameter is the runtime pointer the pass appends; nothing reads
    // it yet, so a null is enough to call through.
    let wasm_param_count = sig.params.len() - 1;

    for (i, args) in calls.iter().enumerate() {
        assert_eq!(
            args.len(),
            wasm_param_count,
            "`{name}` takes {wasm_param_count} wasm parameters, given {}",
            args.len()
        );

        let actuals: Vec<String> = sig.params[..wasm_param_count]
            .iter()
            .zip(args.iter())
            .map(|(ty, a)| format!("{ty} {a}"))
            .chain(std::iter::once("ptr null".to_string()))
            .collect();

        let call = format!("call {} @{}({})", sig.result, sig.name, actuals.join(", "));

        if sig.result == "void" {
            driver.push_str(&format!("  {call}\n"));

            continue;
        }

        // A struct result is printed field by field, which is also what pins the
        // order the pass assembled it in.
        let fields = struct_fields(&sig.result);

        if fields.is_empty() {
            driver.push_str(&format!("  %drv{i} = {call}\n"));
            driver.push_str(&widen_and_print(&sig.result, &format!("%drv{i}"), i, 0));
        } else {
            driver.push_str(&format!("  %drv{i} = {call}\n"));

            for (f, fty) in fields.iter().enumerate() {
                driver.push_str(&format!(
                    "  %drv{i}_f{f} = extractvalue {} %drv{i}, {f}\n",
                    sig.result
                ));
                driver.push_str(&widen_and_print(fty, &format!("%drv{i}_f{f}"), i, f));
            }
        }
    }

    driver.push_str("  ret i32 0\n}\n");

    let ll = scratch(&format!("{name}_driver"), "ll", &driver);
    let bc = ll.with_extension("bc");

    let out = Command::new(llvm_as)
        .arg(&ll)
        .arg("-o")
        .arg(&bc)
        .output()
        .expect("run llvm-as");

    assert!(
        out.status.success(),
        "`{name}` driver does not assemble:\n{}\n--- driver ---\n{driver}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = run_with_timeout(Command::new(lli).arg(&bc), name);

    assert!(
        out.status.success(),
        "`{name}` failed under lli:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let printed: Vec<i64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().parse().expect("a printed integer"))
        .collect();

    // Regroup by call: every call prints one line per result.
    let per_call = if sig.result == "void" {
        0
    } else {
        struct_fields(&sig.result).len().max(1)
    };

    if per_call == 0 {
        return Some(calls.iter().map(|_| vec![]).collect());
    }

    Some(printed.chunks(per_call).map(|c| c.to_vec()).collect())
}

/// The field types of `{ i32, i64 }`, or empty if the type is not a struct.
fn struct_fields(ty: &str) -> Vec<String> {
    let ty = ty.trim();

    if !ty.starts_with('{') {
        return vec![];
    }

    ty.trim_start_matches('{')
        .trim_end_matches('}')
        .split(',')
        .map(|f| f.trim().to_string())
        .filter(|f| !f.is_empty())
        .collect()
}

/// Widens a result to `i64` and prints it, so one format string serves every type.
fn widen_and_print(ty: &str, reg: &str, call: usize, field: usize) -> String {
    let widened = format!("%drv{call}_w{field}");

    let widen = match ty {
        "i64" => format!("  {widened} = add i64 {reg}, 0\n"),
        _ => format!("  {widened} = sext {ty} {reg} to i64\n"),
    };

    format!("{widen}  call i32 (ptr, ...) @printf(ptr @.tracewasm_fmt, i64 {widened})\n")
}

// ---------------------------------------------------------------------------
// The reference: the same module, run by the interpreter

use tracewasm_core::{
    anyhow,
    instance::traits::{ImportRegistry, ImportSignature, ResultVals, Val},
    memory::{MemoryView, linear::LinearMemory},
};

/// A registry for the guests here, none of which import anything.
pub struct NoImports;

impl ImportRegistry for NoImports {
    fn execute<V: MemoryView>(
        &mut self,
        module_name: &str,
        func_name: &str,
        _params: &[Val],
        _memory_view: &mut V,
    ) -> Result<ResultVals, anyhow::Error> {
        unreachable!("guest declares no imports, but called `{module_name}::{func_name}`")
    }

    fn signature(&self, _module_name: &str, _func_name: &str) -> Option<ImportSignature> {
        None
    }

    fn func_count(&self) -> u32 {
        0
    }

    fn global_count(&self) -> u32 {
        0
    }

    fn get_global(&self, module_name: &str, global_name: &str) -> Result<Val, anyhow::Error> {
        unreachable!("guest declares no globals, but read `{module_name}::{global_name}`")
    }
}

/// How a case's export is called through the interpreter.
///
/// One variant per signature the suite uses. The results come back widened to `i64`
/// so they compare directly against what [`run`] parses back out of `lli`.
pub type Interpret = fn(&str, &[i32]) -> Vec<i64>;

macro_rules! interpreter_call {
    ($name:ident, ($($p:ident: $pty:ty),*), ($($rty:ty),*), |$vals:ident| $widen:expr) => {
        /// Calls the export `f` through the interpreter — the reference the lowered
        /// code is checked against.
        pub fn $name(wat: &str, args: &[i32]) -> Vec<i64> {
            let bytes = wat::parse_str(wat).expect("invalid wat");
            let module = Module::<Stack>::compile(&bytes).expect("module compiles");

            let func = module
                .get_typed_func::<($($pty,)*), ($($rty,)*)>("f")
                .expect("export `f`");

            let mut instance = module
                .instantiate::<LinearMemory, _>(NoImports, None)
                .expect("module instantiates");

            let mut it = args.iter();
            $(let $p: $pty = *it.next().expect("an argument") as $pty;)*
            assert!(it.next().is_none(), "too many arguments for `{}`", stringify!($name));

            let $vals = func.call(($($p,)*), &mut instance).expect("call succeeds");

            $widen
        }
    };
}

interpreter_call!(i32_to_i32, (a: i32), (i32), |r| vec![r.0 as i64]);
interpreter_call!(i32x2_to_i32, (a: i32, b: i32), (i32), |r| vec![r.0 as i64]);
interpreter_call!(i32x2_to_i32i64, (a: i32, b: i32), (i32, i64), |r| vec![
    r.0 as i64,
    r.1
]);

/// A `()`-returning export, for the cases whose subject is control flow rather than
/// a value.
pub fn i32_to_void(wat: &str, args: &[i32]) -> Vec<i64> {
    let bytes = wat::parse_str(wat).expect("invalid wat");
    let module = Module::<Stack>::compile(&bytes).expect("module compiles");
    let func = module
        .get_typed_func::<(i32,), ()>("f")
        .expect("export `f`");
    let mut instance = module
        .instantiate::<LinearMemory, _>(NoImports, None)
        .expect("module instantiates");

    func.call((args[0],), &mut instance).expect("call succeeds");

    vec![]
}

// ---------------------------------------------------------------------------
// The three checks, together

/// One lowering case: a module exporting `f`, and the calls to check it with.
pub struct Case {
    /// Names the scratch files and the assertion messages.
    pub name: &'static str,
    pub wat: &'static str,
    /// Argument lists, one per call.
    pub calls: &'static [&'static [i32]],
    /// How to reach `f` through the interpreter. Picks the signature.
    pub interpret: Interpret,
}

/// Lowers the case, checks the IR verifies, and checks it computes what the
/// interpreter computes.
///
/// Returns the IR so a caller can go on to assert on its shape.
pub fn check(case: &Case) -> String {
    let ir = lower(case.wat);

    assert_verifies(case.name, &ir);

    let Some(actual) = run(case.name, &ir, case.calls) else {
        return ir;
    };

    for (args, got) in case.calls.iter().zip(actual) {
        let want = (case.interpret)(case.wat, args);

        assert_eq!(
            got, want,
            "`{}` disagrees with the interpreter for args {args:?}\n--- IR ---\n{ir}",
            case.name
        );
    }

    ir
}

/// Runs a command, failing rather than hanging if it does not finish.
///
/// A lowering bug can turn a terminating wasm loop into an endless LLVM one, and a
/// test that hangs is worse than one that fails: CI waits on it instead of reporting
/// it. The limit is generous — `lli` interprets, so a few seconds is a long time.
fn run_with_timeout(cmd: &mut Command, name: &str) -> std::process::Output {
    use std::{
        io::Read,
        process::Stdio,
        time::{Duration, Instant},
    };

    const LIMIT: Duration = Duration::from_secs(10);

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");

    let started = Instant::now();

    let status = loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => break status,
            None if started.elapsed() > LIMIT => {
                let _ = child.kill();
                let _ = child.wait();

                panic!(
                    "`{name}` did not finish within {LIMIT:?} — the lowered code most likely loops forever"
                );
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    };

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();

    if let Some(mut out) = child.stdout.take() {
        out.read_to_end(&mut stdout).expect("read stdout");
    }

    if let Some(mut err) = child.stderr.take() {
        err.read_to_end(&mut stderr).expect("read stderr");
    }

    std::process::Output {
        status,
        stdout,
        stderr,
    }
}

// ---------------------------------------------------------------------------
// Check 4: the IR is something LLVM can work with

/// Runs `passes` over the IR and returns the result.
///
/// Verifying IR can still be IR the optimiser cannot see through — `mem2reg` only
/// promotes `alloca`s in the entry block, for instance, so a local emitted anywhere
/// else stays in memory forever. Checking what a case *folds to* is how that shows
/// up: a lowering that is merely correct and one that is also canonical look
/// identical until something tries to optimise them.
///
/// `None` if the toolchain is absent.
pub fn optimise(name: &str, ir: &str, passes: &str) -> Option<String> {
    let (Some(llvm_as), Some(opt)) = (llvm_tool("llvm-as"), llvm_tool("opt")) else {
        skip("optimisation", "llvm-as/opt");
        return None;
    };

    let ll = scratch(&format!("{name}_opt"), "ll", ir);
    let bc = ll.with_extension("bc");

    let out = Command::new(llvm_as)
        .arg(&ll)
        .arg("-o")
        .arg(&bc)
        .output()
        .expect("run llvm-as");

    assert!(out.status.success(), "`{name}` does not assemble");

    let out = Command::new(opt)
        .arg(format!("-passes={passes}"))
        .arg(&bc)
        .arg("-S")
        .arg("-o")
        .arg("-")
        .output()
        .expect("run opt");

    assert!(
        out.status.success(),
        "`{name}` failed under `{passes}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    Some(String::from_utf8_lossy(&out.stdout).to_string())
}

/// The body of the first function in `ir`, as trimmed lines.
///
/// For asserting on what a case folds to without depending on register names or
/// indentation.
pub fn function_body(ir: &str) -> Vec<String> {
    ir.lines()
        .skip_while(|l| !l.starts_with("define "))
        .skip(1)
        .take_while(|l| !l.starts_with('}'))
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with(';'))
        .collect()
}
