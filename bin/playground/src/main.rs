use std::fs;
use tracewasm_core::{Stack, memory::linear::LinearMemory, module::Module};
use tracewasm_macros::imports;

pub struct ImportedFunctions;

#[imports]
impl ImportedFunctions {
    #[module("env")]
    fn host_call(&mut self, n: i32) {
        println!("{}", n);
    }
}

/// Where to look for the wasm when no path is given: the scratch crate's debug
/// build, resolved from this file's location so it does not depend on whose
/// checkout it is.
const DEFAULT_WASM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/wasm32-unknown-unknown/debug/tracewasm_scratch.wasm"
);

/// The export the playground calls, and its signature.
const ENTRY: &str = "demo";

fn main() -> Result<(), anyhow::Error> {
    // `tracewasm-scratch` is not a workspace member, so the default path only
    // exists once it has been built for wasm32. Any other module can be run by
    // passing its path instead.
    let path = std::env::args().nth(1).unwrap_or(DEFAULT_WASM.to_string());

    let buf = fs::read(&path).map_err(|err| {
        anyhow::anyhow!(
            "could not read `{path}`: {err}. Pass a path to a .wasm file, or build the \
             scratch crate for wasm32-unknown-unknown first."
        )
    })?;

    let module = Module::<Stack>::compile(&buf)?;
    let registry = ImportedFunctions;
    let func = module.get_typed_func::<(i32,), (i32,)>(ENTRY)?;
    let mut instance = module.instantiate::<LinearMemory, _>(registry, None)?;

    let res = func.call((-1,), &mut instance);

    match res {
        Ok(val) => println!("{}", val.0),
        Err(err) => {
            let trace = err.stack_trace();
            let source_trace = trace.to_source_trace()?;

            println!("{:?}", source_trace.render())
        }
    }

    Ok(())
}
