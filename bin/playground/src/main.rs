use std::fs;
use tracewasm_core::{
    Stack,
    memory::{MemoryView, linear::LinearMemory},
    module::Module,
};
use tracewasm_macros::imports;

/// Example host module. The `#[imports]` macro reads the `#[module("...")]`-tagged
/// (and any `#[global("...")]`-tagged) methods below and generates the entire
/// [`ImportRegistry`] impl (`execute`, `signature`, `func_count`, `global_count`,
/// `get_global`) — the embedder writes only the function bodies.
///
/// Serves as the example host state / import registry for the playground.
pub struct ImportedFunctions {
    /// Call counter mutated by the host functions to demonstrate `&mut self` state.
    count: u32,
}

#[imports]
impl ImportedFunctions {
    #[module("env")]
    fn host1<V: MemoryView>(&mut self, a: i32, b: i32, _memory_view: &mut V) -> (i32,) {
        self.count += 1;
        (a.wrapping_add(b),)
    }

    #[module("env")]
    fn host2<V: MemoryView>(&mut self, a: i32, b: i32, c: i64, _memory_view: &mut V) -> (i32,) {
        (a.wrapping_add(b).wrapping_add(c as i32),)
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
const ENTRY: &str = "bench_bits";

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
    let registry = ImportedFunctions { count: 0 };
    let func = module.get_typed_func::<(i32, i64), (i64,)>(ENTRY)?;
    let mut instance = module.instantiate::<LinearMemory, _>(registry, None)?;

    let res = func.call((1, 2), &mut instance);

    if let Err(err) = res {
        let trace = err.stack_trace();
        let source_trace = trace.to_source_trace()?;

        println!("{:?}", source_trace.render())
    }

    Ok(())
}
