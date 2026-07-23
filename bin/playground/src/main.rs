use std::fs;
use tracewasm_core::{imports, memory::linear::LinearMemory, module::Module};

/// Example host module. The `#[imports]` macro reads the `#[module("...")]`-tagged
/// methods below and generates the entire [`ImportRegistry`] impl (`execute`,
/// `signature`, `size`) — the embedder writes only the function bodies.
pub struct ImportedFunctions {}

#[imports]
impl ImportedFunctions {
    #[module("env")]
    fn host1(&mut self, a: i32, b: i32) -> (i32,) {
        (a.wrapping_add(b),)
    }

    #[module("env")]
    fn host2(&mut self, a: i32, b: i32, c: i64) -> (i32,) {
        (a.wrapping_add(b).wrapping_add(c as i32),)
    }
}

fn main() -> Result<(), anyhow::Error> {
    let buf = fs::read("<PATH TO WASM file>")?;
    let module = Module::compile(&buf)?;
    let registry = ImportedFunctions {};

    let func = module
        .export("bench_control_flow")
        .ok_or(anyhow::Error::msg("export not found!"))?
        .to_typed_func::<(i32,), ()>()?;

    let mut instance = module.instantiate::<LinearMemory, _>(registry)?;
    let _ = func.call((1,), &mut instance)?;

    Ok(())
}
