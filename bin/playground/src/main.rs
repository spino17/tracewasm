use rustc_demangle::demangle;
use std::fs;
use tracewasm_core::{memory::linear::LinearMemory, module::Module};
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
    fn host1(&mut self, a: i32, b: i32) -> (i32,) {
        self.count += 1;
        (a.wrapping_add(b),)
    }

    #[module("env")]
    fn host2(&mut self, a: i32, b: i32, c: i64) -> (i32,) {
        (a.wrapping_add(b).wrapping_add(c as i32),)
    }
}

fn main() -> Result<(), anyhow::Error> {
    let buf = fs::read(
        "/Users/bhavyabhatt/Desktop/bhavya/projects/tracewasm/target/wasm32-unknown-unknown/debug/tracewasm_scratch.wasm",
    )?;

    let module = Module::compile(&buf)?;

    println!("{:?}", module.custom_section.unknown.keys());
    let registry = ImportedFunctions { count: 0 };

    let func = module.get_typed_func::<(i32, i64), (i64,)>("bench_bits")?;

    /*let mut instance = module.instantiate::<LinearMemory, _>(registry, None)?;
    let res = func.call((1, 2), &mut instance)?.0;*/

    let s = "$_RNvNtNtCs3O6bguQwcd4_4core9panicking11panic_const24panic_const_div_overflow";
    let s = s.strip_prefix('$').unwrap_or(s);
    let str = demangle(s);
    // println!("{:#}", str); // core::panicking::panic_const::panic_const_div_overflow

    Ok(())
}
