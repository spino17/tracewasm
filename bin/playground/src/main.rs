use crate::counting::CountingMemory;
use std::fs;
use tracewasm_core::{Stack, memory::linear::LinearMemory, module::Module};
use tracewasm_macros::imports;

mod counting;

pub struct Imports {}

#[imports]
impl Imports {
    #[module("env")]
    fn host(&mut self, n: i32) {
        println!("number passed: {:?}", n)
    }
}

// /Users/bhavyabhatt/Desktop/bhavya/projects/tracewasm/target/wasm32-unknown-unknown/release/tracewasm_scratch.wasm
fn main() -> Result<(), anyhow::Error> {
    let file = fs::read(
        "/Users/bhavyabhatt/Desktop/bhavya/projects/tracewasm/target/wasm32-unknown-unknown/debug/tracewasm_scratch.wasm",
    )?;

    let module = Module::<Stack>::compile(&file)?;
    let mut instance = module.instantiate::<CountingMemory, _>(Imports {}, None)?;

    let demo = module.get_typed_func::<(i32,), (i32,)>("demo")?;

    let res = demo.call((-1,), &mut instance);

    println!("number of memory reads: {:?}", instance.memory_view().reads);

    match res {
        Ok(res) => println!("result is: {:?}", res),
        Err(err) => {
            println!("{}", err.stack_trace().render());
        }
    }

    Ok(())
}
