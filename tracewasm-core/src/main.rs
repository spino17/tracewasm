use crate::parser::TraceWasmParser;
use anyhow::Result;
use std::fs;

pub mod error;
pub mod instruction;
pub mod parser;

fn main() -> Result<(), anyhow::Error> {
    let path = "/Users/bhavyabhatt/Desktop/bhavya/projects/tracewasm/target/wasm32-unknown-unknown/release/tracewasm_scratch.wasm".to_string();
    let wasm_buffer = fs::read(&path)?;
    //let parser = TraceWasmParser;
    //let result = TraceWasmParser::parse(&wasm_buffer)?;
    let mut m = walrus::Module::from_file(&path)?;

    let funcs = m.funcs;
    let func = funcs.by_name("bench_control_flow").unwrap();

    let func_body = funcs.get(func);

    println!("{:?}", func_body);

    Ok(())
}
