#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn host_call(n: i32);
}

#[unsafe(no_mangle)]
pub extern "C" fn demo(n: i32) -> i32 {
    unsafe {
        host_call(n);
    }

    if n > 0 { n + 1 } else { panic!("not allowed!") }
}
