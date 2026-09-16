#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn host(n: i32);
}

#[unsafe(no_mangle)]
pub extern "C" fn demo(n: i32) -> i32 {
    unsafe {
        host(n);
    }

    if n > 0 {
        n + 10
    } else {
        panic!("n less than 0")
    }
}
