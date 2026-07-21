#[unsafe(no_mangle)]
pub extern "C" fn bench_control_flow(a: i32) {
    struct Foo {
        a: Vec<i32>,
    }

    let mut f = Foo { a: vec![] };

    if a > 0 {
        f.a.push(a);
    }
}
