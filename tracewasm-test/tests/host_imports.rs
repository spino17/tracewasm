//! Host-boundary coverage for the `#[imports]` macro, ported from
//! `tracewasm-core/tests/macro_memory_view.rs`.
//!
//! These drive the generated `ImportRegistry` dispatch directly against a bare
//! `LinearMemory`, without building a module at all — the point is the macro's
//! marshalling and its `MemoryView` plumbing, not interpretation.

use tracewasm_core::{
    error::MemoryError,
    instance::traits::{ImportRegistry, Val},
    memory::{MemoryView, linear::LinearMemory},
};
use tracewasm_macros::imports;

struct Host {
    calls: u32,
}

#[imports]
impl Host {
    /// Takes the memory view: writes the byte and reads it straight back.
    #[module("env")]
    fn poke<V: MemoryView>(&mut self, addr: i32, byte: i32, mem: &mut V) -> (i32,) {
        self.calls += 1;
        mem.write_u8(addr as usize, byte as u8).unwrap();

        (mem.read_u8(addr as usize).unwrap() as i32,)
    }

    /// No memory view — must still compile and dispatch.
    #[module("env")]
    fn pure_add(&mut self, a: i32, b: i32) -> (i32,) {
        (a + b,)
    }

    /// Fallible: an out-of-bounds pointer from the guest must trap, not panic.
    /// The `?` converts `MemoryError` into the `anyhow::Error` the registry
    /// reports failures with.
    #[module("env")]
    fn checked_poke<V: MemoryView>(
        &mut self,
        addr: i32,
        byte: i32,
        mem: &mut V,
    ) -> Result<(i32,), tracewasm_core::anyhow::Error> {
        mem.write_u8(addr as usize, byte as u8)?;

        Ok((mem.read_u8(addr as usize)? as i32,))
    }
}

#[test]
fn host_fn_can_read_and_write_guest_memory() {
    let mut host = Host { calls: 0 };
    let mut mem = LinearMemory::new(1); // one page

    let out = host
        .execute("env", "poke", &[Val::I32(64), Val::I32(0xAB)], &mut mem)
        .unwrap();

    assert!(matches!(out.as_ref()[0], Val::I32(0xAB)));
    assert_eq!(host.calls, 1);
    // The write landed in the caller's memory, not a copy.
    assert_eq!(mem.read_u8(64).unwrap(), 0xAB);
}

#[test]
fn host_fn_without_memory_view_still_works() {
    let mut host = Host { calls: 0 };
    let mut mem = LinearMemory::new(1);

    let out = host
        .execute("env", "pure_add", &[Val::I32(2), Val::I32(3)], &mut mem)
        .unwrap();

    assert!(matches!(out.as_ref()[0], Val::I32(5)));
}

#[test]
fn fallible_host_fn_returns_its_value_on_success() {
    let mut host = Host { calls: 0 };
    let mut mem = LinearMemory::new(1);

    let out = host
        .execute(
            "env",
            "checked_poke",
            &[Val::I32(8), Val::I32(0x5A)],
            &mut mem,
        )
        .unwrap();

    assert!(matches!(out.as_ref()[0], Val::I32(0x5A)));
    assert_eq!(mem.read_u8(8).unwrap(), 0x5A);
}

#[test]
fn fallible_host_fn_traps_instead_of_panicking_on_a_bad_pointer() {
    let mut host = Host { calls: 0 };
    let mut mem = LinearMemory::new(1); // 65_536 bytes

    // A guest can hand over any i32; this one is far past the end of memory.
    let res = host.execute(
        "env",
        "checked_poke",
        &[Val::I32(i32::MAX), Val::I32(1)],
        &mut mem,
    );

    // The error reaches the interpreter as a normal trap, and the `MemoryError`
    // it came from survives being carried as an `anyhow::Error`.
    let err = res.expect_err("an out-of-bounds poke must fail");

    assert!(
        err.downcast_ref::<MemoryError>().is_some(),
        "expected a memory trap, got: {err:?}"
    );
}

#[test]
fn signature_excludes_the_memory_view() {
    let host = Host { calls: 0 };

    // `poke` is `(i32, i32) -> (i32)` in wasm terms: the `&mut M` is not a param.
    let (params, results) = host.signature("env", "poke").unwrap();

    assert_eq!(
        params.as_ref().len(),
        2,
        "memory view must not be a wasm arg"
    );
    assert_eq!(results.as_ref().len(), 1);
    assert_eq!(host.func_count(), 3);
}
