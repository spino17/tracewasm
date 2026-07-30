//! Compiles every program in `guests/` to `wasm32-unknown-unknown`.
//!
//! Each guest is a plain Rust source file exporting `#[unsafe(no_mangle)] extern
//! "C"` functions. It is used twice:
//!
//! * here, compiled to a `.wasm` that the interpreter executes;
//! * and `include!`d natively by `tests/differential.rs`, giving every export a
//!   ground truth computed by rustc's own backend.
//!
//! That is the whole point of building rather than checking in fixtures: a
//! checked-in `.wasm` can drift from the source that documents what it should do,
//! and then the "expected" values in the tests are just magic numbers nobody can
//! re-derive.
//!
//! If the wasm target is missing, this emits a warning and sets
//! `cfg(no_guest_wasm)` instead of failing the build, so `cargo test` still runs
//! the parts of the suite that do not need guests.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Guest programs to build, without the `.rs`. Kept explicit rather than globbed
/// so that adding one is a visible change and `cargo` reruns predictably.
const GUESTS: &[&str] = &[
    "arithmetic",
    "control_flow",
    "heap",
    "memory",
    "frames",
    "exotic",
];

const TARGET: &str = "wasm32-unknown-unknown";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_guest_wasm)");
    println!("cargo::rerun-if-changed=guests");
    println!("cargo::rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR always set by cargo"));
    let manifest =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR always set"));
    let guests_dir = manifest.join("guests");

    if !target_available() {
        println!(
            "cargo::warning=target `{TARGET}` is not installed, so the guest programs were not \
             built and the differential tests will be skipped. Install it with: \
             rustup target add {TARGET}"
        );
        println!("cargo::rustc-cfg=no_guest_wasm");

        return;
    }

    for guest in GUESTS {
        let src = guests_dir.join(format!("{guest}.rs"));

        assert!(
            src.exists(),
            "guest `{guest}` is listed in build.rs but {} does not exist",
            src.display()
        );

        build_guest(&src, guest, &out_dir);
    }
}

/// Whether the wasm target's std is actually available to this toolchain.
///
/// `rustc --print target-list` would only prove the target *exists*, not that its
/// std is installed, so this compiles a trivial file instead — the same thing the
/// real build does, minus the guest.
fn target_available() -> bool {
    let probe = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("probe_target.rs");

    if std::fs::write(&probe, "#![crate_type=\"cdylib\"]\npub fn f() {}\n").is_err() {
        return false;
    }

    Command::new(rustc())
        .args(["--target", TARGET, "--emit=metadata", "-o"])
        .arg(probe.with_extension("meta"))
        .arg(&probe)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn build_guest(src: &Path, name: &str, out_dir: &Path) {
    let out = out_dir.join(format!("{name}.wasm"));

    let status = Command::new(rustc())
        .args([
            "--target",
            TARGET,
            "--crate-type",
            "cdylib",
            // Must match the edition this crate is built with. The differential
            // tests `include!` these same files into an edition-2024 crate, so a
            // mismatch would compile the two sides under different prelude and
            // language rules — exactly the kind of skew a differential oracle is
            // supposed to rule out. (Bare `rustc` defaults to edition 2015.)
            "--edition",
            "2024",
            // -O so the guests exercise the instruction mix rustc actually emits
            // for release builds (bulk-memory, sign-ext, trunc_sat, unrolled
            // loops) rather than the much flabbier debug output.
            "-O",
            // Deterministic output: the differential tests compare against a
            // natively compiled copy, so incidental codegen churn is noise.
            "-Cdebuginfo=0",
            "--crate-name",
        ])
        .arg(format!("guest_{name}"))
        .arg("-o")
        .arg(&out)
        .arg(src)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke rustc for guest `{name}`: {e}"));

    assert!(
        status.success(),
        "guest `{name}` failed to compile for {TARGET}; see the rustc output above"
    );

    assert!(
        out.exists(),
        "rustc reported success but did not produce {}",
        out.display()
    );
}

/// The rustc cargo is driving, so a `+toolchain` override is respected.
fn rustc() -> String {
    std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string())
}
