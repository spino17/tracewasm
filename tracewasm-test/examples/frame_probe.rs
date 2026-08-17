//! One recursion probe per process, for measuring native stack cost per frame.
//!
//! Usage: `frame_probe <stack_bytes> <depth>`. Runs `fr_recurse_depth(depth)` on
//! a thread with exactly `stack_bytes` of stack and exits 0 if it returns.
//!
//! A Rust stack overflow is `SIGABRT`, so a probe that exceeds the stack takes
//! the process with it. That is the point: the caller reads the exit status, and
//! one process per probe means a failed probe cannot poison the next one.

#[cfg(not(no_guest_wasm))]
use tracewasm_core::instance::config::Config;
#[cfg(not(no_guest_wasm))]
use tracewasm_test::{Guest, guests};

/// Exit code for "the probe could not run", kept distinct from the 0/1 a real probe
/// reports so a caller cannot read a missing toolchain as a passing measurement.
#[cfg(no_guest_wasm)]
const NO_GUESTS: i32 = 2;

#[cfg(no_guest_wasm)]
fn main() {
    eprintln!(
        "frame_probe needs the wasm32-unknown-unknown target: the `frames` guest was \
         not built. Install it with: rustup target add wasm32-unknown-unknown"
    );
    std::process::exit(NO_GUESTS);
}

#[cfg(not(no_guest_wasm))]
fn main() {
    let mut a = std::env::args().skip(1);
    let stack: usize = a.next().unwrap().parse().unwrap();
    let depth: i32 = a.next().unwrap().parse().unwrap();

    let h = std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            let mut cfg = Config::default();
            // The guard must not be what stops this: the probe is looking for
            // the native cliff.
            cfg.set_max_call_stack_depth(depth as u32 + 1_000);
            let mut g = Guest::with_config(guests::FRAMES, Some(cfg));
            g.i32_i64("fr_recurse_depth", depth)
        })
        .unwrap();

    match h.join() {
        Ok(v) => {
            std::hint::black_box(v);
            std::process::exit(0)
        }
        Err(_) => std::process::exit(1),
    }
}
