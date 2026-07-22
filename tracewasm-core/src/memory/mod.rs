//! The linear-memory abstraction the VM runs against.

/// A module's linear memory, supplied by the embedder.
///
/// The interpreter is generic over this trait so the backing store (a plain
/// `Vec<u8>`, an mmap, a guarded region, …) is the embedder's choice.
pub trait Memory {
    /// Creates a memory pre-allocated to `size` bytes.
    fn allocate_initial_memory(size: usize) -> Self;

    // TODO: add basic read/write methods!
}
