//! Panic messages for invariants that hold by construction.

/// Used when an id fails to resolve in its arena.
///
/// Ids are only handed out by `alloc`, so a lookup can only fail if the id came from
/// a different [`Context`](crate::cfg::context::Context) — a caller bug rather than a
/// runtime condition worth returning an error for.
pub const ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID: &str =
    "valid id are never constructed without `alloc` into the arena";
