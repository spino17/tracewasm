use thiserror::Error;

/// What can go wrong in this crate.
///
/// One case so far: a pool asked to hold more distinct values than its id width can
/// address. Whether that is an error or a panic is the caller's call — see
/// [`try_intern`](crate::interner::Interner::try_intern).
#[derive(Error, Debug)]
pub enum TracewasmUtilsError {
    #[error("too many unique interned values: reached {needed}, over the limit of {limit}")]
    ToManyUniqueValues {
        /// The number of distinct values that pool would have had to hold.
        needed: u32,
        /// The most distinct values the pool's id can name — its `Capacity::val`,
        /// so `u16::MAX`, `u32::MAX` or `u64::MAX` depending on the id width the
        /// pool was declared with.
        limit: u64,
    },
}
