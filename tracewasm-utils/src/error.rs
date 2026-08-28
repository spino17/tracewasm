use thiserror::Error;

#[derive(Error, Debug)]
pub enum TracewasmUtilsError {
    #[error("too many unique interned values: reached {needed}, over the limit of {limit}")]
    ToManyUniqueValues {
        /// The number of distinct values that pool would have had to hold.
        needed: u32,
        /// The most distinct values a 16-bit id can name.
        limit: u64,
    },
}
