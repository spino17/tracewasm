use crate::module::ValType;

/// Formats a value-type list as a parenthesized, comma-separated tuple, e.g.
/// `(I32,F64)`. An empty list renders as `()` (correctly handling void/no-arg
/// signatures — the previous index-based version underflowed on an empty slice).
pub(crate) fn formatted_val_types(types: &[ValType]) -> String {
    let inner = types
        .iter()
        .map(|ty| format!("{ty:?}"))
        .collect::<Vec<_>>()
        .join(",");

    format!("({inner})")
}
