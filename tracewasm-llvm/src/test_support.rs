//! Fixtures shared by the crate's unit tests.
//!
//! The tests live next to what they exercise, so more than one module needs the
//! same two-line setup. Keeping it here rather than repeating it means a change to
//! how a context or a value is built is one edit, not four.

use crate::{
    cfg::{builder::Builder, context::Context, function::FuncId},
    error::ContextError,
    value::Value,
};

/// An empty context and a builder for it.
pub(crate) fn fixture() -> (Context, Builder) {
    (
        Context::default(),
        Builder::new("arm64-apple-macosx".to_string(), String::new()),
    )
}

/// A function taking nothing and returning `void`, for the tests whose subject is
/// the graph rather than the signature.
pub(crate) fn add_fn(
    name: &str,
    builder: &mut Builder,
    ctx: &mut Context,
) -> Result<FuncId, ContextError> {
    let void_ty = ctx.void_ty();

    builder.add_function(name.to_string(), &[], void_ty, ctx)
}

/// A distinct `i32` constant per call, for tests whose subject is the graph rather
/// than the value flowing through it.
pub(crate) fn value(n: i32, ctx: &mut Context) -> Value {
    Value::from_const(n, None, ctx).expect("constant interns")
}
