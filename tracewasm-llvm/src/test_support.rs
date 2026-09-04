//! Fixtures shared by the crate's unit tests.
//!
//! The tests live next to what they exercise, so more than one module needs the
//! same two-line setup. Keeping it here rather than repeating it means a change to
//! how a context or a value is built is one edit, not four.

use crate::{
    cfg::{
        builder::Builder,
        context::Context,
        global::{DefinedFunc, GlobalId},
        module::{DataLayout, Triple},
    },
    error::ContextError,
    value::Value,
};

/// A context targeting `arm64-apple-macosx`, with no data layout, and a builder.
///
/// The triple is fixed so the emitter tests can assert on the `target triple` line;
/// the layout is left unset, so no `target datalayout` line is emitted.
pub(crate) fn fixture() -> Context {
    ctx()
}

/// Just the context, for the tests that never touch a builder.
pub(crate) fn ctx() -> Context {
    Context::new(
        Triple::new(
            "arm64".to_string(),
            "apple".to_string(),
            "macosx".to_string(),
            None,
        ),
        DataLayout::default(),
    )
}

/// A function taking nothing and returning `void`, for the tests whose subject is
/// the graph rather than the signature.
pub(crate) fn add_fn(
    name: &str,
    builder: &mut Builder,
    ctx: &mut Context,
) -> Result<GlobalId<DefinedFunc>, ContextError> {
    let void_ty = ctx.void_ty();

    builder.define_function(name.to_string(), &[], void_ty)
}

/// A distinct `i32` constant per call, for tests whose subject is the graph rather
/// than the value flowing through it.
pub(crate) fn value(n: i32, ctx: &mut Context) -> Value {
    Value::from_const(n, None, ctx).expect("constant interns")
}
