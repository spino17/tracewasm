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
    instruction::cursor::OperandTy,
    interner::{StrId, TyId},
    value::{Register, Value, ValueId, ValueKind},
};

/// A builder over a context targeting `arm64-apple-macosx`, with no data layout.
///
/// The triple is fixed so the emitter tests can assert on the `target triple` line;
/// the layout is left unset, so no `target datalayout` line is emitted.
///
/// Returns the [`Builder`] rather than the [`Context`] because the builder now owns
/// it — and since `Builder` derefs to `Context`, everything a context can do is
/// reachable through it anyway.
pub(crate) fn fixture() -> Builder {
    ctx().builder()
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
) -> Result<GlobalId<DefinedFunc>, ContextError> {
    let void_ty = builder.void_ty();

    builder.define_function(name.to_string(), &[], void_ty)
}

/// A distinct `i32` constant per call, for tests whose subject is the graph rather
/// than the value flowing through it.
pub(crate) fn value(n: i32, ctx: &mut Context) -> ValueId {
    ctx.const_value(n, OperandTy::Inferred)
        .expect("constant interns")
}

/// A named register operand, for a test that needs one to hand rather than one
/// defined by an instruction.
///
/// Goes straight to the arena instead of through
/// [`Value::from_register`](crate::value::Value::from_register), which would want the
/// enclosing function in order to issue a unique name. These stand-ins are only ever
/// read as operands, so the name is taken as given.
pub(crate) fn reg_val(name: &str, ty: TyId, ctx: &mut Context) -> ValueId {
    let name: StrId = ctx.str_interner.intern(name.to_string()).into();

    ctx.alloc_value(Value::new(
        ty,
        ValueKind::Reg(Register {
            name,
            is_unnamed: false,
        }),
    ))
}
