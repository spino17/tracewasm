use crate::{
    cfg::{
        ControlFlowGraph,
        basic_block::BasicBlockId,
        context::Context,
        function::{FuncId, Function},
        module::Module,
    },
    error::ContextError,
    instruction::cursor::Cursor,
    interner::{StrId, TyId},
    value::Value,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Builds a module: adds functions and opens cursors onto their blocks.
///
/// The builder owns the module's own contents — its target settings and the list of
/// functions — while the [`Context`] owns the storage everything is allocated in. The
/// two are threaded together through every call, and
/// [`build`](Self::build) consumes the builder to produce the finished
/// [`ControlFlowGraph`].
pub struct Builder {
    pub(crate) module: Module,
}

impl Builder {
    /// An empty module for the given target.
    ///
    /// Either string may be empty, meaning "unset": the emitter then omits the
    /// corresponding `target` line rather than writing an empty one.
    pub fn new(triple: String, data_layout: String) -> Self {
        Builder {
            module: Module {
                triple,
                data_layout,
                globals: vec![],
                functions: vec![],
                func_names: FxHashSet::default(),
            },
        }
    }

    /// Opens a cursor that writes into `id`.
    ///
    /// A cursor is a position, not a lock: several may be opened at one block over
    /// time. What prevents writing past a terminator is that the terminator builders
    /// consume the cursor, plus the block's own
    /// [`is_locked`](crate::cfg::basic_block::BasicBlock) flag for cursors opened
    /// afterwards.
    pub fn cursor_at_block(&mut self, id: BasicBlockId) -> Cursor {
        Cursor { block: id }
    }

    /// Declares a function and its signature.
    ///
    /// Each parameter is a type and an optional name hint. A named parameter keeps
    /// its hint; an unnamed one draws the next number from the function's counter, so
    /// `&[(i32_ty, None), (i32_ty, None)]` yields `%0` and `%1` and the body's first
    /// unnamed temporary is `%2`. Named parameters consume no numbers.
    ///
    /// The returned [`FuncId`] is how blocks are added and how the function is read
    /// back.
    ///
    /// # Errors
    ///
    /// - [`ContextError::DuplicateFunctionName`] — LLVM identifies a definition by
    ///   name, so two `@name`s in one module is a build error rather than something
    ///   `llvm-as` finds later.
    /// - [`ContextError::FunctionParamTypeNotSized`] — a parameter is passed by
    ///   value, so it needs a size; aggregates are fine, `void` and function types
    ///   are not.
    /// - [`ContextError::FunctionResultTypeInvalid`] — a result may be `void` but not
    ///   a function type.
    /// - [`ContextError::InvalidRegisterName`] — a parameter's name hint is not a
    ///   legal LLVM local.
    pub fn add_function(
        &mut self,
        name: String,
        params: &[(TyId, Option<String>)],
        result: TyId,
        ctx: &mut Context,
    ) -> Result<FuncId, ContextError> {
        let name_id: StrId = ctx.str_interner.intern(name).into();

        if self.module.func_names.contains(&name_id) {
            return Err(ContextError::DuplicateFunctionName(
                ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let void_ty = ctx.void_ty();

        // A parameter is passed by value, so it has to have a size: `llvm-as` refuses
        // a `void` one with "void type only allowed for function results" and a
        // function-typed one with "invalid type for function argument". Aggregates
        // are fine — `define void @f({i32, double} %x)` assembles.
        for (param_ty, _) in params {
            if !param_ty.is_first_class(ctx) {
                return Err(ContextError::FunctionParamTypeNotSized(
                    param_ty.display(ctx).to_string(),
                ));
            }
        }

        // A result may be `void` as well, but still not a function type — that one
        // `llvm-as` refuses with "invalid function return type".
        if result != void_ty && !result.is_first_class(ctx) {
            return Err(ContextError::FunctionResultTypeInvalid(
                result.display(ctx).to_string(),
            ));
        }

        let id = FuncId::new(ctx.funcs.alloc(Function {
            name: name_id,
            blocks: vec![],
            params: vec![],
            result: void_ty,
            block_names: FxHashSet::default(),
        }));

        let mut final_params: Vec<Value> = vec![];

        for (param_ty, param_name) in params {
            let name = ctx.name_for_reg(
                if let Some(param) = param_name {
                    Some(param.as_ref())
                } else {
                    None
                },
                id,
            )?;

            final_params.push(Value::from_register(name, *param_ty, ctx));
        }

        let func = ctx.get_func_mut(id);

        func.params = final_params;
        func.result = result;

        self.module.func_names.insert(name_id);
        self.module.functions.push(id);

        ctx.register_def_instr_index
            .insert(id, FxHashMap::default());

        Ok(id)
    }

    /// Finishes the module.
    ///
    /// Consumes the builder, so nothing more can be added. The [`Context`] is still
    /// needed to read the result, since everything inside it is an id.
    ///
    /// Nothing is verified here: whether each block ends in a terminator, and whether
    /// a phi has one entry per predecessor, are not checked by this crate. `llvm-as`
    /// reports both.
    pub fn build(self) -> ControlFlowGraph {
        ControlFlowGraph {
            module: self.module,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        test_support::fixture,
        value::{FuncSignature, Type, ValueKind},
    };

    /// The name a value's register was issued.
    fn reg_name(value: &Value, ctx: &Context) -> String {
        let ValueKind::Reg(reg) = value.kind() else {
            panic!("expected a register")
        };

        ctx.str_interner.value(reg.name.0).to_string()
    }

    /// The names of a function's parameters, in declaration order.
    fn param_names(f: FuncId, ctx: &Context) -> Vec<String> {
        ctx.get_func(f)
            .params
            .iter()
            .map(|p| reg_name(p, ctx))
            .collect()
    }

    /// An unnamed temporary in `f`'s body, which is what advances the counter.
    fn unnamed_temp(cursor: &Cursor, ctx: &mut Context) -> String {
        let i32_ty = ctx.i32_ty();

        let val = cursor
            .add_alloca(i32_ty, None, None, None, ctx)
            .expect("an i32 is allocatable");

        reg_name(&val, ctx)
    }

    /// LLVM numbers unnamed values per function starting at `%0`, and **parameters
    /// come first** — so the body's first unnamed temporary continues from where the
    /// parameter list stopped rather than restarting.
    ///
    /// `llvm-as` pins the boundary: with two unnamed parameters and a *named* entry
    /// block, `%2` is the first legal instruction number.
    #[test]
    fn unnamed_params_take_the_first_numbers_and_the_body_continues() {
        let (mut ctx, mut builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .add_function(
                "f".to_string(),
                &[(i32_ty, None), (i32_ty, None)],
                void_ty,
                &mut ctx,
            )
            .unwrap();

        assert_eq!(param_names(f, &ctx), ["0", "1"], "parameters number first");

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        assert_eq!(
            unnamed_temp(&cursor, &mut ctx),
            "2",
            "the body continues the parameters' numbering, it does not restart"
        );

        assert_eq!(unnamed_temp(&cursor, &mut ctx), "3");
    }

    /// A *named* parameter takes no number. `llvm-as` is explicit: in
    /// `define i32 @f(i32 %n, i32)` the unnamed parameter is `%0`, and numbering an
    /// instruction `%0` is refused with "expected to be numbered '%1' or greater".
    #[test]
    fn a_named_param_does_not_consume_a_number() {
        let (mut ctx, mut builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .add_function(
                "f".to_string(),
                &[(i32_ty, Some("n".to_string())), (i32_ty, None)],
                void_ty,
                &mut ctx,
            )
            .unwrap();

        assert_eq!(
            param_names(f, &ctx),
            ["n", "0"],
            "only the unnamed parameter draws from the counter"
        );

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        assert_eq!(
            unnamed_temp(&cursor, &mut ctx),
            "1",
            "one number was taken, so the body starts at 1"
        );
    }

    /// All parameters named means none of them took a number, so the body starts at
    /// `%0` — `define i32 @f(i32 %a, i32 %b)` with a first instruction of `%0`
    /// assembles.
    #[test]
    fn all_named_params_leave_the_body_starting_at_zero() {
        let (mut ctx, mut builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .add_function(
                "f".to_string(),
                &[
                    (i32_ty, Some("a".to_string())),
                    (i32_ty, Some("b".to_string())),
                ],
                void_ty,
                &mut ctx,
            )
            .unwrap();

        assert_eq!(param_names(f, &ctx), ["a", "b"]);

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let cursor = builder.cursor_at_block(entry);

        assert_eq!(unnamed_temp(&cursor, &mut ctx), "0");
    }

    /// The counter is per *function*, not per block: it runs on across a branch, which
    /// is what `llvm-as` accepts for `%1` in `entry` and `%2` in `next`.
    #[test]
    fn the_counter_continues_across_blocks() {
        let (mut ctx, mut builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .add_function("f".to_string(), &[(i32_ty, None)], void_ty, &mut ctx)
            .unwrap();

        assert_eq!(param_names(f, &ctx), ["0"]);

        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let next = f.add_basic_block("next".to_string(), &mut ctx).unwrap();

        let in_entry = builder.cursor_at_block(entry);

        assert_eq!(unnamed_temp(&in_entry, &mut ctx), "1");

        let in_next = builder.cursor_at_block(next);

        assert_eq!(
            unnamed_temp(&in_next, &mut ctx),
            "2",
            "a new block does not restart the numbering"
        );

        assert_eq!(
            unnamed_temp(&in_entry, &mut ctx),
            "3",
            "and going back to the first block does not either"
        );
    }

    /// Two functions each number from zero, parameters included — `%0` in one is a
    /// different value from `%0` in the other.
    #[test]
    fn each_function_restarts_the_counter_including_its_params() {
        let (mut ctx, mut builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .add_function(
                "f".to_string(),
                &[(i32_ty, None), (i32_ty, None)],
                void_ty,
                &mut ctx,
            )
            .unwrap();

        let g = builder
            .add_function("g".to_string(), &[(i32_ty, None)], void_ty, &mut ctx)
            .unwrap();

        assert_eq!(param_names(f, &ctx), ["0", "1"]);
        assert_eq!(param_names(g, &ctx), ["0"], "`g` starts over");

        let in_f = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let in_g = g.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        let f_cursor = builder.cursor_at_block(in_f);
        let g_cursor = builder.cursor_at_block(in_g);

        assert_eq!(unnamed_temp(&f_cursor, &mut ctx), "2");
        assert_eq!(
            unnamed_temp(&g_cursor, &mut ctx),
            "1",
            "`g` has one parameter, so its body starts at 1"
        );
    }

    /// A parameter becomes a register of the declared type, named in order, and the
    /// result is recorded as given.
    #[test]
    fn a_signature_becomes_typed_registers_and_a_result() {
        let (mut ctx, mut builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();
        let ptr_ty = ctx.ptr_ty();

        let f = builder
            .add_function(
                "sum".to_string(),
                &[
                    (i32_ty, Some("n".to_string())),
                    (f64_ty, Some("x".to_string())),
                    (ptr_ty, None),
                ],
                i32_ty,
                &mut ctx,
            )
            .expect("a valid signature");

        let func = ctx.get_func(f);

        assert_eq!(func.result, i32_ty, "the result is the one given");
        assert_eq!(func.params.len(), 3);

        let spelled: Vec<String> = func
            .params
            .iter()
            .map(|p| ctx.display(p.ty()).to_string())
            .collect();

        assert_eq!(spelled, ["i32", "double", "ptr"], "in declaration order");

        for param in &func.params {
            assert!(
                matches!(param.kind(), ValueKind::Reg(_)),
                "a parameter is a register, not a constant"
            );
        }

        // The unnamed one falls back to LLVM's `%0` numbering, and the named ones
        // keep their hints — `name_for_reg` is the same path a temporary goes through.
        let names: Vec<String> = func
            .params
            .iter()
            .map(|p| {
                let ValueKind::Reg(reg) = p.kind() else {
                    panic!("a parameter is a register")
                };

                ctx.str_interner.value(reg.name.0).to_string()
            })
            .collect();

        assert_eq!(names, ["n", "x", "0"]);
    }

    /// An aggregate parameter is passed by value and is perfectly legal —
    /// `define void @f({i32, double} %x)` assembles.
    #[test]
    fn an_aggregate_parameter_is_allowed() {
        let (mut ctx, mut builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();
        let void_ty = ctx.void_ty();

        let struct_ty = ctx
            .ty_interner
            .intern(Type::Struct {
                fields: Box::new([i32_ty, f64_ty]),
                packed: false,
            })
            .into();

        assert!(
            builder
                .add_function("f".to_string(), &[(struct_ty, None)], void_ty, &mut ctx)
                .is_ok(),
            "an aggregate is sized, so it can be passed"
        );
    }

    /// A parameter is passed by value, so it has to have a size. `llvm-as` refuses a
    /// `void` one with "void type only allowed for function results" and a
    /// function-typed one with "invalid type for function argument".
    #[test]
    fn an_unsized_parameter_is_refused() {
        let (mut ctx, mut builder) = fixture();

        let void_ty = ctx.void_ty();

        let err = builder
            .add_function("f".to_string(), &[(void_ty, None)], void_ty, &mut ctx)
            .expect_err("`void` is not a parameter type");

        assert!(
            matches!(&err, ContextError::FunctionParamTypeNotSized(t) if t == "void"),
            "the error must name the offending type, got: {err}"
        );

        assert_eq!(
            builder.module.functions.len(),
            0,
            "the refused function must not have been added to the module"
        );
    }

    /// `void` *is* a legal result — that is the one place it is allowed — but a
    /// function type is not: `llvm-as` refuses it with "invalid function return type".
    #[test]
    fn a_result_may_be_void_but_not_a_function_type() {
        let (mut ctx, mut builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        assert!(
            builder
                .add_function("returns_void".to_string(), &[], void_ty, &mut ctx)
                .is_ok(),
            "`void` is a result like any other"
        );

        let signature = FuncSignature::new(vec![i32_ty], i32_ty);
        let func_ty: TyId = ctx.ty_interner.intern(Type::Func(signature)).into();

        let err = builder
            .add_function("returns_fn".to_string(), &[], func_ty, &mut ctx)
            .expect_err("a function cannot return a function type");

        assert!(
            matches!(&err, ContextError::FunctionResultTypeInvalid(t) if t == "i32 (i32)"),
            "the error must name the offending type, got: {err}"
        );
    }
}
