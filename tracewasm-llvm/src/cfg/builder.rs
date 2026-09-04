use std::ops::{Deref, DerefMut};

use crate::{
    cfg::{
        ControlFlowGraph,
        basic_block::BasicBlockId,
        context::Context,
        function::{FuncId, Function},
        global::{
            DeclaredFunc, DefinedFunc, GlobalData, GlobalId, GlobalKind, GlobalVar, GlobalVariable,
            Linkage, Visibility,
        },
    },
    error::ContextError,
    instruction::cursor::{Cursor, RegName},
    interner::{StrId, TyId},
    value::{ConstExpr, FuncSignature, Value},
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Builds a module: adds functions and opens cursors onto their blocks.
///
/// Owns the [`Context`], so no call takes one. Obtain it from
/// [`Context::builder`](Context::builder) and hand it to
/// [`build`](Self::build) at the end for the finished [`ControlFlowGraph`].
///
/// It derefs to [`Context`], so everything a context can do — interning a type,
/// making a constant — is reachable straight through the builder. That is also what
/// lets a `&mut Builder` stand in wherever a `&mut Context` is wanted, which is how
/// [`add_basic_block`](crate::cfg::global::GlobalId::add_basic_block) and friends
/// still take a context without the caller holding one separately.
pub struct Builder {
    pub(crate) ctx: Context,
}

impl Deref for Builder {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl DerefMut for Builder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}

impl Builder {
    /// Returns the immutable reference to the underlying `Context`.
    pub fn ctx(&self) -> &Context {
        &self.ctx
    }

    /// Returns the mutable reference to the underlying `Context`.
    pub fn ctx_mut(&mut self) -> &mut Context {
        &mut self.ctx
    }

    /// Opens a cursor that writes into `id`.
    ///
    /// The cursor borrows the builder for as long as it lives, so exactly one is open
    /// at a time. Opening another at the same block later is fine — a cursor is a
    /// position, not a claim on the block.
    ///
    /// Two things stop anything being written past a terminator: the terminator
    /// builders consume the cursor, and the block carries an
    /// [`is_locked`](crate::cfg::basic_block::BasicBlock) flag for a cursor opened
    /// afterwards.
    pub fn cursor_at_block(&mut self, id: BasicBlockId) -> Cursor<'_> {
        Cursor {
            ctx: &mut self.ctx,
            block: id,
        }
    }

    /// Adds a global variable to the module.
    ///
    /// Emits `@name = global <ty> <initializer>`. The value is a
    /// [`Value::from_global`](crate::value::Value::from_global) away from being usable
    /// as an operand, where its type is `ptr` and `ty` is what that pointer points at.
    ///
    /// The two `Option`s are not independent — one of them has to say what the variable
    /// holds:
    ///
    /// | `ty` | `initializer` | result |
    /// |---|---|---|
    /// | `Some` | `Some` | checked: the initializer's type must equal `ty` |
    /// | `Some` | `None` | a declaration — the type is stated, the value is elsewhere |
    /// | `None` | `Some` | the type is taken from the initializer |
    /// | `None` | `None` | [`ContextError::GlobalTypeAndInitializerBothAbsent`] |
    ///
    /// The match in the first row is *exact*, not merely compatible: `llvm-as` refuses
    /// `@g = global i32 true` with "constant expression type mismatch", even though an
    /// `i1` is an integer.
    ///
    /// # Errors
    ///
    /// [`ContextError::DuplicateGlobalName`] if the name is taken — globals and
    /// functions share one namespace, so a variable may not reuse a function's name.
    /// [`ContextError::GlobalVariableTypeNotSized`] if `ty` is `void` or a function
    /// type, neither of which a variable can hold. Plus the two above.
    pub fn declare_global_variable<T: Into<String>>(
        &mut self,
        name: T,
        ty: Option<TyId>,
        initializer: Option<ConstExpr>,
    ) -> Result<GlobalId<GlobalVar>, ContextError> {
        let name_id: StrId = self.ctx.str_interner.intern(name.into()).into();

        if self.ctx.module.globals.contains_key(&name_id) {
            return Err(ContextError::DuplicateGlobalName(
                self.ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let final_ty = if let Some(ty) = ty {
            // A global holds a value, so its type needs a size — the same `void`-and-
            // function-types exclusion as a parameter or an `alloca`.
            if !ty.is_first_class(&self.ctx) {
                return Err(ContextError::GlobalVariableTypeNotSized(
                    ty.display(&self.ctx).to_string(),
                ));
            }

            // The initializer's type has to match *exactly*, not merely be
            // compatible: `llvm-as` refuses `@g = global i32 true` with "constant
            // expression type mismatch", even though an `i1` is an integer.
            if let Some(initializer) = &initializer {
                let init_ty = initializer.ty(&mut self.ctx);

                if init_ty != ty {
                    return Err(ContextError::GlobalInitializerTypeMismatch(
                        ty.display(&self.ctx).to_string(),
                        init_ty.display(&self.ctx).to_string(),
                    ));
                }
            }

            ty
        } else if let Some(initializer) = &initializer {
            initializer.ty(&mut self.ctx)
        } else {
            return Err(ContextError::GlobalTypeAndInitializerBothAbsent);
        };

        self.ctx.module.globals.insert(
            name_id,
            GlobalData {
                linkage: Linkage::External,
                visibility: Visibility::Default,
                kind: GlobalKind::Variable(GlobalVariable {
                    ty: final_ty,
                    initializer,
                }),
            },
        );

        self.ctx.module.global_variables.push(name_id);

        Ok(GlobalId {
            name: name_id,
            tag: GlobalVar,
        })
    }

    /// Declares a function that this module does not define.
    ///
    /// Emits `declare <result> @name(<param types>)`, which is how a module names
    /// something it links against — a host import, a runtime helper. LLVM requires
    /// it: a call to a name that is neither defined nor declared is refused with
    /// "use of undefined value".
    ///
    /// Parameters are **types only**, with no name hints, because a declaration has
    /// no body for a name to refer to. Nothing is returned either: there is no
    /// [`FuncId`], since there is no function here to add blocks to. The signature is
    /// recorded under the name, so
    /// [`Cursor::build_call`](crate::instruction::cursor::Cursor::build_call)
    /// resolves against it exactly as it would a defined function, and checks
    /// arity, argument types and the return type the same way.
    ///
    /// A declaration shares one namespace with definitions, so a name may be
    /// declared or defined, not both.
    ///
    /// # Errors
    ///
    /// - [`ContextError::DuplicateGlobalName`] — the name is already declared or
    ///   defined in this module.
    /// - [`ContextError::FunctionParamTypeNotSized`] — a parameter needs a size, so
    ///   `void` and function types are refused; aggregates are fine.
    /// - [`ContextError::FunctionResultTypeInvalid`] — a result may be `void` but not
    ///   a function type.
    pub fn declare_function<T: Into<String>>(
        &mut self,
        name: T,
        params: &[TyId],
        result: TyId,
    ) -> Result<GlobalId<DeclaredFunc>, ContextError> {
        let name_id: StrId = self.ctx.str_interner.intern(name.into()).into();

        if self.ctx.module.globals.contains_key(&name_id) {
            return Err(ContextError::DuplicateGlobalName(
                self.ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let void_ty = self.ctx.void_ty();

        for param_ty in params {
            if !param_ty.is_first_class(&self.ctx) {
                return Err(ContextError::FunctionParamTypeNotSized(
                    param_ty.display(&self.ctx).to_string(),
                ));
            }
        }

        if result != void_ty && !result.is_first_class(&self.ctx) {
            return Err(ContextError::FunctionResultTypeInvalid(
                result.display(&self.ctx).to_string(),
            ));
        }

        self.ctx.module.globals.insert(
            name_id,
            GlobalData {
                linkage: Linkage::External,
                visibility: Visibility::Default,
                kind: GlobalKind::Func(FuncSignature::new(params, result)),
            },
        );

        self.ctx.module.imported_functions.push(name_id);

        Ok(GlobalId {
            name: name_id,
            tag: DeclaredFunc,
        })
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
    /// - [`ContextError::DuplicateGlobalName`] — LLVM identifies a definition by
    ///   name, so two `@name`s in one module is a build error rather than something
    ///   `llvm-as` finds later.
    /// - [`ContextError::FunctionParamTypeNotSized`] — a parameter is passed by
    ///   value, so it needs a size; aggregates are fine, `void` and function types
    ///   are not.
    /// - [`ContextError::FunctionResultTypeInvalid`] — a result may be `void` but not
    ///   a function type.
    /// - [`ContextError::InvalidRegisterName`] — a parameter's name hint is not a
    ///   legal LLVM local.
    pub fn define_function<T: Into<String>>(
        &mut self,
        name: T,
        params: &[(TyId, RegName)],
        result: TyId,
    ) -> Result<GlobalId<DefinedFunc>, ContextError> {
        let name_id: StrId = self.ctx.str_interner.intern(name.into()).into();

        if self.ctx.module.globals.contains_key(&name_id) {
            return Err(ContextError::DuplicateGlobalName(
                self.ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let void_ty = self.ctx.void_ty();

        // A parameter is passed by value, so it has to have a size: `llvm-as` refuses
        // a `void` one with "void type only allowed for function results" and a
        // function-typed one with "invalid type for function argument". Aggregates
        // are fine — `define void @f({i32, double} %x)` assembles.
        for (param_ty, _) in params {
            if !param_ty.is_first_class(&self.ctx) {
                return Err(ContextError::FunctionParamTypeNotSized(
                    param_ty.display(&self.ctx).to_string(),
                ));
            }
        }

        // A result may be `void` as well, but still not a function type — that one
        // `llvm-as` refuses with "invalid function return type".
        if result != void_ty && !result.is_first_class(&self.ctx) {
            return Err(ContextError::FunctionResultTypeInvalid(
                result.display(&self.ctx).to_string(),
            ));
        }

        let id = FuncId::new(self.ctx.funcs.alloc(Function {
            name: name_id,
            blocks: vec![],
            params: vec![],
            result: void_ty,
            block_names: FxHashSet::default(),
        }));

        let mut final_params: Vec<Value> = vec![];
        let mut param_tys = vec![];

        for (param_ty, param_name) in params {
            let name = self.ctx.name_for_reg(param_name, id)?;

            param_tys.push(*param_ty);
            final_params.push(Value::from_register(name, *param_ty, &mut self.ctx));
        }

        let func = self.ctx.get_func_mut(id);

        func.params = final_params;
        func.result = result;

        self.ctx.module.globals.insert(
            name_id,
            GlobalData {
                linkage: Linkage::External,
                visibility: Visibility::Default,
                kind: GlobalKind::Func(FuncSignature::new(&param_tys, result)),
            },
        );

        self.ctx.module.functions.push(id);

        self.ctx
            .register_def_instr_index
            .insert(id, FxHashMap::default());

        Ok(GlobalId {
            name: name_id,
            tag: DefinedFunc::new(id),
        })
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
        ControlFlowGraph { context: self.ctx }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cfg::global::{DefinedFunc, GlobalId},
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
    fn param_names(f: GlobalId<DefinedFunc>, ctx: &Context) -> Vec<String> {
        ctx.get_func(f.tag.raw())
            .params
            .iter()
            .map(|p| reg_name(p, ctx))
            .collect()
    }

    /// An unnamed temporary in `f`'s body, which is what advances the counter.
    fn unnamed_temp(cursor: &mut Cursor) -> String {
        let i32_ty = cursor.i32_ty();

        let val = cursor
            .build_alloca(i32_ty, None, None, RegName::Unnamed)
            .expect("an i32 is allocatable");

        reg_name(&val, cursor)
    }

    /// LLVM numbers unnamed values per function starting at `%0`, and **parameters
    /// come first** — so the body's first unnamed temporary continues from where the
    /// parameter list stopped rather than restarting.
    ///
    /// `llvm-as` pins the boundary: with two unnamed parameters and a *named* entry
    /// block, `%2` is the first legal instruction number.
    #[test]
    fn unnamed_params_take_the_first_numbers_and_the_body_continues() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let void_ty = builder.void_ty();

        let f = builder
            .define_function(
                "f".to_string(),
                &[(i32_ty, RegName::Unnamed), (i32_ty, RegName::Unnamed)],
                void_ty,
            )
            .unwrap();

        assert_eq!(
            param_names(f, &builder),
            ["0", "1"],
            "parameters number first"
        );

        let entry = f
            .add_basic_block("entry".to_string(), &mut builder)
            .unwrap();
        let mut cursor = builder.cursor_at_block(entry);

        assert_eq!(
            unnamed_temp(&mut cursor),
            "2",
            "the body continues the parameters' numbering, it does not restart"
        );

        assert_eq!(unnamed_temp(&mut cursor), "3");
    }

    /// A *named* parameter takes no number. `llvm-as` is explicit: in
    /// `define i32 @f(i32 %n, i32)` the unnamed parameter is `%0`, and numbering an
    /// instruction `%0` is refused with "expected to be numbered '%1' or greater".
    #[test]
    fn a_named_param_does_not_consume_a_number() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let void_ty = builder.void_ty();

        let f = builder
            .define_function(
                "f".to_string(),
                &[(i32_ty, "n".to_string().into()), (i32_ty, RegName::Unnamed)],
                void_ty,
            )
            .unwrap();

        assert_eq!(
            param_names(f, &builder),
            ["n", "0"],
            "only the unnamed parameter draws from the counter"
        );

        let entry = f
            .add_basic_block("entry".to_string(), &mut builder)
            .unwrap();
        let mut cursor = builder.cursor_at_block(entry);

        assert_eq!(
            unnamed_temp(&mut cursor),
            "1",
            "one number was taken, so the body starts at 1"
        );
    }

    /// All parameters named means none of them took a number, so the body starts at
    /// `%0` — `define i32 @f(i32 %a, i32 %b)` with a first instruction of `%0`
    /// assembles.
    #[test]
    fn all_named_params_leave_the_body_starting_at_zero() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let void_ty = builder.void_ty();

        let f = builder
            .define_function(
                "f".to_string(),
                &[
                    (i32_ty, "a".to_string().into()),
                    (i32_ty, "b".to_string().into()),
                ],
                void_ty,
            )
            .unwrap();

        assert_eq!(param_names(f, &builder), ["a", "b"]);

        let entry = f
            .add_basic_block("entry".to_string(), &mut builder)
            .unwrap();
        let mut cursor = builder.cursor_at_block(entry);

        assert_eq!(unnamed_temp(&mut cursor), "0");
    }

    /// The counter is per *function*, not per block: it runs on across a branch, which
    /// is what `llvm-as` accepts for `%1` in `entry` and `%2` in `next`.
    #[test]
    fn the_counter_continues_across_blocks() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let void_ty = builder.void_ty();

        let f = builder
            .define_function("f".to_string(), &[(i32_ty, RegName::Unnamed)], void_ty)
            .unwrap();

        assert_eq!(param_names(f, &builder), ["0"]);

        let entry = f
            .add_basic_block("entry".to_string(), &mut builder)
            .unwrap();
        let next = f.add_basic_block("next".to_string(), &mut builder).unwrap();

        let mut in_entry = builder.cursor_at_block(entry);

        assert_eq!(unnamed_temp(&mut in_entry), "1");

        let mut in_next = builder.cursor_at_block(next);

        assert_eq!(
            unnamed_temp(&mut in_next),
            "2",
            "a new block does not restart the numbering"
        );

        let mut in_entry = builder.cursor_at_block(entry);

        assert_eq!(
            unnamed_temp(&mut in_entry),
            "3",
            "and going back to the first block does not either"
        );
    }

    /// Two functions each number from zero, parameters included — `%0` in one is a
    /// different value from `%0` in the other.
    #[test]
    fn each_function_restarts_the_counter_including_its_params() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let void_ty = builder.void_ty();

        let f = builder
            .define_function(
                "f".to_string(),
                &[(i32_ty, RegName::Unnamed), (i32_ty, RegName::Unnamed)],
                void_ty,
            )
            .unwrap();

        let g = builder
            .define_function("g".to_string(), &[(i32_ty, RegName::Unnamed)], void_ty)
            .unwrap();

        assert_eq!(param_names(f, &builder), ["0", "1"]);
        assert_eq!(param_names(g, &builder), ["0"], "`g` starts over");

        let in_f = f
            .add_basic_block("entry".to_string(), &mut builder)
            .unwrap();
        let in_g = g
            .add_basic_block("entry".to_string(), &mut builder)
            .unwrap();

        let mut f_cursor = builder.cursor_at_block(in_f);

        assert_eq!(unnamed_temp(&mut f_cursor), "2");

        let mut g_cursor = builder.cursor_at_block(in_g);

        assert_eq!(
            unnamed_temp(&mut g_cursor),
            "1",
            "`g` has one parameter, so its body starts at 1"
        );
    }

    /// A declaration records its signature so a call resolves against it, exactly as
    /// a definition does — but adds no function to the arena, since there is no body.
    #[test]
    fn a_declaration_records_a_signature_without_a_definition() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();

        builder
            .declare_function("host".to_string(), &[i32_ty], i32_ty)
            .expect("a valid signature");

        assert_eq!(
            builder.module.imported_functions.len(),
            1,
            "the declaration is recorded"
        );

        assert_eq!(
            builder.module.functions.len(),
            0,
            "but it defines no function, so the module has no body for it"
        );

        assert_eq!(
            builder.funcs.len(),
            0,
            "and nothing is allocated in the arena"
        );
    }

    /// Declarations and definitions share one namespace, so a name is one or the
    /// other. Either order collides.
    #[test]
    fn a_declaration_and_a_definition_cannot_share_a_name() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let void_ty = builder.void_ty();

        builder
            .declare_function("f".to_string(), &[], i32_ty)
            .unwrap();

        let err = builder
            .define_function("f".to_string(), &[], i32_ty)
            .expect_err("`f` is already declared");

        assert!(
            matches!(&err, ContextError::DuplicateGlobalName(n) if n == "f"),
            "got: {err}"
        );

        // And the other way round.
        builder
            .define_function("g".to_string(), &[], void_ty)
            .unwrap();

        assert!(
            matches!(
                builder.declare_function("g".to_string(), &[], void_ty),
                Err(ContextError::DuplicateGlobalName(_))
            ),
            "`g` is already defined"
        );
    }

    /// The namespace spans *kinds*, not just functions. LLVM writes every module-level
    /// symbol as `@name`, so a variable and a function collide exactly as two functions
    /// would — which is why the error is
    /// [`DuplicateGlobalName`](ContextError::DuplicateGlobalName) rather than anything
    /// function-specific.
    #[test]
    fn a_variable_and_a_function_cannot_share_a_name() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();

        // A variable, then a function of the same name.
        builder
            .declare_global_variable("shared".to_string(), Some(i32_ty), None)
            .unwrap();

        let err = builder
            .define_function("shared".to_string(), &[], i32_ty)
            .expect_err("`shared` is already a global variable");

        assert!(
            matches!(&err, ContextError::DuplicateGlobalName(n) if n == "shared"),
            "got: {err}"
        );

        // And the other way round: a function, then a variable.
        builder
            .define_function("taken".to_string(), &[], i32_ty)
            .unwrap();

        let err = builder
            .declare_global_variable("taken".to_string(), Some(i32_ty), None)
            .expect_err("`taken` is already a function");

        assert!(
            matches!(&err, ContextError::DuplicateGlobalName(n) if n == "taken"),
            "got: {err}"
        );
    }

    /// A declaration's signature is checked like a definition's: a parameter needs a
    /// size, and a result may be `void` but not a function type.
    #[test]
    fn a_declaration_signature_is_checked() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let void_ty = builder.void_ty();

        let err = builder
            .declare_function("a".to_string(), &[void_ty], i32_ty)
            .expect_err("`void` is not a parameter type");

        assert!(
            matches!(&err, ContextError::FunctionParamTypeNotSized(t) if t == "void"),
            "got: {err}"
        );

        let signature = FuncSignature::new(&[i32_ty], i32_ty);
        let func_ty: TyId = builder.ty_interner.intern(Type::Func(signature)).into();

        assert!(
            matches!(
                builder.declare_function("b".to_string(), &[], func_ty),
                Err(ContextError::FunctionResultTypeInvalid(_))
            ),
            "a function type is not a result"
        );

        // `void` *is* a legal result, and no parameters is a legal signature.
        assert!(
            builder
                .declare_function("c".to_string(), &[], void_ty)
                .is_ok()
        );

        assert_eq!(
            builder.module.imported_functions.len(),
            1,
            "only the accepted declaration was recorded"
        );
    }

    /// A parameter becomes a register of the declared type, named in order, and the
    /// result is recorded as given.
    #[test]
    fn a_signature_becomes_typed_registers_and_a_result() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let f64_ty = builder.f64_ty();
        let ptr_ty = builder.ptr_ty();

        let f = builder
            .define_function(
                "sum".to_string(),
                &[
                    (i32_ty, "n".to_string().into()),
                    (f64_ty, "x".to_string().into()),
                    (ptr_ty, RegName::Unnamed),
                ],
                i32_ty,
            )
            .expect("a valid signature");

        let func = builder.get_func(f.tag.raw());

        assert_eq!(func.result, i32_ty, "the result is the one given");
        assert_eq!(func.params.len(), 3);

        let spelled: Vec<String> = func
            .params
            .iter()
            .map(|p| builder.display(p.ty()).to_string())
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

                builder.str_interner.value(reg.name.0).to_string()
            })
            .collect();

        assert_eq!(names, ["n", "x", "0"]);
    }

    /// An aggregate parameter is passed by value and is perfectly legal —
    /// `define void @f({i32, double} %x)` assembles.
    #[test]
    fn an_aggregate_parameter_is_allowed() {
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let f64_ty = builder.f64_ty();
        let void_ty = builder.void_ty();

        let struct_ty = builder
            .ty_interner
            .intern(Type::Struct {
                fields: Box::new([i32_ty, f64_ty]),
                packed: false,
            })
            .into();

        assert!(
            builder
                .define_function("f".to_string(), &[(struct_ty, RegName::Unnamed)], void_ty)
                .is_ok(),
            "an aggregate is sized, so it can be passed"
        );
    }

    /// A parameter is passed by value, so it has to have a size. `llvm-as` refuses a
    /// `void` one with "void type only allowed for function results" and a
    /// function-typed one with "invalid type for function argument".
    #[test]
    fn an_unsized_parameter_is_refused() {
        let mut builder = fixture();

        let void_ty = builder.void_ty();

        let err = builder
            .define_function("f".to_string(), &[(void_ty, RegName::Unnamed)], void_ty)
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
        let mut builder = fixture();

        let i32_ty = builder.i32_ty();
        let void_ty = builder.void_ty();

        assert!(
            builder
                .define_function("returns_void".to_string(), &[], void_ty)
                .is_ok(),
            "`void` is a result like any other"
        );

        let signature = FuncSignature::new(&[i32_ty], i32_ty);
        let func_ty: TyId = builder.ty_interner.intern(Type::Func(signature)).into();

        let err = builder
            .define_function("returns_fn".to_string(), &[], func_ty)
            .expect_err("a function cannot return a function type");

        assert!(
            matches!(&err, ContextError::FunctionResultTypeInvalid(t) if t == "i32 (i32)"),
            "the error must name the offending type, got: {err}"
        );
    }
}
