//! Function definitions and the handle that addresses them.

use crate::{
    cfg::{
        basic_block::{BasicBlock, BasicBlockId},
        builder::Builder,
        context::Context,
        global::{DefinedFunc, GlobalId},
    },
    error::ContextError,
    interner::{StrId, TyId},
    value::Value,
};
use id_arena::Id;
use rustc_hash::FxHashSet;

/// One function definition.
///
/// Parameters are stored as [`Value`]s rather than bare types, because that is what
/// they are once the function exists: registers the caller supplied, usable directly
/// as operands. `blocks` is in creation order, which is the order the emitter writes
/// them in — the first is the entry block.
pub struct Function {
    pub(crate) name: StrId,
    pub(crate) params: Vec<Value>,
    pub(crate) result: TyId,
    pub(crate) blocks: Vec<BasicBlockId>,
    pub(crate) block_names: FxHashSet<StrId>,
}

/// A handle to a [`Function`] in a [`Context`]'s arena.
///
/// Carries the methods that read or extend the function, so `f.add_basic_block(..)`
/// reads like a method call even though the storage lives in the context.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct FuncId(Id<Function>);

impl Clone for FuncId {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for FuncId {}

impl FuncId {
    /// Wraps an arena id. Only [`Builder::define_function`](crate::cfg::builder::Builder::define_function)
    /// calls this, so an id always names a function that exists.
    pub(crate) fn new(id: Id<Function>) -> Self {
        FuncId(id)
    }

    pub(crate) fn raw(&self) -> Id<Function> {
        self.0
    }
}

impl GlobalId<DefinedFunc> {
    /// The underlying arena id.
    pub(crate) fn raw(&self) -> Id<Function> {
        self.tag.raw().0
    }

    /// Appends a basic block to this function.
    ///
    /// The **first** block added becomes the entry block, which matters: LLVM gives
    /// the entry block no predecessors, so a phi cannot go in it.
    ///
    /// # Errors
    ///
    /// [`ContextError::DuplicateBasicBlockName`] if this function already has a block
    /// with that name — the label a branch names would otherwise be ambiguous. The
    /// check is per function, so an `entry` block in every function is fine.
    pub fn add_basic_block(
        &self,
        name: String,
        ctx: &mut Context,
    ) -> Result<BasicBlockId, ContextError> {
        let name_id: StrId = ctx.str_interner.intern(name).into();
        let func = ctx.get_func_mut(self.tag.raw());

        if func.block_names.contains(&name_id) {
            return Err(ContextError::DuplicateBasicBlockName(
                ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let is_first = func.blocks.is_empty();

        let id = BasicBlockId::new(ctx.blocks.alloc(BasicBlock {
            name: name_id,
            is_first,
            func_id: self.tag.raw(),
            phis: vec![],
            instructions: vec![],
            is_locked: false,
        }));

        let func = ctx.get_func_mut(self.tag.raw());

        func.blocks.push(id);
        func.block_names.insert(name_id);

        Ok(id)
    }

    /// This function's `n`th parameter, or `None` past the end.
    ///
    /// Parameters are registers, usable directly as operands. A pointer parameter has
    /// no defining instruction, so
    /// [`try_inferring_pointee_ty`](crate::value::Value) declines on it and any
    /// `load`, `store` or `getelementptr` through it needs its type given explicitly.
    pub fn nth_param<'a>(&self, n: usize, ctx: &'a Context) -> Option<&'a Value> {
        let func = ctx.get_func(self.tag.raw());
        let params = &func.params;

        if n >= params.len() {
            return None;
        }

        Some(&params[n])
    }

    /// The declared result type, `void` included.
    ///
    /// Every `ret` in this function is checked against it.
    pub fn return_ty(&self, ctx: &Context) -> TyId {
        let func = ctx.get_func(self.tag.raw());

        func.result
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        error::ContextError,
        test_support::{add_fn, fixture},
        value::ValueKind,
    };

    /// Parameters come back by position, typed and named as declared.
    #[test]
    fn nth_param_returns_each_parameter_in_order() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .define_function(
                "f".to_string(),
                &[
                    (i32_ty, Some("n".to_string())),
                    (f64_ty, Some("x".to_string())),
                ],
                void_ty,
                &mut ctx,
            )
            .unwrap();

        let first = f.nth_param(0, &ctx).expect("two parameters were declared");
        let second = f.nth_param(1, &ctx).expect("two parameters were declared");

        assert_eq!(first.ty(), i32_ty);
        assert_eq!(second.ty(), f64_ty);

        // A parameter is a register, which is what makes it usable as an operand.
        let name_of = |v: &crate::value::Value| {
            let ValueKind::Reg(reg) = v.kind() else {
                panic!("a parameter is a register")
            };

            ctx.str_interner.value(reg.name.0).to_string()
        };

        assert_eq!(name_of(first), "n");
        assert_eq!(name_of(second), "x");
    }

    /// Out of range is `None` rather than a panic, so a caller can ask without
    /// knowing the arity — and a function with no parameters has nothing at 0.
    #[test]
    fn nth_param_is_none_past_the_end() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let one_param = builder
            .define_function("f".to_string(), &[(i32_ty, None)], void_ty, &mut ctx)
            .unwrap();

        assert!(one_param.nth_param(0, &ctx).is_some());
        assert!(one_param.nth_param(1, &ctx).is_none(), "one past the end");
        assert!(one_param.nth_param(99, &ctx).is_none());

        let no_params = builder
            .define_function("g".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        assert!(
            no_params.nth_param(0, &ctx).is_none(),
            "a function with no parameters has nothing at 0"
        );
    }

    /// Parameters are per function: asking one for its parameters never reaches
    /// another's.
    #[test]
    fn nth_param_is_scoped_to_its_function() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let f64_ty = ctx.f64_ty();
        let void_ty = ctx.void_ty();

        let f = builder
            .define_function("f".to_string(), &[(i32_ty, None)], void_ty, &mut ctx)
            .unwrap();

        let g = builder
            .define_function("g".to_string(), &[(f64_ty, None)], void_ty, &mut ctx)
            .unwrap();

        assert_eq!(f.nth_param(0, &ctx).unwrap().ty(), i32_ty);
        assert_eq!(g.nth_param(0, &ctx).unwrap().ty(), f64_ty);
    }

    /// The result type comes back as declared, `void` included — that is a real
    /// result, not the absence of one.
    #[test]
    fn return_ty_reports_the_declared_result() {
        let (mut ctx, builder) = fixture();

        let i32_ty = ctx.i32_ty();
        let void_ty = ctx.void_ty();

        let returns_i32 = builder
            .define_function("a".to_string(), &[], i32_ty, &mut ctx)
            .unwrap();

        let returns_void = builder
            .define_function("b".to_string(), &[], void_ty, &mut ctx)
            .unwrap();

        assert_eq!(returns_i32.return_ty(&ctx), i32_ty);
        assert_eq!(returns_void.return_ty(&ctx), void_ty);

        assert_eq!(ctx.display(returns_i32.return_ty(&ctx)).to_string(), "i32");
        assert_eq!(
            ctx.display(returns_void.return_ty(&ctx)).to_string(),
            "void"
        );
    }

    /// `is_first` marks the entry block, which is the one a phi cannot go in. Only
    /// the first block added to a function is it.
    #[test]
    fn only_the_first_block_of_a_function_is_the_entry() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        assert!(ctx.blocks.get(entry.raw()).unwrap().is_first);
        assert!(!ctx.blocks.get(body.raw()).unwrap().is_first);

        // A second function's first block is an entry too — `is_first` is per
        // function, not per module.
        let g = add_fn("g", &builder, &mut ctx).unwrap();
        let g_entry = g.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert!(ctx.blocks.get(g_entry.raw()).unwrap().is_first);
    }

    /// A block records which function it belongs to, and the function records the
    /// block — the two have to agree or a later walk of the graph goes wrong.
    #[test]
    fn a_block_and_its_function_agree() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert_eq!(ctx.blocks.get(entry.raw()).unwrap().func_id.raw(), f.raw());
        assert_eq!(ctx.funcs.get(f.raw()).unwrap().blocks, vec![entry]);
    }

    /// Two blocks sharing a label would make a branch to it ambiguous.
    #[test]
    fn a_duplicate_block_name_in_one_function_is_refused() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();

        f.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        let err = f
            .add_basic_block("entry".to_string(), &mut ctx)
            .expect_err("the name is taken in this function");

        assert!(
            matches!(&err, ContextError::DuplicateBasicBlockName(name) if name == "entry"),
            "the error must name the collision, got: {err}"
        );

        assert_eq!(
            ctx.funcs.get(f.raw()).unwrap().blocks.len(),
            1,
            "the refused block must not have been added"
        );
    }

    /// The check is per function: an `entry` block in every function is the normal
    /// case, so scoping it to the module would reject almost every real program.
    #[test]
    fn the_same_block_name_in_another_function_is_fine() {
        let (mut ctx, builder) = fixture();
        let f = add_fn("f", &builder, &mut ctx).unwrap();
        let g = add_fn("g", &builder, &mut ctx).unwrap();
        let in_f = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let in_g = g.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert_ne!(in_f, in_g, "distinct blocks");

        // The name interns once even so, which is the point of interning it.
        assert_eq!(
            ctx.blocks.get(in_f.raw()).unwrap().name,
            ctx.blocks.get(in_g.raw()).unwrap().name
        );
    }
}
