use crate::{
    cfg::{
        basic_block::BasicBlockId,
        context::Context,
        function::{FuncId, Function},
    },
    error::BuildError,
    interner::StrId,
};
use rustc_hash::FxHashSet;

pub mod basic_block;
pub mod context;
pub mod function;

pub struct Global {}

struct Module {
    triple: String,
    data_layout: String,
    globals: Vec<Global>,
    functions: Vec<FuncId>,
    func_names: FxHashSet<StrId>,
}

pub struct Cursor {
    pub(crate) block: BasicBlockId,
}

pub struct Builder {
    module: Module,
}

impl Builder {
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

    pub fn cursor_at_block(&mut self, id: BasicBlockId) -> Cursor {
        Cursor { block: id }
    }

    pub fn add_function(&mut self, name: String, ctx: &mut Context) -> Result<FuncId, BuildError> {
        let name_id: StrId = ctx.str_interner.intern(name)?.into();

        if self.module.func_names.contains(&name_id) {
            return Err(BuildError::DuplicateFunctionName(
                ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let id = FuncId::new(ctx.funcs.alloc(Function {
            name: name_id,
            blocks: vec![],
            block_names: FxHashSet::default(),
        }));

        self.module.func_names.insert(name_id);
        self.module.functions.push(id);

        Ok(id)
    }

    pub fn build(self) -> ControlFlowGraph {
        ControlFlowGraph {
            module: self.module,
        }
    }
}

pub struct ControlFlowGraph {
    module: Module,
}

impl ControlFlowGraph {
    pub fn emit_ll(&self, ctx: &Context) -> String {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction::Instruction,
        value::{Type, Value},
    };

    /// A context and a builder, which every test below needs.
    fn fixture() -> (Context, Builder) {
        (
            Context::default(),
            Builder::new("arm64-apple-macosx".to_string(), String::new()),
        )
    }

    /// A distinct `Value` per call, for filling phi branches. The value itself is
    /// never inspected — what the tests are about is the graph.
    fn value(n: i32, ctx: &mut Context) -> Value {
        Value::from_const(n, None, &mut ctx.const_interner).expect("constant interns")
    }

    #[test]
    fn simple_api_usage() {
        let (mut ctx, mut builder) = fixture();
        let func = builder.add_function("sum".to_string(), &mut ctx).unwrap();
        let entry = func.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let mut cursor = builder.cursor_at_block(entry);

        cursor.add_unconditional_br(entry, &mut ctx);

        assert_eq!(
            ctx.blocks.get(entry.raw()).unwrap().instructions.len(),
            1,
            "the cursor writes into the block it was opened at"
        );

        let cfg = builder.build();

        assert_eq!(cfg.module.functions.len(), 1);
    }

    // ---- functions -------------------------------------------------------

    /// Every function is its own entry, and the module records them in the order
    /// they were added.
    #[test]
    fn functions_get_distinct_ids_in_order() {
        let (mut ctx, mut builder) = fixture();
        let a = builder.add_function("a".to_string(), &mut ctx).unwrap();
        let b = builder.add_function("b".to_string(), &mut ctx).unwrap();

        assert_ne!(a.raw(), b.raw());
        assert_eq!(builder.module.functions.len(), 2);
        assert_eq!(ctx.funcs.len(), 2);
    }

    /// LLVM identifies a definition by its name, so two `@sum`s in one module is
    /// not something to discover at emit time.
    #[test]
    fn a_duplicate_function_name_is_refused() {
        let (mut ctx, mut builder) = fixture();

        builder.add_function("sum".to_string(), &mut ctx).unwrap();

        let err = builder
            .add_function("sum".to_string(), &mut ctx)
            .expect_err("the name is taken");

        assert!(
            matches!(&err, BuildError::DuplicateFunctionName(name) if name == "sum"),
            "the error must name the collision, got: {err}"
        );

        assert_eq!(
            builder.module.functions.len(),
            1,
            "the refused function must not have been added"
        );
    }

    // ---- basic blocks ----------------------------------------------------

    /// `is_first` marks the entry block, which is the one a phi cannot go in. Only
    /// the first block added to a function is it.
    #[test]
    fn only_the_first_block_of_a_function_is_the_entry() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        assert!(ctx.blocks.get(entry.raw()).unwrap().is_first);
        assert!(!ctx.blocks.get(body.raw()).unwrap().is_first);

        // A second function's first block is an entry too — `is_first` is per
        // function, not per module.
        let g = builder.add_function("g".to_string(), &mut ctx).unwrap();
        let g_entry = g.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert!(ctx.blocks.get(g_entry.raw()).unwrap().is_first);
    }

    /// A block records which function it belongs to, and the function records the
    /// block — the two have to agree or a later walk of the graph goes wrong.
    #[test]
    fn a_block_and_its_function_agree() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert_eq!(ctx.blocks.get(entry.raw()).unwrap().func_id.raw(), f.raw());
        assert_eq!(ctx.funcs.get(f.raw()).unwrap().blocks, vec![entry]);
    }

    /// Two blocks sharing a label would make a branch to it ambiguous.
    #[test]
    fn a_duplicate_block_name_in_one_function_is_refused() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();

        f.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        let err = f
            .add_basic_block("entry".to_string(), &mut ctx)
            .expect_err("the name is taken in this function");

        assert!(
            matches!(&err, BuildError::DuplicateBasicBlockName(name) if name == "entry"),
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
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let g = builder.add_function("g".to_string(), &mut ctx).unwrap();
        let in_f = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let in_g = g.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert_ne!(in_f, in_g, "distinct blocks");

        // The name interns once even so, which is the point of interning it.
        assert_eq!(
            ctx.blocks.get(in_f.raw()).unwrap().name,
            ctx.blocks.get(in_g.raw()).unwrap().name
        );
    }

    // ---- instructions ----------------------------------------------------

    /// Instructions land in the block the cursor names, in order.
    #[test]
    fn instructions_append_in_order_to_the_open_block() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let mut cursor = builder.cursor_at_block(entry);

        cursor.add_unconditional_br(body, &mut ctx);
        cursor.add_unconditional_br(entry, &mut ctx);

        let instrs = &ctx.blocks.get(entry.raw()).unwrap().instructions;

        assert_eq!(instrs.len(), 2);

        assert!(
            matches!(instrs[0], Instruction::UnconditionalBr { label } if label == body),
            "the first instruction is the one added first"
        );

        assert!(
            ctx.blocks.get(body.raw()).unwrap().instructions.is_empty(),
            "the other block is untouched"
        );
    }

    // ---- phis ------------------------------------------------------------

    /// A phi selects between incoming values, so with none there is nothing to
    /// select — and no first branch to take the phi's type from either.
    ///
    /// This is the one phi path that does not reach `Value::ty`, so it is the only
    /// one runnable until that lands.
    #[test]
    fn a_phi_needs_at_least_one_branch() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let _entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let mut cursor = builder.cursor_at_block(body);

        let err = cursor
            .add_phi(&[], "result", &mut ctx)
            .expect_err("a phi with no branches has no value and no type");

        assert!(matches!(err, BuildError::PhiInstructionWithNoBranches));

        assert!(
            ctx.blocks.get(body.raw()).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    // The rest of the phi surface runs through `Cursor::add_phi`, which calls
    // `Value::ty` and `Context::name_for_reg` — both `todo!()` today, so these
    // panic before reaching what they assert. They are written against the current
    // signature so they start working the moment those two land.

    /// The entry block has no predecessors, so a phi there has nothing to choose
    /// between.
    #[test]
    #[ignore = "blocked on Value::ty and Context::name_for_reg"]
    fn a_phi_cannot_go_in_the_entry_block() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let mut cursor = builder.cursor_at_block(entry);

        let err = cursor
            .add_phi(&[(entry, v)], "result", &mut ctx)
            .expect_err("no predecessors to choose between");

        assert!(matches!(
            err,
            BuildError::PhiInstructionCannotBeAddedToEntryBasicBlock
        ));
    }

    /// In a later block a phi is fine, and each one gets its own index — tagged
    /// with the block, since an index alone would mean different phis in different
    /// blocks.
    #[test]
    #[ignore = "blocked on Value::ty and Context::name_for_reg"]
    fn phis_in_a_later_block_are_indexed_within_that_block() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let tail = f.add_basic_block("tail".to_string(), &mut ctx).unwrap();
        let (v1, v2, v3) = (value(1, &mut ctx), value(2, &mut ctx), value(3, &mut ctx));
        let mut cursor = builder.cursor_at_block(body);

        let (first, _) = cursor.add_phi(&[(entry, v1)], "a", &mut ctx).unwrap();
        let (second, _) = cursor.add_phi(&[(entry, v2)], "b", &mut ctx).unwrap();

        assert_eq!((first.index, first.block), (0, body));
        assert_eq!((second.index, second.block), (1, body));

        // A different block starts its own numbering, which is why the id has to
        // carry the block to be unambiguous.
        let mut cursor = builder.cursor_at_block(tail);
        let (elsewhere, _) = cursor.add_phi(&[(entry, v3)], "c", &mut ctx).unwrap();

        assert_eq!((elsewhere.index, elsewhere.block), (0, tail));
        assert_ne!(elsewhere.block, first.block);
    }

    /// The phi's own type comes from its first branch, and the value it hands back
    /// carries that type — that is what makes the result usable downstream.
    #[test]
    #[ignore = "blocked on Value::ty and Context::name_for_reg"]
    fn a_phi_yields_a_value_of_its_branches_type() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let mut cursor = builder.cursor_at_block(body);

        let (_, result) = cursor.add_phi(&[(entry, v)], "merged", &mut ctx).unwrap();

        assert_eq!(result.ty(), Type::I32, "an i32 branch makes an i32 phi");
    }

    /// A phi produces one value, so every incoming value has to have its type.
    #[test]
    #[ignore = "blocked on Value::ty and Context::name_for_reg"]
    fn phi_branches_must_all_share_the_phis_type() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let other = f.add_basic_block("other".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        let an_i32 = value(1, &mut ctx);
        let an_i64 = Value::from_const(1i64, None, &mut ctx.const_interner).unwrap();

        let mut cursor = builder.cursor_at_block(body);

        let err = cursor
            .add_phi(&[(entry, an_i32), (other, an_i64)], "merged", &mut ctx)
            .expect_err("the second branch is an i64");

        assert!(
            matches!(
                &err,
                BuildError::PhiInstructionBranchTypeMismatch(phi, branch)
                    if *phi == Type::I32 && *branch == Type::I64
            ),
            "the error must name both types, got: {err}"
        );
    }

    /// Phis have to precede every other instruction in their block, so once one
    /// has been emitted the window has closed.
    #[test]
    #[ignore = "blocked on Value::ty and Context::name_for_reg"]
    fn a_phi_cannot_follow_an_instruction() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let v = value(1, &mut ctx);
        let mut cursor = builder.cursor_at_block(body);

        cursor.add_unconditional_br(body, &mut ctx);

        let err = cursor
            .add_phi(&[(entry, v)], "result", &mut ctx)
            .expect_err("the block already has an instruction");

        assert!(matches!(err, BuildError::PhiInstructionAddError));

        assert!(
            ctx.blocks.get(body.raw()).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    /// A phi names one value per *predecessor*, so the same predecessor twice is a
    /// bug in the caller — and it is a different bug from an entry-block phi.
    #[test]
    #[ignore = "blocked on Value::ty and Context::name_for_reg"]
    fn a_phi_takes_each_predecessor_once() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let other = f.add_basic_block("other".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let (v1, v2, v3) = (value(1, &mut ctx), value(2, &mut ctx), value(3, &mut ctx));
        let mut cursor = builder.cursor_at_block(body);

        let (phi, _) = cursor
            .add_phi(&[(entry, v1), (other, v2)], "merged", &mut ctx)
            .unwrap();

        let err = phi
            .add_branch((entry, v3), &mut ctx)
            .expect_err("`entry` is already a predecessor of this phi");

        assert!(
            matches!(err, BuildError::BasicBlockBranchAlreadyInPhiInstruction),
            "a repeated predecessor is not an entry-block error, got: {err}"
        );

        // One incoming value per predecessor, and the two branches it was built
        // with are the two it has — not four.
        let stored = &ctx.blocks.get(body.raw()).unwrap().phis[phi.index];

        assert_eq!(stored.branches.len(), 2);
        assert_eq!(stored.blocks.len(), 2);
    }

    // ---- ids across contexts --------------------------------------------

    /// A `Context` owns the arenas an id indexes, so an id from another one is not
    /// silently written into the wrong block — `id_arena` catches it.
    #[test]
    #[should_panic(expected = "valid id are never constructed")]
    fn an_id_from_another_context_panics_rather_than_writing_elsewhere() {
        // A builder per context: a builder's duplicate-name set holds `StrId`s,
        // which only mean anything against the context they were interned in.
        let (mut ctx_a, mut builder_a) = fixture();
        let (mut ctx_b, mut builder_b) = fixture();

        let f = builder_a.add_function("f".to_string(), &mut ctx_a).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx_a).unwrap();
        let g = builder_b.add_function("g".to_string(), &mut ctx_b).unwrap();

        g.add_basic_block("entry".to_string(), &mut ctx_b).unwrap();

        // `entry` indexes `ctx_a`'s block arena, so reaching for it in `ctx_b` must
        // not land on whatever block sits at the same position there.
        let mut cursor = builder_a.cursor_at_block(entry);

        cursor.add_unconditional_br(entry, &mut ctx_b);
    }

    /// The string pool is shared across the whole context, so a name used by both
    /// a function and a block costs one entry.
    #[test]
    fn names_are_interned_once_across_the_context() {
        let (mut ctx, mut builder) = fixture();

        let f = builder
            .add_function("shared".to_string(), &mut ctx)
            .unwrap();

        f.add_basic_block("shared".to_string(), &mut ctx).unwrap();

        assert_eq!(
            ctx.str_interner.len(),
            1,
            "the function and the block share one pooled name"
        );

        let _ = Type::I1;
    }
}
