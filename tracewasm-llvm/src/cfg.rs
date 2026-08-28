use crate::{
    constants::ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID,
    error::BuildError,
    instruction::Instruction,
    interner::{ConstInterner, StrId, StrInterner},
    value::Value,
};
use id_arena::{Arena, Id};
use rustc_hash::FxHashSet;

#[derive(Default)]
pub struct PhiInstruction {
    branches: Vec<(BasicBlockId, Value)>,
    blocks: FxHashSet<BasicBlockId>,
}

impl PhiInstruction {
    pub fn add_branch(&mut self, branch: (BasicBlockId, Value)) -> Result<(), BuildError> {
        let block_id = branch.0;

        if self.blocks.contains(&block_id) {
            return Err(BuildError::BasicBlockBranchAlreadyInPhiInstruction);
        }

        self.blocks.insert(block_id);
        self.branches.push(branch);

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PhiInstrId {
    index: usize,
    block: BasicBlockId,
}

pub struct BasicBlock {
    name: StrId,
    is_first: bool,
    func_id: FuncId,
    phis: Vec<PhiInstruction>,
    instructions: Vec<Instruction>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BasicBlockId(Id<BasicBlock>);

impl Clone for BasicBlockId {
    fn clone(&self) -> Self {
        BasicBlockId(self.0)
    }
}

impl Copy for BasicBlockId {}

impl BasicBlockId {
    fn add_instruction(&self, instr: Instruction, ctx: &mut Context) {
        let block = ctx
            .blocks
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        block.instructions.push(instr);
    }

    fn add_phi(&self, instr: PhiInstruction, ctx: &mut Context) -> Result<PhiInstrId, BuildError> {
        let block = ctx
            .blocks
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        if block.is_first {
            return Err(BuildError::PhiInstructionCannotBeAddedToEntryBasicBlock);
        }

        if !block.instructions.is_empty() {
            return Err(BuildError::PhiInstructionAddError);
        }

        let id = block.phis.len();

        block.phis.push(instr);

        Ok(PhiInstrId {
            index: id,
            block: *self,
        })
    }
}

pub struct Function {
    name: StrId,
    blocks: Vec<BasicBlockId>,
    block_names: FxHashSet<StrId>,
}

#[derive(Debug)]
pub struct FuncId(Id<Function>);

impl Clone for FuncId {
    fn clone(&self) -> Self {
        FuncId(self.0)
    }
}

impl Copy for FuncId {}

impl FuncId {
    pub fn add_basic_block(
        &self,
        name: String,
        ctx: &mut Context,
    ) -> Result<BasicBlockId, BuildError> {
        let name_id: StrId = ctx.str_interner.intern(name)?.into();

        let func = ctx
            .funcs
            .get(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        if func.block_names.contains(&name_id) {
            return Err(BuildError::DuplicateBasicBlockName(
                ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let is_first = func.blocks.is_empty();

        let id = BasicBlockId(ctx.blocks.alloc(BasicBlock {
            name: name_id,
            is_first,
            func_id: *self,
            phis: vec![],
            instructions: vec![],
        }));

        let func = ctx
            .funcs
            .get_mut(self.0)
            .expect(ENTRY_IN_ARENA_SHOULD_EXIST_FOR_ID);

        func.blocks.push(id);
        func.block_names.insert(name_id);

        Ok(id)
    }
}

pub struct Global {}

struct Module {
    triple: String,
    data_layout: String,
    globals: Vec<Global>,
    functions: Vec<FuncId>,
    func_names: FxHashSet<StrId>,
}

pub struct Context {
    blocks: Arena<BasicBlock>,
    funcs: Arena<Function>,
    str_interner: StrInterner,
    const_interner: ConstInterner,
}

impl Default for Context {
    fn default() -> Self {
        Context {
            blocks: Arena::default(),
            funcs: Arena::default(),
            str_interner: StrInterner::default(),
            const_interner: ConstInterner::default(),
        }
    }
}

pub struct Cursor {
    block: BasicBlockId,
}

impl Cursor {
    pub fn add_instruction(&mut self, instr: Instruction, ctx: &mut Context) {
        self.block.add_instruction(instr, ctx);
    }

    pub fn add_phi(
        &mut self,
        instr: PhiInstruction,
        ctx: &mut Context,
    ) -> Result<PhiInstrId, BuildError> {
        self.block.add_phi(instr, ctx)
    }
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

        let id = FuncId(ctx.funcs.alloc(Function {
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
    use crate::value::{Type, Value};

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

        cursor.add_instruction(Instruction::new_unconditional_br(entry), &mut ctx);

        assert_eq!(
            ctx.blocks.get(entry.0).unwrap().instructions.len(),
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

        assert_ne!(a.0, b.0);
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

        assert!(ctx.blocks.get(entry.0).unwrap().is_first);
        assert!(!ctx.blocks.get(body.0).unwrap().is_first);

        // A second function's first block is an entry too — `is_first` is per
        // function, not per module.
        let g = builder.add_function("g".to_string(), &mut ctx).unwrap();
        let g_entry = g.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert!(ctx.blocks.get(g_entry.0).unwrap().is_first);
    }

    /// A block records which function it belongs to, and the function records the
    /// block — the two have to agree or a later walk of the graph goes wrong.
    #[test]
    fn a_block_and_its_function_agree() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert_eq!(ctx.blocks.get(entry.0).unwrap().func_id.0, f.0);
        assert_eq!(ctx.funcs.get(f.0).unwrap().blocks, vec![entry]);
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
            ctx.funcs.get(f.0).unwrap().blocks.len(),
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
            ctx.blocks.get(in_f.0).unwrap().name,
            ctx.blocks.get(in_g.0).unwrap().name
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

        cursor.add_instruction(Instruction::new_unconditional_br(body), &mut ctx);
        cursor.add_instruction(Instruction::new_unconditional_br(entry), &mut ctx);

        let instrs = &ctx.blocks.get(entry.0).unwrap().instructions;

        assert_eq!(instrs.len(), 2);

        assert!(
            matches!(instrs[0], Instruction::UnconditionalBr { label } if label == body),
            "the first instruction is the one added first"
        );

        assert!(
            ctx.blocks.get(body.0).unwrap().instructions.is_empty(),
            "the other block is untouched"
        );
    }

    // ---- phis ------------------------------------------------------------

    /// The entry block has no predecessors, so a phi there has nothing to choose
    /// between.
    #[test]
    fn a_phi_cannot_go_in_the_entry_block() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let mut cursor = builder.cursor_at_block(entry);

        let err = cursor
            .add_phi(PhiInstruction::default(), &mut ctx)
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
    fn phis_in_a_later_block_are_indexed_within_that_block() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let _entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let tail = f.add_basic_block("tail".to_string(), &mut ctx).unwrap();
        let mut cursor = builder.cursor_at_block(body);
        let first = cursor.add_phi(PhiInstruction::default(), &mut ctx).unwrap();
        let second = cursor.add_phi(PhiInstruction::default(), &mut ctx).unwrap();

        assert_eq!((first.index, first.block), (0, body));
        assert_eq!((second.index, second.block), (1, body));

        // A different block starts its own numbering, which is why the id has to
        // carry the block to be unambiguous.
        let mut cursor = builder.cursor_at_block(tail);
        let elsewhere = cursor.add_phi(PhiInstruction::default(), &mut ctx).unwrap();

        assert_eq!((elsewhere.index, elsewhere.block), (0, tail));
        assert_ne!(elsewhere.block, first.block);
    }

    /// Phis have to precede every other instruction in their block, so once one
    /// has been emitted the window has closed.
    #[test]
    fn a_phi_cannot_follow_an_instruction() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let _entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();
        let mut cursor = builder.cursor_at_block(body);

        cursor.add_instruction(Instruction::new_unconditional_br(body), &mut ctx);

        let err = cursor
            .add_phi(PhiInstruction::default(), &mut ctx)
            .expect_err("the block already has an instruction");

        assert!(matches!(err, BuildError::PhiInstructionAddError));

        assert!(
            ctx.blocks.get(body.0).unwrap().phis.is_empty(),
            "the refused phi must not have been added"
        );
    }

    /// A phi names one value per *predecessor*, so the same predecessor twice is a
    /// bug in the caller — and it is a different bug from an entry-block phi.
    #[test]
    fn a_phi_takes_each_predecessor_once() {
        let (mut ctx, mut builder) = fixture();
        let f = builder.add_function("f".to_string(), &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let other = f.add_basic_block("other".to_string(), &mut ctx).unwrap();
        let (v1, v2, v3) = (value(1, &mut ctx), value(2, &mut ctx), value(3, &mut ctx));
        let mut phi = PhiInstruction::default();

        phi.add_branch((entry, v1)).expect("first predecessor");
        phi.add_branch((other, v2)).expect("a second, distinct one");

        let err = phi
            .add_branch((entry, v3))
            .expect_err("`entry` is already a predecessor of this phi");

        assert!(
            matches!(err, BuildError::BasicBlockBranchAlreadyInPhiInstruction),
            "a repeated predecessor is not an entry-block error, got: {err}"
        );

        assert_eq!(phi.branches.len(), 2, "the refused branch was not recorded");
        assert_eq!(phi.blocks.len(), 2);
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

        cursor.add_instruction(Instruction::new_unconditional_br(entry), &mut ctx_b);
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
