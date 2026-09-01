use crate::{
    cfg::{
        basic_block::{BasicBlock, BasicBlockId},
        context::Context,
    },
    error::ContextError,
    interner::{StrId, TyId},
    value::Value,
};
use id_arena::Id;
use rustc_hash::FxHashSet;

pub struct Function {
    pub(crate) name: StrId,
    pub(crate) params: Vec<Value>,
    pub(crate) result: TyId,
    pub(crate) blocks: Vec<BasicBlockId>,
    pub(crate) block_names: FxHashSet<StrId>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct FuncId(Id<Function>);

impl Clone for FuncId {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for FuncId {}

impl FuncId {
    pub(crate) fn new(id: Id<Function>) -> Self {
        FuncId(id)
    }

    pub(crate) fn raw(&self) -> Id<Function> {
        self.0
    }

    pub fn add_basic_block(
        &self,
        name: String,
        ctx: &mut Context,
    ) -> Result<BasicBlockId, ContextError> {
        let name_id: StrId = ctx.str_interner.intern(name).into();
        let func = ctx.get_func_mut(*self);

        if func.block_names.contains(&name_id) {
            return Err(ContextError::DuplicateBasicBlockName(
                ctx.str_interner.value(name_id.0).to_string(),
            ));
        }

        let is_first = func.blocks.is_empty();

        let id = BasicBlockId::new(ctx.blocks.alloc(BasicBlock {
            name: name_id,
            is_first,
            func_id: *self,
            phis: vec![],
            instructions: vec![],
            is_locked: false,
        }));

        let func = ctx.get_func_mut(*self);

        func.blocks.push(id);
        func.block_names.insert(name_id);

        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        error::ContextError,
        test_support::{add_fn, fixture},
    };

    /// `is_first` marks the entry block, which is the one a phi cannot go in. Only
    /// the first block added to a function is it.
    #[test]
    fn only_the_first_block_of_a_function_is_the_entry() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();
        let body = f.add_basic_block("body".to_string(), &mut ctx).unwrap();

        assert!(ctx.blocks.get(entry.raw()).unwrap().is_first);
        assert!(!ctx.blocks.get(body.raw()).unwrap().is_first);

        // A second function's first block is an entry too — `is_first` is per
        // function, not per module.
        let g = add_fn("g", &mut builder, &mut ctx).unwrap();
        let g_entry = g.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert!(ctx.blocks.get(g_entry.raw()).unwrap().is_first);
    }

    /// A block records which function it belongs to, and the function records the
    /// block — the two have to agree or a later walk of the graph goes wrong.
    #[test]
    fn a_block_and_its_function_agree() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let entry = f.add_basic_block("entry".to_string(), &mut ctx).unwrap();

        assert_eq!(ctx.blocks.get(entry.raw()).unwrap().func_id.raw(), f.raw());
        assert_eq!(ctx.funcs.get(f.raw()).unwrap().blocks, vec![entry]);
    }

    /// Two blocks sharing a label would make a branch to it ambiguous.
    #[test]
    fn a_duplicate_block_name_in_one_function_is_refused() {
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();

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
        let (mut ctx, mut builder) = fixture();
        let f = add_fn("f", &mut builder, &mut ctx).unwrap();
        let g = add_fn("g", &mut builder, &mut ctx).unwrap();
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
