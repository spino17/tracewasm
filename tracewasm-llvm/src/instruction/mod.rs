use crate::{
    cfg::{basic_block::BasicBlockId, context::Context},
    error::PhiError,
    interner::TyId,
    value::{I1Value, Value},
};
use rustc_hash::FxHashSet;

pub mod cursor;

pub struct PhiInstruction {
    pub(crate) branches: Vec<(BasicBlockId, Value)>,
    pub(crate) blocks: FxHashSet<BasicBlockId>,
    ref_ty: TyId,
    value: Value,
}

#[derive(Debug, Clone, Copy)]
pub struct PhiInstrHandler {
    pub(crate) index: usize,
    pub(crate) block: BasicBlockId,
}

impl PhiInstrHandler {
    pub fn basic_block(&self) -> BasicBlockId {
        self.block
    }

    pub fn add_branch(
        &self,
        branch: (BasicBlockId, Value),
        ctx: &mut Context,
    ) -> Result<(), PhiError> {
        let index = self.index;

        // Read the phi's type through a shared borrow first: rendering a mismatch
        // needs the type pool, which cannot be reached while the block is borrowed
        // mutably. An id is `Copy`, so nothing is held across the switch.
        let ref_ty = ctx.get_block(self.block).phis[index].ref_ty;
        let branch_ty = branch.1.ty();

        if branch_ty != ref_ty {
            return Err(PhiError::PhiInstructionBranchTypeMismatch(
                ctx.display(ref_ty).to_string(),
                ctx.display(branch_ty).to_string(),
            ));
        }

        let block_id = branch.0;
        let instr = &mut ctx.get_block_mut(self.block).phis[index];

        if instr.blocks.contains(&block_id) {
            return Err(PhiError::BasicBlockBranchAlreadyInPhiInstruction);
        }

        instr.blocks.insert(block_id);
        instr.branches.push(branch);

        Ok(())
    }
}

pub enum InstructionKind {
    UnconditionalBr(UnconditionalBrOperands),
    ConditionalBr(ConditionalBrOperands),
    Load(LoadOperands),
    Store(StoreOperands),
    Alloca(AllocaOperands),
    GetElementPtr(GetElementPtrOperands),
}

pub struct Instruction {
    pub(crate) kind: InstructionKind,
    /// The register this instruction defines, for the `%x =` an emitter writes in
    /// front of it. `None` for the instructions that produce no value.
    pub(crate) value: Option<Value>,
}

pub struct UnconditionalBrOperands {
    pub label: BasicBlockId,
}

pub struct ConditionalBrOperands {
    pub cond: I1Value,
    pub true_label: BasicBlockId,
    pub false_label: BasicBlockId,
}

pub struct LoadOperands {
    pub ty: TyId,
    pub ptr: Value,
    pub align: Option<u32>,
}

pub struct StoreOperands {
    pub value: Value,
    pub ptr: Value,
    pub align: Option<u32>,
}

pub struct AllocaOperands {
    pub ty: TyId,
    pub count: Option<Value>,
    pub align: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct GetElementPtrOperands {
    pub source_ty: TyId,
    pub ptr: Value,
    pub indices: Box<[Value]>,
    pub inbounds: bool,
}

impl GetElementPtrOperands {
    /// What the pointer this `getelementptr` produces points at.
    ///
    /// The first index steps over the source type as pointer arithmetic rather than
    /// descending into it, so with one index or none the pointee is the source type
    /// unchanged; only `indices[1..]` walk inwards.
    ///
    /// `None` when the walk does not typecheck, which a `getelementptr` built through
    /// [`Cursor::add_get_element_ptr`] cannot be — it is validated there.
    pub(crate) fn result_pointee_ty(&self, ctx: &Context) -> Option<TyId> {
        if self.indices.len() <= 1 {
            return Some(self.source_ty);
        }

        self.source_ty
            .walk_pointee_ty_in_gep(&self.indices[1..], ctx)
            .ok()
    }
}
