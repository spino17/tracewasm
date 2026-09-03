//! Instructions and their operands.
//!
//! [`InstructionKind`] is the union of everything this crate can build; each variant
//! carries its own operand struct, so an instruction's fields are named rather than
//! positional. Instructions are written through a
//! [`Cursor`](cursor::Cursor) — see that module for the builders.

use crate::{
    cfg::{basic_block::BasicBlockId, context::Context},
    error::PhiError,
    interner::{StrId, TyId},
    value::{I1Value, Value},
};
use rustc_hash::FxHashSet;

pub mod cursor;

/// A phi node: one incoming value per predecessor block.
///
/// Held separately from a block's other instructions, since LLVM requires phis to
/// come first. Every branch must have the phi's own type, and the same predecessor
/// may appear only once — `blocks` is what makes the second check cheap.
pub struct PhiInstruction {
    /// The incoming edges, as (predecessor, value) pairs.
    pub(crate) branches: Vec<(BasicBlockId, Value)>,
    /// The predecessors already named, so a repeat can be refused.
    pub(crate) blocks: FxHashSet<BasicBlockId>,
    /// The phi's type, taken from its first branch. Every later branch is checked
    /// against it.
    pub(crate) ref_ty: TyId,
    /// The register this phi defines.
    pub(crate) value: Value,
}

/// A handle to a phi already placed in a block, for adding branches to it later.
///
/// A phi is usually built before all its predecessors exist — that is the point of a
/// phi — so branches are added incrementally through this rather than all at once.
#[derive(Debug, Clone, Copy)]
pub struct PhiInstrHandler {
    pub(crate) index: usize,
    pub(crate) block: BasicBlockId,
}

impl PhiInstrHandler {
    /// The block this phi lives in.
    pub fn basic_block(&self) -> BasicBlockId {
        self.block
    }

    /// Adds one incoming edge.
    ///
    /// # Errors
    ///
    /// - [`PhiError::PhiInstructionBranchTypeMismatch`] — the value's type differs
    ///   from the phi's. A phi produces one value, so there is nothing a second type
    ///   could be.
    /// - [`PhiError::BasicBlockBranchAlreadyInPhiInstruction`] — that predecessor is
    ///   already named.
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

/// Every instruction this crate can build.
///
/// Three of these — [`UnconditionalBr`](Self::UnconditionalBr),
/// [`ConditionalBr`](Self::ConditionalBr) and [`Ret`](Self::Ret) — are *terminators*:
/// they end a block, and adding one locks it.
pub enum InstructionKind {
    /// `br label %target`.
    UnconditionalBr(UnconditionalBrOperands),
    /// `br i1 %c, label %t, label %f`.
    ConditionalBr(ConditionalBrOperands),
    /// `%x = load <ty>, ptr %p`.
    Load(LoadOperands),
    /// `store <ty> %v, ptr %p`.
    Store(StoreOperands),
    /// `%x = alloca <ty>` — stack space, yielding a pointer to it.
    Alloca(AllocaOperands),
    /// `%x = getelementptr <ty>, ptr %p, ...` — address arithmetic, computing a
    /// pointer without reading memory.
    GetElementPtr(GetElementPtrOperands),
    /// `ret <ty> %v` or `ret void`.
    Ret(RetOperands),
    /// `%x = call <ty> @f(...)`, or `call void @f(...)` when nothing is returned.
    ///
    /// Not a terminator: control returns to the next instruction, so the block stays
    /// open.
    Call(CallOperands),
    ICmp(ICmpOperadns),
}

/// One instruction: what it does, and the register it defines.
pub struct Instruction {
    pub(crate) kind: InstructionKind,
    /// The register this instruction defines, for the `%x =` an emitter writes in
    /// front of it. `None` for the instructions that produce no value.
    pub(crate) value: Option<Value>,
}

/// Operands of an unconditional branch.
pub struct UnconditionalBrOperands {
    /// The block to jump to.
    pub label: BasicBlockId,
}

/// Operands of a conditional branch.
pub struct ConditionalBrOperands {
    /// The condition. An [`I1Value`] rather than a [`Value`], so the `i1` requirement
    /// is checked once when the value is narrowed rather than here.
    pub cond: I1Value,
    /// Taken when the condition is true.
    pub true_label: BasicBlockId,
    /// Taken when it is false.
    pub false_label: BasicBlockId,
}

/// Operands of a `ret`.
pub struct RetOperands {
    /// The type returned. `void` when nothing is.
    pub ty: TyId,
    /// The value returned, absent for `ret void`.
    pub value: Option<Value>,
}

/// Operands of a `load`.
pub struct LoadOperands {
    /// The type read out of memory. Under opaque pointers this comes from the
    /// instruction, not from the pointer.
    pub ty: TyId,
    /// The address.
    pub ptr: Value,
    /// Explicit alignment. `None` means the ABI default.
    pub align: Option<u32>,
}

/// Operands of a `store`.
pub struct StoreOperands {
    /// The value written.
    pub value: Value,
    /// The address.
    pub ptr: Value,
    /// Explicit alignment. `None` means the ABI default.
    pub align: Option<u32>,
}

/// Operands of an `alloca`.
pub struct AllocaOperands {
    /// The type allocated. The instruction's *result* is a `ptr` to this.
    pub ty: TyId,
    /// Element count, for allocating an array's worth. `None` allocates one.
    pub count: Option<Value>,
    /// Explicit alignment. `None` means the ABI default.
    pub align: Option<u32>,
}

/// Operands of a `getelementptr`.
#[derive(Debug, Clone)]
pub struct GetElementPtrOperands {
    /// The type being indexed into.
    ///
    /// Always resolved: when the caller omits it, the builder infers it from the
    /// pointer and stores what it found, so the instruction records the type it was
    /// actually checked against.
    pub source_ty: TyId,
    /// The base address.
    pub ptr: Value,
    /// The indices. The **first** steps over `source_ty` as pointer arithmetic; only
    /// the rest descend into it.
    pub indices: Box<[Value]>,
    /// Whether to emit `inbounds`, which makes an out-of-range result poison.
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
    /// [`Cursor::build_get_element_ptr`] cannot be — it is validated there.
    pub(crate) fn result_pointee_ty(&self, ctx: &Context) -> Option<TyId> {
        if self.indices.len() <= 1 {
            return Some(self.source_ty);
        }

        self.source_ty
            .walk_pointee_ty_in_gep(&self.indices[1..], ctx)
            .ok()
    }
}

/// Operands of a `call`.
pub struct CallOperands {
    /// The callee, by name. Resolved against the module's function table when the
    /// call is built, so the signature is known to match by the time it is stored.
    pub func_name: StrId,
    /// What the callee returns, `void` included.
    ///
    /// LLVM writes only the return type at a call site — `call i32 @g(i32 7)` — not
    /// the whole function type. The full form is also legal but unnecessary for a
    /// non-variadic callee.
    pub return_ty: TyId,
    /// The arguments, already checked against the callee's parameter types.
    pub params: Vec<Value>,
}

pub struct ICmpOperadns {
    pub cond: ICond,
    pub ty: TyId,
    pub a: Value,
    pub b: Value,
}

#[derive(Clone, Copy)]
pub enum ICond {
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,
}

pub struct FCmpOperands {}
