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
    value::{I1Value, Signedness, Value},
};
use rustc_hash::FxHashSet;
use std::fmt::Display;

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
    /// Compares two integers or pointers, producing an `i1`.
    ///
    /// Not a terminator, and not a branch: the `i1` is an ordinary value that a
    /// conditional branch may later consume, or that may be stored like any other.
    ICmp(ICmpOperands),
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

/// The operands of an `icmp`.
///
/// Emitted as `icmp <cond> <ty> <a>, <b>` — the type is written **once**, and the
/// operands untyped, because LLVM requires both to have it. That requirement is
/// enforced when the instruction is built, so nothing here can disagree.
pub struct ICmpOperands {
    /// Which comparison, and — for the ordered predicates — how to read the operands.
    pub cond: ICond,
    /// The type both operands have.
    ///
    /// An integer or a pointer. `llvm-as` accepts `ptr` with every predicate, signed
    /// ones included, but refuses floats with "icmp requires integer operands".
    pub ty: TyId,
    /// The left operand.
    pub a: Value,
    /// The right operand.
    pub b: Value,
}

/// Which comparison an [`ICmpOperands`] performs.
///
/// Ten predicates, and the signedness lives here rather than in the operand type:
/// LLVM's integer types say nothing about sign, so `ult` and `slt` are what
/// distinguish an unsigned comparison from a signed one over the same bits. See
/// [`signedness`](Self::signedness).
///
/// [`Display`] writes the LLVM keyword, which is what the emitter uses.
#[derive(Clone, Copy)]
pub enum ICond {
    /// Equal. Neither signed nor unsigned — at equal widths the reading cannot
    /// change the answer.
    Eq,
    /// Not equal. Signedness-free for the same reason as [`Eq`](Self::Eq).
    Ne,
    /// Unsigned greater than.
    Ugt,
    /// Unsigned greater than or equal.
    Uge,
    /// Unsigned less than.
    Ult,
    /// Unsigned less than or equal.
    Ule,
    /// Signed greater than.
    Sgt,
    /// Signed greater than or equal.
    Sge,
    /// Signed less than.
    Slt,
    /// Signed less than or equal.
    Sle,
}

impl Display for ICond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ICond::Eq => "eq",
            ICond::Ne => "ne",
            ICond::Ugt => "ugt",
            ICond::Uge => "uge",
            ICond::Ult => "ult",
            ICond::Ule => "ule",
            ICond::Sgt => "sgt",
            ICond::Sge => "sge",
            ICond::Slt => "slt",
            ICond::Sle => "sle",
        })
    }
}

impl ICond {
    /// How to read the operands, or `None` if the predicate does not say.
    ///
    /// The ordered predicates carry it in their name — `ult` compares unsigned, `slt`
    /// signed — which is what decides whether a narrower constant operand is zero- or
    /// sign-extended to the common type.
    ///
    /// `eq` and `ne` return `None`, and that absence is load-bearing rather than a
    /// gap. LLVM has a single `eq`, because at equal widths signedness cannot change
    /// the answer. It only matters when widening, and there the two choices genuinely
    /// disagree: against `i64 4294967295`, a `-1i32` operand is equal after `zext` and
    /// unequal after `sext`. So [`build_icmp`](crate::instruction::cursor::Cursor::build_icmp)
    /// refuses to widen for these rather than pick one.
    pub fn signedness(&self) -> Option<Signedness> {
        let signedness = match self {
            ICond::Eq => return None,
            ICond::Ne => return None,
            ICond::Ugt => Signedness::Unsigned,
            ICond::Uge => Signedness::Unsigned,
            ICond::Ult => Signedness::Unsigned,
            ICond::Ule => Signedness::Unsigned,
            ICond::Sgt => Signedness::Signed,
            ICond::Sge => Signedness::Signed,
            ICond::Slt => Signedness::Signed,
            ICond::Sle => Signedness::Signed,
        };

        Some(signedness)
    }
}

/// The operands of an `fcmp`. Not yet built or emitted.
pub struct FCmpOperands {}
