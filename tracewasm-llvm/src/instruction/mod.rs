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
    /// Integer arithmetic, bitwise logic or a shift, producing a value of the
    /// operand type.
    IArithmetic(IArithmeticOperands),
    /// Compares two floating-point values, producing an `i1`.
    ///
    /// Separate from [`ICmp`](Self::ICmp) because LLVM keeps them separate: `icmp`
    /// refuses floats and `fcmp` refuses integers.
    FCmp(FCmpOperands),
    /// Floating-point arithmetic, producing a value of the operand type.
    FArithmetic(FArithmeticOperands),
    /// Negates a floating-point value.
    ///
    /// Its own kind rather than an [`FArithmetic`](Self::FArithmetic) op, because
    /// `fneg` is the one arithmetic instruction that takes a **single** operand —
    /// `llvm-as` refuses `fneg double %a, %b`. Integers have no counterpart; negating
    /// one is `sub 0, %x`.
    FNeg(FNegOperands),
    /// Converts a value from one type to another.
    ///
    /// Not a computation on the value's *number* so much as on how those bits are
    /// read — except where LLVM says otherwise, as `sitofp` and `fptosi` genuinely
    /// recompute.
    Cast(CastOperands),
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

/// The operands of an integer arithmetic, bitwise or shift instruction.
///
/// Emitted as `<op> <ty> <a>, <b>` — the type written once, since LLVM requires both
/// operands to have it. The result has that same type, unlike a comparison's `i1`.
pub struct IArithmeticOperands {
    /// Which operation.
    pub op: IArithmeticOp,
    /// The type both operands have, and the type of the result.
    pub ty: TyId,
    /// The left operand.
    pub a: Value,
    /// The right operand. For a shift, the shift *amount*.
    pub b: Value,
}

/// Which integer operation an [`IArithmeticOperands`] performs.
///
/// Only six of these carry a signedness, and LLVM is the authority on which: it spells
/// `sdiv`/`udiv`, `srem`/`urem` and `ashr`/`lshr` as distinct opcodes, and has exactly
/// **one** `add`, `and`, `shl` and so on. There is no `sadd` — two's-complement
/// addition produces the same bits under either reading, which is why one opcode
/// suffices. See [`signedness`](Self::signedness).
///
/// [`Display`] writes the LLVM keyword, which is what the emitter uses.
#[derive(Clone, Copy)]
pub enum IArithmeticOp {
    /// Addition. No signedness — see the type-level note.
    Add,
    /// Subtraction. No signedness.
    Sub,
    /// Multiplication. No signedness.
    Mul,
    /// Unsigned division.
    Udiv,
    /// Signed division.
    Sdiv,
    /// Unsigned remainder.
    Urem,
    /// Signed remainder.
    Srem,
    /// Shift left. No signedness: vacated bits are always zero.
    Shl,
    /// Logical shift right — vacated bits are zero, so the value is read unsigned.
    Lshr,
    /// Arithmetic shift right — vacated bits copy the sign, so the value is read
    /// signed.
    Ashr,
    /// Bitwise and. No signedness.
    And,
    /// Bitwise or. No signedness.
    Or,
    /// Bitwise exclusive or. No signedness.
    Xor,
}

impl Display for IArithmeticOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            IArithmeticOp::Add => "add",
            IArithmeticOp::Sub => "sub",
            IArithmeticOp::Mul => "mul",
            IArithmeticOp::Udiv => "udiv",
            IArithmeticOp::Sdiv => "sdiv",
            IArithmeticOp::Urem => "urem",
            IArithmeticOp::Srem => "srem",
            IArithmeticOp::Shl => "shl",
            IArithmeticOp::Lshr => "lshr",
            IArithmeticOp::Ashr => "ashr",
            IArithmeticOp::And => "and",
            IArithmeticOp::Or => "or",
            IArithmeticOp::Xor => "xor",
        })
    }
}

impl IArithmeticOp {
    /// How to read the operands, or `None` if the operation does not say.
    ///
    /// `None` for `add`, `sub`, `mul`, `shl`, `and`, `or` and `xor` — LLVM has a
    /// single opcode for each, because the *result* bits are the same either way.
    /// That absence is load-bearing: it is what makes
    /// [`build_iarithmetic`](crate::instruction::cursor::Cursor::build_iarithmetic)
    /// refuse to widen a narrower constant rather than guess. The result may not
    /// depend on the reading, but the widening does — `add i64 100, -1` is 99, while
    /// the same `i32` constant zero-extended gives 4294967395.
    ///
    /// The six that do carry one are the pairs LLVM spells separately.
    pub fn signedness(&self) -> Option<Signedness> {
        let v = match self {
            IArithmeticOp::Add => return None,
            IArithmeticOp::Sub => return None,
            IArithmeticOp::Mul => return None,
            IArithmeticOp::And => return None,
            IArithmeticOp::Or => return None,
            IArithmeticOp::Xor => return None,
            IArithmeticOp::Shl => return None,
            IArithmeticOp::Udiv => Signedness::Unsigned,
            IArithmeticOp::Sdiv => Signedness::Signed,
            IArithmeticOp::Urem => Signedness::Unsigned,
            IArithmeticOp::Srem => Signedness::Signed,
            IArithmeticOp::Lshr => Signedness::Unsigned,
            IArithmeticOp::Ashr => Signedness::Signed,
        };

        Some(v)
    }
}

/// The operands of a floating-point arithmetic instruction.
///
/// Emitted as `<op> <ty> <a>, <b>`. Every variant here is binary; `fneg` is unary and
/// lives in [`FNegOperands`] instead.
pub struct FArithmeticOperands {
    /// Which operation.
    pub op: FArithmeticOp,
    /// The type both operands have, and the type of the result.
    pub ty: TyId,
    /// The left operand.
    pub a: Value,
    /// The right operand.
    pub b: Value,
}

/// Which floating-point operation an [`FArithmeticOperands`] performs.
///
/// None of these carry a signedness — a float's sign is part of its format, so there
/// is no signed/unsigned split to make.
///
/// [`Display`] writes the LLVM keyword, which is what the emitter uses.
#[derive(Clone, Copy)]
pub enum FArithmeticOp {
    /// Addition.
    FAdd,
    /// Subtraction.
    FSub,
    /// Multiplication.
    FMul,
    /// Division.
    FDiv,
    /// Remainder.
    FRem,
}

impl Display for FArithmeticOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FArithmeticOp::FAdd => "fadd",
            FArithmeticOp::FSub => "fsub",
            FArithmeticOp::FMul => "fmul",
            FArithmeticOp::FDiv => "fdiv",
            FArithmeticOp::FRem => "frem",
        })
    }
}

/// The operand of an `fneg`.
///
/// Emitted as `fneg <ty> <value>`. One operand, not two: `llvm-as` refuses
/// `fneg double %a, %b` with "expected metadata after comma". That is why `fneg` is
/// not an [`FArithmeticOp`] — giving it a second operand would make an
/// unrepresentable instruction constructible.
pub struct FNegOperands {
    /// The type of the operand, and of the result.
    pub ty: TyId,
    /// The value to negate.
    pub value: Value,
}

/// The operands of an `fcmp`.
///
/// Emitted as `fcmp <cond> <ty> <a>, <b>` — the same shape as [`ICmpOperands`], with
/// the type written once because LLVM requires both operands to have it.
pub struct FCmpOperands {
    /// Which comparison, and how it treats a NaN operand.
    pub cond: FCond,
    /// The type both operands have.
    ///
    /// A floating-point type: `half`, `bfloat`, `float` or `double`. Integers are
    /// compared with `icmp` instead.
    pub ty: TyId,
    /// The left operand.
    pub a: Value,
    /// The right operand.
    pub b: Value,
}

/// Which comparison an [`FCmpOperands`] performs.
///
/// Sixteen predicates, in three groups, and the distinction is entirely about **NaN**:
///
/// - **Ordered** (`o…`) — false whenever either operand is a NaN, *and* the comparison
///   must also hold. `Ord` alone asks only "is neither a NaN".
/// - **Unordered** (`u…`) — true whenever either operand is a NaN, *or* the comparison
///   holds. `Uno` alone asks only "is either a NaN".
/// - **Constant** ([`True`](Self::True), [`False`](Self::False)) — neither group.
///   These ignore their operands entirely, NaN included, and still take two of them:
///   `llvm-as` refuses `fcmp true` written without operands.
///
/// So the two constants are *not* the complement of "ordered". Measured against a NaN
/// operand, `True` yields 1 like an unordered predicate and `False` yields 0 like an
/// ordered one.
///
/// [`Display`] writes the LLVM keyword, which is what the emitter uses.
#[derive(Clone, Copy)]
pub enum FCond {
    /// Ordered and equal.
    Oeq,
    /// Ordered and greater than.
    Ogt,
    /// Ordered and greater than or equal.
    Oge,
    /// Ordered and less than.
    Olt,
    /// Ordered and less than or equal.
    Ole,
    /// Ordered and not equal.
    One,
    /// Ordered: neither operand is a NaN.
    Ord,
    /// Unordered or equal.
    Ueq,
    /// Unordered or greater than.
    Ugt,
    /// Unordered or greater than or equal.
    Uge,
    /// Unordered or less than.
    Ult,
    /// Unordered or less than or equal.
    Ule,
    /// Unordered or not equal.
    Une,
    /// Unordered: either operand is a NaN.
    Uno,
    /// Always true, whatever the operands — NaN included.
    True,
    /// Always false, whatever the operands — NaN included.
    False,
}

impl Display for FCond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FCond::Oeq => "oeq",
            FCond::Ogt => "ogt",
            FCond::Oge => "oge",
            FCond::Olt => "olt",
            FCond::Ole => "ole",
            FCond::One => "one",
            FCond::Ord => "ord",
            FCond::Ueq => "ueq",
            FCond::Ugt => "ugt",
            FCond::Uge => "uge",
            FCond::Ult => "ult",
            FCond::Ule => "ule",
            FCond::Une => "une",
            FCond::Uno => "uno",
            FCond::True => "true",
            FCond::False => "false",
        })
    }
}

/// The operands of a conversion.
///
/// Emitted as `<op> <src_ty> <value> to <dest_ty>` — the source type is written out
/// even though the operand carries it, because that is the syntax LLVM reads.
pub struct CastOperands {
    /// Which conversion.
    pub op: CastOp,
    /// The type being converted from. Always the operand's own type: it is read off
    /// the value rather than supplied, so the two cannot disagree.
    pub src_ty: TyId,
    /// The value being converted.
    pub value: Value,
    /// The type being converted to, and the type of the result.
    pub dest_ty: TyId,
}

/// Which conversion a [`CastOperands`] performs.
///
/// LLVM has one opcode per case rather than a general "convert", and each covers a
/// narrow pairing — see [`is_cast_allowed`](Self::is_cast_allowed). Picking the wrong
/// one is refused with "invalid cast opcode" rather than silently reinterpreted.
///
/// [`Display`] writes the LLVM keyword, which is what the emitter uses.
#[derive(Clone, Copy)]
pub enum CastOp {
    /// Drop the high bits of a narrower integer. The one integer narrowing — there is
    /// no signed variant, because dropping bits needs no interpretation.
    Trunc,
    /// Widen an integer, filling the new high bits with zeros.
    Zext,
    /// Widen an integer, copying the sign bit into the new high bits.
    Sext,
    /// Narrow a float, rounding to the nearest value the smaller format holds.
    Fptrunc,
    /// Widen a float. Exact — every value of the smaller format is one of the larger.
    Fpext,
    /// Float to unsigned integer, truncating toward zero.
    Fptoui,
    /// Float to signed integer, truncating toward zero.
    Fptosi,
    /// Unsigned integer to float.
    Uitofp,
    /// Signed integer to float.
    Sitofp,
    /// A pointer to an integer holding its address.
    Ptrtoint,
    /// An integer to the pointer at that address.
    Inttoptr,
    /// Reread the same bits as another type of the same width — `i32` as `float`, say.
    /// The only conversion that changes nothing about the value.
    Bitcast,
}

impl Display for CastOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            CastOp::Trunc => "trunc",
            CastOp::Zext => "zext",
            CastOp::Sext => "sext",
            CastOp::Fptrunc => "fptrunc",
            CastOp::Fpext => "fpext",
            CastOp::Fptoui => "fptoui",
            CastOp::Fptosi => "fptosi",
            CastOp::Uitofp => "uitofp",
            CastOp::Sitofp => "sitofp",
            CastOp::Ptrtoint => "ptrtoint",
            CastOp::Inttoptr => "inttoptr",
            CastOp::Bitcast => "bitcast",
        })
    }
}

impl CastOp {
    /// Whether this opcode connects those two types.
    ///
    /// Each answer matches `llvm-as`, which refuses everything else with "invalid cast
    /// opcode" rather than reinterpreting:
    ///
    /// - `trunc` / `zext` / `sext` — integers whose widths **differ** in the right
    ///   direction. Equal widths are refused: `trunc i32 to i32` is not a no-op, it is
    ///   an error.
    /// - `fptrunc` / `fpext` — the same, for floats. So `fpext half to bfloat` is
    ///   refused even though both are floats: at 16 bits each, neither is wider.
    /// - `fptoui` / `fptosi` / `uitofp` / `sitofp` — across the two families, in the
    ///   direction the name gives.
    /// - `ptrtoint` / `inttoptr` — the only ways to reach a `ptr`.
    /// - `bitcast` — two sized types of **equal** width, so `i32` to `float` is fine.
    ///   A `ptr` cannot appear at either end: it has no width here, and under opaque
    ///   pointers a pointer `bitcast` would name a conversion that no longer exists.
    ///   Aggregates are excluded for the same reason — no width.
    pub fn is_cast_allowed(&self, src: TyId, dest: TyId, ctx: &Context) -> bool {
        match self {
            CastOp::Trunc => {
                src.is_integer(ctx)
                    && dest.is_integer(ctx)
                    && (src.width(ctx).unwrap() > dest.width(ctx).unwrap())
            }
            CastOp::Zext | CastOp::Sext => {
                src.is_integer(ctx)
                    && dest.is_integer(ctx)
                    && (src.width(ctx).unwrap() < dest.width(ctx).unwrap())
            }
            CastOp::Fptrunc => {
                src.is_float(ctx)
                    && dest.is_float(ctx)
                    && (src.width(ctx).unwrap() > dest.width(ctx).unwrap())
            }
            CastOp::Fpext => {
                src.is_float(ctx)
                    && dest.is_float(ctx)
                    && (src.width(ctx).unwrap() < dest.width(ctx).unwrap())
            }
            CastOp::Fptoui => src.is_float(ctx) && dest.is_integer(ctx),
            CastOp::Fptosi => src.is_float(ctx) && dest.is_integer(ctx),
            CastOp::Uitofp => src.is_integer(ctx) && dest.is_float(ctx),
            CastOp::Sitofp => src.is_integer(ctx) && dest.is_float(ctx),
            CastOp::Ptrtoint => src.is_ptr(ctx) && dest.is_integer(ctx),
            CastOp::Inttoptr => src.is_integer(ctx) && dest.is_ptr(ctx),
            CastOp::Bitcast => {
                if let Some(src_width) = src.width(ctx)
                    && let Some(dest_width) = dest.width(ctx)
                    && src_width == dest_width
                {
                    true
                } else {
                    false
                }
            }
        }
    }
}
