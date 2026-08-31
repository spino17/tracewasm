//! Lowering of a WebAssembly operator stream into a register machine.
//!
//! WebAssembly is a stack machine, but the stack traffic it implies is mostly
//! bookkeeping: `local.get; local.get; i32.add` describes two pushes and two pops
//! that a register machine does not need. This pass consumes the same operator
//! stream as [`crate::instruction::stack`] and produces
//! [`RegInstruction`]s whose operands name where a value *is* — a frame slot, or a
//! global — so nothing moves unless it has to.
//!
//! The stack pass remains the reference for tracing fidelity; this one is for
//! running the same module quickly.
//!
//! ## The frame: four regions, one index space
//!
//! A [`Slot`] operand carries one absolute frame index and no tag, so an execution
//! reads it as `registers[frame_base + index]` — one add, one load. What the index
//! names depends only on which range of the frame it falls in:
//!
//! ```text
//!   0                                    locals — params first, then declared
//!   locals_count                         consts — the body's interned pool
//!   locals_count + consts                spills
//!   locals_count + consts + spills       operand registers
//!   registers + consts + spills          end of frame
//! ```
//!
//! The sizes travel with the body in [`RegFrameLayout`], which is what lets
//! [`Slot::render`] name a region again and what the runtime uses to place the
//! constant pool at frame entry. Note that [`RegFrameLayout::registers`] counts from
//! the frame base and so *includes* the locals.
//!
//! The order is deliberate. A callee's frame is based at its caller's `caller_base`,
//! which is an operand register index, so putting the constants and spills *below*
//! the operand registers is what stops a callee from overwriting them — see
//! [`runtime::reg`](crate::runtime::reg) for the runtime half.
//!
//! ## Provisional indices, and the pass that resolves them
//!
//! Those region sizes are not known until the body ends: how many constants get
//! interned, and how many spill slots are needed, both depend on operators not yet
//! seen. So lowering works in a **provisional** index space and fixes it up once.
//!
//! While walking the body, an operand register index counts from the frame base but
//! ignores the constant and spill regions — `curr_register_index` starts at
//! `locals_count` and rises from there. A local is already final, since the locals
//! region never moves. But no operand can be given a frame index yet, so an
//! instruction is built with `Slot(u16::MAX)` in every operand position and the real
//! operands are recorded in a [`BackpatchMap`], keyed by the index of the instruction
//! that carries them. Each recorded operand is a [`BackPatchableSlot`]: a constant by
//! pool id, a spill by pool index, an operand register by provisional index, or a
//! local, which is final already.
//!
//! At the end of the body [`BackpatchMap::apply`] walks those entries once and writes
//! the frame index into the instruction itself: a constant lands at
//! `locals_count + id`, a spill at `locals_count + consts + slot`, and a provisional
//! register index — along with every destination start and every `caller_base` —
//! shifts up by `consts + spills`. The register shift carries no `locals_count` term,
//! because a provisional register index already includes it.
//!
//! **Two heights, and they are not the same number.** `stack.height()` counts
//! simulated slots; `curr_register_index` is a provisional frame index, so it rises
//! only for the slots that occupy an operand register. A slot naming a constant, a
//! local, a global or a spill occupies a stack position and no register. Anything
//! restoring the stack to a label's entry state must use the slot height; anything
//! naming a destination register must use the register index. Both are recorded per
//! block — the slot height in `Block::recorded_height`, the register index derived
//! by counting register slots back down to it.
//!
//! ## Lazily forwarded locals
//!
//! A slot naming a local is only valid while that local still holds the value; a
//! later write has to materialize every operand still reading it. That is [`lazy`]'s
//! job — see its module docs for how one shared entry per borrowed value makes
//! multiple simultaneous borrows, and multiple live snapshots of the same local, fall
//! out of the representation.
//!
//! Globals are *not* forwarded. [`RegInstruction::GlobalGet`] reads one into a
//! register eagerly, which is what lets an operand be a bare frame index — a global
//! lives in the instance, not the frame, so it could never be one — and which means
//! no write and no call can invalidate a value already read.
//!
//! ## Operands live in the instruction
//!
//! A [`RegInstruction`] carries its own operands. [`InputRegisters<I>`] is `I`
//! [`Slot`]s laid out in the variant, and [`OutputRegisters<O>`] is the single frame
//! index the `O` destinations start at — one field suffices because an instruction's
//! destinations are always consecutive. Reading either at execution is a field
//! access, not an indirection.
//!
//! That is what keeps the enum at 8 bytes: a `Slot` is two, so a binary operator's
//! whole signature is six. Seven shapes do not fit and carry an [`Id`] into an arena
//! instead — `if`, `br_if`, `br_table`, `call_indirect`, `select`, `memory.init`, and
//! [`RegInstruction::Move`], whose arity its opcode does not fix. Those arenas ship
//! inside [`FrameLayout`], because every id in the instruction list points into one.
//!
//! A memory offset is a `u32`, too wide to sit beside a signature, so a load or a
//! store carries an [`InternedId`] into [`RegFrameLayout::memory_offsets`] —
//! interned, so the `0` that every bare pointer deref loads through costs one entry
//! however many instructions use it.
//!
//! ## Labels and [`RegInstruction::Move`]
//!
//! A block's params and results have to live in the *same* registers no matter which
//! path reaches the label, so control flow can merge without the consumer knowing
//! which way it came. A [`RegInstruction::Move`] materializes them into a contiguous
//! run based at the label's register height.
//!
//! One `Move` is emitted before every *exit* from a block, and there are three:
//!
//! * falling through into the block's `end`;
//! * each `br`/`br_if`/`br_table` arm targeting the label;
//! * falling out of an `if`'s then-branch into its `else`, which jumps to the `end`
//!   and so bypasses the `end`'s own `Move`.
//!
//! Branch targets are the instruction *after* the fallthrough `Move`, so each path
//! performs its own copy exactly once. `br_if` is the exception: its `Move` is a
//! field of the instruction rather than a preceding one, because it must only run on
//! the taken path.
//!
//! Entering a block with params materializes them the same way, so the body starts
//! from a known layout regardless of what the operands were before.
//!
//! ## Branch lowering does not disturb the stack
//!
//! A branch has to know what the *target* label's layout is — how far to unwind and
//! which registers the values land in — but the operators after it still belong to
//! the enclosing block, which is not being left. So
//! `br_truncation_registers` reads the stack through
//! `simulated_pop`, which peeks by depth instead of popping. It
//! computes the destination base by counting register slots down to the target's
//! height rather than by unwinding to it.
//!
//! This matters most for `br_table`, where every arm is computed against the same
//! unchanged stack, and for outward branches, whose target base lies *below* the
//! enclosing block's — unwinding there would discard slots, lazy borrows, and spill
//! slots that the enclosing block still owns.
//!
//! Real state changes still go through `pop`, because that is
//! where lazy reference counts are released and spill slots freed. After an
//! unconditional branch the enclosing block is reset with
//! `pops_and_pushes` so its `end` sees the layout it expects.
//!
//! ## Unreachable operators
//!
//! Everything between an unconditional branch and the enclosing block's `else` or
//! `end` is dead. Rather than lowering it against a fictional stack,
//! `UnreachableTrackingControlStack` filters it out before the match: one check at
//! the top of the loop, and every arm below can assume it is reachable.
//!
//! The tracker keeps its own small stack of blocks opened *while* dead. `else` and
//! `end` clear the dead state only when that stack is empty — meaning the construct
//! they close was opened while reachable. So an `if` opened inside dead code keeps
//! both of its arms dead, and blocks opened in dead code never reach
//! `add_block` at all, which is what keeps the real control stack
//! balanced without a reconciliation step.

use crate::{
    error::{
        CallIndirectError, InstructionExecutionError, MemoryAccessKind, MemoryError, TraceWasmError,
    },
    instance::{Instance, traits::ImportRegistry},
    instruction::{
        Block, BlockKind, CallerBaseData, FrameLayout, Instruction, check_memory_index,
        params_and_results_from_blockty,
        register::{
            arena::{Arena, Id},
            backpatch::{BackPatchableSlot, BackpatchMap, InstructionSource},
            lazy::{
                LazyArena, LazyEntryDropResult, LazyLocation, LazySlot, Local, LocalSlot,
                SpillArena, SpillIndex,
            },
        },
    },
    memory::Memory,
    module::{FuncDecl, FuncIndex, FuncType, GlobalIndex, LocalIndex, TableIndex, TyIndex},
    runtime::{
        I32_TRUNC_HIGH, I32_TRUNC_LOW, I64_TRUNC_HIGH, I64_TRUNC_LOW, Step, U32_TRUNC_HIGH,
        U64_TRUNC_HIGH,
        reg::RegFrame,
        signature_mismatch,
        stack::Stack,
        trunc_float_to_int,
        value::{DataVal, Value},
    },
};
use ordered_float::OrderedFloat;
use smallvec::{SmallVec, smallvec};
use std::{
    collections::hash_map::Entry,
    hash::{Hash, Hasher},
    mem::discriminant,
    vec,
};
use tracewasm_utils::interner::{InternedId, Interner};
// The bitwise and negation arms name these as methods, as the stack machine's do.
use std::ops::{BitAnd, BitOr, BitXor, Neg};
use wasmparser::{BlockType, Operator, OperatorsReader};

pub mod arena;
pub mod backpatch;
pub mod lazy;
pub mod llvm;
pub mod render;

#[cfg(test)]
mod tests;

// What a 16-bit index can name, per region. All three are at the type's ceiling
// because none of them is the real constraint: a frame's four regions laid end to end
// are, and that sum is checked once at the end of the body — three individually legal
// regions can still add up to something no `Slot` could name.
//
// Locals cannot reach these — `MAX_WASM_FUNCTION_LOCALS` is 50,000 — but the two
// interned pools can, and do. A distinct constant or memory offset costs only a few
// bytes of body, so a module that `wasmparser` validates can hold 65,536 of either;
// `too_many_constants_is_reported_not_truncated` and
// `too_many_memory_offsets_is_reported_not_truncated` are exactly that module. Which
// is why both pools are reached through `Interner::try_intern` and not the panicking
// `Interner::intern`: overflowing one is an input this crate has to answer for, and
// the module still lowers under `Stack`.

/// Distinct constants one body may intern.
const MAX_CONSTS: u16 = u16::MAX;
/// Locals plus operand registers one frame may name, counted from the frame base.
const MAX_REGISTER_SLOTS: u16 = u16::MAX;
/// Distinct memory offsets one body's loads and stores may name between them.
pub(crate) const MAX_MEMORY_OFFSETS: u16 = u16::MAX;

/// The static byte offset of one load or store, as its own type so that the ids the
/// pool hands out cannot be confused with another pool's.
///
/// A bare `u32` would work as the interned value, but [`InternedId`] is generic over
/// what was interned — so a second `Interner<u32, _>` for anything else would issue
/// ids of the very same type, and one could be read against the wrong pool. The
/// newtype makes that a compile error.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MemoryOffset(pub u32);

/// Which kind of label a block opens, before its instruction index is known.
///
/// [`SimulatedStack::add_block`] turns this into a [`BlockKind`], filling in the
/// index itself so the caller cannot get it wrong: whether a params [`Move`] precedes
/// the block's own instruction determines that index, and the params count is
/// derived from the block type inside `add_block`.
///
/// [`Move`]: RegInstruction::Move
enum BlockVariant {
    /// `if`: its condition sits above the params, so its entry height is one lower
    /// than the others'.
    If,
    /// `loop`: the only label whose branch target is its start rather than its
    /// `end`.
    Loop,
    /// `block`.
    Block,
    /// The implicit label around the whole body, which `return` targets.
    ///
    /// Never constructed today: [`RegInstruction::emit_instructions_for_func`] pushes
    /// that [`Block`] itself rather than going through [`SimulatedStack::add_block`],
    /// since a function frame has no operator to open it and no block type to read.
    /// The variant exists for the mapping in `add_block` to be total.
    #[allow(dead_code, reason = "kept so `add_block`'s mapping stays total")]
    Func,
}

/// A constant the body uses, interned into a per-body pool.
///
/// An immediate has no origin to be read from, so it is given one: the pool becomes
/// the frame's constant region, and an operand naming constant `i` is just the frame
/// slot `locals_count + i`. Reading one therefore costs exactly what reading a local
/// costs, and no instruction has to carry a value inline.
///
/// Identity is by **variant and bit pattern**, not by numeric equality — see the
/// hand-written [`PartialEq`] below for why that is not merely pedantic.
///
/// The float arms hold `OrderedFloat` only because `f32`/`f64` are not `Ord`; its
/// own `Hash` is not used, since it canonicalises `-0.0` to `+0.0` and every NaN
/// alike — see the hand-written [`Hash`] below.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Const {
    /// A 32-bit integer immediate.
    I32(i32),
    /// A 64-bit integer immediate.
    I64(i64),
    /// A 32-bit float immediate; the bit pattern is preserved exactly.
    F32(OrderedFloat<f32>),
    /// A 64-bit float immediate; the bit pattern is preserved exactly.
    F64(OrderedFloat<f64>),
    /// A function reference, `None` being null.
    ///
    /// `ref.func` and `ref.null` are constants like any other: the operator
    /// describes the whole value, so nothing is emitted for one and it is read out of
    /// the frame's constant region by whatever consumes it.
    ///
    /// The heap type `ref.null` names is deliberately not carried, matching
    /// [`StackInstruction::RefNull`](crate::instruction::stack::StackInstruction::RefNull) in
    /// the stack pass and
    /// [`Val::Ref`](crate::runtime::value::Val::Ref) at execution: a null is a null,
    /// validation has already established that each one reached a slot willing to
    /// hold it, and nothing downstream can distinguish a null `funcref` from a null
    /// `externref`. A *non-null* `externref` would not fit here — none can exist
    /// while references are `Option<FuncIndex>` end to end.
    Ref(Option<FuncIndex>),
}

/// Two constants are the same constant only if they are the same *type* and the same
/// *bits*. Both halves are load-bearing, because this is the interner's dedup key and
/// a merge means two operands sharing one frame slot.
///
/// **Same type.** A slot holds one value, and the runtime materialises it by matching
/// the variant — so merging an `i32` with an `f64` that happens to share a bit
/// pattern would write the wrong kind of value into the slot the other one reads.
///
/// **Same bits, not numerically equal.** Numeric equality would merge `+0.0` with
/// `-0.0`, and every NaN with every other, silently dropping a sign or a payload that
/// `f64.min`, `copysign` and plain arithmetic all propagate. Comparing bits is also
/// what makes the [`Eq`] impl below sound: a NaN constant has to equal itself, or the
/// pool could neither find nor dedup it.
impl PartialEq for Const {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Const::I32(first) => {
                if let Const::I32(second) = other {
                    *first as u32 == *second as u32
                } else {
                    false
                }
            }
            Const::I64(first) => {
                if let Const::I64(second) = other {
                    *first as u64 == *second as u64
                } else {
                    false
                }
            }
            Const::F32(first) => {
                if let Const::F32(second) = other {
                    first.into_inner().to_bits() == second.into_inner().to_bits()
                } else {
                    false
                }
            }
            Const::F64(first) => {
                if let Const::F64(second) = other {
                    first.into_inner().to_bits() == second.into_inner().to_bits()
                } else {
                    false
                }
            }
            Const::Ref(first) => {
                if let Const::Ref(second) = other {
                    first == second
                } else {
                    false
                }
            }
        }
    }
}

impl Eq for Const {}

/// Hashes the same thing [`PartialEq`] compares: the variant, then the bits.
///
/// Written out because the derive would hash the float arms through
/// `OrderedFloat`, which canonicalises — `-0.0` would land in `+0.0`'s bucket and
/// every NaN in one. That is *sound* against a bit-comparing `PartialEq`, since
/// unequal values may share a hash, but it puts values the pool deliberately keeps
/// apart into the same bucket. Hashing the bits keeps them apart there too.
impl Hash for Const {
    fn hash<H: Hasher>(&self, state: &mut H) {
        discriminant(self).hash(state);

        match self {
            Const::I32(v) => v.hash(state),
            Const::I64(v) => v.hash(state),
            Const::F32(v) => v.into_inner().to_bits().hash(state),
            Const::F64(v) => v.into_inner().to_bits().hash(state),
            Const::Ref(v) => v.hash(state),
        }
    }
}

impl Const {
    /// The value as it appears in a rendered instruction.
    ///
    /// Suffixed by width for everything but `i32`, so a rendering says which
    /// immediate it is without the reader inferring it from the operator.
    fn render(&self) -> String {
        match self {
            Const::I32(v) => format!("{v}"),
            Const::I64(v) => format!("{v}i64"),
            Const::F32(v) => format!("{v}f32"),
            Const::F64(v) => format!("{v}f64"),
            Const::Ref(Some(f)) => format!("({})ref", f.0),
            Const::Ref(None) => "(null)ref".to_string(),
        }
    }
}

/// Where an instruction reads one operand from.
///
/// The resolved, executable form: the lowering pass has already decided that this
/// operand needs no copy and can be read in place. One untagged frame index covers
/// every case, because there is only one place to read from — a constant, a local, a
/// spill and an operand register are told apart solely by which of the frame's four
/// regions their index falls in, and nothing at execution has to ask.
///
/// The lowering-time counterpart is [`BackPatchableSlot`], which additionally has to
/// represent operands whose frame index is not known yet.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Slot(u16);

impl Slot {
    /// Whether this operand is an operand register rather than a local.
    ///
    /// **Only meaningful on a provisional index**, i.e. during lowering, before the
    /// end-of-body pass shifts registers above the constant and spill regions. On a
    /// resolved [`Slot`] a constant and a spill also satisfy `>= locals_count`, so
    /// the answer would be wrong; use the region boundaries in [`RegFrameLayout`]
    /// instead, as [`Slot::render`] does.
    #[allow(
        dead_code,
        reason = "the live register-height test is BackPatchableSlot::is_register, \
                  which discriminates by variant; this is the range test, kept for \
                  the provisional indices the lowering works in"
    )]
    fn is_register(&self, locals_count: u16) -> bool {
        self.0 >= locals_count
    }

    /// Names the location this operand reads, by the region its frame index falls in.
    ///
    /// The whole layout is needed because the index alone carries no tag: the four
    /// regions are `locals | consts | spills | registers`, and only their sizes say
    /// which one an index names. Execution never does this — it adds the index to the
    /// frame base and loads — so this exists for rendering and for tests.
    fn render(&self, frame_layout: &RegFrameLayout) -> String {
        let frame_index = self.0;

        // The placeholder the pass emits for a spill or constant before the
        // end-of-body backpatch resolves it. Rendering one means it escaped.
        if frame_index == u16::MAX {
            return "<unresolved>".to_string();
        }

        let locals_count = frame_layout.locals_count;
        let spills_base = locals_count + frame_layout.consts.len() as u16;
        let registers_base = spills_base + frame_layout.spills;

        if frame_index < locals_count {
            format!("local{frame_index}")
        } else if frame_index < spills_base {
            match frame_layout
                .consts
                .get((frame_index - locals_count) as usize)
            {
                Some(c) => c.render(),
                None => format!("<const {} past the pool>", frame_index - locals_count),
            }
        } else if frame_index < registers_base {
            format!("spill{}", frame_index - spills_base)
        } else {
            format!("r{}", frame_index - registers_base)
        }
    }
}

impl Default for Slot {
    /// Frame index 0, which is a local in any frame that has one.
    ///
    /// Not the placeholder an unresolved operand carries: that is `Slot(u16::MAX)`,
    /// which [`Slot::render`] names `<unresolved>` and which the runtime cannot
    /// mistake for a location. A default reads as frame slot 0, so using it for an
    /// operand still to be filled in would hide the omission rather than show it.
    fn default() -> Self {
        Slot(0)
    }
}

/// One entry on the simulated operand stack during lowering.
///
/// The internal counterpart of [`Slot`]. The difference is the lazy cases: where a
/// `Slot` has already resolved a local to either `Local(n)` or `Spilled(s)`, a
/// `StackSlot` holds the shared [`LocalSlot`] handle, so a spill that happens
/// *after* this slot was pushed is still observed when it is finally read. Resolving
/// to a `Slot` therefore happens at pop or peek time, never at push time.
#[derive(Clone, Copy)]
enum StackSlot {
    /// An immediate. Occupies a stack position and no register.
    Const(InternedId<Const, u16>),
    /// A value an earlier instruction produced, in the register named here. The
    /// only variant [`SimulatedStack::curr_register_index`] counts.
    Register(u16),
    /// A borrow of a local, resolving to the local itself or to the spill slot it
    /// was rescued into, whichever holds the value when this is finally read.
    Local(LocalSlot),
}

/// The `I` operands an instruction reads, in wasm push order — `registers[0]` is the
/// deepest, the one pushed first.
///
/// Two bytes apiece and no indirection, which is what lets a signature fit inside an
/// eight-byte instruction. Every entry is `Slot(u16::MAX)` until the end-of-body pass
/// resolves it.
#[derive(Debug)]
pub(crate) struct InputRegisters<const I: usize> {
    /// The operands, deepest first.
    registers: [Slot; I],
}

/// The `O` registers an instruction writes, named by the first of them.
///
/// One field is enough because an instruction's destinations are always consecutive:
/// they are allocated one after another as the results are pushed, so the run is
/// `start .. start + O`. `O` is a compile-time witness only — nothing reads it at
/// execution, which is why an `O` of 0 still costs the two bytes and why a store
/// carries [`InputRegisters`] alone instead.
///
/// `start` is provisional until the end-of-body pass shifts it clear of the constant
/// and spill regions.
#[derive(Debug)]
pub(crate) struct OutputRegisters<const O: usize> {
    /// Frame index of the first destination; the run continues for `O`.
    start: u16,
}

/// The operands of an instruction whose arity is fixed by its opcode.
///
/// **Inputs may alias outputs.** Destinations are allocated after the inputs are
/// consumed, so a result frequently reuses the register one of its operands occupied.
/// An executor must read every operand before writing any destination.
#[derive(Debug)]
pub(crate) struct Signature<const I: usize, const O: usize> {
    /// Where the operands are read from.
    pub input: InputRegisters<I>,
    /// Where the results are written to.
    pub output: OutputRegisters<O>,
}

/// The operands of a [`RegInstruction::Move`], whose arity is a label's and so is
/// only known at lowering time.
///
/// One length covers both halves because a move is 1:1 — it writes exactly as many
/// destinations as it reads sources, at `output_start ..  output_start + input.len()`.
/// Storing a second length would admit a mismatched pair that cannot occur.
///
/// Too wide to sit in an instruction, so it lives in
/// [`RegFrameLayout::dyn_signatures`] and the instruction carries an [`Id`].
#[derive(Debug)]
pub(crate) struct DynSignature {
    /// The sources, deepest first. Its length is the move's arity.
    input: Vec<Slot>,
    /// Frame index of the first destination; the run continues for `input.len()`.
    output_start: u16,
}

impl DynSignature {
    /// A move of `input` into the run based at `output_start`.
    pub fn new(input: Vec<Slot>, output_start: u16) -> Self {
        DynSignature {
            input,
            output_start,
        }
    }

    /// Whether this move transfers nothing — a label with no params or results,
    /// which is the common case.
    ///
    /// Load-bearing in two places: lowering skips emitting the [`Move`] entirely
    /// rather than emitting an empty one, and because it therefore records no
    /// operands, [`BackpatchMap::apply`] has to ask before reaching for a patch.
    ///
    /// [`Move`]: RegInstruction::Move
    pub fn is_empty(&self) -> bool {
        self.input.is_empty()
    }
}

/// What [`SimulatedStack::pops_and_pushes_registers`] hands back for the instruction
/// being built: an operand run to store in it, and the frame index its destinations
/// start at.
///
/// The run is all placeholders. The real operands go into the backpatch map under
/// this instruction's index, so the caller's job is to put the run in the variant and
/// let the end-of-body pass fill it in.
///
/// `#[must_use]` because dropping it leaves the map holding operands for an
/// instruction that never carried the run they belong in, and resolution would then
/// write them into whatever else sits at that index. A caller that wants only the
/// stack and register bookkeeping wants [`SimulatedStack::pops_and_pushes`], which
/// records nothing.
#[must_use]
struct PopsPushesResult {
    /// One `Slot(u16::MAX)` per operand popped, to store in the instruction. The
    /// operands themselves are in the backpatch map.
    input: Vec<Slot>,
    /// Provisional frame index of the first destination allocated.
    output_start: u16,
}

/// The open labels, innermost last. Index 0 is the implicit function frame.
///
/// A `br relative_depth` resolves to `stack[len - 1 - relative_depth]`.
#[derive(Default)]
struct ControlStack {
    /// Open labels, outermost first.
    stack: Vec<Block>,
}

impl ControlStack {
    /// How many labels are open, the function frame included. A `br` resolves its
    /// relative depth against this.
    fn len(&self) -> usize {
        self.stack.len()
    }
}

/// Filters out operators that cannot execute, so the lowering match never sees them.
///
/// After an unconditional branch, everything up to the enclosing block's `else` or
/// `end` is dead. Dead code is stack-polymorphic — it may pop more than it pushed —
/// so lowering it against the simulated stack is both meaningless and prone to
/// underflow. Skipping it entirely means every arm of the match can assume its
/// operands exist.
///
/// The only state needed is a flag plus a count of constructs opened *while* dead:
///
/// * `block`/`loop`/`if` while dead push onto [`Self::blocks`] and are skipped, so
///   they never reach [`SimulatedStack::add_block`] and the real [`ControlStack`]
///   never sees them.
/// * `end` while dead pops one and stays dead, unless [`Self::blocks`] is empty — in
///   which case it closes a construct that was opened while *reachable*, so it is the
///   point where liveness resumes and the operator is processed normally.
/// * `else` behaves the same way, which is what makes an `if` opened inside dead code
///   keep both arms dead: its `else` finds a non-empty [`Self::blocks`] and does not
///   resurrect anything.
///
/// Because dead constructs are only ever counted here, the real control stack stays
/// balanced with no reconciliation: every block it holds was pushed while reachable
/// and is popped by an `end` that is also processed.
struct UnreachableTrackingControlStack {
    /// Constructs opened while dead, innermost last. Only the kind is kept; nothing
    /// about a dead block is needed beyond knowing when it closes.
    blocks: Vec<BlockVariant>,
    /// Whether the operators being read cannot execute, i.e. an unconditional
    /// transfer has been lowered and the enclosing block has not closed yet.
    unreachable: bool,
}

/// What the lowering loop should do with an operator.
enum UnreachableCheckResult {
    /// Skip it — it cannot execute.
    Continue,
    /// Lower it normally.
    Reachable,
}

impl UnreachableTrackingControlStack {
    /// A tracker for a body that starts out reachable, as every body does.
    fn new() -> Self {
        UnreachableTrackingControlStack {
            blocks: vec![],
            unreachable: false,
        }
    }

    /// Marks the rest of the enclosing block dead. Called after every unconditional
    /// transfer — `br`, `br_table`, `return`, `unreachable` — each of which pairs it
    /// with [`SimulatedStack::reset_enclosing_block_layout`]. Not after `br_if`,
    /// whose fall-through is reachable.
    fn set_unreachable(&mut self) {
        self.unreachable = true;
    }

    /// Resumes lowering. Only correct when [`Self::blocks`] is empty, i.e. the
    /// construct being closed was opened while reachable.
    fn unset_unreachable(&mut self) {
        self.unreachable = false;
    }

    /// Records a construct opened while dead, so its `else`/`end` is recognised as
    /// closing it rather than as closing the construct that died.
    fn add_block(&mut self, block: BlockVariant) {
        self.blocks.push(block);
    }

    /// Closes the innermost construct opened while dead.
    ///
    /// Panics if none is open, which would mean [`Self::check_unreachablity`] let an
    /// `end` through the wrong arm — the real [`ControlStack`] would already be
    /// unbalanced by then.
    fn pop_block(&mut self) -> BlockVariant {
        self.blocks.pop().unwrap()
    }

    /// Classifies one operator, updating the dead-code state as a side effect.
    ///
    /// Called for every operator before the lowering match; returns
    /// [`UnreachableCheckResult::Reachable`] immediately when nothing is dead.
    fn check_unreachablity(&mut self, operator: &Operator<'_>) -> UnreachableCheckResult {
        if !self.unreachable {
            return UnreachableCheckResult::Reachable;
        }

        if let Some(block) = Self::is_block(operator) {
            self.add_block(block);

            UnreachableCheckResult::Continue
        } else if Self::is_else(operator) {
            if self.is_empty() {
                self.unset_unreachable();

                UnreachableCheckResult::Reachable
            } else {
                debug_assert!(matches!(self.blocks.last().unwrap(), BlockVariant::If));

                UnreachableCheckResult::Continue
            }
        } else if Self::is_end(operator) {
            if self.is_empty() {
                self.unset_unreachable();

                UnreachableCheckResult::Reachable
            } else {
                self.pop_block();

                UnreachableCheckResult::Continue
            }
        } else {
            UnreachableCheckResult::Continue
        }
    }

    /// The kind of label this operator opens, or `None` if it opens none.
    ///
    /// [`BlockVariant::Func`] is not among the answers: the function frame is opened
    /// by the pass itself, never by an operator.
    fn is_block(operator: &Operator<'_>) -> Option<BlockVariant> {
        match operator {
            Operator::Block { .. } => Some(BlockVariant::Block),
            Operator::If { .. } => Some(BlockVariant::If),
            Operator::Loop { .. } => Some(BlockVariant::Loop),
            _ => None,
        }
    }

    /// Whether this operator opens the second arm of an `if`.
    fn is_else(operator: &Operator<'_>) -> bool {
        matches!(operator, Operator::Else)
    }

    /// Whether this operator closes a label.
    fn is_end(operator: &Operator<'_>) -> bool {
        matches!(operator, Operator::End)
    }

    /// Whether every construct opened while dead has since closed — and so whether
    /// the next `else`/`end` closes the construct that *died*, which is what makes
    /// the code after it live again.
    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// The arena entry behind [`RegInstruction::If`]: a condition and both of the jump
/// targets an `if` needs.
#[derive(Debug)]
pub(crate) struct IfOperands {
    /// Read when the instruction executes; a zero takes the else path.
    cond: InputRegisters<1>,
    /// Index of the matching [`RegInstruction::Else`], backpatched at `end`.
    else_index: Option<u32>,
    /// Index of the matching `end`, backpatched.
    end_index: u32,
}

/// The arena entry behind [`RegInstruction::BrIf`].
#[derive(Debug)]
pub(crate) struct BrIfOperands {
    /// Read before the move is performed, since the move's destinations may include
    /// the register the condition is in.
    cond: InputRegisters<1>,
    /// Performed only on the taken path, which is why it is a field here rather than
    /// a preceding [`RegInstruction::Move`]. Empty when the target label carries
    /// nothing.
    mov: DynSignature,
    /// Absolute jump target, backpatched when the label closes.
    target_index: u32,
}

/// One resolved arm of a [`RegInstruction::BrTable`].
///
/// Each arm carries its own move and jump target because a single `br_table` may mix
/// loop and non-loop labels: validation only requires the label *types* to match, so
/// the arities agree but the unwind heights — and therefore the destination registers
/// — differ per arm.
#[derive(Debug)]
pub(crate) struct RegBrTableTarget {
    /// Values transferred to this arm's label, on the same terms as
    /// [`RegInstruction::Move`]. Empty when the label carries nothing.
    pub mov: DynSignature,
    /// Absolute jump target: a loop's start for a back-edge, otherwise the label's
    /// `end`, backpatched when that `end` is reached.
    pub target_index: u32,
}

/// The arena entry behind [`RegInstruction::BrTable`].
#[derive(Debug)]
pub(crate) struct BrTableOperands {
    /// The arm selector. Out-of-range picks the default, which is the last arm.
    index: InputRegisters<1>,
    /// One per arm, the default last. Each carries its own move and target because a
    /// single table may mix loop and non-loop labels, whose heights differ even
    /// though validation makes their arities agree.
    br_targets: Vec<RegBrTableTarget>,
}

/// The arena entry behind [`RegInstruction::CallIndirect`]: two operand runs, two
/// immediates and a frame base, several times what fits inline.
#[derive(Debug)]
pub(crate) struct CallIndirectOperands {
    /// Index into the module's type section of the signature this call site
    /// expects.
    ///
    /// Two jobs: the callee's own type is checked structurally against it at
    /// execution — a mismatch traps — and its param count is how many of the
    /// operands at `operands` are arguments.
    ty_index: TyIndex,
    /// The table the callee index is resolved through, and what a trap out of
    /// this call reports.
    table_index: TableIndex,
    /// Where the callee index is read from. Resolved *before* the move is
    /// performed; see the note above.
    slot: InputRegisters<1>,
    /// This call's arguments, in wasm push order. The `ty_index` signature's param
    /// count says how many of them there are, and they are moved to `caller_base`,
    /// `caller_base + 1`, … one apiece.
    ///
    /// A [`DynSignature`] because the arity is not fixed by the opcode. Its
    /// `output_start` is unused here — [`Self::caller_base`] is the destination, and
    /// the end-of-body pass shifts that one.
    operands: DynSignature,
    /// Frame index the callee's frame is based at, on the same terms as
    /// [`RegInstruction::Call`]'s — including the placeholder and the invariant it
    /// ends up satisfying.
    caller_base: u16,
}

/// The arena entry behind [`RegInstruction::Select`].
///
/// `select` is the only value operator that needs three operands *and* a
/// destination, which is one slot more than eight bytes hold.
#[derive(Debug)]
pub(crate) struct SelectOperands(Signature<3, 1>);

/// The arena entry behind [`RegInstruction::MemoryInit`]: three operands plus an
/// immediate, which is two bytes past what fits inline.
#[derive(Debug)]
pub(crate) struct MemoryInitOperands {
    /// Index of the data segment to copy from.
    data_index: u32,
    /// `dest`, `src`, `len`.
    operands: InputRegisters<3>,
}

/// The whole lowering state for one function body.
///
/// "Simulated" because [`Self::stack`] holds descriptions of operands rather than
/// values: the pass walks the operator stream maintaining the stack wasm *would*
/// have, and reads off where each value lives instead of moving it.
struct SimulatedStack {
    /// The operand stack, one [`StackSlot`] per value wasm would have pushed.
    stack: Stack<StackSlot>,
    /// The next operand register to allocate, as a *provisional* frame index.
    ///
    /// Seeded to `locals_count` and moved only by [`StackSlot::Register`] entries, so
    /// it is the locals count plus the number of live operand registers — an index,
    /// not a count. Distinct from `stack.height()`, which counts every simulated slot
    /// including those that occupy no register. See the module docs.
    ///
    /// Provisional because it does not yet clear the constant and spill regions; the
    /// end-of-body pass shifts every index derived from it.
    curr_register_index: usize,
    /// Peak [`Self::curr_register_index`], i.e. one past the highest operand register
    /// the body reaches.
    ///
    /// Ships verbatim as [`RegFrameLayout::registers`], so it **includes the locals**
    /// and the frame is `max_registers + spills + consts` slots wide.
    max_registers: u16,
    /// Lazy borrows of locals; see [`lazy`].
    lazy_locals: LazyArena<Local>,
    /// Frame slots holding locals materialized ahead of a write.
    spills: SpillArena,
    /// Open labels, for resolving branch depths and backpatching at `end`.
    control_stack: ControlStack,
    // One arena per operand shape too wide to sit inside an eight-byte instruction.
    // All seven move to the [`RegFrameLayout`] unchanged when the body ends; they are
    // separate rather than one arena of an enum so an [`Id`] cannot be read against
    // the wrong shape.
    /// `if` conditions and their two jump targets.
    if_arena: Arena<IfOperands>,
    /// `br_if` conditions, each with the move performed only on the taken path.
    br_if_arena: Arena<BrIfOperands>,
    /// `br_table` index operands, each with its own `Vec` of arms.
    br_table_arena: Arena<BrTableOperands>,
    /// `call_indirect` callee slots, arguments and callee frame bases.
    call_indirect_arena: Arena<CallIndirectOperands>,
    /// `select`'s three operands and its destination.
    select_arena: Arena<SelectOperands>,
    /// `memory.init`'s three operands and its segment index.
    memory_init_arena: Arena<MemoryInitOperands>,
    /// The label moves a [`RegInstruction::Move`] performs. Separate from the six
    /// above because a move's arity is a label's, not an opcode's.
    dyn_signatures: Arena<DynSignature>,
    /// Static byte offsets of the body's loads and stores, deduped so a repeated
    /// offset costs one entry.
    memory_offsets: Interner<MemoryOffset, u16>,
    /// Every operand recorded against the instruction that will carry it, for the
    /// end-of-body pass. Lowering-time only: it is what fills the instructions in,
    /// and is dropped once it has.
    backpatch_map: BackpatchMap,
    /// The body's constant pool. Becomes [`RegFrameLayout::consts`], and its length is
    /// one of the two terms every register index is shifted by.
    const_interner: Interner<Const, u16>,
}

impl SimulatedStack {
    /// Empty lowering state for one body.
    ///
    /// The two counts size the lazy origin tables, which are indexed by local and
    /// global index without bounds checks — so `locals_count` must include the
    /// params, not just the declared locals. The arenas, the backpatch map and the
    /// control stack start empty and grow as the body is walked; the caller pushes
    /// the function frame onto the control stack.
    ///
    /// The two register counters do **not** start at zero. Seeding both to
    /// `locals_count` is what puts locals and operand registers in one index space,
    /// which is what lets an operand be a bare frame index with no tag: below
    /// `locals_count` it is a local, at or above it an operand register. It is also
    /// why [`RegFrameLayout::registers`] includes the locals.
    fn new(locals_count: u32) -> Self {
        SimulatedStack {
            stack: Stack::new_with_capacity(0),
            curr_register_index: locals_count as usize,
            max_registers: locals_count as u16,
            lazy_locals: LazyArena::new(locals_count),
            spills: SpillArena::default(),
            control_stack: ControlStack::default(),
            if_arena: Arena::default(),
            br_if_arena: Arena::default(),
            br_table_arena: Arena::default(),
            call_indirect_arena: Arena::default(),
            select_arena: Arena::default(),
            memory_init_arena: Arena::default(),
            dyn_signatures: Arena::default(),
            memory_offsets: Interner::default(),
            backpatch_map: BackpatchMap::default(),
            const_interner: Interner::default(),
        }
    }

    /// Params plus declared locals, which is also where the operand registers begin
    /// in the provisional index space.
    ///
    /// Read off the lazy origin table rather than stored again, since that table is
    /// sized to exactly this and the two could otherwise disagree.
    fn locals_count(&self) -> u16 {
        self.lazy_locals.origin.len() as u16
    }

    /// Allocates one register, keeping [`Self::max_registers`] a true high-water mark.
    ///
    /// Every path that allocates a register must either come through here or fold its
    /// destinations into `max_registers` by hand, or the frame will be sized smaller
    /// than the registers instructions actually write. `br_truncation_registers` is
    /// the one that does it by hand, because its destinations are not allocated by
    /// pushing.
    ///
    /// # Errors
    ///
    /// [`TraceWasmError::RegisterFrameTooLarge`] once the region would outgrow a
    /// 16-bit frame index.
    fn advanced_register_index(&mut self) -> Result<(), TraceWasmError> {
        self.curr_register_index += 1;

        if self.curr_register_index > self.max_registers as usize {
            // Widened for the same reason as `SpillArena::reserve_slot`: the cap
            // is at `u16::MAX`, so a same-width `>=` is an equality in disguise.
            if self.max_registers as u32 >= MAX_REGISTER_SLOTS as u32 {
                return Err(TraceWasmError::RegisterFrameTooLarge {
                    what: "locals and operand registers",
                    needed: self.max_registers as u32 + 1,
                    limit: MAX_REGISTER_SLOTS as u32,
                });
            }

            self.max_registers += 1;
        }

        Ok(())
    }

    /// Frees the top register, when a [`StackSlot::Register`] is popped.
    fn recede_register_index(&mut self) {
        self.curr_register_index -= 1;
    }

    /// Opens a label, recording the layout its `else`/`end`/branches restore to.
    ///
    /// Returns the block's `(params, results)`, which the caller needs to decide
    /// whether to emit a params [`Move`]. Taking a [`BlockVariant`] rather than a
    /// [`BlockKind`] keeps that decision in one place: the params count is derived
    /// from `blockty` here, and the same value picks both the `Move` and the
    /// instruction index recorded in the `BlockKind` — `instr_len + 1` when a `Move`
    /// precedes the block's own instruction, `instr_len` otherwise.
    ///
    /// `recorded_height` is a **slot** height, not a register index: it is what the
    /// stack is popped back down to, and popping is what releases lazy borrows and
    /// spill slots. The register base is recovered when needed by counting register
    /// slots back down to it. An `if` subtracts one more than the others because its
    /// condition sits above the params and is consumed by the `if` itself.
    ///
    /// Only ever called for reachable operators — dead blocks are counted by
    /// [`UnreachableTrackingControlStack`] and never reach here.
    ///
    /// [`Move`]: RegInstruction::Move
    fn add_block(
        &mut self,
        kind: BlockVariant,
        blockty: &BlockType,
        types: &[FuncType],
        instr_len: usize,
    ) -> (u32, u32) {
        let (params, results) = params_and_results_from_blockty(blockty, types);

        let kind = match kind {
            BlockVariant::Func => BlockKind::Func,
            BlockVariant::Block => BlockKind::Block,
            BlockVariant::If => BlockKind::If {
                index: if params != 0 {
                    instr_len + 1 // a move is emitted when params != 0, so the actual instruction lands at `len + 1`
                } else {
                    instr_len
                } as u32,
                else_index: None,
            },
            BlockVariant::Loop => BlockKind::Loop {
                index: if params != 0 {
                    instr_len + 1 // see above.
                } else {
                    instr_len
                } as u32,
            },
        };

        let recorded_height = match kind {
            BlockKind::Func => 0,
            BlockKind::Block => self.stack.height() - params,
            BlockKind::Loop { .. } => self.stack.height() - params,
            BlockKind::If { .. } => {
                // top is the `if` condition and then params
                self.stack.height() - params - 1
            }
        };

        self.control_stack.stack.push(Block {
            kind,
            recorded_height,
            params,
            results,
            attached_breaks: vec![],

            // below two fields are not used in register lowering!
            // they are just placeholders
            is_unreachable_traversing: false,
            has_inherited: false,
        });

        (params, results)
    }

    /// Closes the innermost label and hands back its record, for the `end` that
    /// backpatches every branch attached to it.
    ///
    /// Returns the [`Block`] by value because its `attached_breaks` are needed after
    /// it has left the stack, and it is the last reader of them.
    fn pop_block(&mut self) -> Block {
        self.control_stack
            .stack
            .pop()
            .expect("hitting this means the control stack logic for pushing blocks is incorrect")
    }

    /// The innermost open label — the block the operators being lowered belong to.
    fn get_curr_block(&self) -> &Block {
        debug_assert!(!self.control_stack.stack.is_empty());

        &self.control_stack.stack[self.control_stack.stack.len() - 1]
    }

    /// Mutable [`Self::get_curr_block`], for attaching a branch to the label it
    /// targets.
    fn get_curr_block_mut(&mut self) -> &mut Block {
        debug_assert!(!self.control_stack.stack.is_empty());
        let len = self.control_stack.stack.len();

        &mut self.control_stack.stack[len - 1]
    }

    /// A label by absolute position, index 0 being the function frame.
    ///
    /// Branches index by *relative depth*, so a caller converts first:
    /// `control_stack.len() - 1 - relative_depth`.
    fn get_block(&self, index: usize) -> &Block {
        &self.control_stack.stack[index]
    }

    /// Mutable [`Self::get_block`], for attaching a branch to an outer label.
    fn get_block_mut(&mut self, index: usize) -> &mut Block {
        &mut self.control_stack.stack[index]
    }

    /// Releases one borrow of a lazily forwarded local and reports where
    /// its value was.
    ///
    /// The location is read *before* the reference count drops, because the answer
    /// this returns is where the popped operand reads from, while the cleanup that
    /// follows depends on that same location: the last borrow of an `Original` entry
    /// clears its origin slot, the last borrow of a `Spilled` one returns its spill
    /// slot to the pool.
    ///
    /// An associated function rather than a method so the caller can pass one arena
    /// and the spill pool without borrowing all of `self`.
    fn pop_lazy<T>(
        slot: LazySlot<T>,
        arena: &mut LazyArena<T>,
        spills: &mut SpillArena,
    ) -> LazyLocation {
        let location = slot.location(arena);

        if matches!(slot.decrease_ref_count(arena), LazyEntryDropResult::Dropped) {
            match location {
                LazyLocation::Original(local_index) => arena.origin[local_index as usize] = None,
                LazyLocation::Spilled(spill_index) => spills.free_slot(spill_index),
            }
        }

        location
    }

    /// Starts or joins a lazy borrow of one local.
    ///
    /// If something already borrows this origin, the new stack slot shares that
    /// entry, so a later spill redirects both at once. Otherwise a fresh entry is
    /// allocated and recorded as the origin's live borrow.
    fn push_lazy<T>(location: u16, arena: &mut LazyArena<T>) -> LazySlot<T> {
        match arena.origin[location as usize] {
            Some(slot) => {
                slot.advanced_ref_count(arena);

                slot
            }
            None => {
                let slot = arena.allocate(location);
                arena.origin[location as usize] = Some(slot);

                slot
            }
        }
    }

    /// Consumes the top operand and resolves it to the [`Slot`] an instruction will
    /// read.
    ///
    /// This is the only path that releases state, and everything that has to be
    /// released rides on it: the register is freed, the lazy reference count drops,
    /// and a dropped entry gives back its origin slot or its spill slot. Bulk
    /// operations therefore unwind by repeated popping rather than by truncating —
    /// truncation would skip all of it. [`Self::simulated_pop`] is the read-only
    /// counterpart for cases that must not release anything.
    fn pop(&mut self) -> BackPatchableSlot {
        let val = self.stack.pop();

        match val {
            StackSlot::Const(val) => BackPatchableSlot::Const(val),
            StackSlot::Register(index) => {
                self.recede_register_index();

                BackPatchableSlot::Register(index)
            }
            StackSlot::Local(slot) => {
                let location = Self::pop_lazy(slot, &mut self.lazy_locals, &mut self.spills);

                match location {
                    LazyLocation::Original(local_index) => Slot(local_index).into(),
                    LazyLocation::Spilled(spill_index) => BackPatchableSlot::Spill(spill_index),
                }
            }
        }
    }

    /// Resolves the operand `depth` entries below the top *without* consuming it.
    ///
    /// `depth == 0` is the top, so walking `0..n` visits slots in the order repeated
    /// popping would, which is what lets a caller compute an instruction's operands
    /// as if it had unwound the stack.
    ///
    /// This exists for branch lowering. A branch has to describe the *target* label's
    /// layout — how far to unwind, and which registers the carried values land in —
    /// but the operators after it still belong to the enclosing block, which is not
    /// being left. Actually unwinding would release lazy borrows and spill slots the
    /// enclosing block still owns, and for a branch to an *outer* label it would pop
    /// below the enclosing block's own base entirely.
    ///
    /// Reading instead of popping also makes `br_table` work: every arm is computed
    /// against the same unchanged stack, where the first arm would otherwise have
    /// destroyed the state the rest need.
    ///
    /// The one thing a peek cannot do is maintain [`Self::curr_register_index`], so
    /// [`Self::br_truncation_registers`] tracks the register base itself, decrementing
    /// per register slot seen.
    fn simulated_pop(&self, depth: u32) -> BackPatchableSlot {
        let val = *self.stack.peek_from_top(depth);

        match val {
            StackSlot::Const(val) => BackPatchableSlot::Const(val),
            StackSlot::Register(index) => BackPatchableSlot::Register(index),
            StackSlot::Local(slot) => {
                let location = slot.location(&self.lazy_locals);

                match location {
                    LazyLocation::Original(local_index) => Slot(local_index).into(),
                    LazyLocation::Spilled(spill_index) => BackPatchableSlot::Spill(spill_index),
                }
            }
        }
    }

    /// Pushes one operand onto the simulated stack.
    ///
    /// A local becomes a shared lazy borrow rather than a resolved
    /// location, so a spill occurring before it is read still reaches it. A register
    /// allocates one.
    ///
    /// The two are told apart by `frame_index < locals_count`, which works only
    /// because register indices are provisional here: a resolved index in that range
    /// could also be a constant or a spill.
    ///
    /// A spill is never pushed. It is a *transition* of an existing borrow, applied
    /// through [`Self::set_lazy`], not a value that arrives from nothing.
    fn push(&mut self, slot: Slot) -> Result<(), TraceWasmError> {
        let slot = if slot.0 < self.locals_count() {
            let slot = Self::push_lazy(slot.0, &mut self.lazy_locals);

            StackSlot::Local(slot)
        } else {
            self.advanced_register_index()?;

            StackSlot::Register(slot.0)
        };

        self.stack.push(slot);

        Ok(())
    }

    /// Resolves the top operand without consuming it, for `local.tee`.
    ///
    /// `local.tee n` writes the value *and* leaves it on the stack, so the operand
    /// stays where it is. That is sound only because the caller spills first: any
    /// borrow of local `n` still on the stack — including this one — is redirected to
    /// a spill slot before the write, so the slot left behind no longer names the
    /// local being overwritten. Reading the location here, after the spill, is what
    /// picks up that redirection.
    fn tee(&self) -> BackPatchableSlot {
        let top_slot = &self.stack.top();

        match top_slot {
            StackSlot::Const(val) => BackPatchableSlot::Const(*val),
            StackSlot::Register(index) => BackPatchableSlot::Register(*index),
            StackSlot::Local(slot) => match slot.location(&self.lazy_locals) {
                LazyLocation::Original(local_index) => Slot(local_index).into(),
                LazyLocation::Spilled(spill_index) => BackPatchableSlot::Spill(spill_index),
            },
        }
    }

    /// `i32.const` and friends: records the immediate, emitting nothing.
    fn push_const(&mut self, val: Const) -> Result<(), TraceWasmError> {
        let id = self.const_interner.try_intern(val)?;

        self.stack.push(StackSlot::Const(id));

        Ok(())
    }

    /// `local.get`: starts or joins a lazy borrow of the local, emitting nothing.
    /// The value is read in place by whatever consumes it, unless a write to the
    /// local intervenes and spills it first.
    fn push_local(&mut self, index: u16) -> Result<(), TraceWasmError> {
        self.push(Slot(index))
    }

    /// Applies an instruction's stack effect and records its operands for
    /// resolution.
    ///
    /// Pops `pops` operands — deepest first, so `input[0]` is the first value pushed
    /// — into an entry keyed by `instr_index` in the backpatch map, then allocates
    /// `pushes` destination registers and pushes them back. Allocating the outputs
    /// *after* consuming the inputs is what lets a result reuse an operand's
    /// register, and is why an executor must read all operands before writing any
    /// destination.
    ///
    /// `instr_index` must be the index the instruction carrying these operands will
    /// occupy, since that is the only thing tying the two together;
    /// `source` distinguishes several runs recorded against one instruction, as
    /// `br_if`'s condition and move are.
    ///
    /// Use [`Self::pops_and_pushes`] when only the stack effect is wanted; this
    /// variant leaves an entry the end-of-body pass will look for.
    fn pops_and_pushes_registers(
        &mut self,
        pops: u32,
        pushes: u32,
        instr_index: usize,
        source: InstructionSource,
    ) -> Result<PopsPushesResult, TraceWasmError> {
        let pops = pops as usize;
        let pushes = pushes as usize;

        let mut backpatchable_slots = vec![BackPatchableSlot::Register(u16::MAX); pops];
        let slots = vec![Slot(u16::MAX); pops]; // will be backpatched!

        for i in 0..pops {
            let slot = self.pop();

            backpatchable_slots[pops - 1 - i] = slot;
        }

        match self.backpatch_map.0.entry(instr_index) {
            Entry::Occupied(mut occ) => {
                let v = occ.get_mut();

                v.push((source, backpatchable_slots));
            }
            Entry::Vacant(vac) => {
                let v = vec![(source, backpatchable_slots)];

                vac.insert(v);
            }
        }

        let res = PopsPushesResult {
            input: slots,
            output_start: self.curr_register_index as u16,
        };

        for _ in 0..pushes {
            let out = Slot(self.curr_register_index as u16);

            self.push(out)?;
        }

        Ok(res)
    }

    /// The same stack effect as [`Self::pops_and_pushes_registers`], recording
    /// nothing.
    ///
    /// For resets that reshape the model without describing an instruction — the
    /// layout an `else` hands to its arm, or the layout an unconditional branch
    /// leaves the enclosing block in. Those emit no operands, so a map entry would be
    /// keyed to an instruction that does not carry them, and resolution would write
    /// them into whatever else occupies that index.
    ///
    /// The pops still go through [`Self::pop`], so lazy borrows and spill slots are
    /// released as they should be.
    fn pops_and_pushes(&mut self, pops: u32, pushes: u32) -> Result<(), TraceWasmError> {
        let pops = pops as usize;
        let pushes = pushes as usize;

        for _ in 0..pops {
            self.pop();
        }

        for _ in 0..pushes {
            let out = Slot(self.curr_register_index as u16);

            self.push(out)?;
        }

        Ok(())
    }

    /// Restores the enclosing block to the layout its `else`/`end` expects, after an
    /// operator that leaves the rest of that block dead.
    ///
    /// Every unconditional transfer ends with this — `br`, `br_table`, `return`,
    /// `unreachable`. The operators between here and the enclosing block's `else` or
    /// `end` are filtered out by [`UnreachableTrackingControlStack`], so nothing is
    /// lowered against the stack in between; but that `else`/`end` *is* lowered, and
    /// it materializes the block's results and asserts on the height it arrives at.
    /// This is what puts the model where that assertion expects it: the block's own
    /// entry height, with its results in registers above it.
    ///
    /// The block being restored is the one the operators belong to, *not* the one a
    /// branch targets — an outward branch leaves the enclosing block without closing
    /// it, and its `end` still has to be lowered.
    ///
    /// Pops go through [`Self::pops_and_pushes`], so the operands abandoned here
    /// release their lazy borrows and spill slots. Nothing downstream can read them:
    /// the path that would have is the one just taken away.
    fn reset_enclosing_block_layout(&mut self) -> Result<(), TraceWasmError> {
        let block = self.get_curr_block();
        let recorded_height = block.recorded_height;
        let results = block.results;
        let unwind = self.stack.height() - recorded_height;

        self.pops_and_pushes(unwind, results)?;

        Ok(())
    }

    /// [`Self::pops_and_pushes_registers`] for an instruction whose arity is fixed by
    /// its opcode, returning the [`Signature`] to store in the variant.
    fn registers_for<const I: usize, const O: usize>(
        &mut self,
        instr_index: usize,
        source: InstructionSource,
    ) -> Result<Signature<I, O>, TraceWasmError> {
        let result = self.pops_and_pushes_registers(I as u32, O as u32, instr_index, source)?;

        Ok(Signature {
            input: InputRegisters {
                registers: result.input.try_into().unwrap(),
            },
            output: OutputRegisters {
                start: result.output_start,
            },
        })
    }

    /// Forces the top `depth` operands into a contiguous register run, returning the
    /// [`RegInstruction::Move`] that performs it.
    ///
    /// Used wherever several control paths have to agree on where values live: a
    /// block's params at entry, and its results at each exit. Whatever the operands
    /// were — constants, lazy locals, registers — afterwards they are registers based
    /// at the label's register height, so a merge does not have to know which path
    /// arrived.
    ///
    /// Pops and re-pushes from the same base, so the destination range overlaps the
    /// sources; see [`RegInstruction::Move`] for what that requires of an executor.
    fn materialize_stack_slots_in_registers(
        &mut self,
        depth: u32,
        instr_index: usize,
        source: InstructionSource,
    ) -> Result<DynSignature, TraceWasmError> {
        let result = self.pops_and_pushes_registers(depth, depth, instr_index, source)?;

        Ok(DynSignature {
            input: result.input,
            output_start: result.output_start,
        })
    }

    /// Builds the [`RegInstruction::Move`] a branch performs on its way to a label,
    /// **without changing the simulated stack**.
    ///
    /// A taken branch unwinds to `base_height` while carrying the top
    /// `arity_to_preserve` operands into the label's registers. This computes that:
    /// the carried operands become the move's inputs, and its destinations are the
    /// `arity_to_preserve` registers based at the target's register height.
    ///
    /// Everything is read through [`Self::simulated_pop`], so no state is released.
    /// The stack must survive because the branch does not end the enclosing block —
    /// the operators after it are still lowered against it, and for a branch to an
    /// outer label `base_height` lies *below* the enclosing block's own base. It is
    /// also what lets `br_table` compute every arm against the same stack.
    ///
    /// Because nothing pops, [`Self::curr_register_index`] cannot do the counting, so
    /// `register_index` walks down one per register slot seen — arriving at exactly
    /// the value popping to `base_height` would have produced. Destination registers
    /// are then folded into [`Self::max_registers`] by hand, since they are allocated
    /// without going through [`Self::advanced_register_index`]; the branch's operands
    /// need not be registers already, so its destinations can sit above anything
    /// allocated so far.
    fn br_truncation_registers(
        &mut self,
        base_height: u32,
        arity_to_preserve: u32,
        instr_index: usize,
        source: InstructionSource,
    ) -> Result<DynSignature, TraceWasmError> {
        let arity_to_preserve = arity_to_preserve as usize;
        let curr_stack_height = self.stack.height();
        let popped_count = (curr_stack_height - base_height) as usize;
        let mut register_index = self.curr_register_index as u16;

        let mut backpatchable_slots =
            vec![BackPatchableSlot::Register(u16::MAX); arity_to_preserve];
        let slots = vec![Slot(u16::MAX); arity_to_preserve];

        for i in 0..popped_count {
            let slot = self.simulated_pop(i as u32);

            if slot.is_register() {
                register_index -= 1;
            }

            if i < arity_to_preserve {
                backpatchable_slots[arity_to_preserve - 1 - i] = slot;
            }
        }

        if arity_to_preserve > 0 {
            match self.backpatch_map.0.entry(instr_index) {
                Entry::Occupied(mut occ) => {
                    let v = occ.get_mut();

                    v.push((source, backpatchable_slots));
                }
                Entry::Vacant(vac) => {
                    let v = vec![(source, backpatchable_slots)];

                    vac.insert(v);
                }
            }
        }

        let res = DynSignature {
            input: slots,
            output_start: register_index,
        };

        let peak = register_index as u32 + arity_to_preserve as u32;

        if peak > self.max_registers as u32 {
            if peak > MAX_REGISTER_SLOTS as u32 {
                return Err(TraceWasmError::RegisterFrameTooLarge {
                    what: "locals and operand registers",
                    needed: peak,
                    limit: MAX_REGISTER_SLOTS as u32,
                });
            }

            self.max_registers = peak as u16;
        }

        Ok(res)
    }

    /// Rescues every operand still forwarding to `location`, ahead of a write to it.
    ///
    /// Returns the spill slot the caller must emit a
    /// [`LocalSpill`](RegInstruction::LocalSpill) into — *before* the writing
    /// instruction, so the copy captures the pre-write value — or `None` when nothing
    /// borrows the origin and no rescue is needed.
    ///
    /// One redirect covers all of the borrows, however many there are, because they
    /// share one entry. Clearing the origin afterwards is what lets a subsequent
    /// `local.get` of the same index start a fresh borrow of the *new* value while
    /// the older operands keep reading the spill slot.
    fn set_lazy<T>(
        location: u32,
        arena: &mut LazyArena<T>,
        spills: &mut SpillArena,
    ) -> Result<Option<SpillIndex>, TraceWasmError> {
        let Some(slot) = arena.origin[location as usize] else {
            return Ok(None);
        };

        let spill_index = spills.reserve_slot()?;

        slot.spill(spill_index, arena);
        arena.origin[location as usize] = None;

        Ok(Some(spill_index))
    }

    /// What [`Self::curr_register_index`] would be after popping `depth` operands,
    /// without popping any of them.
    ///
    /// The two heights differ, so this cannot be subtraction: only
    /// [`StackSlot::Register`] entries in the window give a register back, and a
    /// `Const`, `Local`, or `Global` occupies a stack position and none. See the
    /// module docs.
    ///
    /// Callers use it to name a register base *before* consuming the operands that
    /// determine it — a call's `caller_base`, which has to be known when the
    /// instruction is built. The window must span every operand the instruction
    /// consumes: `call_indirect` counts one deeper than its signature's arity,
    /// because the callee index is popped along with the arguments.
    fn register_index_at_depth(&self, depth: u32) -> usize {
        let mut register_index = self.curr_register_index;

        for i in 0..depth {
            if matches!(self.stack.peek_from_top(i), StackSlot::Register(_)) {
                register_index -= 1;
            }
        }

        register_index
    }
}

/// The storage one lowered body needs, and the region sizes that give an operand
/// index its meaning.
///
/// # The four regions
///
/// A frame is one contiguous run of slots, divided as:
///
/// ```text
///   0                                    locals — params first, then declared
///   locals_count                         consts — the interned pool, in id order
///   locals_count + consts                spills
///   locals_count + consts + spills       operand registers
///   registers + consts + spills          end of frame
/// ```
///
/// [`Self::registers`] and [`Self::spills`] are high-water marks over the whole body
/// rather than counts at any one point, so a frame sized to them never has to grow
/// mid-execution.
///
/// **[`Self::registers`] counts from the frame base, so it includes the locals.**
/// The lowering seeds its register index at `locals_count` — that is what lets a
/// local and an operand register share one index space — so the frame is
/// `registers + spills + consts` slots wide, *not* `locals + registers + …`.
///
/// # Why these counts ship with the body
///
/// A [`Slot`] carries one absolute frame index and no tag, so execution never needs
/// to know which region an index names: it adds the index to the frame base and
/// loads. The region *sizes* are the only thing that can recover the distinction,
/// which is what [`Slot::render`] uses them for, and what the runtime uses to place
/// the constant pool and the spill region at frame entry.
pub(crate) struct RegFrameLayout {
    /// Peak frame index the body's operand registers reach, one past the last.
    ///
    /// **Includes the locals**, since the lowering's register index starts at
    /// [`Self::locals_count`] — so the operand registers themselves are the range
    /// `[locals_count + consts + spills, registers + consts + spills)`.
    pub registers: u16,
    /// Spill slots holding locals rescued from a later write by
    /// [`RegInstruction::LocalSpill`].
    ///
    /// Zero for a body that never overwrites a lazily-forwarded local, which is the
    /// common case — and then the region is simply empty, closing the gap between the
    /// constants and the operand registers.
    pub spills: u16,
    // The seven operand shapes that do not fit inside an eight-byte instruction. Each
    // of those instructions holds only an [`Id`], so these have to ship with the body
    // — an id is meaningless without the arena it indexes.
    /// `if` conditions, and the `else`/`end` indices a false condition jumps to.
    pub if_arena: Arena<IfOperands>,
    /// `br_if` conditions, each with the move its taken path performs.
    pub br_if_arena: Arena<BrIfOperands>,
    /// `br_table` index operands, each with its own arms.
    pub br_table_arena: Arena<BrTableOperands>,
    /// `call_indirect` callee slots, arguments, type index and callee frame base.
    pub call_indirect_arena: Arena<CallIndirectOperands>,
    /// `select`'s three operands and its destination.
    pub select_arena: Arena<SelectOperands>,
    /// `memory.init`'s three operands and its segment index.
    pub memory_init_arena: Arena<MemoryInitOperands>,
    /// The label moves [`RegInstruction::Move`] performs, whose arity a label fixes
    /// rather than an opcode.
    pub dyn_signatures: Arena<DynSignature>,
    /// Static byte offsets of the body's loads and stores, in allocation order.
    ///
    /// A `memarg.offset` is a `u32`, too wide to sit in an 8-byte instruction next to
    /// a signature, so the instruction carries an [`InternedId<MemoryOffset>`] into
    /// this pool instead. Interned, so the `0` that every bare pointer deref loads
    /// through costs one entry no matter how many loads use it.
    pub memory_offsets: Interner<MemoryOffset, u16>,
    /// Params plus declared locals, i.e. the width of the frame's first region.
    ///
    /// Doubles as the base of the constant region, and as the boundary that tells a
    /// local index from an operand register index during lowering.
    pub locals_count: u16,
    /// The body's interned constants, in [`InternedId`] order.
    ///
    /// Constant `i` occupies frame slot `locals_count + i`, and
    /// [`RuntimeFrame::enter_frame`](crate::instruction::RuntimeFrame::enter_frame)
    /// writes the whole pool there on every frame entry. Its length is one of the two
    /// terms the end-of-body pass shifts operand registers by.
    pub consts: Box<[Const]>,
}

impl FrameLayout for RegFrameLayout {
    type BrTableTarget = RegBrTableTarget;
}

/// The three parallel outputs of lowering one function body into register form: the
/// instruction list, the source-offset sidecar indexed alongside it, and the frame
/// required to execute it.
///
/// The first two come out of an [`Instructions`], which is what keeps them the same
/// length; see [`RegInstruction::emit_instructions_for_func`].
type RegLoweredFuncBody = (Vec<RegInstruction>, Vec<u32>, RegFrameLayout);

/// One lowered instruction.
///
/// Operands are inline: a variant carries them as a [`Signature`], or as
/// [`InputRegisters`] alone where there is no destination to name. The seven shapes
/// that do not fit eight bytes carry an [`Id`] into an arena on [`FrameLayout`]
/// instead. Jump fields are absolute indices into the containing
/// `Vec<RegInstruction>`, i.e. runtime program counters.
///
/// [`Kind`](tracewasm_macros::Kind) derives the fieldless [`RegInstructionKind`]
/// alongside this, so a table keyed by kind is an exhaustive `match` *and* is
/// visited in full — see the derive's docs for why both halves matter.
#[derive(Debug, tracewasm_macros::Kind)]
pub(crate) enum RegInstruction {
    /// `global.get`: read a global into an operand register.
    ///
    /// Unlike `local.get`, which forwards lazily and is read in place by whatever
    /// consumes it, this materialises **eagerly** — and that is what makes the
    /// operand encoding possible. An operand is a bare frame index with no tag, so a
    /// global, which lives in the instance rather than the frame, cannot be one. It
    /// also means nothing can invalidate the value afterwards: a later `global.set`,
    /// or a call that writes any global, leaves this register alone, so globals need
    /// no rescue machinery of the kind [`Self::LocalSpill`] provides for locals.
    ///
    /// The register survives a call because it sits below the `caller_base` any call
    /// in this body is based at.
    GlobalGet {
        /// The global read, in the module's global index space.
        index: GlobalIndex,
        /// The register the value is written to.
        output: OutputRegisters<1>,
    },
    /// `global.set`: write the operand into a global.
    ///
    /// Needs no rescue of earlier reads: [`Self::GlobalGet`] already materialised
    /// each one into a register, so this cannot invalidate an operand.
    GlobalSet {
        index: GlobalIndex,
        input: InputRegisters<1>,
    },
    /// `local.set`: write the operand into a local. See [`Self::GlobalSet`] on
    /// rescues.
    LocalSet {
        index: LocalIndex,
        input: InputRegisters<1>,
    },
    /// Copies a local into a spill slot, immediately before a write that would
    /// otherwise invalidate operands still reading it. See [`lazy`].
    LocalSpill {
        index: LocalIndex,
        spill_index: SpillIndex,
    },
    /// `local.tee`: write the operand into a local, leaving it on the operand stack.
    ///
    /// Arity is `<1, 0>` rather than `<1, 1>` because the value stays where it
    /// already is — no destination register is allocated.
    LocalTee {
        /// The local written.
        index: LocalIndex,
        /// The value to write, left on the simulated stack for the consumer.
        input: InputRegisters<1>,
    },
    /// `loop`: a branch target and nothing else.
    ///
    /// Carries no operands and does no work — a `br` to this label is already an
    /// absolute jump to it. It exists so the label occupies an instruction index for
    /// a back-edge to name, and so a body renders with its structure visible.
    Loop,
    /// `if`: fall through when the condition is non-zero, otherwise jump past
    /// `else_index` to the else-arm — or to `end_index` when there is none.
    ///
    /// Block params, if any, were materialized by a [`Self::Move`] emitted just
    /// before this instruction.
    If(Id<IfOperands>),
    /// Reached only by falling out of a taken then-branch, which must skip the
    /// else-arm: control jumps straight to `end_index`.
    ///
    /// A false condition never lands here — [`Self::If`] jumps past it.
    Else { end_index: u32 },
    /// `br`: jump to a label unconditionally.
    ///
    /// Carries no operands. Values transferred to the label were materialized by a
    /// [`Self::Move`] emitted immediately before this instruction, so `target_index`
    /// points past the label's own fallthrough `Move` and each path copies once.
    Br {
        /// Absolute jump target: a loop's start for a back-edge, otherwise the
        /// label's `end`, backpatched.
        target_index: u32,
    },
    /// `br_if`: jump when the condition is non-zero, otherwise fall through.
    ///
    /// The move is a **field rather than a preceding instruction** because it must
    /// only run on the taken path — an unconditional `Move` ahead of the branch would
    /// overwrite registers the fall-through still needs. `mov` is empty when the
    /// label carries no values.
    ///
    /// The condition is read before the move is performed, so a move whose
    /// destinations include the condition's register is harmless.
    BrIf(Id<BrIfOperands>),
    /// `br_table`: jump to the arm selected by the operand, or to the default when it
    /// is out of range.
    ///
    /// Arms are a `Vec<RegBrTableTarget>` on the arena entry; the default is the last
    /// element. Each arm carries its own move and target, since the labels may sit at
    /// different heights even though their arities agree.
    BrTable(Id<BrTableOperands>),
    // The numeric instructions, grouped by the type they are named for and then
    // by shape: loads, stores, unary, binary. Same order as the lowering match, so
    // the two read side by side.
    //
    // A load pops an address and pushes the value read; a store pops the address
    // and the value and pushes nothing. `offset` is the static displacement folded
    // into the address, and the narrow forms name the width they touch and how it
    // is widened: `i64.load8_s` reads one byte and sign-extends it to 64 bits. The
    // memarg's alignment is deliberately not carried — it is a hint with no effect
    // on semantics, and the stack pass keeps a field the interpreter ignores at
    // every use site.
    //
    // The rest carry nothing but their operand runs, and their arity lives in the
    // `Signature<I, O>` here — with `emit!`, this declaration is the only place it
    // is written down, which is why `arity_case` in the tests checks every one of
    // them against the spec. Three things the names do not say. A comparison
    // pushes an `i32` rather than nothing, so it is `<2, 1>` like the arithmetic.
    // `div_s`/`div_u`/`rem_s`/`rem_u` and the non-saturating `trunc_f*` trap at
    // execution, which lowering does not model — an executor must not commit the
    // destination register before the trap. The `trunc_sat_f*` forms do not trap;
    // that is the whole difference.

    // i32 — loads
    /// `i32.load`.
    I32Load {
        /// Static byte offset added to the popped address.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i32.load8_s`.
    I32Load8S {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i32.load8_u`.
    I32Load8U {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i32.load16_s`.
    I32Load16S {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i32.load16_u`.
    I32Load16U {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },

    // i32 — stores
    /// `i32.store`.
    I32Store {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        input: InputRegisters<2>,
    },
    /// `i32.store8`.
    I32Store8 {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        input: InputRegisters<2>,
    },
    /// `i32.store16`.
    I32Store16 {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        input: InputRegisters<2>,
    },

    // i32 — unary
    /// `i32.clz`.
    I32Clz(Signature<1, 1>),
    /// `i32.ctz`.
    I32Ctz(Signature<1, 1>),
    /// `i32.eqz`.
    I32Eqz(Signature<1, 1>),
    /// `i32.extend16_s`.
    I32Extend16S(Signature<1, 1>),
    /// `i32.extend8_s`.
    I32Extend8S(Signature<1, 1>),
    /// `i32.popcnt`.
    I32Popcnt(Signature<1, 1>),
    /// `i32.reinterpret_f32`.
    I32ReinterpretF32(Signature<1, 1>),
    /// `i32.trunc_f32_s`.
    I32TruncF32S(Signature<1, 1>),
    /// `i32.trunc_f32_u`.
    I32TruncF32U(Signature<1, 1>),
    /// `i32.trunc_f64_s`.
    I32TruncF64S(Signature<1, 1>),
    /// `i32.trunc_f64_u`.
    I32TruncF64U(Signature<1, 1>),
    /// `i32.trunc_sat_f32_s`.
    I32TruncSatF32S(Signature<1, 1>),
    /// `i32.trunc_sat_f32_u`.
    I32TruncSatF32U(Signature<1, 1>),
    /// `i32.trunc_sat_f64_s`.
    I32TruncSatF64S(Signature<1, 1>),
    /// `i32.trunc_sat_f64_u`.
    I32TruncSatF64U(Signature<1, 1>),
    /// `i32.wrap_i64`.
    I32WrapI64(Signature<1, 1>),

    // i32 — binary
    /// `i32.add`.
    I32Add(Signature<2, 1>),
    /// `i32.and`.
    I32And(Signature<2, 1>),
    /// `i32.div_s`.
    I32DivS(Signature<2, 1>),
    /// `i32.div_u`.
    I32DivU(Signature<2, 1>),
    /// `i32.eq`.
    I32Eq(Signature<2, 1>),
    /// `i32.ge_s`.
    I32GeS(Signature<2, 1>),
    /// `i32.ge_u`.
    I32GeU(Signature<2, 1>),
    /// `i32.gt_s`.
    I32GtS(Signature<2, 1>),
    /// `i32.gt_u`.
    I32GtU(Signature<2, 1>),
    /// `i32.le_s`.
    I32LeS(Signature<2, 1>),
    /// `i32.le_u`.
    I32LeU(Signature<2, 1>),
    /// `i32.lt_s`.
    I32LtS(Signature<2, 1>),
    /// `i32.lt_u`.
    I32LtU(Signature<2, 1>),
    /// `i32.mul`.
    I32Mul(Signature<2, 1>),
    /// `i32.ne`.
    I32Ne(Signature<2, 1>),
    /// `i32.or`.
    I32Or(Signature<2, 1>),
    /// `i32.rem_s`.
    I32RemS(Signature<2, 1>),
    /// `i32.rem_u`.
    I32RemU(Signature<2, 1>),
    /// `i32.rotl`.
    I32Rotl(Signature<2, 1>),
    /// `i32.rotr`.
    I32Rotr(Signature<2, 1>),
    /// `i32.shl`.
    I32Shl(Signature<2, 1>),
    /// `i32.shr_s`.
    I32ShrS(Signature<2, 1>),
    /// `i32.shr_u`.
    I32ShrU(Signature<2, 1>),
    /// `i32.sub`.
    I32Sub(Signature<2, 1>),
    /// `i32.xor`.
    I32Xor(Signature<2, 1>),

    // i64 — loads
    /// `i64.load`.
    I64Load {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i64.load8_s`.
    I64Load8S {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i64.load8_u`.
    I64Load8U {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i64.load16_s`.
    I64Load16S {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i64.load16_u`.
    I64Load16U {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i64.load32_s`.
    I64Load32S {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },
    /// `i64.load32_u`.
    I64Load32U {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },

    // i64 — stores
    /// `i64.store`.
    I64Store {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        input: InputRegisters<2>,
    },
    /// `i64.store8`.
    I64Store8 {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        input: InputRegisters<2>,
    },
    /// `i64.store16`.
    I64Store16 {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        input: InputRegisters<2>,
    },
    /// `i64.store32`.
    I64Store32 {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        input: InputRegisters<2>,
    },

    // i64 — unary
    /// `i64.clz`.
    I64Clz(Signature<1, 1>),
    /// `i64.ctz`.
    I64Ctz(Signature<1, 1>),
    /// `i64.eqz`.
    I64Eqz(Signature<1, 1>),
    /// `i64.extend16_s`.
    I64Extend16S(Signature<1, 1>),
    /// `i64.extend32_s`.
    I64Extend32S(Signature<1, 1>),
    /// `i64.extend8_s`.
    I64Extend8S(Signature<1, 1>),
    /// `i64.extend_i32_s`.
    I64ExtendI32S(Signature<1, 1>),
    /// `i64.extend_i32_u`.
    I64ExtendI32U(Signature<1, 1>),
    /// `i64.popcnt`.
    I64Popcnt(Signature<1, 1>),
    /// `i64.reinterpret_f64`.
    I64ReinterpretF64(Signature<1, 1>),
    /// `i64.trunc_f32_s`.
    I64TruncF32S(Signature<1, 1>),
    /// `i64.trunc_f32_u`.
    I64TruncF32U(Signature<1, 1>),
    /// `i64.trunc_f64_s`.
    I64TruncF64S(Signature<1, 1>),
    /// `i64.trunc_f64_u`.
    I64TruncF64U(Signature<1, 1>),
    /// `i64.trunc_sat_f32_s`.
    I64TruncSatF32S(Signature<1, 1>),
    /// `i64.trunc_sat_f32_u`.
    I64TruncSatF32U(Signature<1, 1>),
    /// `i64.trunc_sat_f64_s`.
    I64TruncSatF64S(Signature<1, 1>),
    /// `i64.trunc_sat_f64_u`.
    I64TruncSatF64U(Signature<1, 1>),

    // i64 — binary
    /// `i64.add`.
    I64Add(Signature<2, 1>),
    /// `i64.and`.
    I64And(Signature<2, 1>),
    /// `i64.div_s`.
    I64DivS(Signature<2, 1>),
    /// `i64.div_u`.
    I64DivU(Signature<2, 1>),
    /// `i64.eq`.
    I64Eq(Signature<2, 1>),
    /// `i64.ge_s`.
    I64GeS(Signature<2, 1>),
    /// `i64.ge_u`.
    I64GeU(Signature<2, 1>),
    /// `i64.gt_s`.
    I64GtS(Signature<2, 1>),
    /// `i64.gt_u`.
    I64GtU(Signature<2, 1>),
    /// `i64.le_s`.
    I64LeS(Signature<2, 1>),
    /// `i64.le_u`.
    I64LeU(Signature<2, 1>),
    /// `i64.lt_s`.
    I64LtS(Signature<2, 1>),
    /// `i64.lt_u`.
    I64LtU(Signature<2, 1>),
    /// `i64.mul`.
    I64Mul(Signature<2, 1>),
    /// `i64.ne`.
    I64Ne(Signature<2, 1>),
    /// `i64.or`.
    I64Or(Signature<2, 1>),
    /// `i64.rem_s`.
    I64RemS(Signature<2, 1>),
    /// `i64.rem_u`.
    I64RemU(Signature<2, 1>),
    /// `i64.rotl`.
    I64Rotl(Signature<2, 1>),
    /// `i64.rotr`.
    I64Rotr(Signature<2, 1>),
    /// `i64.shl`.
    I64Shl(Signature<2, 1>),
    /// `i64.shr_s`.
    I64ShrS(Signature<2, 1>),
    /// `i64.shr_u`.
    I64ShrU(Signature<2, 1>),
    /// `i64.sub`.
    I64Sub(Signature<2, 1>),
    /// `i64.xor`.
    I64Xor(Signature<2, 1>),

    // f32 — loads
    /// `f32.load`.
    F32Load {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },

    // f32 — stores
    /// `f32.store`.
    F32Store {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        input: InputRegisters<2>,
    },

    // f32 — unary
    /// `f32.abs`.
    F32Abs(Signature<1, 1>),
    /// `f32.ceil`.
    F32Ceil(Signature<1, 1>),
    /// `f32.convert_i32_s`.
    F32ConvertI32S(Signature<1, 1>),
    /// `f32.convert_i32_u`.
    F32ConvertI32U(Signature<1, 1>),
    /// `f32.convert_i64_s`.
    F32ConvertI64S(Signature<1, 1>),
    /// `f32.convert_i64_u`.
    F32ConvertI64U(Signature<1, 1>),
    /// `f32.demote_f64`.
    F32DemoteF64(Signature<1, 1>),
    /// `f32.floor`.
    F32Floor(Signature<1, 1>),
    /// `f32.nearest`.
    F32Nearest(Signature<1, 1>),
    /// `f32.neg`.
    F32Neg(Signature<1, 1>),
    /// `f32.reinterpret_i32`.
    F32ReinterpretI32(Signature<1, 1>),
    /// `f32.sqrt`.
    F32Sqrt(Signature<1, 1>),
    /// `f32.trunc`.
    F32Trunc(Signature<1, 1>),

    // f32 — binary
    /// `f32.add`.
    F32Add(Signature<2, 1>),
    /// `f32.copysign`.
    F32Copysign(Signature<2, 1>),
    /// `f32.div`.
    F32Div(Signature<2, 1>),
    /// `f32.eq`.
    F32Eq(Signature<2, 1>),
    /// `f32.ge`.
    F32Ge(Signature<2, 1>),
    /// `f32.gt`.
    F32Gt(Signature<2, 1>),
    /// `f32.le`.
    F32Le(Signature<2, 1>),
    /// `f32.lt`.
    F32Lt(Signature<2, 1>),
    /// `f32.max`.
    F32Max(Signature<2, 1>),
    /// `f32.min`.
    F32Min(Signature<2, 1>),
    /// `f32.mul`.
    F32Mul(Signature<2, 1>),
    /// `f32.ne`.
    F32Ne(Signature<2, 1>),
    /// `f32.sub`.
    F32Sub(Signature<2, 1>),

    // f64 — loads
    /// `f64.load`.
    F64Load {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        sig: Signature<1, 1>,
    },

    // f64 — stores
    /// `f64.store`.
    F64Store {
        /// Static byte offset added to the popped address, held in
        /// [`RegFrameLayout::memory_offsets`] because a `u32` does not fit here
        /// beside the operands.
        offset: InternedId<MemoryOffset, u16>,
        input: InputRegisters<2>,
    },

    // f64 — unary
    /// `f64.abs`.
    F64Abs(Signature<1, 1>),
    /// `f64.ceil`.
    F64Ceil(Signature<1, 1>),
    /// `f64.convert_i32_s`.
    F64ConvertI32S(Signature<1, 1>),
    /// `f64.convert_i32_u`.
    F64ConvertI32U(Signature<1, 1>),
    /// `f64.convert_i64_s`.
    F64ConvertI64S(Signature<1, 1>),
    /// `f64.convert_i64_u`.
    F64ConvertI64U(Signature<1, 1>),
    /// `f64.floor`.
    F64Floor(Signature<1, 1>),
    /// `f64.nearest`.
    F64Nearest(Signature<1, 1>),
    /// `f64.neg`.
    F64Neg(Signature<1, 1>),
    /// `f64.promote_f32`.
    F64PromoteF32(Signature<1, 1>),
    /// `f64.reinterpret_i64`.
    F64ReinterpretI64(Signature<1, 1>),
    /// `f64.sqrt`.
    F64Sqrt(Signature<1, 1>),
    /// `f64.trunc`.
    F64Trunc(Signature<1, 1>),

    // f64 — binary
    /// `f64.add`.
    F64Add(Signature<2, 1>),
    /// `f64.copysign`.
    F64Copysign(Signature<2, 1>),
    /// `f64.div`.
    F64Div(Signature<2, 1>),
    /// `f64.eq`.
    F64Eq(Signature<2, 1>),
    /// `f64.ge`.
    F64Ge(Signature<2, 1>),
    /// `f64.gt`.
    F64Gt(Signature<2, 1>),
    /// `f64.le`.
    F64Le(Signature<2, 1>),
    /// `f64.lt`.
    F64Lt(Signature<2, 1>),
    /// `f64.max`.
    F64Max(Signature<2, 1>),
    /// `f64.min`.
    F64Min(Signature<2, 1>),
    /// `f64.mul`.
    F64Mul(Signature<2, 1>),
    /// `f64.ne`.
    F64Ne(Signature<2, 1>),
    /// `f64.sub`.
    F64Sub(Signature<2, 1>),
    /// `select`: `input[2] != 0 ? input[0] : input[1]`.
    Select(Id<SelectOperands>),
    /// `return`: leave the function, carrying its results.
    ///
    /// Lowered as a branch to the outermost label, so it behaves like any other
    /// exit: the results, if there are any, are materialized by a [`Self::Move`]
    /// emitted immediately before, and this is recorded in the function block's
    /// `attached_breaks` for backpatching. Carries no operands of its own.
    Return {
        /// Absolute index of the function's `end`, backpatched when that `end` is
        /// reached. `u32::MAX` until then, which is what an unpatched target would
        /// show up as.
        target_index: u32,
    },
    /// `call`: invoke a function by index.
    ///
    /// The arguments were staged into `[caller_base, caller_base + params)` by a
    /// [`Self::Move`] emitted immediately before — omitted for a callee that takes
    /// none — so they become the callee's locals in place. Both arities come from
    /// the callee's declared type rather than from this variant.
    Call {
        /// The callee, in the module's function index space.
        func_index: FuncIndex,
        /// Frame index the callee's frame is based at: where its arguments were
        /// staged, and where its results come back to.
        ///
        /// `u32::MAX` until the end-of-body pass resolves it, like every other
        /// operand-register index. Once resolved it is always at or above
        /// `locals_count + consts + spills`, so the callee's frame — which starts
        /// here — cannot overwrite this frame's locals, constants or spills.
        caller_base: u16,
    },
    /// `call_indirect`: resolve a callee through a table at execution and call it.
    ///
    /// The callee index is pushed *above* the arguments, so it is popped first and
    /// the arguments are staged below it. That extra operand is the only structural
    /// difference from [`Self::Call`] — and the reason both of this call's heights
    /// count one deeper than the signature's arity.
    ///
    /// **The argument destinations are not stored.** The arguments themselves are
    /// [`CallIndirectOperands::operands`], of which the first `params` are the call's
    /// — `params` coming from the type [`CallIndirectOperands::ty_index`] names. Their
    /// destinations are that many registers based at
    /// [`CallIndirectOperands::caller_base`], contiguous because that is how they
    /// were allocated, so nothing has to name them.
    ///
    /// **The move is a field rather than a preceding [`Self::Move`]**, and unlike
    /// [`Self::BrIf`] the reason is not a control path: the staged arguments land in
    /// `[caller_base, caller_base + params)`, which may *include* the register
    /// `slot` reads the callee index from. A `Move` ahead of the instruction would
    /// destroy the index before it was used. An executor must therefore resolve
    /// `slot` before performing the move — the same read-before-write contract
    /// [`Self::BrIf`] has with its condition.
    CallIndirect(Id<CallIndirectOperands>),
    /// `ref.is_null`: `1` if the reference is null, else `0`.
    ///
    /// The result is an `i32`, not a reference — like [`Self::I32Eqz`] this is a
    /// predicate and follows the comparison convention, so it can feed a `br_if`
    /// directly. Consumes the reference; it does not peek.
    ///
    /// Its operand is frequently a [`Const::Ref`] rather than a register, since
    /// `ref.null` and `ref.func` are immediates — the answer is known at lowering
    /// time in that case, but folding it is left to whatever optimizes the stream.
    RefIsNull(Signature<1, 1>),
    /// `unreachable`: trap.
    ///
    /// Lowered like the branches in everything but the branch: it ends the block's
    /// reachable code, so the operators after it are dropped and the enclosing block
    /// is reset to the layout its `end` expects. It names no label, though, so there
    /// is no move to perform, no break to attach, and no target to backpatch —
    /// nothing survives the trap to be carried anywhere.
    ///
    /// A block whose only exit is a trap still reserves registers for its results,
    /// because its `end` is lowered as if the block could fall into it. Those
    /// registers are never written.
    Unreachable,
    /// Copies each input slot into the register named by the output at the same
    /// position, materializing block params and results so every path into a label
    /// leaves its values in the same registers.
    ///
    /// **Must be executed as a gather-then-scatter, not an in-place copy.**
    ///
    /// Two things rule out `copy_within` or any single-pass loop:
    ///
    /// * The inputs are not all registers. A [`Slot`] may name any region of the
    ///   frame — a constant, a local, a spill — or a global, which is not in the
    ///   frame at all, so this gathers into one contiguous destination range rather
    ///   than moving a block within one slice.
    ///
    /// * The register-sourced inputs *overlap* the destinations, and the two callers
    ///   need opposite copy directions — so there is no ordering that is correct for
    ///   both.
    ///
    /// `materialize_stack_slots_in_registers` pops `depth` slots and pushes `depth`
    /// registers from the same base, so a source's index never exceeds its own
    /// destination. Slots `[Const(5), Register(b)]` into `[b, b + 1]`:
    ///
    /// ```text
    /// ascending :  reg[b]   = 5           ;  reg[b+1] = reg[b]  -> reads 5   WRONG
    /// descending:  reg[b+1] = reg[b]  ok  ;  reg[b]   = 5       ok
    /// ```
    ///
    /// `br_truncation_registers` discards the slots between the target's base and the
    /// branch operands, which shifts the operands *down*, so there a source index can
    /// exceed its destination. Stack `[Register(b), Register(b + 1), Const]` above the
    /// base with `arity == 2` gives inputs `[Register(b + 1), Const]` into `[b, b + 1]`:
    ///
    /// ```text
    /// ascending :  reg[b]   = reg[b+1] ok ;  reg[b+1] = C       ok
    /// descending:  reg[b+1] = C           ;  reg[b]   = reg[b+1] -> reads C  WRONG
    /// ```
    ///
    /// Reading every input before writing any output is correct for both, and stays
    /// correct if the allocation pattern changes — neither direction rule is visible
    /// from the instruction itself, so relying on one is a trap for the next reader:
    ///
    /// ```text
    /// let mut tmp: SmallVec<[Value; 4]> = SmallVec::new();
    /// for slot in &sig.input { tmp.push(read(slot)); }
    /// for (i, val) in tmp.into_iter().enumerate() { regs[sig.output_start + i] = val; }
    /// ```
    ///
    /// The buffer costs nothing in practice: arities are the label's params or
    /// results, which are one or two values for anything rustc emits.
    Move(Id<DynSignature>),
    // The bulk-memory operators. Each takes its operands in wasm push order, so the
    // run reads left to right as the operands are written in the text format —
    // `input[0]` is the deepest, not the top of the stack. All of them address memory
    // 0; lowering rejects any other index, so none carries one.
    /// `memory.size`: write the memory's current size in pages to the destination
    /// register.
    ///
    /// The only variant here with no inputs, so it carries a bare output run rather
    /// than a [`Signature`].
    MemorySize(OutputRegisters<1>),
    /// `memory.grow`: grow the memory by the page delta in the input, writing the
    /// size *before* the growth to the output.
    ///
    /// Does **not** trap when the request cannot be satisfied — it writes `-1` and
    /// execution continues. The ceiling is the module's declared maximum, narrowed by
    /// the instance [`Config`](crate::instance::config::Config).
    MemoryGrow(Signature<1, 1>),
    /// `memory.copy`: copy within linear memory. Operands are `dest`, `src`, `len`.
    ///
    /// The ranges may overlap (`memmove` semantics). Traps if either runs past the
    /// end of memory, with nothing written — so an executor must bounds-check both
    /// before it moves a byte.
    MemoryCopy(InputRegisters<3>),
    /// `memory.fill`: set a range to the low byte of a value. Operands are `dest`,
    /// `value`, `len`.
    ///
    /// Only the low byte of `value` is used, though the operand is a full `i32`.
    /// Traps if the range runs past the end of memory.
    MemoryFill(InputRegisters<3>),
    /// `memory.init`: copy from a passive data segment into linear memory. Operands
    /// are `dest`, `src`, `len`, where `src` indexes the *segment*.
    ///
    /// Traps if the source range exceeds the segment or the destination exceeds
    /// memory. A segment already released by [`Self::DataDrop`] reads as empty, so a
    /// zero-length init still succeeds where any non-empty one traps.
    MemoryInit(Id<MemoryInitOperands>),
    /// `data.drop`: release a passive data segment's bytes. The field is the segment
    /// index.
    ///
    /// The segment becomes empty rather than invalid, so a later [`Self::MemoryInit`]
    /// traps only if it asks for a non-empty range. Dropping twice is harmless. Takes
    /// no operands, which is why it is the one memory variant with no register run.
    DataDrop(u32),
    /// Closes a label, one per `end` operator.
    ///
    /// This is where branches to a `block` or `if` label land — a `loop`'s
    /// back-edge targets its start instead. The fallthrough [`Self::Move`] is
    /// emitted *before* it, so a taken branch, which performed its own move on the
    /// way, jumps here and skips the copy it would otherwise repeat. The outermost
    /// one closes the function body.
    End,
}

// One `RegInstruction` per lowered operator, so this size is multiplied across every
// compiled module — the same budget, and the same reasoning, as `Instruction` in the
// stack pass.
//
// What holds it is that a `Slot` is two bytes and a destination run is named by its
// start alone, so `Signature<2, 1>` — the widest shape a value operator needs — is
// six, and the tag fits in the seventh. A load or store spends two of those six on
// its offset id, which is why a store carries `InputRegisters<2>` and not
// `Signature<2, 0>`: an `OutputRegisters` it has no use for would still cost two
// bytes and push the variant to ten.
//
// The seven shapes that cannot fit — `if`, `br_if`, `br_table`, `call_indirect`,
// `select`, `memory.init`, `Move` — carry a four-byte `Id` into an arena, which is
// also how an instruction whose arity its opcode does not fix is handled at all.
// Anything added later has the same two choices: fit the eight bytes, or go to an
// arena and pay one indirection for it.
const _: () = assert!(
    size_of::<RegInstruction>() <= 8,
    "RegInstruction grew past 8 bytes. Either fit the new shape, or move its operands \
     to an arena and carry an `Id` instead."
);

impl RegInstruction {
    /// Materializes every live lazy borrow in `arena`, emitting one spill
    /// instruction per borrow.
    ///
    /// # Why a rescue cannot live at the write
    ///
    /// A borrow of a local reads its origin in place (see [`lazy`]), and
    /// stays valid only until something writes that origin. The write emits a spill
    /// that copies the old value aside, and the borrow is redirected to it — so from
    /// then on the operand resolves to that spill slot *for every path*, because
    /// lowering resolves it once.
    ///
    /// That is only sound if the copy satisfies two properties. Emitting it at the
    /// write satisfies neither in general:
    ///
    /// * **It must run on every path that reads it.** A write inside one arm of an
    ///   `if`, or after a `br_if` that was taken, is skipped on the other path — and
    ///   that path still reads the spill slot, which nothing wrote. The value it
    ///   wanted is untouched in the local, but the operand is no longer looking
    ///   there.
    ///
    /// * **It must not run again after the write.** A write inside a loop body puts
    ///   the copy inside the repeated region, so the back-edge re-runs it and the
    ///   second pass captures what the first pass wrote — destroying the snapshot it
    ///   existed to preserve.
    ///
    /// # Where it lives instead
    ///
    /// The rescue is hoisted to the constructs that can break one of those:
    /// `if`, `br_if` and `br_table`, where control diverges, and `loop`, where it
    /// repeats. Called before the construct is emitted, the spill dominates every
    /// reader and runs exactly once.
    ///
    /// A `block` needs nothing: entering one is unconditional and it never repeats,
    /// so a spill inside it already has both properties.
    ///
    /// Once hoisted, the write inside the construct finds
    /// [`LazyArena::origin`]`[n] == None` and adds nothing — the borrow is already
    /// materialized, so there is no second copy and no second slot.
    ///
    /// # Cost
    ///
    /// Nothing is emitted unless a borrow is *live across* the construct, which
    /// means a `local.get` result still resting on the operand stack when control
    /// reaches it. Compiled Rust keeps such values in locals rather than on the
    /// stack, so in practice this loop finds nothing and emits nothing.
    ///
    /// When it does fire it is deliberately conservative: it spills whether or not
    /// any arm actually writes that origin, because a single forward pass cannot
    /// know what a construct writes before lowering it. Narrowing that would take a
    /// pre-pass collecting each construct's written origins, and the waste it would
    /// recover is one copy — the operand itself is no slower, since a spill slot and
    /// a local are both frame reads.
    ///
    /// # Ordering
    ///
    /// Call this *before* the arm reads `instructions.len()` — for the block index
    /// recorded by [`SimulatedStack::add_block`], or for the `attached_breaks` entry
    /// a branch registers. Spills emitted after that point shift the instruction the
    /// index was meant to name.
    fn spill_lazy<T, F: Fn(SpillIndex, u32) -> RegInstruction>(
        arena: &mut LazyArena<T>,
        spills: &mut SpillArena,
        instructions: &mut Instructions,
        instruction_emitter: F,
        offset: u32,
    ) -> Result<(), TraceWasmError> {
        let lazy_count = arena.origin.len() as u32;

        for index in 0..lazy_count {
            if let Some(spill_index) = SimulatedStack::set_lazy(index, arena, spills)? {
                instructions.push(instruction_emitter(spill_index, index), offset);
            }
        }

        Ok(())
    }

    /// Rescues every live local borrow ahead of a diverging or repeating construct.
    ///
    /// See [`Self::spill_lazy`] for why the rescue cannot be left at the write. Locals
    /// are the only origin space that needs this: a global is read into a register by
    /// [`Self::GlobalGet`], so no construct can strand a borrow of one.
    ///
    /// `offset` is the construct's own byte offset, so a spill hoisted above it
    /// traces back to the operator that forced it rather than to the write it
    /// rescued.
    fn spill_live_locals(
        simulated_stack: &mut SimulatedStack,
        instructions: &mut Instructions,
        offset: u32,
    ) -> Result<(), TraceWasmError> {
        let instruction_emitter =
            |spill_index: SpillIndex, index: u32| RegInstruction::LocalSpill {
                index: LocalIndex(index),
                spill_index,
            };

        Self::spill_lazy(
            &mut simulated_stack.lazy_locals,
            &mut simulated_stack.spills,
            instructions,
            instruction_emitter,
            offset,
        )
    }

    /// Claims the registers an operator of arity `I` -> `O` needs, hands them to
    /// `emitter` to build the instruction, and appends it at `offset`.
    ///
    /// The arity is carried by the const parameters, so it comes from the variant's
    /// own declaration — see the `emit!` macro in
    /// [`Self::emit_instructions_for_func`], which is how every arm reaches this.
    fn emit<const I: usize, const O: usize, F: FnOnce(Signature<I, O>) -> RegInstruction>(
        simulated_stack: &mut SimulatedStack,
        instructions: &mut Instructions,
        offset: u32,
        emitter: F,
    ) -> Result<(), TraceWasmError> {
        let registers =
            simulated_stack.registers_for::<I, O>(instructions.len(), InstructionSource::Emit)?;

        instructions.push(emitter(registers), offset);

        Ok(())
    }
}

/// Where the running frame begins in the instance's register file.
///
/// One field, because a positional machine needs one base: every [`Slot`] index is
/// added to it, and the four regions are found by offsetting from it with sizes taken
/// from [`RegFrameLayout`]. There is no second base to record, unlike the stack
/// machine's operand height.
///
/// A callee's is its caller's plus the calling instruction's `caller_base`.
pub(crate) struct RegCallerBaseData {
    /// Absolute index of the frame's first slot — its local 0 — in
    /// [`RegFrame::registers`](crate::runtime::reg::RegFrame).
    pub base_register_index: u32,
}

impl CallerBaseData for RegCallerBaseData {
    /// The outermost frame starts at 0, the bottom of an empty register file — which
    /// is what [`RuntimeFrame::reset`](crate::instruction::RuntimeFrame::reset)
    /// guarantees before every call.
    fn initial_data() -> Self {
        RegCallerBaseData {
            base_register_index: 0,
        }
    }

    /// The frame's first slot, which is all this machine needs: every [`Slot`] is an
    /// offset from it.
    fn base_offset(&self) -> u32 {
        self.base_register_index
    }
}

/// A body under construction: the instruction list and its source-offset sidecar,
/// which have to stay the same length.
///
/// The two are paired in a type rather than kept as two `Vec`s because this pass
/// appends from around thirty places — every operator arm, plus the spill and move
/// helpers — and several emit more than one instruction per operator. Nothing but
/// [`Self::push`] can change the length, so "an offset for every instruction" holds
/// by construction instead of by remembering it at each site. Diagnostics index the
/// sidecar by program counter, so a list that fell short would not be caught until a
/// trap reached the missing entry.
#[derive(Default)]
struct Instructions {
    /// The lowered body, in execution order. An index into it is a program counter.
    inner: Vec<RegInstruction>,
    /// Byte offset of the operator each instruction was lowered from, index for index
    /// with `inner`. Several instructions from one operator repeat its offset.
    offsets: Vec<u32>,
}

impl Instructions {
    /// Appends `instr`, recording `offset` as the byte offset of the operator it was
    /// lowered from. Several instructions from one operator share its offset.
    fn push(&mut self, instr: RegInstruction, offset: u32) {
        self.inner.push(instr);
        self.offsets.push(offset);
    }

    /// The index the next pushed instruction will take — which is what a block or a
    /// branch records as its target. See [`RegInstruction::spill_lazy`] on why spills
    /// have to be emitted before an arm reads this.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether anything has been emitted yet.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Instruction for RegInstruction {
    type Vm = crate::Register;
    type BrTableTarget = RegBrTableTarget;
    type FrameLayout = RegFrameLayout;
    type RuntimeFrame = RegFrame;
    type CallerBaseData = RegCallerBaseData;

    /// Lowers one function body's operator stream into register form.
    ///
    /// `params`/`results` are the body's own arity, seeding the implicit function
    /// frame at the root of the control stack. `locals_count` must cover params plus
    /// declared locals, since it sizes the lazy origin table, which is indexed
    /// unchecked. `globals_count` is unused — globals stopped being forwarded when
    /// [`RegInstruction::GlobalGet`] began materialising them eagerly — and stays on
    /// the signature only because [`Instruction`] declares it.
    ///
    /// Returns the instruction list, the byte offset of the operator each instruction
    /// was lowered from, and the [`FrameLayout`] needed to run it. The list and the
    /// layout are inseparable: every operand in the list is a frame index only the
    /// layout's region sizes can place, and every [`Id`] in it points into one of the
    /// layout's arenas.
    ///
    /// The offsets are parallel to the instructions, index for index, as the trait
    /// requires. This pass emits several instructions for some operators — a spill,
    /// a move and then the operator itself — and all of them carry that operator's
    /// offset. Both lists come from one [`Instructions`], which is what makes their
    /// lengths agree without every emit site having to.
    ///
    /// Rejects any operator the pass does not model as
    /// [`TraceWasmError::Unsupported`].
    fn emit_instructions_for_func(
        mut operator_reader: OperatorsReader<'_>,
        params: u32,
        results: u32,
        types: &[FuncType],
        func_decls: &[FuncDecl],
        locals_count: u32,
        _globals_count: u32,
    ) -> Result<RegLoweredFuncBody, TraceWasmError> {
        let mut instructions = Instructions::default();
        let mut simulated_stack = SimulatedStack::new(locals_count);
        let mut unreachable_tracking_stack = UnreachableTrackingControlStack::new();
        let mut call_instr_backpatches: Vec<(usize, u16)> = vec![];

        simulated_stack.control_stack.stack.push(Block {
            kind: BlockKind::Func,
            recorded_height: 0, // functions always have recorded height to be 0, so they leave stack with just its results
            params,
            results,
            is_unreachable_traversing: false,
            has_inherited: false,
            attached_breaks: vec![],
        });

        while !operator_reader.eof() {
            let (operator, offset) = operator_reader.read_with_offset()?;
            let offset = offset as u32;

            /// [`Self::emit`] against this body's `simulated_stack` and
            /// `instructions`, at the current operator's `offset`.
            ///
            /// Declared inside the loop rather than at function scope so it can name
            /// `offset`, which changes per operator; the other two are the same
            /// throughout. Naming all three here is what keeps an arm to one line —
            /// spelling them out per arm is the only thing that made these five lines
            /// long instead of one. Expansion is compile-time, so redeclaring it each
            /// iteration costs nothing.
            ///
            /// The argument is normally the variant itself — a tuple-variant
            /// constructor *is* a `Fn(Signature<I, O>) -> RegInstruction`, so the arity
            /// comes from the variant's own declaration and no arm restates it. Pass a
            /// closure where the variant carries an immediate beside its signature.
            // The `?` lives here rather than in each arm: every one of them reaches
            // `Self::emit`, so this is the single place a register-allocation failure
            // has to propagate from.
            macro_rules! emit {
                ($build:expr) => {
                    Self::emit(&mut simulated_stack, &mut instructions, offset, $build)?
                };
            }

            // A load or store keeps its byte offset in `memory_offsets`, so the id has
            // to be allocated before the instruction can be built — which `emit!`'s
            // closure cannot do, since it can neither borrow the stack nor propagate.
            macro_rules! emit_load {
                ($memarg:expr, $variant:ident) => {{
                    let offset = simulated_stack
                        .memory_offsets
                        .try_intern(MemoryOffset($memarg.offset as u32))?;

                    emit!(|sig| RegInstruction::$variant { offset, sig })
                }};
            }

            // As `emit_load!`, but a store has no destination, so only the operand
            // half of the signature reaches the variant. The arity is annotated
            // because discarding the output leaves `O` otherwise unconstrained.
            macro_rules! emit_store {
                ($memarg:expr, $variant:ident) => {{
                    let offset = simulated_stack
                        .memory_offsets
                        .try_intern(MemoryOffset($memarg.offset as u32))?;

                    emit!(|sig: Signature<2, 0>| RegInstruction::$variant {
                        offset,
                        input: sig.input
                    })
                }};
            }

            if !matches!(
                unreachable_tracking_stack.check_unreachablity(&operator),
                UnreachableCheckResult::Reachable
            ) {
                continue;
            }

            match operator {
                Operator::GlobalGet { global_index } => {
                    let registers = simulated_stack
                        .registers_for::<0, 1>(instructions.len(), InstructionSource::Emit)?
                        .output;

                    instructions.push(
                        RegInstruction::GlobalGet {
                            index: GlobalIndex(global_index),
                            output: registers,
                        },
                        offset,
                    );
                }
                Operator::GlobalSet { global_index } => {
                    let registers = simulated_stack
                        .registers_for::<1, 0>(instructions.len(), InstructionSource::Emit)?
                        .input;

                    instructions.push(
                        RegInstruction::GlobalSet {
                            index: GlobalIndex(global_index),
                            input: registers,
                        },
                        offset,
                    );
                }
                Operator::LocalGet { local_index } => {
                    simulated_stack.push_local(local_index as u16)?;
                }
                Operator::LocalSet { local_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        local_index,
                        &mut simulated_stack.lazy_locals,
                        &mut simulated_stack.spills,
                    )? {
                        instructions.push(
                            RegInstruction::LocalSpill {
                                index: LocalIndex(local_index),
                                spill_index,
                            },
                            offset,
                        );
                    }

                    let registers = simulated_stack
                        .registers_for::<1, 0>(instructions.len(), InstructionSource::Emit)?
                        .input;

                    instructions.push(
                        RegInstruction::LocalSet {
                            index: LocalIndex(local_index),
                            input: registers,
                        },
                        offset,
                    );
                }
                Operator::LocalTee { local_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        local_index,
                        &mut simulated_stack.lazy_locals,
                        &mut simulated_stack.spills,
                    )? {
                        instructions.push(
                            RegInstruction::LocalSpill {
                                index: LocalIndex(local_index),
                                spill_index,
                            },
                            offset,
                        );
                    }

                    let slot = simulated_stack.tee();

                    match simulated_stack.backpatch_map.0.entry(instructions.len()) {
                        Entry::Occupied(mut occ) => {
                            let v = occ.get_mut();

                            v.push((InstructionSource::Emit, vec![slot]));
                        }
                        Entry::Vacant(vac) => {
                            let v = vec![(InstructionSource::Emit, vec![slot])];

                            vac.insert(v);
                        }
                    }

                    let registers = InputRegisters {
                        registers: [Slot(u16::MAX)],
                    };

                    instructions.push(
                        RegInstruction::LocalTee {
                            index: LocalIndex(local_index),
                            input: registers,
                        },
                        offset,
                    );
                }
                Operator::RefNull { hty: _ } => {
                    simulated_stack.push_const(Const::Ref(None))?;
                }
                Operator::RefFunc { function_index } => {
                    simulated_stack.push_const(Const::Ref(Some(FuncIndex(function_index))))?;
                }
                Operator::RefIsNull => emit!(RegInstruction::RefIsNull),

                Operator::MemorySize { mem } => {
                    check_memory_index(mem)?;

                    emit!(|sig: Signature<0, 1>| { RegInstruction::MemorySize(sig.output) })
                }
                Operator::MemoryGrow { mem } => {
                    check_memory_index(mem)?;

                    emit!(RegInstruction::MemoryGrow)
                }
                Operator::MemoryCopy { dst_mem, src_mem } => {
                    check_memory_index(dst_mem)?;
                    check_memory_index(src_mem)?;

                    emit!(|sig: Signature<3, 0>| { RegInstruction::MemoryCopy(sig.input) })
                }
                Operator::MemoryFill { mem } => {
                    check_memory_index(mem)?;

                    emit!(|sig: Signature<3, 0>| { RegInstruction::MemoryFill(sig.input) })
                }
                Operator::MemoryInit { data_index, mem } => {
                    check_memory_index(mem)?;

                    let registers = simulated_stack
                        .registers_for::<3, 0>(instructions.len(), InstructionSource::Emit)?
                        .input;

                    instructions.push(
                        RegInstruction::MemoryInit(simulated_stack.memory_init_arena.alloc(
                            MemoryInitOperands {
                                data_index,
                                operands: registers,
                            },
                        )),
                        offset,
                    );
                }
                Operator::DataDrop { data_index } => {
                    instructions.push(RegInstruction::DataDrop(data_index), offset);
                }

                Operator::I32Const { value } => {
                    simulated_stack.push_const(Const::I32(value))?;
                }
                Operator::I32Load { memarg } => emit_load!(memarg, I32Load),
                Operator::I32Load8S { memarg } => emit_load!(memarg, I32Load8S),
                Operator::I32Load8U { memarg } => emit_load!(memarg, I32Load8U),
                Operator::I32Load16S { memarg } => emit_load!(memarg, I32Load16S),
                Operator::I32Load16U { memarg } => emit_load!(memarg, I32Load16U),
                Operator::I32Store { memarg } => emit_store!(memarg, I32Store),
                Operator::I32Store8 { memarg } => emit_store!(memarg, I32Store8),
                Operator::I32Store16 { memarg } => emit_store!(memarg, I32Store16),
                Operator::I32Clz => emit!(RegInstruction::I32Clz),
                Operator::I32Ctz => emit!(RegInstruction::I32Ctz),
                Operator::I32Eqz => emit!(RegInstruction::I32Eqz),
                Operator::I32Extend16S => emit!(RegInstruction::I32Extend16S),
                Operator::I32Extend8S => emit!(RegInstruction::I32Extend8S),
                Operator::I32Popcnt => emit!(RegInstruction::I32Popcnt),
                Operator::I32ReinterpretF32 => emit!(RegInstruction::I32ReinterpretF32),
                Operator::I32TruncF32S => emit!(RegInstruction::I32TruncF32S),
                Operator::I32TruncF32U => emit!(RegInstruction::I32TruncF32U),
                Operator::I32TruncF64S => emit!(RegInstruction::I32TruncF64S),
                Operator::I32TruncF64U => emit!(RegInstruction::I32TruncF64U),
                Operator::I32TruncSatF32S => emit!(RegInstruction::I32TruncSatF32S),
                Operator::I32TruncSatF32U => emit!(RegInstruction::I32TruncSatF32U),
                Operator::I32TruncSatF64S => emit!(RegInstruction::I32TruncSatF64S),
                Operator::I32TruncSatF64U => emit!(RegInstruction::I32TruncSatF64U),
                Operator::I32WrapI64 => emit!(RegInstruction::I32WrapI64),

                Operator::I32Add => emit!(RegInstruction::I32Add),
                Operator::I32And => emit!(RegInstruction::I32And),
                Operator::I32DivS => emit!(RegInstruction::I32DivS),
                Operator::I32DivU => emit!(RegInstruction::I32DivU),
                Operator::I32Eq => emit!(RegInstruction::I32Eq),
                Operator::I32GeS => emit!(RegInstruction::I32GeS),
                Operator::I32GeU => emit!(RegInstruction::I32GeU),
                Operator::I32GtS => emit!(RegInstruction::I32GtS),
                Operator::I32GtU => emit!(RegInstruction::I32GtU),
                Operator::I32LeS => emit!(RegInstruction::I32LeS),
                Operator::I32LeU => emit!(RegInstruction::I32LeU),
                Operator::I32LtS => emit!(RegInstruction::I32LtS),
                Operator::I32LtU => emit!(RegInstruction::I32LtU),
                Operator::I32Mul => emit!(RegInstruction::I32Mul),
                Operator::I32Ne => emit!(RegInstruction::I32Ne),
                Operator::I32Or => emit!(RegInstruction::I32Or),
                Operator::I32RemS => emit!(RegInstruction::I32RemS),
                Operator::I32RemU => emit!(RegInstruction::I32RemU),
                Operator::I32Rotl => emit!(RegInstruction::I32Rotl),
                Operator::I32Rotr => emit!(RegInstruction::I32Rotr),
                Operator::I32Shl => emit!(RegInstruction::I32Shl),
                Operator::I32ShrS => emit!(RegInstruction::I32ShrS),
                Operator::I32ShrU => emit!(RegInstruction::I32ShrU),
                Operator::I32Sub => emit!(RegInstruction::I32Sub),
                Operator::I32Xor => emit!(RegInstruction::I32Xor),

                Operator::I64Const { value } => {
                    simulated_stack.push_const(Const::I64(value))?;
                }
                Operator::I64Load { memarg } => emit_load!(memarg, I64Load),
                Operator::I64Load8S { memarg } => emit_load!(memarg, I64Load8S),
                Operator::I64Load8U { memarg } => emit_load!(memarg, I64Load8U),
                Operator::I64Load16S { memarg } => emit_load!(memarg, I64Load16S),
                Operator::I64Load16U { memarg } => emit_load!(memarg, I64Load16U),
                Operator::I64Load32S { memarg } => emit_load!(memarg, I64Load32S),
                Operator::I64Load32U { memarg } => emit_load!(memarg, I64Load32U),
                Operator::I64Store { memarg } => emit_store!(memarg, I64Store),
                Operator::I64Store8 { memarg } => emit_store!(memarg, I64Store8),
                Operator::I64Store16 { memarg } => emit_store!(memarg, I64Store16),
                Operator::I64Store32 { memarg } => emit_store!(memarg, I64Store32),
                Operator::I64Clz => emit!(RegInstruction::I64Clz),
                Operator::I64Ctz => emit!(RegInstruction::I64Ctz),
                Operator::I64Eqz => emit!(RegInstruction::I64Eqz),
                Operator::I64Extend16S => emit!(RegInstruction::I64Extend16S),
                Operator::I64Extend32S => emit!(RegInstruction::I64Extend32S),
                Operator::I64Extend8S => emit!(RegInstruction::I64Extend8S),
                Operator::I64ExtendI32S => emit!(RegInstruction::I64ExtendI32S),
                Operator::I64ExtendI32U => emit!(RegInstruction::I64ExtendI32U),
                Operator::I64Popcnt => emit!(RegInstruction::I64Popcnt),
                Operator::I64ReinterpretF64 => emit!(RegInstruction::I64ReinterpretF64),
                Operator::I64TruncF32S => emit!(RegInstruction::I64TruncF32S),
                Operator::I64TruncF32U => emit!(RegInstruction::I64TruncF32U),
                Operator::I64TruncF64S => emit!(RegInstruction::I64TruncF64S),
                Operator::I64TruncF64U => emit!(RegInstruction::I64TruncF64U),
                Operator::I64TruncSatF32S => emit!(RegInstruction::I64TruncSatF32S),
                Operator::I64TruncSatF32U => emit!(RegInstruction::I64TruncSatF32U),
                Operator::I64TruncSatF64S => emit!(RegInstruction::I64TruncSatF64S),
                Operator::I64TruncSatF64U => emit!(RegInstruction::I64TruncSatF64U),

                Operator::I64Add => emit!(RegInstruction::I64Add),
                Operator::I64And => emit!(RegInstruction::I64And),
                Operator::I64DivS => emit!(RegInstruction::I64DivS),
                Operator::I64DivU => emit!(RegInstruction::I64DivU),
                Operator::I64Eq => emit!(RegInstruction::I64Eq),
                Operator::I64GeS => emit!(RegInstruction::I64GeS),
                Operator::I64GeU => emit!(RegInstruction::I64GeU),
                Operator::I64GtS => emit!(RegInstruction::I64GtS),
                Operator::I64GtU => emit!(RegInstruction::I64GtU),
                Operator::I64LeS => emit!(RegInstruction::I64LeS),
                Operator::I64LeU => emit!(RegInstruction::I64LeU),
                Operator::I64LtS => emit!(RegInstruction::I64LtS),
                Operator::I64LtU => emit!(RegInstruction::I64LtU),
                Operator::I64Mul => emit!(RegInstruction::I64Mul),
                Operator::I64Ne => emit!(RegInstruction::I64Ne),
                Operator::I64Or => emit!(RegInstruction::I64Or),
                Operator::I64RemS => emit!(RegInstruction::I64RemS),
                Operator::I64RemU => emit!(RegInstruction::I64RemU),
                Operator::I64Rotl => emit!(RegInstruction::I64Rotl),
                Operator::I64Rotr => emit!(RegInstruction::I64Rotr),
                Operator::I64Shl => emit!(RegInstruction::I64Shl),
                Operator::I64ShrS => emit!(RegInstruction::I64ShrS),
                Operator::I64ShrU => emit!(RegInstruction::I64ShrU),
                Operator::I64Sub => emit!(RegInstruction::I64Sub),
                Operator::I64Xor => emit!(RegInstruction::I64Xor),

                Operator::F32Const { value } => {
                    simulated_stack
                        .push_const(Const::F32(OrderedFloat(f32::from_bits(value.bits()))))?;
                }
                Operator::F32Load { memarg } => emit_load!(memarg, F32Load),
                Operator::F32Store { memarg } => emit_store!(memarg, F32Store),
                Operator::F32Abs => emit!(RegInstruction::F32Abs),
                Operator::F32Ceil => emit!(RegInstruction::F32Ceil),
                Operator::F32ConvertI32S => emit!(RegInstruction::F32ConvertI32S),
                Operator::F32ConvertI32U => emit!(RegInstruction::F32ConvertI32U),
                Operator::F32ConvertI64S => emit!(RegInstruction::F32ConvertI64S),
                Operator::F32ConvertI64U => emit!(RegInstruction::F32ConvertI64U),
                Operator::F32DemoteF64 => emit!(RegInstruction::F32DemoteF64),
                Operator::F32Floor => emit!(RegInstruction::F32Floor),
                Operator::F32Nearest => emit!(RegInstruction::F32Nearest),
                Operator::F32Neg => emit!(RegInstruction::F32Neg),
                Operator::F32ReinterpretI32 => emit!(RegInstruction::F32ReinterpretI32),
                Operator::F32Sqrt => emit!(RegInstruction::F32Sqrt),
                Operator::F32Trunc => emit!(RegInstruction::F32Trunc),

                Operator::F32Add => emit!(RegInstruction::F32Add),
                Operator::F32Copysign => emit!(RegInstruction::F32Copysign),
                Operator::F32Div => emit!(RegInstruction::F32Div),
                Operator::F32Eq => emit!(RegInstruction::F32Eq),
                Operator::F32Ge => emit!(RegInstruction::F32Ge),
                Operator::F32Gt => emit!(RegInstruction::F32Gt),
                Operator::F32Le => emit!(RegInstruction::F32Le),
                Operator::F32Lt => emit!(RegInstruction::F32Lt),
                Operator::F32Max => emit!(RegInstruction::F32Max),
                Operator::F32Min => emit!(RegInstruction::F32Min),
                Operator::F32Mul => emit!(RegInstruction::F32Mul),
                Operator::F32Ne => emit!(RegInstruction::F32Ne),
                Operator::F32Sub => emit!(RegInstruction::F32Sub),

                Operator::F64Const { value } => {
                    simulated_stack
                        .push_const(Const::F64(OrderedFloat(f64::from_bits(value.bits()))))?;
                }
                Operator::F64Load { memarg } => emit_load!(memarg, F64Load),
                Operator::F64Store { memarg } => emit_store!(memarg, F64Store),
                Operator::F64Abs => emit!(RegInstruction::F64Abs),
                Operator::F64Ceil => emit!(RegInstruction::F64Ceil),
                Operator::F64ConvertI32S => emit!(RegInstruction::F64ConvertI32S),
                Operator::F64ConvertI32U => emit!(RegInstruction::F64ConvertI32U),
                Operator::F64ConvertI64S => emit!(RegInstruction::F64ConvertI64S),
                Operator::F64ConvertI64U => emit!(RegInstruction::F64ConvertI64U),
                Operator::F64Floor => emit!(RegInstruction::F64Floor),
                Operator::F64Nearest => emit!(RegInstruction::F64Nearest),
                Operator::F64Neg => emit!(RegInstruction::F64Neg),
                Operator::F64PromoteF32 => emit!(RegInstruction::F64PromoteF32),
                Operator::F64ReinterpretI64 => emit!(RegInstruction::F64ReinterpretI64),
                Operator::F64Sqrt => emit!(RegInstruction::F64Sqrt),
                Operator::F64Trunc => emit!(RegInstruction::F64Trunc),

                Operator::F64Add => emit!(RegInstruction::F64Add),
                Operator::F64Copysign => emit!(RegInstruction::F64Copysign),
                Operator::F64Div => emit!(RegInstruction::F64Div),
                Operator::F64Eq => emit!(RegInstruction::F64Eq),
                Operator::F64Ge => emit!(RegInstruction::F64Ge),
                Operator::F64Gt => emit!(RegInstruction::F64Gt),
                Operator::F64Le => emit!(RegInstruction::F64Le),
                Operator::F64Lt => emit!(RegInstruction::F64Lt),
                Operator::F64Max => emit!(RegInstruction::F64Max),
                Operator::F64Min => emit!(RegInstruction::F64Min),
                Operator::F64Mul => emit!(RegInstruction::F64Mul),
                Operator::F64Ne => emit!(RegInstruction::F64Ne),
                Operator::F64Sub => emit!(RegInstruction::F64Sub),
                Operator::Nop => {
                    continue;
                }
                Operator::Select => {
                    let registers = simulated_stack
                        .registers_for::<3, 1>(instructions.len(), InstructionSource::Emit)?;

                    instructions.push(
                        RegInstruction::Select(
                            simulated_stack
                                .select_arena
                                .alloc(SelectOperands(registers)),
                        ),
                        offset,
                    );
                }
                Operator::Drop => {
                    simulated_stack.pop();

                    continue;
                }
                Operator::Block { blockty } => {
                    let (block_params, _) = simulated_stack.add_block(
                        BlockVariant::Block,
                        &blockty,
                        types,
                        instructions.len(),
                    );

                    if block_params != 0 {
                        let move_registers = simulated_stack.materialize_stack_slots_in_registers(
                            block_params,
                            instructions.len(),
                            InstructionSource::Emit,
                        )?;

                        instructions.push(
                            RegInstruction::Move(
                                simulated_stack.dyn_signatures.alloc(move_registers),
                            ),
                            offset,
                        );
                    }
                }
                Operator::Loop { blockty } => {
                    // A loop repeats: a rescue left inside the body would re-run on
                    // the back-edge and capture what the previous iteration wrote.
                    // Hoisting it above the header runs it exactly once, on entry.
                    Self::spill_live_locals(&mut simulated_stack, &mut instructions, offset)?;

                    let (block_params, _) = simulated_stack.add_block(
                        BlockVariant::Loop,
                        &blockty,
                        types,
                        instructions.len(),
                    );

                    if block_params != 0 {
                        let move_registers = simulated_stack.materialize_stack_slots_in_registers(
                            block_params,
                            instructions.len(),
                            InstructionSource::Emit,
                        )?;

                        instructions.push(
                            RegInstruction::Move(
                                simulated_stack.dyn_signatures.alloc(move_registers),
                            ),
                            offset,
                        );
                    }

                    instructions.push(RegInstruction::Loop, offset);
                }
                Operator::If { blockty } => {
                    // Control diverges here, and a write in either arm would leave
                    // the other path reading a spill slot nothing wrote.
                    Self::spill_live_locals(&mut simulated_stack, &mut instructions, offset)?;

                    // the simulated stack would have layout like this at `if` instruction: [...other...][...params...][cond]
                    // to obtain recorded_height, we should pop params + 1 number of stack slots, and measure the `curr_register_index`
                    // which will be the recorded_height. So after the `end` of this `if` we should leave the stack
                    // at: recorded_height + results. We should materalize all the popped values by pushing it back on the stack
                    // making them materialized in registers. Same would be for results. So no matter what branch control flow takes
                    // the layout of frame in the start of the instruction and at the end of the instruction is same.

                    let (block_params, _) = simulated_stack.add_block(
                        BlockVariant::If,
                        &blockty,
                        types,
                        instructions.len(),
                    );

                    if block_params != 0 {
                        let move_registers = simulated_stack.materialize_stack_slots_in_registers(
                            block_params + 1,
                            instructions.len(),
                            InstructionSource::Emit,
                        )?;

                        instructions.push(
                            RegInstruction::Move(
                                simulated_stack.dyn_signatures.alloc(move_registers),
                            ),
                            offset,
                        );
                    }

                    let cond = simulated_stack
                        .registers_for::<1, 0>(instructions.len(), InstructionSource::Emit)?
                        .input;

                    instructions.push(
                        RegInstruction::If(simulated_stack.if_arena.alloc(IfOperands {
                            cond,
                            else_index: None,
                            end_index: u32::MAX,
                        })),
                        offset,
                    );
                }
                Operator::Else => {
                    let if_block = simulated_stack.get_curr_block_mut();
                    let recorded_height = if_block.recorded_height;
                    let block_params = if_block.params;
                    let block_results = if_block.results;

                    let BlockKind::If {
                        index: _,
                        else_index,
                    } = &mut if_block.kind
                    else {
                        unreachable!(
                            "hitting this means TraceWasm has a bug recording the instructions"
                        )
                    };

                    *else_index = Some(if block_results == 0 {
                        instructions.len() as u32
                    } else {
                        instructions.len() as u32 + 1 // mov instruction also emitted!
                    });

                    // materialize the results produced by the if arm.
                    // The else instruction is only reached by the if arm, if the condition is false
                    // then the pc is jumped directly to the first instruction of the else arm skipping
                    // the else instruction itself.
                    if block_results != 0 {
                        let move_registers = simulated_stack.materialize_stack_slots_in_registers(
                            block_results,
                            instructions.len(),
                            InstructionSource::Emit,
                        )?;

                        instructions.push(
                            RegInstruction::Move(
                                simulated_stack.dyn_signatures.alloc(move_registers),
                            ),
                            offset,
                        );
                    }

                    // reset the frame layout with params on top for else instructions.
                    simulated_stack.pops_and_pushes(
                        simulated_stack.stack.height() - recorded_height,
                        block_params,
                    )?;

                    instructions.push(
                        RegInstruction::Else {
                            end_index: u32::MAX,
                        },
                        offset,
                    );
                }
                Operator::Br { relative_depth } => {
                    let block_index =
                        simulated_stack.control_stack.len() - 1 - relative_depth as usize;
                    let block = simulated_stack.get_block(block_index);
                    let params = block.params;
                    let results = block.results;
                    let recorded_height = block.recorded_height;

                    let (move_registers, target_index) =
                        if let Some(loop_index) = block.kind.is_loop() {
                            (
                                simulated_stack.br_truncation_registers(
                                    recorded_height,
                                    params,
                                    instructions.len(),
                                    InstructionSource::Emit,
                                )?,
                                loop_index,
                            )
                        } else {
                            let move_registers = simulated_stack.br_truncation_registers(
                                recorded_height,
                                results,
                                instructions.len(),
                                InstructionSource::Emit,
                            )?;

                            simulated_stack
                                .get_block_mut(block_index)
                                .attached_breaks
                                .push((
                                    if move_registers.is_empty() {
                                        instructions.len() as u32
                                    } else {
                                        instructions.len() as u32 + 1
                                    },
                                    u32::MAX,
                                ));

                            (move_registers, u32::MAX)
                        };

                    if !move_registers.is_empty() {
                        instructions.push(
                            RegInstruction::Move(
                                simulated_stack.dyn_signatures.alloc(move_registers),
                            ),
                            offset,
                        );
                    }

                    instructions.push(RegInstruction::Br { target_index }, offset);

                    simulated_stack.reset_enclosing_block_layout()?;
                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::BrIf { relative_depth } => {
                    // Taking the branch skips everything after it, including any
                    // write that would have rescued a borrow still live here.
                    Self::spill_live_locals(&mut simulated_stack, &mut instructions, offset)?;

                    let block_index =
                        simulated_stack.control_stack.len() - 1 - relative_depth as usize;
                    let block = simulated_stack.get_block(block_index);
                    let params = block.params;
                    let results = block.results;
                    let recorded_height = block.recorded_height;
                    let block_kind = block.kind;

                    let cond = simulated_stack
                        .registers_for::<1, 0>(instructions.len(), InstructionSource::BrIfCond)?
                        .input;

                    let (move_registers, target_index) =
                        if let Some(loop_index) = block_kind.is_loop() {
                            (
                                simulated_stack.br_truncation_registers(
                                    recorded_height,
                                    params,
                                    instructions.len(),
                                    InstructionSource::BrIfMov,
                                )?,
                                loop_index,
                            )
                        } else {
                            let move_registers = simulated_stack.br_truncation_registers(
                                recorded_height,
                                results,
                                instructions.len(),
                                InstructionSource::BrIfMov,
                            )?;

                            simulated_stack
                                .get_block_mut(block_index)
                                .attached_breaks
                                .push((instructions.len() as u32, u32::MAX));

                            (move_registers, u32::MAX)
                        };

                    instructions.push(
                        RegInstruction::BrIf(simulated_stack.br_if_arena.alloc(BrIfOperands {
                            cond,
                            mov: move_registers,
                            target_index,
                        })),
                        offset,
                    );
                }
                Operator::BrTable { targets: table } => {
                    // As for `br_if`: every arm jumps away, so a later write cannot
                    // be relied on to have rescued anything.
                    Self::spill_live_locals(&mut simulated_stack, &mut instructions, offset)?;

                    let mut br_targets = vec![];
                    let targets = table.targets();
                    let mut targets = targets.collect::<Result<Vec<_>, _>>()?;

                    targets.push(table.default());

                    let table_index = simulated_stack
                        .registers_for::<1, 0>(instructions.len(), InstructionSource::BrTableIndex)?
                        .input; // targets index

                    for (i, &relative_depth) in targets.iter().enumerate() {
                        let block_index =
                            simulated_stack.control_stack.len() - 1 - relative_depth as usize;
                        let block = simulated_stack.get_block_mut(block_index);
                        let params = block.params;
                        let results = block.results;
                        let recorded_height = block.recorded_height;
                        let block_kind = block.kind;

                        let (move_registers, target_index) =
                            if let Some(loop_index) = block_kind.is_loop() {
                                (
                                    simulated_stack.br_truncation_registers(
                                        recorded_height,
                                        params,
                                        instructions.len(),
                                        InstructionSource::BrTableMov,
                                    )?,
                                    loop_index,
                                )
                            } else {
                                let move_registers = simulated_stack.br_truncation_registers(
                                    recorded_height,
                                    results,
                                    instructions.len(),
                                    InstructionSource::BrTableMov,
                                )?;

                                simulated_stack
                                    .get_block_mut(block_index)
                                    .attached_breaks
                                    .push((instructions.len() as u32, i as u32));

                                (move_registers, u32::MAX)
                            };

                        let br_target = RegBrTableTarget {
                            mov: move_registers,
                            target_index,
                        };

                        br_targets.push(br_target);
                    }

                    instructions.push(
                        RegInstruction::BrTable(simulated_stack.br_table_arena.alloc(
                            BrTableOperands {
                                index: table_index,
                                br_targets,
                            },
                        )),
                        offset,
                    );

                    simulated_stack.reset_enclosing_block_layout()?;
                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::Return => {
                    let move_registers = simulated_stack.br_truncation_registers(
                        0,
                        results,
                        instructions.len(),
                        InstructionSource::Emit,
                    )?;

                    simulated_stack.control_stack.stack[0]
                        .attached_breaks
                        .push((
                            if move_registers.is_empty() {
                                instructions.len() as u32
                            } else {
                                instructions.len() as u32 + 1
                            },
                            u32::MAX,
                        ));

                    if !move_registers.is_empty() {
                        instructions.push(
                            RegInstruction::Move(
                                simulated_stack.dyn_signatures.alloc(move_registers),
                            ),
                            offset,
                        );
                    }

                    instructions.push(
                        RegInstruction::Return {
                            target_index: u32::MAX,
                        },
                        offset,
                    );

                    simulated_stack.reset_enclosing_block_layout()?;
                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::Call { function_index } => {
                    let func_decl = &func_decls[function_index as usize];
                    let func_ty = &types[func_decl.ty.0 as usize];
                    let params = func_ty.params.len() as u32;
                    let results = func_ty.results.len() as u32;
                    let recorded_height = simulated_stack.stack.height() - params;
                    let caller_base = simulated_stack.register_index_at_depth(params) as u16;

                    if params != 0 {
                        let move_registers = simulated_stack.materialize_stack_slots_in_registers(
                            params,
                            instructions.len(),
                            InstructionSource::Emit,
                        )?;

                        instructions.push(
                            RegInstruction::Move(
                                simulated_stack.dyn_signatures.alloc(move_registers),
                            ),
                            offset,
                        );
                    }

                    call_instr_backpatches.push((instructions.len(), caller_base));

                    instructions.push(
                        RegInstruction::Call {
                            func_index: FuncIndex(function_index),
                            caller_base: u16::MAX, // will be backpatched!
                        },
                        offset,
                    );

                    simulated_stack.pops_and_pushes(
                        simulated_stack.stack.height() - recorded_height,
                        results,
                    )?;
                }
                Operator::CallIndirect {
                    type_index,
                    table_index,
                } => {
                    let ty = &types[type_index as usize];
                    let params = ty.params.len() as u32;
                    let results = ty.results.len() as u32;
                    let recorded_height = simulated_stack.stack.height() - params - 1;
                    let caller_base = simulated_stack.register_index_at_depth(params + 1) as u16;

                    let slot = simulated_stack
                        .registers_for::<1, 0>(
                            instructions.len(),
                            InstructionSource::CallIndirectSlot,
                        )?
                        .input;

                    let move_registers = simulated_stack.materialize_stack_slots_in_registers(
                        params,
                        instructions.len(),
                        InstructionSource::CallIndirectMov,
                    )?;

                    call_instr_backpatches.push((instructions.len(), caller_base));

                    instructions.push(
                        RegInstruction::CallIndirect(simulated_stack.call_indirect_arena.alloc(
                            CallIndirectOperands {
                                ty_index: TyIndex(type_index),
                                table_index: TableIndex(table_index),
                                slot,
                                operands: move_registers,
                                caller_base: u16::MAX,
                            },
                        )),
                        offset,
                    );

                    simulated_stack.pops_and_pushes(
                        simulated_stack.stack.height() - recorded_height,
                        results,
                    )?;
                }
                Operator::Unreachable => {
                    instructions.push(RegInstruction::Unreachable, offset);

                    simulated_stack.reset_enclosing_block_layout()?;
                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::End => {
                    let block = simulated_stack.pop_block();
                    let results = block.results;
                    let attached_breaks = &block.attached_breaks;

                    if results != 0 {
                        let move_registers = simulated_stack.materialize_stack_slots_in_registers(
                            results,
                            instructions.len(),
                            InstructionSource::Emit,
                        )?;

                        instructions.push(
                            RegInstruction::Move(
                                simulated_stack.dyn_signatures.alloc(move_registers),
                            ),
                            offset,
                        );
                    }

                    debug_assert!(
                        simulated_stack.stack.height() == block.recorded_height + results
                    );

                    let index = instructions.len() as u32;

                    for (br_index, br_targets_index) in attached_breaks {
                        match &mut instructions.inner[*br_index as usize] {
                            RegInstruction::Br { target_index } => {
                                *target_index = index;
                            }
                            RegInstruction::BrIf(id) => {
                                let entry = simulated_stack.br_if_arena.get_mut(*id);

                                entry.target_index = index;
                            }
                            RegInstruction::BrTable(id) => {
                                let entry = simulated_stack.br_table_arena.get_mut(*id);
                                // Each table owns its arms, so the recorded index is
                                // the arm's position within this table.
                                entry.br_targets[*br_targets_index as usize].target_index = index;
                            }
                            RegInstruction::Return { target_index } => {
                                *target_index = index;
                            }
                            _ => unreachable!(
                                "hitting this means TraceWasm has a bug recording the instructions"
                            ),
                        }
                    }

                    // Backpatch this block's own structural indices. `func`/`loop` need none: a function's
                    // `end` is not referenced by index, and a loop's branch target is its start, not its end.
                    match block.kind {
                        BlockKind::Func | BlockKind::Loop { .. } => {}
                        BlockKind::Block => {} // no backpatching require
                        BlockKind::If {
                            index: if_index,
                            else_index: ei,
                        } => {
                            // Fill the `if`'s `else_index` and `end_index` ...
                            let RegInstruction::If(id) = &mut instructions.inner[if_index as usize]
                            else {
                                unreachable!(
                                    "hitting this means TraceWasm has a bug recording the instructions"
                                )
                            };

                            let entry = simulated_stack.if_arena.get_mut(*id);

                            entry.else_index = ei;
                            entry.end_index = index;

                            // ... and point the `else` (if present) at this same `end`, so a then-branch
                            // that falls through into `else` knows where the construct closes.
                            if let Some(else_index) = ei {
                                let RegInstruction::Else { end_index } =
                                    &mut instructions.inner[else_index as usize]
                                else {
                                    unreachable!(
                                        "hitting this means TraceWasm has a bug recording the instructions"
                                    )
                                };

                                *end_index = index;
                            }
                        }
                    }

                    instructions.push(RegInstruction::End, offset);
                }
                _ => {
                    return Err(TraceWasmError::Unsupported(format!(
                        "instruction `{:?}`",
                        operator
                    )));
                }
            }
        }

        // ---- Resolving the frame layout -------------------------------------
        //
        // Only now are all three region sizes known, so only now can an operand be
        // given a frame index. Walking the body could not do it: a constant or spill
        // sits below the operand registers, so placing either needs the *final*
        // count of both, and a register index has to clear both regions, so placing
        // one needs their totals too. Every operand was therefore built as a
        // `Slot(u16::MAX)` placeholder, with the real one recorded in the backpatch
        // map under the index of the instruction carrying it.
        //
        // The frame these produce is:
        //
        //   | locals | consts | spills | registers ... |
        //
        let locals_count = simulated_stack.lazy_locals.origin.len() as u16;
        let spills = simulated_stack.spills.allocation_len();
        let consts_len = simulated_stack.const_interner.len() as u16;

        // The per-region limits above each bound one region; this bounds the frame,
        // which is all four laid end to end. Three individually-legal regions can
        // still add up to something no 16-bit index can name, and this runs before
        // the backpatch loops below so no such index is ever written.
        let frame_width = simulated_stack.max_registers as u32 + spills as u32 + consts_len as u32;

        if frame_width > u16::MAX as u32 {
            return Err(TraceWasmError::RegisterFrameTooLarge {
                what: "the whole frame",
                needed: frame_width,
                limit: u16::MAX as u32,
            });
        }

        // A `caller_base` is an operand register index too — the one the arguments
        // were staged at — so it takes the same shift. The consequence is the
        // invariant the whole layout rests on: `caller_base` ends up at or above
        // `locals_count + consts + spills`, so a callee's frame, which begins there,
        // starts above everything its caller must keep across the call.
        for (instr_index, relative_caller_base) in call_instr_backpatches {
            let instr = &mut instructions.inner[instr_index];

            match instr {
                RegInstruction::Call {
                    func_index: _,
                    caller_base,
                } => *caller_base = relative_caller_base + spills + consts_len,
                RegInstruction::CallIndirect(id) => {
                    let entry = simulated_stack.call_indirect_arena.get_mut(*id);

                    entry.caller_base = relative_caller_base + spills + consts_len;
                }
                _ => unreachable!(
                    "instruction backpatching is inserted only for call and call_indirect instruction!"
                ),
            }
        }

        // Both counts are already high-water marks — `max_registers` is raised by
        // `advanced_register_index` and by `br_truncation_registers`, and
        // `allocation_len` only grows when no freed spill slot can be reused — so they
        // are read off directly rather than recomputed from the instruction list.
        //
        // `registers` goes across verbatim, still counting from the frame base and so
        // still including the locals. That is what makes the frame
        // `registers + spills + consts` wide, and `locals_count` and `consts` travel
        // with it because they are what invert the shifts above: given an index, they
        // are the only way to say which region it names.
        let mut frame = RegFrameLayout {
            registers: simulated_stack.max_registers,
            spills,
            if_arena: simulated_stack.if_arena,
            br_if_arena: simulated_stack.br_if_arena,
            br_table_arena: simulated_stack.br_table_arena,
            call_indirect_arena: simulated_stack.call_indirect_arena,
            select_arena: simulated_stack.select_arena,
            memory_init_arena: simulated_stack.memory_init_arena,
            dyn_signatures: simulated_stack.dyn_signatures,
            memory_offsets: simulated_stack.memory_offsets,
            locals_count,
            consts: simulated_stack
                .const_interner
                .into_values()
                .into_boxed_slice(),
        };

        // apply all backpatches
        simulated_stack.backpatch_map.apply(
            &mut instructions.inner,
            locals_count,
            consts_len,
            spills,
            &mut frame,
        );

        Ok((instructions.inner, instructions.offsets, frame))
    }

    #[inline(always)]
    fn execute<M: crate::memory::Memory, I: crate::instance::traits::ImportRegistry>(
        &self,
        module: &crate::module::Module<Self::Vm>,
        instance: &mut crate::instance::Instance<M, I, Self::Vm>,
        frame_layout: &Self::FrameLayout,
        caller_base_data: &Self::CallerBaseData,
        imported_func_count: u32,
    ) -> Result<crate::runtime::Step<Self>, Box<crate::error::InstructionExecutionError>> {
        let res = match self {
            RegInstruction::GlobalGet { index, output } => {
                Self::set_value_to_register(
                    output.start,
                    instance.global_vals[index.0 as usize].into(),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::GlobalSet { index, input } => {
                let ty = module.globals[index.0 as usize].ty.content_type();

                instance.global_vals[index.0 as usize] =
                    Self::slot_value(input.registers[0], caller_base_data, instance).into_val(&ty);

                Step::Next
            }
            RegInstruction::LocalSet { index, input } => {
                Self::set_local(
                    *index,
                    Self::slot_value(input.registers[0], caller_base_data, instance),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::LocalSpill { index, spill_index } => {
                Self::set_value_to_spills(
                    spill_index.raw_value(),
                    Self::local(*index, caller_base_data, instance),
                    caller_base_data,
                    frame_layout,
                    instance,
                );

                Step::Next
            }
            RegInstruction::LocalTee { index, input } => {
                Self::set_local(
                    *index,
                    Self::slot_value(input.registers[0], caller_base_data, instance),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::Loop => Step::Next,
            RegInstruction::If(id) => {
                let entry = frame_layout.if_arena.get(*id);
                let cond =
                    Self::slot_value(entry.cond.registers[0], caller_base_data, instance).as_i32();

                if cond != 0 {
                    Step::Next
                } else {
                    if let Some(else_index) = entry.else_index {
                        Step::JumpTo(else_index + 1) // first instruction of the else branch
                    } else {
                        Step::JumpTo(entry.end_index)
                    }
                }
            }
            RegInstruction::Else { end_index } => Step::JumpTo(*end_index),
            RegInstruction::Br { target_index } => Step::JumpTo(*target_index),
            RegInstruction::BrIf(id) => {
                let entry = frame_layout.br_if_arena.get(*id);

                let cond =
                    Self::slot_value(entry.cond.registers[0], caller_base_data, instance).as_i32();

                if cond != 0 {
                    Self::execute_mov(&entry.mov, caller_base_data, instance);

                    Step::JumpTo(entry.target_index)
                } else {
                    Step::Next
                }
            }
            RegInstruction::BrTable(id) => {
                let entry = frame_layout.br_table_arena.get(*id);
                let targets = &entry.br_targets;

                let index = Self::slot_value(entry.index.registers[0], caller_base_data, instance)
                    .as_i32() as u32 as usize;

                let target_count = targets.len() - 1;

                let branch = if target_count <= index {
                    &targets[target_count] // always the last element of targets
                } else {
                    &targets[index]
                };

                Self::execute_mov(&branch.mov, caller_base_data, instance);

                Step::JumpTo(branch.target_index)
            }
            RegInstruction::I32Load { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_i32(effective_offset)?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Load8S { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_i8(effective_offset)? as i32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Load8U { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_u8(effective_offset)? as i32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Load16S { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_i16(effective_offset)? as i32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Load16U { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_u16(effective_offset)? as i32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Store { offset, input } => {
                let input = &input.registers;
                let val = Self::slot_value(input[1], caller_base_data, instance).as_i32();

                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    input[0],
                    caller_base_data,
                    instance,
                )?;

                instance.memory.write_u32(effective_offset, val as u32)?;

                Step::Next
            }
            RegInstruction::I32Store8 { offset, input } => {
                let inputs = &input.registers;
                let val = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    inputs[0],
                    caller_base_data,
                    instance,
                )?;

                instance.memory.write_u8(effective_offset, val as u8)?;

                Step::Next
            }
            RegInstruction::I32Store16 { offset, input } => {
                let inputs = &input.registers;
                let val = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    inputs[0],
                    caller_base_data,
                    instance,
                )?;

                instance.memory.write_u16(effective_offset, val as u16)?;

                Step::Next
            }
            RegInstruction::I32Clz(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.leading_zeros() as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Ctz(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.trailing_zeros() as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Eqz(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(if a == 0 { 1 } else { 0 }),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Extend16S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a as i16 as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Extend8S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a as i8 as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Popcnt(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32();

                // Counts set bits in the two's-complement representation, so a
                // negative operand counts its sign bits too — which is what the
                // spec's bit-level definition asks for.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.count_ones() as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32ReinterpretF32(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.to_bits() as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32TruncF32S(sig) => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_f32() as f64;

                let truncated = trunc_float_to_int(a, I32_TRUNC_LOW, I32_TRUNC_HIGH, "i32")?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(truncated as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32TruncF32U(sig) => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_f32() as f64;

                let truncated = trunc_float_to_int(a, 0.0, U32_TRUNC_HIGH, "u32")?;

                // The result is the `u32` bit pattern held in an `i32`, so values
                // above `i32::MAX` come back out negative.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(truncated as u32 as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32TruncF64S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                let truncated = trunc_float_to_int(a, I32_TRUNC_LOW, I32_TRUNC_HIGH, "i32")?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(truncated as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32TruncF64U(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                let truncated = trunc_float_to_int(a, 0.0, U32_TRUNC_HIGH, "u32")?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(truncated as u32 as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32TruncSatF32S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32TruncSatF32U(sig) => {
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_f32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32TruncSatF64S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32TruncSatF64U(sig) => {
                // Saturate to `u32`, the *target* width — going through `u64` here
                // would clamp at the wrong bound and then wrap on the way down.
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_f64() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32WrapI64(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Add(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.wrapping_add(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32And(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.bitand(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32DivS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.checked_div(b).ok_or(
                        InstructionExecutionError::Division {
                            num: a.to_string(),
                            deno: b.to_string(),
                        },
                    )?),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32DivU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(
                        a.checked_div(b)
                            .ok_or(InstructionExecutionError::Division {
                                num: a.to_string(),
                                deno: b.to_string(),
                            })? as i32,
                    ),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Eq(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a == b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32GeS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a >= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32GeU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a >= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32GtS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a > b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32GtU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a > b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32LeS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a <= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32LeU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a <= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32LtS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a < b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32LtU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a < b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Mul(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.wrapping_mul(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Ne(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a != b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Or(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.bitor(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32RemS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                // A zero divisor is the *only* trap here. Unlike `i32.div_s`,
                // `rem_s` does not trap on overflow: the spec defines
                // `i32::MIN % -1` as `0`, which is what `wrapping_rem` returns.
                // `checked_rem` would wrongly report that case as a failure.
                if b == 0 {
                    return Err(Box::new(InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    }));
                }

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.wrapping_rem(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32RemU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.checked_rem(b).ok_or(
                        InstructionExecutionError::Remainder {
                            left: a.to_string(),
                            right: b.to_string(),
                        },
                    )? as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Rotl(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.rotate_left(b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Rotr(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.rotate_right(b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Shl(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.wrapping_shl(b as u32)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32ShrS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                // Arithmetic shift: on `i32` the sign bit is replicated.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.wrapping_shr(b as u32)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32ShrU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32;

                // Logical shift: done on `u32` so the vacated high bits are zeros.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.wrapping_shr(b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Sub(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.wrapping_sub(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I32Xor(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(a.bitxor(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Load { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_i64(effective_offset)?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Load8S { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_i8(effective_offset)? as i64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Load8U { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_u8(effective_offset)? as i64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Load16S { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_i16(effective_offset)? as i64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Load16U { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_u16(effective_offset)? as i64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Load32S { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_i32(effective_offset)? as i64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Load32U { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_u32(effective_offset)? as i64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Store { offset, input } => {
                let inputs = &input.registers;
                let val = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    inputs[0],
                    caller_base_data,
                    instance,
                )?;

                instance.memory.write_u64(effective_offset, val as u64)?;

                Step::Next
            }
            RegInstruction::I64Store8 { offset, input } => {
                let inputs = &input.registers;
                let val = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    inputs[0],
                    caller_base_data,
                    instance,
                )?;

                instance.memory.write_u8(effective_offset, val as u8)?;

                Step::Next
            }
            RegInstruction::I64Store16 { offset, input } => {
                let inputs = &input.registers;
                let val = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    inputs[0],
                    caller_base_data,
                    instance,
                )?;

                instance.memory.write_u16(effective_offset, val as u16)?;

                Step::Next
            }
            RegInstruction::I64Store32 { offset, input } => {
                let inputs = &input.registers;
                let val = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    inputs[0],
                    caller_base_data,
                    instance,
                )?;

                instance.memory.write_u32(effective_offset, val as u32)?;

                Step::Next
            }
            RegInstruction::I64Clz(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.leading_zeros() as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Ctz(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.trailing_zeros() as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Eqz(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(if a == 0 { 1 } else { 0 }),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Extend16S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a as i16 as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Extend32S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a as i32 as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Extend8S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a as i8 as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64ExtendI32S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64ExtendI32U(sig) => {
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Popcnt(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                // See `I32Popcnt`. The count is at most 64, but the result type is
                // `i64` — unary integer ops keep their operand's width, unlike the
                // comparisons.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.count_ones() as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64ReinterpretF64(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.to_bits() as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64TruncF32S(sig) => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_f32() as f64;

                let truncated = trunc_float_to_int(a, I64_TRUNC_LOW, I64_TRUNC_HIGH, "i64")?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(truncated as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64TruncF32U(sig) => {
                // `f32` promotes to `f64` losslessly, keeping the bounds exact.
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_f32() as f64;

                let truncated = trunc_float_to_int(a, 0.0, U64_TRUNC_HIGH, "u64")?;

                // As with the `i32` forms, the result is the unsigned bit pattern
                // held in a signed value.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(truncated as u64 as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64TruncF64S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                let truncated = trunc_float_to_int(a, I64_TRUNC_LOW, I64_TRUNC_HIGH, "i64")?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(truncated as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64TruncF64U(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                let truncated = trunc_float_to_int(a, 0.0, U64_TRUNC_HIGH, "u64")?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(truncated as u64 as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64TruncSatF32S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64TruncSatF32U(sig) => {
                // Saturate to `u64`, the *target* width — clamping at `u32::MAX`
                // first would lose every value an `i64` can still represent.
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_f32() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64TruncSatF64S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64TruncSatF64U(sig) => {
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_f64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Add(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.wrapping_add(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64And(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.bitand(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64DivS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.checked_div(b).ok_or(
                        InstructionExecutionError::Division {
                            num: a.to_string(),
                            deno: b.to_string(),
                        },
                    )?),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64DivU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64() as u64;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(
                        a.checked_div(b)
                            .ok_or(InstructionExecutionError::Division {
                                num: a.to_string(),
                                deno: b.to_string(),
                            })? as i64,
                    ),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Eq(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a == b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64GeS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a >= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64GeU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64() as u64;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a >= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64GtS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a > b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64GtU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64() as u64;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a > b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64LeS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a <= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64LeU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64() as u64;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a <= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64LtS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a < b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64LtU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64() as u64;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a < b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Mul(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.wrapping_mul(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Ne(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a != b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Or(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.bitor(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64RemS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                // See `I32RemS`: only a zero divisor traps; `i64::MIN % -1` is `0`.
                if b == 0 {
                    return Err(Box::new(InstructionExecutionError::Remainder {
                        left: a.to_string(),
                        right: b.to_string(),
                    }));
                }

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.wrapping_rem(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64RemU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64() as u64;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.checked_rem(b).ok_or(
                        InstructionExecutionError::Remainder {
                            left: a.to_string(),
                            right: b.to_string(),
                        },
                    )? as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Rotl(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64() as u64;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.rotate_left(b as u32) as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Rotr(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64() as u64;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.rotate_right(b as u32) as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Shl(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.wrapping_shl(b as u32)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64ShrS(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                // Arithmetic shift: on `i64` the sign bit is replicated.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.wrapping_shr(b as u32)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64ShrU(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64() as u64;
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64() as u64;

                // Logical shift: done on `u64` so the vacated high bits are zeros.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.wrapping_shr(b as u32) as i64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Sub(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.wrapping_sub(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::I64Xor(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_i64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i64(a.bitxor(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Load { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;
                let val = instance.memory.read_f32(effective_offset)?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Store { offset, input } => {
                let inputs = &input.registers;
                let val = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    inputs[0],
                    caller_base_data,
                    instance,
                )?;

                instance.memory.write_f32(effective_offset, val)?;

                Step::Next
            }
            RegInstruction::F32Abs(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a.abs()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Ceil(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a.ceil()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32ConvertI32S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a as f32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32ConvertI32U(sig) => {
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a as f32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32ConvertI64S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a as f32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32ConvertI64U(sig) => {
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a as f32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32DemoteF64(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a as f32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Floor(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a.floor()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Nearest(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a.round_ties_even()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Neg(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a.neg()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32ReinterpretI32(sig) => {
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(f32::from_bits(a)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Sqrt(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a.sqrt()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Trunc(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a.trunc()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Add(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a + b),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Copysign(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                // Purely a sign-bit transplant: the magnitude of `a` with the sign
                // of `b`. Defined even when either operand is NaN — the sign is
                // copied without inspecting the payload — so unlike `min`/`max`
                // this needs no NaN special case, and Rust's method matches.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a.copysign(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Div(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                // Unlike the integer divides this never traps: IEEE 754 gives
                // `±inf` for a non-zero numerator over zero, and NaN for `0.0/0.0`.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a / b),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Eq(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a == b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Ge(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a >= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Gt(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a > b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Le(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a <= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Lt(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a < b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Max(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                let r = if a.is_nan() || b.is_nan() {
                    f32::NAN
                } else if a == b {
                    // -0.0 and +0.0 compare equal, so pick by sign: max wants +0.0
                    if a.is_sign_positive() { a } else { b }
                } else if a > b {
                    a
                } else {
                    b
                };

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(r),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Min(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                let r = if a.is_nan() || b.is_nan() {
                    f32::NAN
                } else if a == b {
                    // -0.0 and +0.0 compare equal, so pick by sign: min wants -0.0
                    if a.is_sign_negative() { a } else { b }
                } else if a < b {
                    a
                } else {
                    b
                };

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(r),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Mul(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a * b),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Ne(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a != b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F32Sub(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f32();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f32(a - b),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Load { offset, sig } => {
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    sig.input.registers[0],
                    caller_base_data,
                    instance,
                )?;

                let val = instance.memory.read_f64(effective_offset)?;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(val),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Store { offset, input } => {
                let inputs = &input.registers;
                let val = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let effective_offset = Self::effective_address(
                    *offset,
                    frame_layout,
                    inputs[0],
                    caller_base_data,
                    instance,
                )?;

                instance.memory.write_f64(effective_offset, val)?;

                Step::Next
            }
            RegInstruction::F64Abs(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a.abs()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Ceil(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a.ceil()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64ConvertI32S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a as f64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64ConvertI32U(sig) => {
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_i32() as u32;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a as f64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64ConvertI64S(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a as f64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64ConvertI64U(sig) => {
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a as f64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Floor(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a.floor()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Nearest(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a.round_ties_even()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Neg(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a.neg()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64PromoteF32(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f32();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a as f64),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64ReinterpretI64(sig) => {
                let a = Self::slot_value(sig.input.registers[0], caller_base_data, instance)
                    .as_i64() as u64;

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(f64::from_bits(a)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Sqrt(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a.sqrt()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Trunc(sig) => {
                let a =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a.trunc()),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Add(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a + b),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Copysign(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                // See `F32Copysign`: magnitude of `a`, sign of `b`, NaN included.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a.copysign(b)),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Div(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                // See `F32Div`: division by zero yields an infinity or NaN, never
                // a trap.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a / b),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Eq(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a == b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Ge(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a >= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Gt(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a > b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Le(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a <= b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Lt(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a < b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Max(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                let r = if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else if a == b {
                    // -0.0 and +0.0 compare equal, so pick by sign: max wants +0.0
                    if a.is_sign_positive() { a } else { b }
                } else if a > b {
                    a
                } else {
                    b
                };

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(r),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Min(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                let r = if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else if a == b {
                    // -0.0 and +0.0 compare equal, so pick by sign: min wants -0.0
                    if a.is_sign_negative() { a } else { b }
                } else if a < b {
                    a
                } else {
                    b
                };

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(r),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Mul(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a * b),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Ne(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32((a != b) as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::F64Sub(sig) => {
                let inputs = &sig.input.registers;
                let b = Self::slot_value(inputs[1], caller_base_data, instance).as_f64();
                let a = Self::slot_value(inputs[0], caller_base_data, instance).as_f64();

                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_f64(a - b),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::Select(sig) => {
                let entry = frame_layout.select_arena.get(*sig);
                let inputs = &entry.0.input.registers;
                let output_register_index = entry.0.output.start;

                let a = Self::slot_value(inputs[0], caller_base_data, instance);
                let b = Self::slot_value(inputs[1], caller_base_data, instance);
                let cond = Self::slot_value(inputs[2], caller_base_data, instance).as_i32();

                // true condition
                if cond != 0 {
                    Self::set_value_to_register(
                        output_register_index,
                        a,
                        caller_base_data,
                        instance,
                    );
                } else {
                    Self::set_value_to_register(
                        output_register_index,
                        b,
                        caller_base_data,
                        instance,
                    );
                }

                Step::Next
            }
            RegInstruction::Return { target_index } => Step::JumpTo(*target_index),
            RegInstruction::Call {
                func_index: callee_func_index,
                caller_base,
            } => {
                let callee_caller_base_data = RegCallerBaseData {
                    base_register_index: *caller_base as u32 + caller_base_data.base_register_index,
                };

                if callee_func_index.0 >= imported_func_count {
                    Step::Call {
                        func_index: *callee_func_index,
                        caller_base_data: callee_caller_base_data,
                        is_indirect: None,
                    }
                } else {
                    crate::runtime::TraceVM::call_imported::<M, I, Self::Vm>(
                        *callee_func_index,
                        module,
                        instance,
                        None,
                        &callee_caller_base_data,
                    )?;

                    Step::Next
                }
            }
            RegInstruction::CallIndirect(id) => {
                let entry = frame_layout.call_indirect_arena.get(*id);
                let table = &instance.table_vals[entry.table_index.0 as usize];

                let slot = Self::slot_value(entry.slot.registers[0], caller_base_data, instance)
                    .as_i32() as u32 as usize;

                let Some(func_ref) = table.table.get(slot).copied() else {
                    return Err(Box::new(InstructionExecutionError::CallIndirect(
                        entry.table_index,
                        CallIndirectError::TableSlotOutOfBounds,
                    )));
                };

                let Some(callee_func_index) = func_ref else {
                    return Err(Box::new(InstructionExecutionError::CallIndirect(
                        entry.table_index,
                        CallIndirectError::NullElementInTable,
                    )));
                };

                let func_ty = &module.types[entry.ty_index.0 as usize];
                let params = &func_ty.params;
                let results = &func_ty.results;
                let func = &module.func_decls[callee_func_index.0 as usize];
                let ty = &module.types[func.ty.0 as usize];
                let declared_params = &ty.params;
                let declared_results = &ty.results;

                if params.as_ref() != declared_params.as_ref()
                    || results.as_ref() != declared_results.as_ref()
                {
                    return Err(Box::new(signature_mismatch(
                        entry.table_index,
                        declared_params,
                        declared_results,
                        params,
                        results,
                    )));
                }

                let param_slots = &entry.operands.input[..declared_params.len()];

                let mut tmp: SmallVec<[Value; 3]> = smallvec![];

                for slot in param_slots {
                    tmp.push(Self::slot_value(*slot, caller_base_data, instance));
                }

                for (i, val) in tmp.into_iter().enumerate() {
                    Self::set_value_to_register(
                        entry.caller_base + i as u16,
                        val,
                        caller_base_data,
                        instance,
                    );
                }

                let callee_caller_base_data = RegCallerBaseData {
                    base_register_index: entry.caller_base as u32
                        + caller_base_data.base_register_index,
                };

                if callee_func_index.0 >= imported_func_count {
                    Step::Call {
                        func_index: callee_func_index,
                        caller_base_data: callee_caller_base_data,
                        is_indirect: Some(entry.table_index),
                    }
                } else {
                    crate::runtime::TraceVM::call_imported::<M, I, Self::Vm>(
                        callee_func_index,
                        module,
                        instance,
                        Some(entry.table_index),
                        &callee_caller_base_data,
                    )?;

                    Step::Next
                }
            }
            RegInstruction::RefIsNull(sig) => {
                let func_ref =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_ref();

                // A predicate, so the result is an `i32` and not a reference — the
                // same convention `iNN.eqz` follows, which is what lets it feed a
                // `br_if` directly.
                Self::set_value_to_register(
                    sig.output.start,
                    Value::from_i32(func_ref.is_none() as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::Unreachable => {
                return Err(Box::new(InstructionExecutionError::Unreachable));
            }
            RegInstruction::Move(id) => {
                Self::execute_mov(
                    frame_layout.dyn_signatures.get(*id),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::MemorySize(outputs) => {
                Self::set_value_to_register(
                    outputs.start,
                    Value::from_i32(instance.memory.size_in_pages() as i32),
                    caller_base_data,
                    instance,
                );

                Step::Next
            }
            RegInstruction::MemoryGrow(sig) => {
                let delta_in_pages =
                    Self::slot_value(sig.input.registers[0], caller_base_data, instance).as_i32()
                        as u32;

                let max_pages = instance.config.get_max_memory_size_in_pages();

                match instance.memory.grow(delta_in_pages, max_pages) {
                    Ok(old_page) => Self::set_value_to_register(
                        sig.output.start,
                        Value::from_i32(old_page as i32),
                        caller_base_data,
                        instance,
                    ),
                    Err(_) => Self::set_value_to_register(
                        sig.output.start,
                        Value::from_i32(-1),
                        caller_base_data,
                        instance,
                    ),
                }

                Step::Next
            }
            RegInstruction::MemoryCopy(inputs) => {
                let inputs = &inputs.registers;
                let dest = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32
                    as usize;
                let src = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32
                    as usize;
                let len = Self::slot_value(inputs[2], caller_base_data, instance).as_i32() as u32
                    as usize;

                instance.memory.copy_within(dest, src, len)?;

                Step::Next
            }
            RegInstruction::MemoryFill(inputs) => {
                let inputs = &inputs.registers;
                let dest = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32
                    as usize;
                let val = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32;
                let len = Self::slot_value(inputs[2], caller_base_data, instance).as_i32() as u32
                    as usize;

                instance.memory.fill(dest, val, len)?;

                Step::Next
            }
            RegInstruction::MemoryInit(id) => {
                let entry = frame_layout.memory_init_arena.get(*id);
                let inputs = &entry.operands.registers;
                let dest = Self::slot_value(inputs[0], caller_base_data, instance).as_i32() as u32
                    as usize;
                let src = Self::slot_value(inputs[1], caller_base_data, instance).as_i32() as u32
                    as usize;
                let len = Self::slot_value(inputs[2], caller_base_data, instance).as_i32() as u32
                    as usize;

                let segment: &[u8] = match &instance.data_vals[entry.data_index as usize] {
                    DataVal::Dropped => &[],
                    DataVal::Passive(segment) => segment,
                };

                let end = src
                    .checked_add(len)
                    .filter(|end| *end <= segment.len())
                    .ok_or(MemoryError::OutOfBoundsAccess(
                        MemoryAccessKind::Read,
                        src,
                        segment.len(),
                    ))?;

                instance.memory.write(dest, &segment[src..end])?;

                Step::Next
            }
            RegInstruction::DataDrop(data_index) => {
                instance.data_vals[*data_index as usize] = DataVal::Dropped;

                Step::Next
            }
            RegInstruction::End => Step::Next,
        };

        Ok(res)
    }
}

/// Frame and memory accessors shared by the `execute` arms.
///
/// All of them resolve an index against `caller_base_data.base_register_index`, the
/// absolute base of the running frame. Only [`Self::set_value_to_spills`] has to know
/// anything about the layout's regions; every other frame access is one add, which is
/// the whole point of a single untagged index space.
impl RegInstruction {
    /// Reads local `index`.
    ///
    /// The locals are the frame's first region, so a local index *is* a frame index.
    #[inline(always)]
    fn local<M: Memory, I: ImportRegistry>(
        index: LocalIndex,
        caller_base_data: &RegCallerBaseData,
        instance: &Instance<M, I, crate::Register>,
    ) -> Value {
        instance.frame.registers[(caller_base_data.base_register_index + index.0) as usize]
    }

    /// Writes local `index`, on the same terms as [`Self::local`].
    #[inline(always)]
    fn set_local<M: Memory, I: ImportRegistry>(
        index: LocalIndex,
        val: Value,
        caller_base_data: &RegCallerBaseData,
        instance: &mut Instance<M, I, crate::Register>,
    ) {
        instance.frame.registers[(caller_base_data.base_register_index + index.0) as usize] = val;
    }

    /// Reads one operand.
    ///
    /// The function the whole operand design exists for: a constant, a local, a spill
    /// and an operand register are all one indexed load, because they share one frame
    /// index space and the index needs no tag to say which. Only a global is
    /// elsewhere, and that is the one branch here.
    #[inline(always)]
    fn slot_value<M: Memory, I: ImportRegistry>(
        slot: Slot,
        caller_base_data: &RegCallerBaseData,
        instance: &Instance<M, I, crate::Register>,
    ) -> Value {
        instance.frame.registers[(caller_base_data.base_register_index + slot.0 as u32) as usize]
    }

    /// Writes a frame slot named by an already-resolved index.
    ///
    /// Named for its usual caller — an instruction's destination register — but it
    /// writes any frame index, which is how `Call` and `CallIndirect` stage their
    /// arguments at `caller_base`.
    #[inline(always)]
    fn set_value_to_register<M: Memory, I: ImportRegistry>(
        relative_index: u16,
        val: Value,
        caller_base_data: &RegCallerBaseData,
        instance: &mut Instance<M, I, crate::Register>,
    ) {
        instance.frame.registers
            [(caller_base_data.base_register_index + relative_index as u32) as usize] = val;
    }

    /// Writes spill slot `relative_index`.
    ///
    /// The one accessor that computes a region base at execution, because a
    /// `LocalSpill` carries a spill-pool index rather than a frame
    /// index. The spill region begins above the locals and the constants, hence
    /// `locals_count + consts.len()`.
    ///
    /// **This must match the spill backpatch in `emit_instructions_for_func` term for
    /// term.** That pass resolves a spill *operand* to the same address; if the two
    /// disagree, a spill is written to one slot and read from another, and the read
    /// returns whatever was left there.
    #[inline(always)]
    fn set_value_to_spills<M: Memory, I: ImportRegistry>(
        relative_index: u16,
        val: Value,
        caller_base_data: &RegCallerBaseData,
        frame_layout: &RegFrameLayout,
        instance: &mut Instance<M, I, crate::Register>,
    ) {
        instance.frame.registers[caller_base_data.base_register_index as usize
            + relative_index as usize
            + frame_layout.locals_count as usize
            + frame_layout.consts.len()] = val;
    }

    /// Performs a label move: reads every source, then writes every destination.
    ///
    /// Both passes are needed because the two runs may overlap — a move to a label
    /// whose registers sit under the operands it carries is the common case — so
    /// writing as it reads would clobber a source it had not reached yet. The buffer
    /// costs nothing in practice: an arity is a label's params or results, which is
    /// one or two values for anything rustc emits.
    #[inline(always)]
    fn execute_mov<M: Memory, I: ImportRegistry>(
        signature: &DynSignature,
        caller_base_data: &RegCallerBaseData,
        instance: &mut Instance<M, I, crate::Register>,
    ) {
        let mut tmp: SmallVec<[Value; 3]> = smallvec![];
        let inputs = &signature.input;
        let output_start = signature.output_start;

        // memmove: input and output can overlap!
        for slot in inputs {
            tmp.push(Self::slot_value(*slot, caller_base_data, instance));
        }

        for (i, val) in tmp.into_iter().enumerate() {
            Self::set_value_to_register(output_start + i as u16, val, caller_base_data, instance);
        }
    }

    /// The address a load or store touches: the operand in `slot`, plus the static
    /// offset `offset` names in [`RegFrameLayout::memory_offsets`].
    ///
    /// # Errors
    ///
    /// [`MemoryError::EffectiveAddressOverflow`] when the sum leaves the 32-bit
    /// address space, which is out of bounds by definition and so a trap rather than
    /// a wrap.
    #[inline(always)]
    fn effective_address<M: Memory, I: ImportRegistry>(
        offset: InternedId<MemoryOffset, u16>,
        frame_layout: &RegFrameLayout,
        slot: Slot,
        caller_base_data: &RegCallerBaseData,
        instance: &mut Instance<M, I, crate::Register>,
    ) -> Result<usize, MemoryError> {
        let MemoryOffset(memarg_offset) = *frame_layout.memory_offsets.value(offset);

        let addr = Self::slot_value(slot, caller_base_data, instance).as_i32() as u32;

        let effective_offset = addr
            .checked_add(memarg_offset)
            .ok_or(MemoryError::EffectiveAddressOverflow(addr, memarg_offset))?;

        Ok(effective_offset as usize)
    }
}

/// The wasm mnemonic for a value-op kind: `I32TruncSatF32U` → `i32.trunc_sat_f32_u`.
///
/// Derived rather than tabulated, because 136 hand-written strings is 136 chances
/// to write `i32.extend_8_s` for `i32.extend8_s`. The rule is one line: start a
/// word at each capital, so digits stay attached to the word they follow, and the
/// first word is the type prefix.
///
/// It is not taken on trust — `arity_case` in the tests builds each operator's
/// `.wat` from this, and that `.wat` has to assemble, validate, and contain the
/// operator the kind is named after. A wrong mnemonic fails those rather than
/// quietly rendering an instruction under a name that does not exist.
pub(crate) fn mnemonic(kind: RegInstructionKind) -> String {
    let mut words: Vec<String> = vec![];

    for character in format!("{kind:?}").chars() {
        if character.is_ascii_uppercase() || words.is_empty() {
            words.push(String::new());
        }

        words
            .last_mut()
            .expect("just pushed")
            .push(character.to_ascii_lowercase());
    }

    format!("{}.{}", words[0], words[1..].join("_"))
}
