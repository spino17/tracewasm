//! Lowering of a WebAssembly operator stream into a register machine.
//!
//! WebAssembly is a stack machine, but the stack traffic it implies is mostly
//! bookkeeping: `local.get; local.get; i32.add` describes two pushes and two pops
//! that a register machine does not need. This pass consumes the same operator
//! stream as [`crate::instruction::stack`] and produces
//! [`RegInstruction`]s whose operands name where a value *is* — a constant, a local,
//! a global, a spill slot, or a register — so nothing moves unless it has to.
//!
//! The stack pass remains the reference for tracing fidelity; this one is for
//! running the same module quickly.
//!
//! ## The simulated stack
//!
//! Lowering walks the operators once, maintaining a `SimulatedStack`: a stack of
//! `StackSlot`s standing for the operands wasm would have pushed. Nothing is
//! emitted for `local.get` or `i32.const` — they push a slot describing where the
//! value lives. An instruction that consumes operands pops the slots and records
//! them as its inputs, so the consumer reads the original locations directly.
//!
//! **Two heights, and they are not the same number.** `stack.height()` counts
//! simulated slots; `curr_register_index` counts only the slots that occupy a
//! machine register. A `Const`, `Local`, `Global`, or `Spilled` slot occupies a
//! stack position and no register. Anything restoring the stack to a label's entry
//! state must use the slot height; anything naming a destination register must use
//! the register index. Both are recorded per block — the slot height in
//! `Block::recorded_height`, the register index derived by counting register slots
//! back down to it.
//!
//! ## Lazily forwarded locals and globals
//!
//! A slot naming a local or global is only valid while that origin still holds the
//! value; a later write has to materialize every operand still reading it. That is
//! [`lazy`]'s job — see its module docs for how one shared entry per borrowed value
//! makes multiple simultaneous borrows, and multiple live snapshots of the same
//! local, fall out of the representation.
//!
//! ## Operands live in flat side tables
//!
//! A [`RegInstruction`] never stores its operands inline. Inputs go into
//! [`FrameLayout::input_registers_arena`] and destination registers into
//! [`FrameLayout::output_registers_arena`]; the variant holds only a start index and
//! (for the variable-arity forms) a length. That is what keeps the enum at 24 bytes
//! — inline operands would put `Select` alone at 56 — and it means resolving an
//! operand at execution is a slice index rather than a pointer chase.
//!
//! The arenas ship inside [`FrameLayout`], because every index in the instruction
//! list points into them.
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
    error::TraceWasmError,
    instruction::{
        Block, BlockKind, params_and_results_from_blockty,
        register::lazy::{
            Global, GlobalSlot, LazyArena, LazyEntryDropResult, LazyLocation, LazySlot, Local,
            LocalSlot, SpillArena, SpillIndex,
        },
    },
    module::{FuncDecl, FuncIndex, FuncType, GlobalIndex, LocalIndex, TableIndex, TyIndex},
    vm::stack::Stack,
};
use std::marker::PhantomData;
use wasmparser::{BlockType, Operator, OperatorsReader};

pub mod lazy;

#[cfg(test)]
mod tests;

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
    /// Never constructed today: [`RegInstruction::emit_instruction_for_func`] pushes
    /// that [`Block`] itself rather than going through [`SimulatedStack::add_block`],
    /// since a function frame has no operator to open it and no block type to read.
    /// The variant exists for the mapping in `add_block` to be total.
    Func,
}

/// An immediate operand, carried inline because it has no home to be read from.
#[derive(Debug, Clone, Copy)]
pub enum Const {
    /// A 32-bit integer immediate.
    I32(i32),
    /// A 64-bit integer immediate.
    I64(i64),
    /// A 32-bit float immediate; the bit pattern is preserved exactly.
    F32(f32),
    /// A 64-bit float immediate; the bit pattern is preserved exactly.
    F64(f64),
}

/// Where an instruction reads one operand from.
///
/// This is the resolved, executable form: the lowering pass has already decided that
/// this operand needs no copy and can be read in place. Only [`Self::Register`]
/// values were produced by a preceding instruction; the rest are read straight out of
/// the frame's constants, locals, globals, or spill area.
#[derive(Debug, Clone, Copy)]
pub enum Slot {
    /// An immediate, needing no load at all.
    Const(Const),
    /// Read local `n` in place. Valid because nothing writes that local between the
    /// `local.get` and this instruction — a write would have spilled it first.
    Local(u32),
    /// Read global `n` in place, on the same terms as [`Self::Local`].
    Global(u32),
    /// Read the frame spill slot holding a local or global that was materialized
    /// before a write to its origin. See [`lazy`].
    Spilled(SpillIndex),
    /// Read an operand register, i.e. a value some earlier instruction produced.
    Register(u32),
}

impl Slot {
    /// Whether this operand occupies a machine register — the distinction between
    /// the simulated stack's two heights. See the module docs.
    fn is_register(&self) -> bool {
        matches!(self, Slot::Register(_))
    }
}

impl Default for Slot {
    /// Filler for arena entries that are about to be overwritten.
    ///
    /// `SimulatedStack::pops_and_pushes_registers` resizes the input arena before
    /// popping, because it fills the run back-to-front. Every slot it reserves is
    /// written before anything reads it, so the value only has to be cheap and
    /// harmless — a constant reads nothing and names no location.
    fn default() -> Self {
        Slot::Const(Const::I32(0))
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
    Const(Const),
    /// A value an earlier instruction produced, in the register named here. The
    /// only variant [`SimulatedStack::curr_register_index`] counts.
    Register(u32),
    /// A borrow of a local, resolving to the local itself or to the spill slot it
    /// was rescued into, whichever holds the value when this is finally read.
    Local(LocalSlot),
    /// [`Self::Local`] for a global.
    Global(GlobalSlot),
}

/// A fixed-length run of `L` entries in one of the flat arenas, named by its start.
///
/// `T` is [`Slot`] for input runs and `u32` for output-register runs; the parameter
/// keeps the two from being resolved against the wrong arena. Four bytes whatever `L`
/// is, which is what lets instruction variants stay small.
pub struct Registers<const L: usize, T> {
    /// Index of the run's first entry in its arena; it covers `L` from there.
    start: u32,
    phantom: PhantomData<T>,
}

impl<const L: usize, T> Registers<L, T> {
    /// Resolves this run against the arena it indexes.
    ///
    /// Returns a fixed-size array, so callers index it without bounds checks. Panics
    /// if `arena` is not the arena this run was recorded in — for a lowered body,
    /// the matching field of its [`FrameLayout`].
    pub fn registers<'a>(&self, arena: &'a [T]) -> &'a [T; L] {
        let start = self.start as usize;

        arena[start..(start + L)].try_into().unwrap()
    }
}

/// The operands of an instruction whose arity is fixed by its opcode.
///
/// `I` inputs and `O` destination registers, each a run in the corresponding arena.
///
/// **Inputs may alias outputs.** Destination registers are allocated after the
/// inputs are consumed, so an instruction's output frequently reuses the register one
/// of its inputs occupied. An executor must read every operand before writing any
/// destination.
pub struct Signature<const I: usize, const O: usize> {
    /// Operands in wasm push order — `input[0]` is the deepest, the one pushed first.
    pub input: Registers<I, Slot>,
    /// Destination registers, in the order results are pushed.
    pub output: Registers<O, u32>,
}

/// The operands of an instruction whose arity is only known at lowering time — a
/// label's params or results.
///
/// One `len` covers both runs because every user is 1:1: a [`RegInstruction::Move`]
/// writes exactly as many destinations as it reads sources. Storing two lengths would
/// admit a mismatched pair that cannot occur, and would push [`RegInstruction`] past
/// its size budget.
pub struct DynSignature {
    /// Start of the source run in the input arena.
    input: u32,
    /// Start of the destination run in the output arena.
    output: u32,
    /// Length of both runs; see why there is only one on the type.
    len: u32,
}

impl DynSignature {
    /// The source operands, in push order.
    pub fn input_registers<'a>(&self, arena: &'a [Slot]) -> &'a [Slot] {
        let start = self.input as usize;

        &arena[start..(start + self.len as usize)]
    }

    /// The destination registers, positionally matched to
    /// [`Self::input_registers`].
    pub fn output_registers<'a>(&self, arena: &'a [u32]) -> &'a [u32] {
        let start = self.output as usize;

        &arena[start..(start + self.len as usize)]
    }

    /// Whether this signature moves nothing — a label with no params or results,
    /// which is the common case. Callers use it to skip emitting a `Move` entirely
    /// rather than emitting an empty one.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Where the operands recorded by [`SimulatedStack::pops_and_pushes_registers`]
/// landed in the flat arenas.
///
/// `#[must_use]` because dropping it means the entries just written are
/// unreferenced: the arenas ship inside [`FrameLayout`], so a discarded result is
/// dead weight in every compiled function. A caller that only needs the stack and
/// register bookkeeping wants [`SimulatedStack::pops_and_pushes`], which does the
/// same thing without touching the arenas.
#[must_use]
struct PopsPushesResult {
    input_start: u32,
    output_start: u32,
}

/// The open labels, innermost last. Index 0 is the implicit function frame.
///
/// A `br relative_depth` resolves to `stack[len - 1 - relative_depth]`.
#[derive(Default)]
struct ControlStack {
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

/// One resolved arm of a [`RegInstruction::BrTable`].
///
/// Each arm carries its own move and jump target because a single `br_table` may mix
/// loop and non-loop labels: validation only requires the label *types* to match, so
/// the arities agree but the unwind heights — and therefore the destination registers
/// — differ per arm.
pub struct BrTarget {
    /// Values transferred to this arm's label, on the same terms as
    /// [`RegInstruction::Move`]. Empty when the label carries nothing.
    pub mov: DynSignature,
    /// Absolute jump target: a loop's start for a back-edge, otherwise the label's
    /// `end`, backpatched when that `end` is reached.
    pub target_index: u32,
}

/// The whole lowering state for one function body.
///
/// "Simulated" because [`Self::stack`] holds descriptions of operands rather than
/// values: the pass walks the operator stream maintaining the stack wasm *would*
/// have, and reads off where each value lives instead of moving it.
struct SimulatedStack {
    /// The operand stack, one [`StackSlot`] per value wasm would have pushed.
    stack: Stack<StackSlot>,
    /// How many machine registers are live. Distinct from `stack.height()`: only
    /// [`StackSlot::Register`] entries consume one. See the module docs.
    curr_register_index: usize,
    /// Peak [`Self::curr_register_index`], i.e. the register count the frame needs.
    max_registers: u32,
    /// Lazy borrows of locals; see [`lazy`].
    lazy_locals: LazyArena<Local>,
    /// Lazy borrows of globals; see [`lazy`].
    lazy_globals: LazyArena<Global>,
    /// Frame slots holding locals and globals materialized ahead of a write.
    spills: SpillArena,
    /// Flat arena of every instruction's input operands, indexed by
    /// [`Registers`]/[`DynSignature`] and shipped in [`FrameLayout`].
    input_registers: Vec<Slot>,
    /// Flat arena of every instruction's destination registers, on the same terms.
    output_registers: Vec<u32>,
    /// Open labels, for resolving branch depths and backpatching at `end`.
    control_stack: ControlStack,
    /// Flat arena of `br_table` arms, indexed by
    /// [`RegInstruction::BrTable`]'s `(targets_start, targets_len)` range.
    br_targets: Vec<BrTarget>,
}

impl SimulatedStack {
    /// Empty lowering state for one body.
    ///
    /// The two counts size the lazy origin tables, which are indexed by local and
    /// global index without bounds checks — so `locals_count` must include the
    /// params, not just the declared locals. Everything else starts empty and grows
    /// as the body is walked; the control stack in particular is empty here, and the
    /// caller pushes the function frame onto it.
    fn new(locals_count: u32, globals_count: u32) -> Self {
        SimulatedStack {
            stack: Stack::new_with_capacity(0),
            curr_register_index: 0,
            max_registers: 0,
            lazy_locals: LazyArena::new(locals_count),
            lazy_globals: LazyArena::new(globals_count),
            spills: SpillArena::default(),
            input_registers: vec![],
            output_registers: vec![],
            control_stack: ControlStack::default(),
            br_targets: vec![],
        }
    }

    /// Allocates one register, keeping [`Self::max_registers`] a true high-water
    /// mark. Every path that allocates a register must come through here, or the
    /// frame will be sized smaller than the registers instructions actually write.
    fn advanced_register_index(&mut self) {
        self.curr_register_index += 1;

        if self.curr_register_index as u32 > self.max_registers {
            self.max_registers += 1;
        }
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
                    instr_len + 1 // move instruction is emitted if params != 0 so the actuall instruction lands at `len + 1`
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
            BlockKind::Block { .. } => self.stack.height() - params,
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

    /// Releases one borrow of a lazily forwarded local or global and reports where
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
        let location = slot.location(&arena);

        if matches!(slot.decrease_ref_count(arena), LazyEntryDropResult::Dropped) {
            match location {
                LazyLocation::Original(local_index) => arena.origin[local_index as usize] = None,
                LazyLocation::Spilled(spill_index) => spills.free_slot(spill_index),
            }
        }

        location
    }

    /// Starts or joins a lazy borrow of one local or global.
    ///
    /// If something already borrows this origin, the new stack slot shares that
    /// entry, so a later spill redirects both at once. Otherwise a fresh entry is
    /// allocated and recorded as the origin's live borrow.
    fn push_lazy<T>(location: u32, arena: &mut LazyArena<T>) -> LazySlot<T> {
        let slot = match arena.origin[location as usize] {
            Some(slot) => {
                slot.advanced_ref_count(arena);

                slot
            }
            None => {
                let slot = arena.allocate(location);
                arena.origin[location as usize] = Some(slot);

                slot
            }
        };

        slot
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
    fn pop(&mut self) -> Slot {
        let val = self.stack.pop();

        let slot = match val {
            StackSlot::Const(val) => Slot::Const(val),
            StackSlot::Register(val) => {
                self.recede_register_index();

                Slot::Register(val)
            }
            StackSlot::Local(slot) => {
                let location = Self::pop_lazy(slot, &mut self.lazy_locals, &mut self.spills);

                match location {
                    LazyLocation::Original(local_index) => Slot::Local(local_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
            StackSlot::Global(slot) => {
                let location = Self::pop_lazy(slot, &mut self.lazy_globals, &mut self.spills);

                match location {
                    LazyLocation::Original(global_index) => Slot::Global(global_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
        };

        slot
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
    fn simulated_pop(&self, depth: u32) -> Slot {
        let val = *self.stack.peek_from_top(depth);

        let slot = match val {
            StackSlot::Const(val) => Slot::Const(val),
            StackSlot::Register(val) => Slot::Register(val),
            StackSlot::Local(slot) => {
                let location = slot.location(&self.lazy_locals);

                match location {
                    LazyLocation::Original(local_index) => Slot::Local(local_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
            StackSlot::Global(slot) => {
                let location = slot.location(&self.lazy_globals);

                match location {
                    LazyLocation::Original(global_index) => Slot::Global(global_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
        };

        slot
    }

    /// Pushes one operand onto the simulated stack.
    ///
    /// A local or global becomes a shared lazy borrow rather than a resolved
    /// location, so a spill occurring before it is read still reaches it. A register
    /// allocates one.
    ///
    /// [`Slot::Spilled`] is rejected: a spill is a *transition* of an existing
    /// borrow, applied through [`Self::set_lazy`], never a value pushed from nothing.
    fn push(&mut self, val: Slot) {
        let slot = match val {
            Slot::Const(val) => StackSlot::Const(val),
            Slot::Register(val) => {
                self.advanced_register_index();

                StackSlot::Register(val)
            }
            Slot::Local(index) => {
                let slot = Self::push_lazy(index, &mut self.lazy_locals);

                StackSlot::Local(slot)
            }
            Slot::Global(index) => {
                let slot = Self::push_lazy(index, &mut self.lazy_globals);

                StackSlot::Global(slot)
            }
            Slot::Spilled(_) => unreachable!("spilled slots are never produced for push!"),
        };

        self.stack.push(slot);
    }

    /// Resolves the top operand without consuming it, for `local.tee`.
    ///
    /// `local.tee n` writes the value *and* leaves it on the stack, so the operand
    /// stays where it is. That is sound only because the caller spills first: any
    /// borrow of local `n` still on the stack — including this one — is redirected to
    /// a spill slot before the write, so the slot left behind no longer names the
    /// local being overwritten. Reading the location here, after the spill, is what
    /// picks up that redirection.
    fn tee(&self) -> Slot {
        let top_slot = &self.stack.top();

        match top_slot {
            StackSlot::Const(val) => Slot::Const(*val),
            StackSlot::Register(val) => Slot::Register(*val),
            StackSlot::Local(slot) => match slot.location(&self.lazy_locals) {
                LazyLocation::Original(local_index) => Slot::Local(local_index),
                LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
            },
            StackSlot::Global(slot) => match slot.location(&self.lazy_globals) {
                LazyLocation::Original(global_index) => Slot::Global(global_index),
                LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
            },
        }
    }

    /// `i32.const` and friends: records the immediate, emitting nothing.
    fn push_const(&mut self, val: Const) {
        self.push(Slot::Const(val));
    }

    /// `local.get`: starts or joins a lazy borrow of the local, emitting nothing.
    /// The value is read in place by whatever consumes it, unless a write to the
    /// local intervenes and spills it first.
    fn push_local(&mut self, index: u32) {
        self.push(Slot::Local(index));
    }

    /// `global.get`: [`Self::push_local`] for a global. Also spilled by a call,
    /// which may write any global.
    fn push_global(&mut self, index: u32) {
        self.push(Slot::Global(index));
    }

    /// Applies an instruction's stack effect and records its operands in the arenas.
    ///
    /// Pops `pops` operands into a fresh input run — deepest first, so `input[0]` is
    /// the first value pushed — then allocates `pushes` destination registers and
    /// pushes them back. Allocating the outputs *after* consuming the inputs is what
    /// lets a result reuse an operand's register, and is why an executor must read
    /// all operands before writing any destination.
    ///
    /// Use [`Self::pops_and_pushes`] when only the stack effect is wanted; this
    /// variant grows the arenas that ship in [`FrameLayout`].
    fn pops_and_pushes_registers(&mut self, pops: u32, pushes: u32) -> PopsPushesResult {
        let pops = pops as usize;
        let pushes = pushes as usize;
        let input_start = self.input_registers.len();
        let output_start = self.output_registers.len();

        self.input_registers
            .resize(input_start + pops, Slot::default());

        for i in 0..pops {
            self.input_registers[input_start + pops - 1 - i] = self.pop();
        }

        self.output_registers.resize(output_start + pushes, 0);

        for i in 0..pushes {
            self.output_registers[output_start + i] = self.curr_register_index as u32;
            let out = Slot::Register(self.curr_register_index as u32);

            self.push(out);
        }

        PopsPushesResult {
            input_start: input_start as u32,
            output_start: output_start as u32,
        }
    }

    /// The same stack effect as [`Self::pops_and_pushes_registers`], recording
    /// nothing.
    ///
    /// For resets that reshape the model without describing an instruction — the
    /// layout an `else` hands to its arm, or the layout an unconditional branch
    /// leaves the enclosing block in. Those emit no operands, so writing runs into
    /// the arenas would leave entries nothing indexes, in tables that ship with every
    /// compiled function.
    ///
    /// The pops still go through [`Self::pop`], so lazy borrows and spill slots are
    /// released as they should be.
    fn pops_and_pushes(&mut self, pops: u32, pushes: u32) {
        let pops = pops as usize;
        let pushes = pushes as usize;

        for _ in 0..pops {
            self.pop();
        }

        for _ in 0..pushes {
            let out = Slot::Register(self.curr_register_index as u32);

            self.push(out);
        }
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
    fn reset_enclosing_block_layout(&mut self) {
        let block = self.get_curr_block();
        let recorded_height = block.recorded_height;
        let results = block.results;
        let unwind = self.stack.height() - recorded_height;

        self.pops_and_pushes(unwind, results);
    }

    /// [`Self::pops_and_pushes_registers`] for an instruction whose arity is fixed by
    /// its opcode, returning the [`Signature`] to store in the variant.
    fn registers_for<const I: usize, const O: usize>(&mut self) -> Signature<I, O> {
        let result = self.pops_and_pushes_registers(I as u32, O as u32);

        Signature {
            input: Registers {
                start: result.input_start as u32,
                phantom: PhantomData,
            },
            output: Registers {
                start: result.output_start as u32,
                phantom: PhantomData,
            },
        }
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
    fn materialize_stack_slots_in_registers(&mut self, depth: u32) -> DynSignature {
        let result = self.pops_and_pushes_registers(depth, depth);

        DynSignature {
            input: result.input_start,
            output: result.output_start,
            len: depth,
        }
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
    ) -> DynSignature {
        let input_start = self.input_registers.len();
        let output_start = self.output_registers.len();
        let arity_to_preserve = arity_to_preserve as usize;
        let curr_stack_height = self.stack.height();
        let popped_count = (curr_stack_height - base_height) as usize;
        let mut register_index = self.curr_register_index as u32;

        self.input_registers
            .resize(input_start + arity_to_preserve, Slot::default());

        self.output_registers
            .resize(output_start + arity_to_preserve, 0);

        for i in 0..popped_count {
            let slot = self.simulated_pop(i as u32);

            if slot.is_register() {
                register_index -= 1;
            }

            if i < arity_to_preserve {
                self.input_registers[input_start + arity_to_preserve - 1 - i] = slot;
            }
        }

        // output registers for the branch results
        for i in 0..arity_to_preserve {
            self.output_registers[output_start + i] = register_index;
            register_index += 1;
        }

        if register_index > self.max_registers {
            self.max_registers = register_index;
        }

        DynSignature {
            input: input_start as u32,
            output: output_start as u32,
            len: arity_to_preserve as u32,
        }
    }

    /// Rescues every operand still forwarding to `location`, ahead of a write to it.
    ///
    /// Returns the spill slot the caller must emit a
    /// [`LocalSpill`](RegInstruction::LocalSpill) /
    /// [`GlobalSpill`](RegInstruction::GlobalSpill) into — *before* the writing
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
    ) -> Option<SpillIndex> {
        let Some(slot) = arena.origin[location as usize] else {
            return None;
        };

        let spill_index = spills.reserve_slot();

        slot.spill(spill_index, arena);
        arena.origin[location as usize] = None;

        Some(spill_index)
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

/// The storage one lowered body needs, in slot counts.
///
/// Both fields are high-water marks over the whole body rather than counts at any
/// one point, so a frame sized to them never has to grow mid-execution.
///
/// **Operands only.** Locals are not counted here: like the stack pass's
/// [`max_height`](crate::instruction::stack), these are measured from the frame's
/// operand base, so a consumer laying out storage needs
/// `locals_len + registers + spills`.
pub struct FrameLayout {
    /// Operand registers, i.e. the peak `curr_register_index`.
    pub registers: u32,
    /// Spill slots holding locals and globals rescued from a later write by
    /// [`RegInstruction::LocalSpill`] / [`RegInstruction::GlobalSpill`].
    ///
    /// Zero for a body that never overwrites a lazily-forwarded local or global,
    /// which is the common case.
    pub spills: u32,
    /// Every instruction's input operands, concatenated in lowering order.
    ///
    /// [`Signature::input`] and [`DynSignature::input_registers`] name runs here.
    /// Shipped with the body because every one of those indices is meaningless
    /// without it.
    pub input_registers_arena: Box<[Slot]>,
    /// Every instruction's destination registers, on the same terms as
    /// [`Self::input_registers_arena`].
    pub output_registers_arena: Box<[u32]>,
    /// Every `br_table`'s arms, concatenated in lowering order.
    ///
    /// A [`RegInstruction::BrTable`] owns the contiguous run named by its
    /// `(targets_start, targets_len)`, with the default arm last. Empty, and
    /// unallocated, for the common case of a body with no `br_table`.
    pub br_targets_arena: Box<[BrTarget]>,
}

/// The two outputs of lowering one function body into register form: the
/// instruction list, and the frame required to execute it.
pub type LoweredRegFuncBody = (Vec<RegInstruction>, FrameLayout);

/// One lowered instruction.
///
/// Operands are never inline: a variant carries indices into the arenas on
/// [`FrameLayout`], resolved through [`Signature`] or [`DynSignature`]. Jump fields
/// are absolute indices into the containing `Vec<RegInstruction>`, i.e. runtime
/// program counters.
pub enum RegInstruction {
    /// `i32.load`: read four bytes at `address + offset`.
    I32Load {
        /// Static byte offset added to the popped address.
        offset: u32,
        sig: Signature<1, 1>,
    },
    /// `global.set`: write the operand into a global.
    ///
    /// Any operand still forwarding to this global was rescued by a preceding
    /// [`Self::GlobalSpill`].
    GlobalSet {
        index: GlobalIndex,
        sig: Signature<1, 0>,
    },
    /// `local.set`: write the operand into a local. See [`Self::GlobalSet`] on
    /// rescues.
    LocalSet {
        index: LocalIndex,
        sig: Signature<1, 0>,
    },
    /// `local.tee`: write the operand into a local, leaving it on the operand stack.
    ///
    /// Arity is `<1, 0>` rather than `<1, 1>` because the value stays where it
    /// already is — no destination register is allocated.
    LocalTee {
        index: LocalIndex,
        sig: Signature<1, 0>,
    },
    /// `i32.store`: write `sig.input[1]` to `sig.input[0] + offset`.
    I32Store {
        /// Static byte offset added to the popped address.
        offset: u32,
        sig: Signature<2, 0>,
    },
    /// Copies a local into a spill slot, immediately before a write that would
    /// otherwise invalidate operands still reading it. See [`lazy`].
    LocalSpill {
        index: LocalIndex,
        spill_index: SpillIndex,
    },
    /// [`Self::LocalSpill`] for a global.
    GlobalSpill {
        index: GlobalIndex,
        spill_index: SpillIndex,
    },
    /// `if`: fall through when the condition is non-zero, otherwise jump past
    /// `else_index` to the else-arm — or to `end_index` when there is none.
    ///
    /// Block params, if any, were materialized by a [`Self::Move`] emitted just
    /// before this instruction.
    If {
        cond: Signature<1, 0>,
        /// Index of the matching [`Self::Else`], backpatched at `end`.
        else_index: Option<u32>,
        /// Index of the matching `end`, backpatched.
        end_index: u32,
    },
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
    BrIf {
        cond: Registers<1, Slot>,
        mov: DynSignature,
        target_index: u32,
    },
    /// `br_table`: jump to the arm selected by the operand, or to the default when it
    /// is out of range.
    ///
    /// Arms are a contiguous run of `BrTarget` in the body's target arena; the
    /// default is the last element. Each arm carries its own move and target, since
    /// the labels may sit at different heights even though their arities agree.
    BrTable {
        index: Registers<1, Slot>,
        targets_start: u32,
        targets_len: u32,
    },
    /// `i32.add`.
    I32Add(Signature<2, 1>),
    /// `i32.eqz`: `1` if the operand is zero, else `0`.
    I32Eqz(Signature<1, 1>),
    /// `select`: `input[2] != 0 ? input[0] : input[1]`.
    Select(Signature<3, 1>),
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
        /// Frame-relative register index the callee's frame is based at: where its
        /// arguments were staged, and where its results come back to.
        caller_base: u32,
    },
    /// `call_indirect`: resolve a callee through a table at execution and call it.
    ///
    /// The callee index is pushed *above* the arguments, so it is popped first and
    /// the arguments are staged below it. That extra operand is the only structural
    /// difference from [`Self::Call`] — and the reason both of this call's heights
    /// count one deeper than the signature's arity.
    ///
    /// **Neither operand run is stored.** The arguments are the `params` operands
    /// starting at `operands`, `params` coming from the signature [`Self::ty_index`]
    /// names; their destinations are the same many registers based at `caller_base`,
    /// contiguous because that is how they were allocated. Reconstructing both from
    /// the signature is what keeps this variant inside the size budget — see the
    /// note on [`RegInstruction`]'s size assertion.
    ///
    /// **The move is a field rather than a preceding [`Self::Move`]**, and unlike
    /// [`Self::BrIf`] the reason is not a control path: the staged arguments land in
    /// `[caller_base, caller_base + params)`, which may *include* the register
    /// `slot` reads the callee index from. A `Move` ahead of the instruction would
    /// destroy the index before it was used. An executor must therefore resolve
    /// `slot` before performing the move — the same read-before-write contract
    /// [`Self::BrIf`] has with its condition.
    ///
    /// [`Self::ty_index`]: RegInstruction::CallIndirect::ty_index
    CallIndirect {
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
        slot: Registers<1, Slot>,
        /// Start of this call's arguments in the input arena. They run for the
        /// `ty_index` signature's param count, in wasm push order, and are moved to
        /// `caller_base`, `caller_base + 1`, ... one apiece.
        operands: u32,
        /// Frame-relative register index the callee's frame is based at, on the
        /// same terms as [`Self::Call`]'s.
        caller_base: u32,
    },
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
    /// * The inputs are not all registers. A [`Slot`] reads from a constant, the
    ///   locals array, the globals table, the spill area, or the register file, so
    ///   this gathers from five places into one contiguous destination range rather
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
    /// for slot in sig.input_registers(arena) { tmp.push(read(slot)); }
    /// for (i, &dst) in sig.output_registers(arena).iter().enumerate() { regs[dst] = tmp[i]; }
    /// ```
    ///
    /// The buffer costs nothing in practice: arities are the label's params or
    /// results, which are one or two values for anything rustc emits.
    Move(DynSignature),
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
// What holds it here is that operands live in the flat side tables rather than in the
// variant: a `Registers<I, O>` is a pair of `u32` starts (8 bytes) whatever `I` and
// `O` are, so the widest variant is `I32Load(u32, Registers<1, 1>)` at 12 bytes plus
// tag. Inlining the operands instead would put `Select(Registers<3, 1>)` alone at 56.
//
// The constraint this places on what comes next: an instruction whose arity is not a
// compile-time constant — `call`, `call_indirect`, the block param/result moves — must
// stay within the same 8-byte shape. Either derive both arities at execution from an
// index the variant already carries (as `CallIndirect` does with its `ty_index` in the
// stack pass), or store an explicit `len` and drop something else to pay for it.
const _: () = assert!(
    size_of::<RegInstruction>() <= 24,
    "RegInstruction grew past 24 bytes. Need to keep it compact."
);

impl RegInstruction {
    /// Materializes every live lazy borrow in `arena`, emitting one spill
    /// instruction per borrow.
    ///
    /// # Why a rescue cannot live at the write
    ///
    /// A borrow of a local or global reads its origin in place (see [`lazy`]), and
    /// stays valid only until something writes that origin. The write emits a spill
    /// that copies the old value aside, and the borrow is redirected to it — so from
    /// then on the operand resolves to [`Slot::Spilled`] *for every path*, because
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
        instructions: &mut Vec<RegInstruction>,
        instruction_emitter: F,
    ) {
        let lazy_count = arena.origin.len() as u32;

        for index in 0..lazy_count {
            if let Some(spill_index) = SimulatedStack::set_lazy(index, arena, spills) {
                instructions.push(instruction_emitter(spill_index, index));
            }
        }
    }

    /// Rescues every live local borrow ahead of a diverging or repeating construct.
    ///
    /// See [`Self::spill_lazy`] for why the rescue cannot be left at the write, and
    /// [`Self::spill_live_globals`] for the other half — a construct needs both, and
    /// a call site that makes only one of them leaves the other origin space exposed.
    fn spill_live_locals(
        simulated_stack: &mut SimulatedStack,
        instructions: &mut Vec<RegInstruction>,
    ) {
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
        );
    }

    /// [`Self::spill_live_locals`] for globals, which use their own arena but share
    /// the frame's spill pool.
    fn spill_live_globals(
        simulated_stack: &mut SimulatedStack,
        instructions: &mut Vec<RegInstruction>,
    ) {
        let instruction_emitter =
            |spill_index: SpillIndex, index: u32| RegInstruction::GlobalSpill {
                index: GlobalIndex(index),
                spill_index,
            };

        Self::spill_lazy(
            &mut simulated_stack.lazy_globals,
            &mut simulated_stack.spills,
            instructions,
            instruction_emitter,
        );
    }

    /// Lowers one function body's operator stream into register form.
    ///
    /// `params`/`results` are the body's own arity, seeding the implicit function
    /// frame at the root of the control stack. `locals_count` must cover params plus
    /// declared locals, and `globals_count` the module's globals — both size the lazy
    /// origin tables, which are indexed unchecked.
    ///
    /// Returns the instruction list alongside the [`FrameLayout`] needed to run it.
    /// The two are inseparable: every operand in the list is an index into the
    /// arenas the layout carries.
    ///
    /// Rejects any operator the pass does not model as
    /// [`TraceWasmError::Unsupported`].
    pub fn emit_instruction_for_func(
        mut operator_reader: OperatorsReader<'_>,
        params: u32,
        results: u32,
        types: &[FuncType],
        func_decls: &[FuncDecl],
        locals_count: u32,
        globals_count: u32,
    ) -> Result<LoweredRegFuncBody, TraceWasmError> {
        let mut instructions: Vec<RegInstruction> = vec![];
        let mut simulated_stack = SimulatedStack::new(locals_count, globals_count);
        let mut unreachable_tracking_stack = UnreachableTrackingControlStack::new();

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
            let (operator, _offset) = operator_reader.read_with_offset()?;

            if !matches!(
                unreachable_tracking_stack.check_unreachablity(&operator),
                UnreachableCheckResult::Reachable
            ) {
                continue;
            }

            match operator {
                Operator::GlobalGet { global_index } => {
                    simulated_stack.push_global(global_index);
                }
                Operator::GlobalSet { global_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        global_index,
                        &mut simulated_stack.lazy_globals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::GlobalSpill {
                            index: GlobalIndex(global_index),
                            spill_index,
                        });
                    }

                    let registers = simulated_stack.registers_for::<1, 0>();

                    instructions.push(RegInstruction::GlobalSet {
                        index: GlobalIndex(global_index),
                        sig: registers,
                    });
                }
                Operator::LocalGet { local_index } => {
                    simulated_stack.push_local(local_index);
                }
                Operator::LocalSet { local_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        local_index,
                        &mut simulated_stack.lazy_locals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::LocalSpill {
                            index: LocalIndex(local_index),
                            spill_index,
                        });
                    }

                    let registers = simulated_stack.registers_for::<1, 0>();

                    instructions.push(RegInstruction::LocalSet {
                        index: LocalIndex(local_index),
                        sig: registers,
                    });
                }
                Operator::LocalTee { local_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        local_index,
                        &mut simulated_stack.lazy_locals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::LocalSpill {
                            index: LocalIndex(local_index),
                            spill_index,
                        });
                    }

                    let input_start = simulated_stack.input_registers.len();

                    simulated_stack.input_registers.push(simulated_stack.tee());

                    let registers = Signature {
                        input: Registers {
                            start: input_start as u32,
                            phantom: PhantomData,
                        },
                        output: Registers {
                            start: simulated_stack.output_registers.len() as u32,
                            phantom: PhantomData,
                        },
                    };

                    instructions.push(RegInstruction::LocalTee {
                        index: LocalIndex(local_index),
                        sig: registers,
                    });
                }
                Operator::I32Const { value } => {
                    simulated_stack.push_const(Const::I32(value));
                }
                Operator::I32Load { memarg } => {
                    let registers = simulated_stack.registers_for::<1, 1>();

                    instructions.push(RegInstruction::I32Load {
                        offset: memarg.offset as u32,
                        sig: registers,
                    });
                }
                Operator::I32Store { memarg } => {
                    let registers = simulated_stack.registers_for::<2, 0>();

                    instructions.push(RegInstruction::I32Store {
                        offset: memarg.offset as u32,
                        sig: registers,
                    });
                }
                Operator::I32Add => {
                    let registers = simulated_stack.registers_for::<2, 1>();

                    instructions.push(RegInstruction::I32Add(registers));
                }
                Operator::I32Eqz => {
                    let registers = simulated_stack.registers_for::<1, 1>();

                    instructions.push(RegInstruction::I32Eqz(registers));
                }
                Operator::Nop => {
                    continue;
                }
                Operator::Select => {
                    let registers = simulated_stack.registers_for::<3, 1>();

                    instructions.push(RegInstruction::Select(registers));
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
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(block_params);

                        instructions.push(RegInstruction::Move(move_registers));
                    }
                }
                Operator::Loop { blockty } => {
                    // A loop repeats: a rescue left inside the body would re-run on
                    // the back-edge and capture what the previous iteration wrote.
                    // Hoisting it above the header runs it exactly once, on entry.
                    Self::spill_live_locals(&mut simulated_stack, &mut instructions);
                    Self::spill_live_globals(&mut simulated_stack, &mut instructions);

                    let (block_params, _) = simulated_stack.add_block(
                        BlockVariant::Loop,
                        &blockty,
                        types,
                        instructions.len(),
                    );

                    if block_params != 0 {
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(block_params);

                        instructions.push(RegInstruction::Move(move_registers));
                    }
                }
                Operator::If { blockty } => {
                    // Control diverges here, and a write in either arm would leave
                    // the other path reading a spill slot nothing wrote.
                    Self::spill_live_locals(&mut simulated_stack, &mut instructions);
                    Self::spill_live_globals(&mut simulated_stack, &mut instructions);

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
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(block_params + 1);

                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    instructions.push(RegInstruction::If {
                        cond: simulated_stack.registers_for::<1, 0>(),
                        else_index: None,
                        end_index: u32::MAX,
                    });
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
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(block_results);

                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    // reset the frame layout with params on top for else instructions.
                    simulated_stack.pops_and_pushes(
                        simulated_stack.stack.height() - recorded_height,
                        block_params,
                    );

                    instructions.push(RegInstruction::Else {
                        end_index: u32::MAX,
                    });
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
                                simulated_stack.br_truncation_registers(recorded_height, params),
                                loop_index,
                            )
                        } else {
                            let move_registers =
                                simulated_stack.br_truncation_registers(recorded_height, results);

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
                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    instructions.push(RegInstruction::Br { target_index });

                    simulated_stack.reset_enclosing_block_layout();
                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::BrIf { relative_depth } => {
                    // Taking the branch skips everything after it, including any
                    // write that would have rescued a borrow still live here.
                    Self::spill_live_locals(&mut simulated_stack, &mut instructions);
                    Self::spill_live_globals(&mut simulated_stack, &mut instructions);

                    let block_index =
                        simulated_stack.control_stack.len() - 1 - relative_depth as usize;
                    let block = simulated_stack.get_block(block_index);
                    let params = block.params;
                    let results = block.results;
                    let recorded_height = block.recorded_height;
                    let block_kind = block.kind;

                    let cond = simulated_stack.registers_for::<1, 0>().input;

                    let (move_registers, target_index) =
                        if let Some(loop_index) = block_kind.is_loop() {
                            (
                                simulated_stack.br_truncation_registers(recorded_height, params),
                                loop_index,
                            )
                        } else {
                            let move_registers =
                                simulated_stack.br_truncation_registers(recorded_height, results);

                            simulated_stack
                                .get_block_mut(block_index)
                                .attached_breaks
                                .push((instructions.len() as u32, u32::MAX));

                            (move_registers, u32::MAX)
                        };

                    instructions.push(RegInstruction::BrIf {
                        cond,
                        mov: move_registers,
                        target_index,
                    });
                }
                Operator::BrTable { targets: table } => {
                    // As for `br_if`: every arm jumps away, so a later write cannot
                    // be relied on to have rescued anything.
                    Self::spill_live_locals(&mut simulated_stack, &mut instructions);
                    Self::spill_live_globals(&mut simulated_stack, &mut instructions);

                    let targets_start = simulated_stack.br_targets.len() as u32;
                    let mut targets_len = 0;

                    let targets = table.targets();
                    let mut targets = targets.collect::<Result<Vec<_>, _>>()?;

                    targets.push(table.default());

                    let table_index = simulated_stack.registers_for::<1, 0>().input; // targets index

                    for (i, &relative_depth) in targets.iter().enumerate() {
                        let block_index =
                            simulated_stack.control_stack.len() - 1 - relative_depth as usize;
                        let block = simulated_stack.get_block_mut(block_index);
                        let params = block.params;
                        let results = block.results;
                        let recorded_height = block.recorded_height;
                        let block_kind = block.kind;

                        let (move_registers, target_index) = if let Some(loop_index) =
                            block_kind.is_loop()
                        {
                            (
                                simulated_stack.br_truncation_registers(recorded_height, params),
                                loop_index,
                            )
                        } else {
                            let move_registers =
                                simulated_stack.br_truncation_registers(recorded_height, results);

                            simulated_stack
                                .get_block_mut(block_index)
                                .attached_breaks
                                .push((instructions.len() as u32, targets_start + i as u32));

                            (move_registers, u32::MAX)
                        };

                        let br_target = BrTarget {
                            mov: move_registers,
                            target_index,
                        };

                        simulated_stack.br_targets.push(br_target);

                        targets_len += 1;
                    }

                    instructions.push(RegInstruction::BrTable {
                        index: table_index,
                        targets_start,
                        targets_len,
                    });

                    simulated_stack.reset_enclosing_block_layout();
                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::Return => {
                    let move_registers = simulated_stack.br_truncation_registers(0, results);

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
                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    instructions.push(RegInstruction::Return {
                        target_index: u32::MAX,
                    });

                    simulated_stack.reset_enclosing_block_layout();
                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::Call { function_index } => {
                    Self::spill_live_globals(&mut simulated_stack, &mut instructions);

                    let func_decl = &func_decls[function_index as usize];
                    let func_ty = &types[func_decl.ty.0 as usize];
                    let params = func_ty.params.len() as u32;
                    let results = func_ty.results.len() as u32;
                    let recorded_height = simulated_stack.stack.height() - params;
                    let caller_base = simulated_stack.register_index_at_depth(params) as u32;

                    if params != 0 {
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(params);

                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    instructions.push(RegInstruction::Call {
                        func_index: FuncIndex(function_index),
                        caller_base,
                    });

                    simulated_stack
                        .pops_and_pushes(simulated_stack.stack.height() - recorded_height, results);
                }
                Operator::CallIndirect {
                    type_index,
                    table_index,
                } => {
                    Self::spill_live_globals(&mut simulated_stack, &mut instructions);

                    let ty = &types[type_index as usize];
                    let params = ty.params.len() as u32;
                    let results = ty.results.len() as u32;
                    let recorded_height = simulated_stack.stack.height() - params - 1;
                    let caller_base = simulated_stack.register_index_at_depth(params + 1) as u32;

                    let slot = simulated_stack.registers_for::<1, 0>().input;

                    let move_registers =
                        simulated_stack.materialize_stack_slots_in_registers(params);

                    instructions.push(RegInstruction::CallIndirect {
                        ty_index: TyIndex(type_index),
                        table_index: TableIndex(table_index),
                        slot,
                        operands: move_registers.input,
                        caller_base,
                    });

                    simulated_stack
                        .pops_and_pushes(simulated_stack.stack.height() - recorded_height, results);
                }
                Operator::Unreachable => {
                    instructions.push(RegInstruction::Unreachable);

                    simulated_stack.reset_enclosing_block_layout();
                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::End => {
                    let block = simulated_stack.pop_block();
                    let results = block.results;
                    let attached_breaks = &block.attached_breaks;

                    if results != 0 {
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(results);

                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    debug_assert!(
                        simulated_stack.stack.height() == block.recorded_height + results
                    );

                    let index = instructions.len() as u32;

                    for (br_index, br_targets_index) in attached_breaks {
                        match &mut instructions[*br_index as usize] {
                            RegInstruction::Br { target_index } => {
                                *target_index = index;
                            }
                            RegInstruction::BrIf {
                                cond: _,
                                mov: _,
                                target_index,
                            } => {
                                *target_index = index;
                            }
                            RegInstruction::BrTable { .. } => {
                                // `br_targets_index` is already absolute, so the table's own
                                // range is not needed here.
                                simulated_stack.br_targets[*br_targets_index as usize]
                                    .target_index = index;
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
                            let RegInstruction::If {
                                cond: _,
                                else_index,
                                end_index,
                            } = &mut instructions[if_index as usize]
                            else {
                                unreachable!(
                                    "hitting this means TraceWasm has a bug recording the instructions"
                                )
                            };

                            *else_index = ei;
                            *end_index = index;

                            // ... and point the `else` (if present) at this same `end`, so a then-branch
                            // that falls through into `else` knows where the construct closes.
                            if let Some(else_index) = ei {
                                let RegInstruction::Else { end_index } =
                                    &mut instructions[else_index as usize]
                                else {
                                    unreachable!(
                                        "hitting this means TraceWasm has a bug recording the instructions"
                                    )
                                };

                                *end_index = index;
                            }
                        }
                    }

                    instructions.push(RegInstruction::End);
                }
                _ => {
                    return Err(TraceWasmError::Unsupported(format!(
                        "instruction `{:?}`",
                        operator
                    )));
                }
            }
        }

        // Both counts are already high-water marks — `max_registers` is maintained
        // by `advanced_register_index` and `allocation_len` only grows when no
        // freed spill slot can be reused — so they are read off directly here
        // rather than recomputed from the instruction list.
        let frame = FrameLayout {
            // Bounded by the operand-stack depth, which a function body's size in
            // the binary already bounds well below `u32::MAX`, so this cannot
            // truncate for any module that could be loaded at all.
            registers: simulated_stack.max_registers as u32,
            spills: simulated_stack.spills.allocation_len(),
            input_registers_arena: simulated_stack.input_registers.into_boxed_slice(),
            output_registers_arena: simulated_stack.output_registers.into_boxed_slice(),
            br_targets_arena: simulated_stack.br_targets.into_boxed_slice(),
        };

        Ok((instructions, frame))
    }
}
