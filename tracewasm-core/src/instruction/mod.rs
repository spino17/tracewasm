//! Lowering a WebAssembly operator stream into an executable instruction list.
//!
//! [`Instruction`] is the crate's central abstraction: a *lowering strategy*,
//! bundling an instruction set with the frame it runs in and the calling
//! convention that connects them. Two implement it — [`stack::StackInstruction`],
//! which keeps wasm's own operand stack, and [`register::RegInstruction`], which
//! lowers the same operators into a register machine. [`Module`] and [`Instance`]
//! are generic over the trait, so a module is compiled for one machine or the
//! other and the rest of the crate does not care which.
//!
//! Four traits, because a machine is four decisions that have to agree:
//!
//! | Trait | Decides |
//! |---|---|
//! | [`Instruction`] | the instruction set, and how a body is lowered and executed |
//! | [`FrameLayout`] | what lowering hands execution besides the instructions |
//! | [`RuntimeFrame`] | where live values are kept |
//! | [`CallerBaseData`] | how a callee's frame is located inside the caller's |
//!
//! [`Instruction`]'s associated types tie them together, so the halves of one
//! machine cannot be paired with the halves of the other.
//!
//! This module also holds what both passes share: `Block` and `BlockKind`,
//! the control-flow bookkeeping a lowering keeps per open label, and the small
//! helpers around block types and memory indices. The passes themselves are in
//! [`stack`] and [`register`].

use crate::{
    error::{InstructionExecutionError, TraceWasmError},
    instance::{Instance, traits::ImportRegistry},
    memory::Memory,
    module::{FuncDecl, FuncType, Module, ValType},
    runtime::{
        Step,
        value::{Val, Value},
    },
};
use smallvec::SmallVec;
use wasmparser::{BlockType, OperatorsReader};

pub mod register;
pub mod stack;

/// Where in the caller's frame a callee's own frame begins.
///
/// The two machines measure that base in different units — the stack machine in
/// operand-stack slots, the register machine in registers — so the driver holds
/// it behind this trait rather than as a `u32` it would have to interpret.
pub trait CallerBaseData {
    /// The callee's frame base, in whatever unit the machine counts.
    ///
    /// A positional machine adds this to every local and operand index it
    /// touches. A stack machine, whose positions are implicit in the stack
    /// pointer, mostly ignores it and uses it only to bound an unwind.
    fn base_offset(&self) -> u32;

    /// Records how many locals the callee declares, fixing the boundary between
    /// its locals and its operands.
    ///
    /// Called once per call, after the callee's body is known and before its
    /// first instruction runs. Branch targets are computed against the resulting
    /// boundary, so an implementation that leaves it unset leaves those
    /// computations reading whatever it was initialised to.
    fn set_callee_locals_count(&mut self, count: u32);
}

/// One function activation's value storage, as the calling convention sees it.
///
/// **The two implementations differ in kind, not just in layout**, and callers
/// have to respect both:
///
/// * The stack machine's `Stack<Value>` is *consuming and implicit*.
///   [`get_params`](Self::get_params) and [`results`](Self::results) pop what
///   they return, and position comes from the stack pointer, so the
///   `caller_base_data` argument is unused.
/// * The register machine's `RegFrame` is *positional and
///   non-consuming*. Nothing is removed; every access is an index relative to
///   [`CallerBaseData::base_offset`].
///
/// So a driver must call the reading methods exactly once per call and must not
/// assume either behaviour — reading twice yields the same values on one machine
/// and underflows the other.
pub trait RuntimeFrame {
    /// How this machine names a callee's frame base. See [`CallerBaseData`].
    type CallerBaseData: CallerBaseData;

    /// Places a call's arguments where the callee expects to find its params.
    fn set_params(&mut self, params: &[Val]);
    /// Reads a callee's arguments back out, to hand to an imported function.
    ///
    /// **Consumes them on the stack machine** and leaves them in place on the
    /// register machine; see the trait docs.
    fn get_params(
        &mut self,
        params_count: u32,
        caller_base_data: &Self::CallerBaseData,
    ) -> SmallVec<[Value; 5]>;

    /// Writes a call's results back where the caller will look for them.
    fn set_results<R: IntoIterator<Item = Val>>(
        &mut self,
        results: R,
        caller_base_data: &Self::CallerBaseData,
    );

    /// Takes the results a finished function left behind, in push order
    /// (`result0..resultN-1`).
    ///
    /// **Consumes them on the stack machine**, as [`Self::get_params`] does.
    ///
    /// Unlike every other accessor here this takes no [`CallerBaseData`], so a
    /// positional implementation has no frame base to read against and can only
    /// answer for the outermost activation.
    fn results(&mut self, results_count: u32) -> SmallVec<[Value; 3]>;

    /// Empties the frame, so an instance survives a trap and can be called
    /// again.
    ///
    /// A failing call returns with values still live; the next call clears them
    /// rather than the failing one unwinding tidily.
    fn reset(&mut self);

    /// Initialises the callee's declared locals — everything after its params —
    /// to the zero of their type, as wasm requires.
    ///
    /// `locals_ty` covers the whole addressable range, params included, so the
    /// declared locals are `locals_ty[params_count..]`.
    fn set_zero_values_in_locals_after_params(
        &mut self,
        params_count: u32,
        locals_ty: &[ValType],
        caller_base_data: &Self::CallerBaseData,
    );

    /// Drops a returning callee's frame and leaves its results where the caller
    /// expects them — the counterpart of [`Self::set_zero_values_in_locals_after_params`]
    /// at the other end of a call.
    fn tear_callee_frame_and_set_results(
        &mut self,
        results_count: u32,
        caller_base_data: &Self::CallerBaseData,
    );
}

/// The per-function storage plan a lowering produces alongside its instructions.
///
/// What a layout holds differs sharply — the stack machine needs little beyond a
/// peak height, the register machine carries operand arenas and register counts —
/// so only the part the driver reads uniformly is named here.
pub trait FrameLayout {
    /// This machine's resolved `br_table` arm.
    type BrTableTarget;

    /// Every `br_table` arm in the body, concatenated in lowering order.
    ///
    /// A branch-table instruction owns a contiguous `(start, len)` run of this
    /// slice rather than an inline list, which is what keeps the instruction
    /// small enough to stay within its size budget.
    fn br_table_targets(&self) -> &[Self::BrTableTarget];
}

/// One lowering strategy: an instruction set, the frame it runs in, and how a
/// body is produced and executed.
///
/// This is the seam between the two machines. [`StackInstruction`](stack::StackInstruction)
/// keeps wasm's own operand stack and is the reference for tracing fidelity;
/// [`RegInstruction`](register::RegInstruction) lowers the same operators into a
/// register machine that moves values only when it must. Everything downstream —
/// [`Module`], [`Instance`], the error types — is generic over this trait rather
/// than over the two concrete sets.
///
/// The associated types are tied together so a machine cannot be assembled from
/// mismatched halves: a body's [`FrameLayout`] must yield the same
/// `BrTableTarget` the instructions index, and its [`RuntimeFrame`] must speak
/// the same [`CallerBaseData`] the instructions are executed against.
pub trait Instruction: Sized {
    /// A resolved `br_table` arm, as stored in the body's flat target array.
    type BrTableTarget;
    /// The storage plan lowering produces for one body.
    type FrameLayout: FrameLayout<BrTableTarget = Self::BrTableTarget>;
    /// Where this machine keeps live values during execution.
    type RuntimeFrame: Default + RuntimeFrame<CallerBaseData = Self::CallerBaseData>;
    /// How this machine locates a callee's frame within the caller's.
    type CallerBaseData: CallerBaseData;

    /// Lowers one function body's operator stream into this instruction set.
    ///
    /// Returns the instructions, the source byte offset of the operator each one
    /// came from, and the frame the body needs.
    ///
    /// **The offsets are parallel to the instructions** — same length, index for
    /// index — because diagnostics index them by program counter to name the
    /// operator that trapped. An implementation that emits several instructions
    /// for one operator must record an offset for each.
    fn emit_instruction_for_func(
        operator_reader: OperatorsReader<'_>,
        params: u32,
        results: u32,
        types: &[FuncType],
        func_decls: &[FuncDecl],
        locals_count: u32,
        globals_count: u32,
    ) -> Result<(Vec<Self>, Vec<u32>, Self::FrameLayout), TraceWasmError<Self>>;

    /// Executes this one instruction against `instance`.
    ///
    /// Returns a `Step` telling the driver where to go next — fall through,
    /// jump, call, or return — rather than moving a program counter itself, so
    /// the driver owns control flow and this stays a pure per-instruction
    /// operation.
    ///
    /// `caller_base_data` locates this activation's frame; `br_table_targets` is
    /// the body's arm array, which a branch table indexes by its own range.
    fn execute<M: Memory, I: ImportRegistry>(
        &self,
        module: &Module<Self>,
        instance: &mut Instance<M, I, Self>,
        br_table_targets: &[Self::BrTableTarget],
        caller_base_data: &Self::CallerBaseData,
        imported_func_count: u32,
    ) -> Result<Step<Self>, Box<InstructionExecutionError<Self>>>;
}

/// What kind of label a control-stack entry represents, plus the data needed to
/// backpatch its originating instruction once its `end` is seen.
#[derive(Clone, Copy)]
pub(crate) enum BlockKind {
    /// The implicit outermost frame for a function body. Not backpatched (its
    /// `end` is the final instruction and no branch instruction stores its
    /// index directly).
    Func,
    /// A `block`. `index` is the position of its `Instruction::Block`, so the
    /// `end_index` field can be filled in later.
    Block,
    /// A `loop`. `index` is the position of its `Instruction::Loop`; this is the
    /// back-edge target used directly by branches (no backpatching needed).
    Loop { index: u32 },
    /// An `if`. `index` locates the `Instruction::If`; `else_index` is filled in
    /// when the `Else` operator is seen (if any) so both can be backpatched at
    /// `end`.
    If { index: u32, else_index: Option<u32> },
}

impl BlockKind {
    /// Returns the loop's start-instruction index iff this label is a `loop`.
    ///
    /// Branch lowering keys off this: a branch to a loop is a back-edge whose
    /// target is known immediately (the loop start) and whose arity is the
    /// loop's *params*; a branch to any other label is a forward jump to its
    /// `end` whose arity is the label's *results*.
    fn is_loop(&self) -> Option<u32> {
        if let BlockKind::Loop { index } = self {
            Some(*index)
        } else {
            None
        }
    }
}

/// A live control-flow label on a lowering pass's control stack.
///
/// Shared by both passes: each keeps its own `ControlStack` of these, since the
/// bookkeeping a label needs — entry height, arity, dead-code state, the branches
/// waiting to be backpatched — is the same whichever machine is being emitted.
struct Block {
    /// Which label this is, carrying the instruction index its opener sits at so
    /// that index can be backpatched at the matching `end`. See [`BlockKind`].
    kind: BlockKind,
    /// Operand-stack height this label unwinds to, *excluding* the label's own
    /// arity values.
    ///
    /// Invariant: when control reaches this label's target (the `end` of a
    /// block/if/func, or the start of a `loop`), the stack is truncated to
    /// `recorded_height` and then exactly `arity` values remain on top — results
    /// for block/if/func, params for a loop. It is captured at label entry as
    /// "height below the params" (`curr_height - params`, and additionally minus
    /// the condition for `if`). Meaningless while [`Self::has_inherited`] is set
    /// (dead code), where it is stored as `0`.
    recorded_height: u32,
    /// Arity of the label's input type (block params). For a loop this is also
    /// the branch arity.
    params: u32,
    /// Arity of the label's result type. For block/if/func this is the branch
    /// arity and the height delta applied at `end`.
    results: u32,
    /// True while the remainder of this block's body is unreachable (dead code),
    /// e.g. after `unreachable`, `br`, or `br_table`. While set, height tracking
    /// is frozen (the pass's `set_height` ignores writes) because dead code has a
    /// stack-polymorphic type and tracking it is both meaningless and prone to
    /// underflow.
    is_unreachable_traversing: bool,
    /// True iff this block was *opened while its parent was already dead*, i.e.
    /// it is unreachable for its entire lifetime.
    ///
    /// This distinguishes two reasons `is_unreachable_traversing` can be set:
    /// - locally dead (a `br`/`unreachable` inside a live block) — recoverable
    ///   at the block's `else`;
    /// - inherited dead (born inside dead code) — must NEVER be cleared, because
    ///   *both* arms of such an `if` are dead. The pass consults this flag when an
    ///   `else` would otherwise resurrect genuinely dead code.
    has_inherited: bool,
    /// Branches (`br`/`br_if`/`br_table` arms, and `return` targeting the
    /// function frame) that target this block's `end` and therefore need their
    /// `target_index` backpatched once that `end` is reached.
    ///
    /// Each entry is `(instruction_index, brtable_target_slot)`. The second field
    /// is an **absolute** index into the function's flat `br_table` target array —
    /// not an arm number relative to the table's own `start_index` — so patching is
    /// a direct write and never has to read the range back out of the instruction.
    /// It is `u32::MAX` (unused) for `Br`/`BrIf`/`Return`, which have a single
    /// `target_index` in the variant itself.
    attached_breaks: Vec<(u32, u32)>,
}

/// Returns `(params, results)` *counts* for a block type. Only the arities
/// matter for height tracking, not the concrete value types.
///
/// `BlockType::Type(_)` is the shorthand single-result form (`[] -> [t]`),
/// hence `(0, 1)`.
fn params_and_results_from_blockty(blockty: &BlockType, types: &[FuncType]) -> (u32, u32) {
    match blockty {
        BlockType::Empty => (0, 0),
        BlockType::Type(_) => (0, 1),
        BlockType::FuncType(index) => {
            let ty = &types[*index as usize];

            (ty.params.len() as u32, ty.results.len() as u32)
        }
    }
}

/// Rejects a memory index other than 0.
///
/// TraceWasm allows at most one memory, which [`Module::compile`](crate::module::Module::compile)
/// already enforces at the section level; this catches the same thing at the
/// instruction level, where the multi-memory proposal puts an index on every
/// memory operator. Generic only so it can name the caller's error type.
fn check_memory_index<Instr: Instruction>(index: u32) -> Result<(), TraceWasmError<Instr>> {
    if index != 0 {
        return Err(TraceWasmError::Unsupported(
            "more than one memory".to_string(),
        ));
    }

    Ok(())
}
