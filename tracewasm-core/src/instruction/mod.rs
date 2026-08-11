use crate::module::FuncType;
use wasmparser::BlockType;

pub mod register;
pub mod stack;

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

/// A live control-flow label on the [`ControlStack`].
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
    /// is frozen (see [`ControlStack::set_height`]) because dead code has a
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
    ///   *both* arms of such an `if` are dead. [`ControlStack::end_unreachable_traversing`]
    ///   consults this flag so an `else` does not resurrect genuinely dead code.
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
