//! Lowering of a WebAssembly operator stream into TraceWasm's flat instruction list.
//!
//! `Instruction::emit_instruction_from_operator_reader` consumes a
//! [`wasmparser::OperatorsReader`] (one function body, or one constant
//! expression) and produces a `Vec<Instruction>` in which **structured control
//! flow has been resolved into absolute indices** and **operand-stack heights
//! have been precomputed**. The goal is that a downstream interpreter never has
//! to re-scan for matching `end`s or rebuild block types at runtime: every
//! branch already knows the exact instruction index (`pc`) it jumps to and the
//! stack height it must unwind to.
//!
//! ## Two jobs performed in a single linear pass
//!
//! 1. **Backpatching of forward references.** WebAssembly control flow is
//!    structured (`block`/`loop`/`if`/`else`/`end`), but a *forward* branch or a
//!    block's own `end` position is not known until that `end` is reached later
//!    in the stream. Such fields are emitted with the sentinel [`usize::MAX`] and
//!    filled in when the matching `End` operator is processed. `usize::MAX` (not
//!    `0`) is used deliberately: `0` is a valid instruction index, so a missed
//!    backpatch would silently jump to the first instruction, whereas `usize::MAX`
//!    makes the bug trap on an out-of-bounds access.
//!
//! 2. **Operand-stack height precomputation.** `ControlStack::curr_height`
//!    tracks the operand-stack depth as the pass advances. For every label we
//!    record the height the stack unwinds to when that label is targeted (see
//!    `Block::recorded_height`). A branch then stores `recorded_height` plus
//!    the label's `arity`, which is all an interpreter needs to truncate the
//!    value stack in O(1) on a taken branch or on `end`.
//!
//! ## Height-tracking invariant (load-bearing)
//!
//! The height model only works if **every** operator that changes the operand
//! stack updates `curr_height`. The control operators handled below do so; any
//! value/numeric/memory/local operator added later MUST record its net stack
//! effect — normally via `ControlStack::apply_stack_effects_to_height`, or
//! `ControlStack::set_height` where the exact resulting height is known. Both
//! are no-ops while the current block is traversing dead code, so height updates
//! placed in unreachable code are safely dropped rather than underflowing the
//! `u32` height on a stack-polymorphic operand.

use crate::{
    error::TraceWasmError,
    module::{FuncDecl, FuncIndex, FuncType},
};
use wasmparser::{BlockType, Operator, OperatorsReader};

/// A lowered TraceWasm instruction.
///
/// Only structured control flow is modelled today; value/numeric/memory
/// operators are rejected as unsupported by the lowering pass. Index fields
/// (`end_index`, `else_index`, `target_index`, ...) are *absolute* positions
/// into the containing `Vec<Instruction>`, i.e. runtime program counters.
#[derive(Debug, Clone)]
pub enum Instruction {
    Unreachable,
    Nop,
    Block {
        /// Absolute index of this block's matching `End`. Backpatched; a branch
        /// that targets this block jumps here.
        end_index: usize,
    },
    Loop,
    If {
        /// Absolute index of the matching `Else`, if one exists. Backpatched at
        /// `End`.
        else_index: Option<usize>,
        /// Absolute index of this `if`'s matching `End`. Backpatched.
        end_index: usize,
    },
    Else {
        /// Absolute index of the owning `if`'s `End`. When the then-branch falls
        /// through into `else`, control skips to this `End`. Backpatched.
        if_end_index: usize,
    },
    End {
        /// Number of result values the just-closed block leaves on the stack.
        arity: u32,
        /// Height the stack unwinds to before the `arity` results are kept. See
        /// `Block::recorded_height`.
        recorded_height: u32,
    },
    Br {
        /// Absolute jump target. For a `loop` label this is the `Loop`
        /// instruction (a back-edge / "continue"); otherwise it is the label's
        /// `End`. Backpatched (with `usize::MAX` sentinel) for non-loop targets.
        target_index: usize,
        /// Number of values transferred to the label (loop params, else results).
        arity: u32,
        /// Stack height the target label unwinds to; see `Block::recorded_height`.
        recorded_height: u32,
    },
    BrIf {
        /// See `Br::target_index`. Same target rules as `Br`.
        target_index: usize,
        arity: u32,
        recorded_height: u32,
    },
    BrTable {
        /// One [`TargetBranch`] per explicit label, in order, followed by the
        /// default label as the final element.
        targets: Vec<TargetBranch>,
    },
    Return {
        /// Absolute index of the function's `End`. Backpatched (`usize::MAX`
        /// sentinel) — `return` is a branch to the outermost function label.
        target_index: usize,
        /// Number of result values the function returns.
        arity: u32,
        /// Stack height the function frame unwinds to before the `arity` results
        /// are kept; always 0 for the function frame.
        recorded_height: u32,
    },
    Call {
        func_index: FuncIndex,
        params_count: u32,
    },
}

/// One resolved arm of a `br_table`: where to jump and how to reshape the stack.
///
/// Each arm carries its own `recorded_height`/`arity` because a single
/// `br_table` may legally mix loop and non-loop targets (validation only
/// requires the label *types* to match); their unwind targets and heights
/// differ even though the value counts agree.
#[derive(Debug, Clone)]
pub struct TargetBranch {
    /// Absolute jump target (loop start or label `End`). Backpatched for
    /// non-loop targets.
    pub target_index: usize,
    pub arity: u32,
    pub recorded_height: u32,
}

/// What kind of label a control-stack entry represents, plus the data needed to
/// backpatch its originating instruction once its `end` is seen.
enum BlockKind {
    /// The implicit outermost frame for a function body. Not backpatched (its
    /// `end` is the final instruction and no branch instruction stores its
    /// index directly).
    Func,
    /// A `block`. `index` is the position of its `Instruction::Block`, so the
    /// `end_index` field can be filled in later.
    Block { index: usize },
    /// A `loop`. `index` is the position of its `Instruction::Loop`; this is the
    /// back-edge target used directly by branches (no backpatching needed).
    Loop { index: usize },
    /// An `if`. `index` locates the `Instruction::If`; `else_index` is filled in
    /// when the `Else` operator is seen (if any) so both can be backpatched at
    /// `end`.
    If {
        index: usize,
        else_index: Option<usize>,
    },
}

impl BlockKind {
    /// Returns the loop's start-instruction index iff this label is a `loop`.
    ///
    /// Branch lowering keys off this: a branch to a loop is a back-edge whose
    /// target is known immediately (the loop start) and whose arity is the
    /// loop's *params*; a branch to any other label is a forward jump to its
    /// `end` whose arity is the label's *results*.
    fn is_loop(&self) -> Option<usize> {
        if let BlockKind::Loop { index } = self {
            Some(*index)
        } else {
            None
        }
    }
}

/// A live control-flow label on the [`ControlStack`].
struct Block {
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
    /// Each entry is `(instruction_index, brtable_target_slot)`. The second
    /// field selects which arm of a `BrTable::targets` vec to patch; it is
    /// `usize::MAX` (unused) for `Br`/`BrIf`, which have a single `target_index`.
    attached_breaks: Vec<(usize, usize)>,
}

/// The stack of currently-open control-flow labels, plus the running
/// operand-stack height.
///
/// Index 0 (when present) is the implicit function frame; the last element is
/// the innermost open label. A `br relative_depth` resolves to
/// `inner[len - 1 - relative_depth]`.
#[derive(Default)]
struct ControlStack {
    inner: Vec<Block>,
    /// Current operand-stack depth at the point the pass has reached. See the
    /// module-level "Height-tracking invariant".
    curr_height: u32,
}

impl ControlStack {
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

    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Pushes a new label for a `block`/`loop`/`if`, capturing its
    /// `recorded_height` from the current stack state.
    ///
    /// If the parent is already dead, the child inherits deadness
    /// (`has_inherited = true`) and its `recorded_height` is left `0` because it
    /// will never be consulted at runtime. Otherwise `recorded_height` is the
    /// height *below* the label's params (the params are allowed to be consumed
    /// by the block, so they are not part of the unwind height). An `if`
    /// additionally has the branch condition sitting on top of the params, so it
    /// subtracts one more for it. (The condition is popped from `curr_height` by
    /// the `If` arm's `PopPush` stack effect, not here.)
    fn add_block(&mut self, kind: BlockKind, blockty: &BlockType, types: &[FuncType]) {
        let (params, results) = Self::params_and_results_from_blockty(blockty, types);

        let is_unreachable_traversing = self
            .inner
            .last()
            .is_some_and(|b| b.is_unreachable_traversing);

        if is_unreachable_traversing {
            self.inner.push(Block {
                kind,
                recorded_height: 0, // this won't be used at runtime because of unreachablity
                params,
                results,
                is_unreachable_traversing,
                has_inherited: true,
                attached_breaks: vec![],
            });

            return;
        }

        let recorded_height = match kind {
            BlockKind::Func => 0,
            BlockKind::Block { .. } => self.curr_height - params,
            BlockKind::Loop { .. } => self.curr_height - params,
            BlockKind::If { .. } => {
                // top is the `if` condition and then params
                self.curr_height - params - 1
            }
        };

        self.inner.push(Block {
            kind,
            recorded_height,
            params,
            results,
            is_unreachable_traversing: false,
            has_inherited: false,
            attached_breaks: vec![],
        });
    }

    /// Marks the current (innermost) block's remaining body as dead code. Called
    /// after unconditional control transfers (`unreachable`, `br`, `br_table`,
    /// `return`).
    fn set_unreachable_traversing(&mut self) {
        let curr_block = self.get_curr_block_mut();
        curr_block.is_unreachable_traversing = true;
    }

    /// Clears the current block's dead-code flag — but only if the block became
    /// dead *locally*.
    ///
    /// A block that was born dead (`has_inherited`) stays dead: both arms of an
    /// `if` opened inside unreachable code are unreachable, so an intervening
    /// `else` must not mark the else-arm live. Skipping the clear here keeps
    /// `curr_height` frozen through the whole dead subtree, so it is restored
    /// correctly only when a genuinely-live ancestor's `end` runs.
    fn end_unreachable_traversing(&mut self) {
        let curr_block = self.get_curr_block_mut();

        if curr_block.has_inherited {
            return;
        }

        curr_block.is_unreachable_traversing = false;
    }

    fn get_block_mut(&mut self, index: usize) -> &mut Block {
        &mut self.inner[index]
    }

    fn get_curr_block(&self) -> &Block {
        debug_assert!(!self.inner.is_empty());
        &self.inner[self.inner.len() - 1]
    }

    fn get_curr_block_mut(&mut self) -> &mut Block {
        debug_assert!(!self.inner.is_empty());
        let len = self.inner.len();
        &mut self.inner[len - 1]
    }

    fn pop(&mut self) -> Option<Block> {
        self.inner.pop()
    }

    /// Sets `curr_height`, unless the current block is traversing dead code.
    ///
    /// The dead-code guard is what makes it safe for branch/`end` handlers to
    /// compute heights unconditionally: once a block goes unreachable, its
    /// height is frozen until the block's `else`/`end` recomputes it from
    /// `recorded_height`, so any writes attempted by dead instructions are
    /// dropped here rather than corrupting the model (or underflowing).
    ///
    /// NOTE: use this when the exact resulting height is already known — e.g. at
    /// `else`/`end`, which reset to `recorded_height + arity`. For an operator
    /// described by its pop/push counts, prefer
    /// [`Self::apply_stack_effects_to_height`], which derives the new height from
    /// the current one instead of requiring the caller to compute it.
    fn set_height(&mut self, height: u32) {
        if self.inner.is_empty() {
            self.curr_height = height;

            return;
        }

        // height is not changed by the instructions which are unreachable.
        // These instructions typically occur after unconditional br instructions.
        if self.get_curr_block().is_unreachable_traversing {
            return;
        }

        self.curr_height = height;
    }

    /// Applies an operator's net stack effect: `curr_height -= pops; curr_height += pushes`.
    /// This is the default for ordinary operators described by their pop/push counts.
    ///
    /// NOTE: the dead-code guard is load-bearing, not just an optimization. The
    /// arithmetic is skipped entirely while the current block is traversing dead
    /// code, where `curr_height` is frozen (and may be below `pops`, since dead
    /// code is stack-polymorphic) — evaluating `curr_height - pops` there would
    /// underflow the `u32`. Guarding before the subtraction is why callers like
    /// `br_if`/`call` can invoke this unconditionally.
    fn apply_stack_effects_to_height(&mut self, pops: u32, pushes: u32) {
        if self.inner.is_empty() {
            return;
        }

        if self.get_curr_block().is_unreachable_traversing {
            return;
        }

        self.curr_height = self.curr_height - pops + pushes;
    }
}

/// How an operator affects the tracked operand-stack height, returned by every
/// match arm of the lowering pass so the effect is applied in exactly one place.
///
/// Requiring each arm to produce one of these makes it impossible to silently
/// forget a height update — the compiler forces the arm to state its effect.
enum StackEffectResult {
    /// The operator pops `pops` values and pushes `pushes`; the net change is
    /// applied to `curr_height` (skipped while traversing dead code).
    PopPush { pops: u32, pushes: u32 },
    /// The operator resets the height to a known absolute value, e.g. `else`/`end`
    /// restoring `recorded_height + arity`.
    SetHeight(u32),
    /// The operator leaves the stack height unchanged.
    NoEffect,
    /// No height needs recording. Returned by unconditional branches
    /// (`br`/`br_table`/`return`): the instructions following them up to the
    /// enclosing `end` are unreachable anyway, and reaching that `end` always
    /// resets the height correctly to the block's `recorded_height + results`.
    Unreachable,
}

impl Instruction {
    /// Lowers one operator stream into a flat `Vec<Instruction>` with control
    /// flow resolved and stack heights precomputed.
    ///
    /// `is_func` is `Some((params, results))` for a function body, in which case
    /// an implicit [`BlockKind::Func`] frame is pushed to catch top-level
    /// branches and the trailing `end`. It is `None` for constant expressions
    /// (global/table/element/data init), which carry no root frame; their
    /// terminating `end` has nothing to pop and simply ends the pass (see the
    /// `Operator::End` arm).
    ///
    /// `types` is the module's type section, used to resolve `BlockType::FuncType`
    /// arities. `func_decls` is the module's function declarations, used by
    /// `Call` to resolve a callee's parameter count.
    pub(crate) fn emit_instruction_from_operator_reader(
        mut operator_reader: OperatorsReader<'_>,
        is_func: Option<(u32, u32)>, // arity of the function
        types: &[FuncType],
        func_decls: &[FuncDecl],
    ) -> Result<Vec<Instruction>, TraceWasmError> {
        let mut instructions: Vec<Instruction> = vec![];
        let mut control_stack: ControlStack = ControlStack::default();

        if let Some((params, results)) = is_func {
            control_stack.inner.push(Block {
                kind: BlockKind::Func,
                recorded_height: 0, // functions always have recorded height to be 0, so they leave stack with just its results
                params,
                results,
                is_unreachable_traversing: false,
                has_inherited: false,
                attached_breaks: vec![],
            });
        }

        while !operator_reader.eof() {
            let operator = operator_reader.read()?;

            let (instruction, stack_effect): (Instruction, StackEffectResult) = match operator {
                Operator::Unreachable => {
                    // all instructions after this is unreachable until the end of the current block
                    control_stack.set_unreachable_traversing();

                    (Instruction::Unreachable, StackEffectResult::NoEffect)
                }
                Operator::Nop => (Instruction::Nop, StackEffectResult::NoEffect),
                Operator::Block { blockty } => {
                    control_stack.add_block(
                        BlockKind::Block {
                            index: instructions.len(),
                        },
                        &blockty,
                        types,
                    );

                    (
                        Instruction::Block {
                            end_index: usize::MAX, // dummy value! will backpath when we see END for this block
                        },
                        StackEffectResult::NoEffect,
                    )
                }
                Operator::Loop { blockty } => {
                    control_stack.add_block(
                        BlockKind::Loop {
                            index: instructions.len(),
                        },
                        &blockty,
                        types,
                    );

                    (Instruction::Loop, StackEffectResult::NoEffect)
                }
                Operator::If { blockty } => {
                    control_stack.add_block(
                        BlockKind::If {
                            index: instructions.len(),
                            else_index: None,
                        },
                        &blockty,
                        types,
                    );

                    (
                        Instruction::If {
                            else_index: None,
                            end_index: usize::MAX, // dummy value! will backpath when we see END for this `if`
                        },
                        StackEffectResult::PopPush { pops: 1, pushes: 0 },
                    )
                }
                Operator::Else => {
                    let index = instructions.len();
                    let block = control_stack.get_curr_block_mut();
                    let recorded_height = block.recorded_height;
                    let params = block.params;

                    let BlockKind::If {
                        index: _index,
                        else_index,
                    } = &mut block.kind
                    else {
                        unreachable!(
                            "hitting this means TraceWasm has a bug recording the instructions"
                        )
                    };

                    *else_index = Some(index); // backpatching the `else` index in the `if` block

                    // `else` instruction ends the unreachable traversing because those instructions
                    // at runtime can execute if the `if` branch is not taken! The else block first instruction
                    // would see the height to be `recorded_heigh (at the if) + params` (condition is already popped).
                    //
                    // `end_unreachable_traversing` is a no-op when the `if` was born in dead code
                    // (`has_inherited`), so a dead `if` correctly keeps both arms dead; and `set_height`'s
                    // own guard then leaves `curr_height` frozen in that case.
                    control_stack.end_unreachable_traversing();

                    (
                        Instruction::Else {
                            if_end_index: usize::MAX,
                        }, // dummy value! will backpatch when we see END for the `if` of this `else`
                        StackEffectResult::SetHeight(recorded_height + params),
                    )
                }
                Operator::Br { relative_depth } => {
                    // NOTE on Branching: each branch instruction will resolve to a particular block based on the relative_depth provided.
                    // This block dictates the recorded_height and arity which this branch instruction should leave the stack in.
                    // - If the resolved block is a loop, then the target for the branch is back to the `loop` instruction (this means `continue`).
                    // - If the resolver block is not a loop (i.e. block/if/function), then the target for the branch is `end` of that block.
                    // Executing `end` always leave the stack with heigh = recorded_heigh + results, even for loops!
                    let block_index = control_stack.len() - 1 - relative_depth as usize;
                    let block = control_stack.get_block_mut(block_index); // extract the block to which this `br` applies to using `relative_depth`
                    let params = block.params;
                    let results = block.results;
                    let recorded_height = block.recorded_height;
                    let index = instructions.len();

                    // brs with a depth resolved to a "loop" block targets the loop start and so the arity
                    // will be params of the loop. For other blocks, the br targets the end of that block
                    let instr = if let Some(loop_index) = block.kind.is_loop() {
                        Instruction::Br {
                            target_index: loop_index, // correct target index,
                            arity: params,
                            recorded_height,
                        }
                    } else {
                        block.attached_breaks.push((index, usize::MAX));

                        Instruction::Br {
                            target_index: usize::MAX, // dummy value! will backpatch when we see END for the block this `br` is attached to
                            arity: results,
                            recorded_height,
                        }
                    };

                    // `br` is unconditional, so no height update is needed here: the call below freezes
                    // `curr_height` until this block's `else`/`end` recomputes it from `recorded_height`.
                    // Any write we made now would land in dead code and be discarded — this is also why
                    // `br_table` (equally unconditional) omits it while `br_if` (conditional) does not.
                    // all the instructions after this till the `end` of the current block are unreachable!
                    control_stack.set_unreachable_traversing();

                    (instr, StackEffectResult::Unreachable)
                }
                Operator::BrIf { relative_depth } => {
                    // Same target/arity resolution as `Br` (see its notes), but `br_if` is *conditional*:
                    // it pops an i32 predicate and, when not taken, falls through. The fall-through path is
                    // reachable, so we must NOT mark the block unreachable; we only account for the popped
                    // condition below. The label's values remain on the stack on fall-through, hence the
                    // net effect on `curr_height` is exactly -1.
                    let block_index = control_stack.len() - 1 - relative_depth as usize;
                    let block = control_stack.get_block_mut(block_index); // extract the block to which this `br` applies to using `relative_depth`
                    let params = block.params;
                    let results = block.results;
                    let recorded_height = block.recorded_height;
                    let index = instructions.len();

                    let instr = if let Some(loop_index) = block.kind.is_loop() {
                        Instruction::BrIf {
                            target_index: loop_index, // correct target index,
                            arity: params,
                            recorded_height,
                        }
                    } else {
                        block.attached_breaks.push((index, usize::MAX));

                        Instruction::BrIf {
                            target_index: usize::MAX,
                            arity: results,
                            recorded_height,
                        } // dummy value! will backpatch when we see END for the block this `br` is attached to
                    };

                    // instructions following the br_if means branch was not taken and those instruction would see the above height
                    (instr, StackEffectResult::PopPush { pops: 1, pushes: 0 })
                }
                Operator::BrTable { targets: table } => {
                    // `br_table` selects among N explicit labels plus a default. All arms share the same
                    // label type, but each arm is lowered independently because their targets/heights
                    // differ (a loop arm jumps back with `params`; a block arm jumps forward to `end` with
                    // `results`). Like `br`, it is unconditional, so no height update precedes the
                    // `set_unreachable_traversing` below.
                    let targets = table.targets();
                    let mut targets = targets.collect::<Result<Vec<_>, _>>()?;

                    targets.push(table.default()); // default (last element) is taken when the popped index is out of range, i.e. i >= number of explicit targets

                    let index = instructions.len();
                    let mut br_targets = vec![];

                    for (i, &relative_depth) in targets.iter().enumerate() {
                        let block_index = control_stack.len() - 1 - relative_depth as usize;
                        let block = control_stack.get_block_mut(block_index); // extract the block to which this `br` applies to using `relative_depth`
                        let params = block.params;
                        let results = block.results;
                        let recorded_height = block.recorded_height;

                        if let Some(loop_index) = block.kind.is_loop() {
                            br_targets.push(TargetBranch {
                                target_index: loop_index,
                                arity: params,
                                recorded_height,
                            });
                        } else {
                            // record which arm (`i`) of this `br_table` to backpatch at the target's `end`
                            block.attached_breaks.push((index, i));

                            // dummy value! will backpatch when we see END for the block this `br` is attached to
                            br_targets.push(TargetBranch {
                                target_index: usize::MAX,
                                arity: results,
                                recorded_height,
                            });
                        };
                    }

                    control_stack.set_unreachable_traversing();

                    (
                        Instruction::BrTable {
                            targets: br_targets,
                        },
                        StackEffectResult::Unreachable,
                    )
                }
                Operator::Return => {
                    // `return` is an unconditional branch to the outermost function label: it targets
                    // the function's `end`, transfers the function results, and unwinds to the frame's
                    // base (recorded_height 0). Handled like a `br` to block 0 — attached for backpatch
                    // and, being unconditional, followed by `set_unreachable_traversing`.
                    let func_block = control_stack.get_block_mut(0); // function is top-most block
                    let results = func_block.results;
                    let recorded_height = func_block.recorded_height;
                    let index = instructions.len();

                    func_block.attached_breaks.push((index, usize::MAX));
                    control_stack.set_unreachable_traversing();

                    (
                        Instruction::Return {
                            target_index: usize::MAX,
                            arity: results,
                            recorded_height,
                        },
                        StackEffectResult::Unreachable,
                    )
                }
                Operator::Call { function_index } => {
                    let func_decl = &func_decls[function_index as usize];
                    let ty = &types[func_decl.ty.0 as usize];
                    let params = &ty.params;
                    let results = &ty.results;

                    (
                        Instruction::Call {
                            func_index: FuncIndex(function_index),
                            params_count: params.len() as u32,
                        },
                        StackEffectResult::PopPush {
                            pops: params.len() as u32,
                            pushes: results.len() as u32,
                        },
                    )
                }
                Operator::End => {
                    // the non-function operator stream in Table init/Global init/Element/Data ends with an extra `end` which has no matching block start
                    // so popping from the control stack might give a `None`, we just return the lowered instructions.
                    let Some(block) = control_stack.pop() else {
                        // this is entered only when is_func is `None` i.e. Table init/Global init/Element/Data
                        debug_assert!(is_func.is_none());

                        return Ok(instructions);
                    };

                    let results = block.results;
                    let recorded_height = block.recorded_height;
                    let attached_breaks = &block.attached_breaks;
                    let index = instructions.len();

                    // Backpatch every forward branch that targeted this block: its jump target is this
                    // `end`. For `br`/`br_if` there is a single `target_index`; for `br_table` the second
                    // tuple field selects the specific arm to patch. Loops never appear here because a
                    // branch to a loop resolves to the loop start immediately and is not attached.
                    for (br_index, br_targets_index) in attached_breaks {
                        match &mut instructions[*br_index] {
                            Instruction::Br {
                                target_index,
                                arity: _arity,
                                recorded_height: _recorded_height,
                            } => {
                                *target_index = index;
                            }
                            Instruction::BrIf {
                                target_index,
                                arity: _arity,
                                recorded_height: _recorded_height,
                            } => {
                                *target_index = index;
                            }
                            Instruction::BrTable { targets } => {
                                targets[*br_targets_index].target_index = index;
                            }
                            Instruction::Return {
                                target_index,
                                arity: _arity,
                                recorded_height: _recorded_height,
                            } => {
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
                        BlockKind::Func | BlockKind::Loop { .. } => {} // no backpatching required
                        BlockKind::Block { index: block_index } => {
                            let Instruction::Block { end_index } = &mut instructions[block_index]
                            else {
                                unreachable!(
                                    "hitting this means TraceWasm has a bug recording the instructions"
                                )
                            };

                            *end_index = index;
                        }
                        BlockKind::If {
                            index: if_index,
                            else_index: ei,
                        } => {
                            // Fill the `if`'s `else_index` and `end_index` ...
                            let Instruction::If {
                                else_index,
                                end_index,
                            } = &mut instructions[if_index]
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
                                let Instruction::Else { if_end_index } =
                                    &mut instructions[else_index]
                                else {
                                    unreachable!(
                                        "hitting this means TraceWasm has a bug recording the instructions"
                                    )
                                };

                                *if_end_index = index;
                            }
                        }
                    }

                    (
                        Instruction::End {
                            arity: results,
                            recorded_height,
                        },
                        StackEffectResult::SetHeight(recorded_height + results),
                    )
                }
                _ => {
                    return Err(TraceWasmError::Unsupported(format!(
                        "instruction `{:?}`",
                        operator
                    )));
                }
            };

            match stack_effect {
                StackEffectResult::PopPush { pops, pushes } => {
                    control_stack.apply_stack_effects_to_height(pops, pushes)
                }
                StackEffectResult::SetHeight(height) => control_stack.set_height(height),
                StackEffectResult::NoEffect | StackEffectResult::Unreachable => {}
            }

            instructions.push(instruction);
        }

        Ok(instructions)
    }
}
