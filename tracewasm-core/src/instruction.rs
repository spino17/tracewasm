use crate::{ast::FuncType, error::TraceWasmError};
use wasmparser::{BlockType, Operator, OperatorsReader};

pub enum Instruction {
    Unreachable,
    Nop,
    Block {
        blockty: BlockType,
        end_index: usize,
    },
    Loop {
        blockty: BlockType,
    },
    If {
        blockty: BlockType,
        else_index: Option<usize>,
        end_index: usize,
    },
    Else {
        if_end_index: usize,
    },
    End {
        arity: u32,
        recorded_height: u32,
    },
    Br {
        target_index: usize,
        arity: u32,
        recorded_height: u32,
    },
    BrIf {
        target_index: usize,
        arity: u32,
        recorded_height: u32,
    },
    BrTable {
        targets: Vec<TargetBranch>, // (target, arity, recorded_height)
    },
}

pub struct TargetBranch {
    pub target_index: usize,
    pub arity: u32,
    pub recorded_height: u32,
}

enum BlockKind {
    Func,
    Block {
        index: usize,
    },
    Loop {
        index: usize,
    },
    If {
        index: usize,
        else_index: Option<usize>,
    },
}

impl BlockKind {
    fn is_loop(&self) -> Option<usize> {
        if let BlockKind::Loop { index } = self {
            Some(*index)
        } else {
            None
        }
    }
}

struct Block {
    kind: BlockKind,
    recorded_height: u32,
    params: u32,
    results: u32,
    is_unreachable_traversing: bool,
    has_inherited: bool,
    // brs which branched to this block and hence needs to be backpatched their
    // target index with the `end` index of this block
    attached_breaks: Vec<(usize, usize)>, // second index is for BrTable, targets index
}

#[derive(Default)]
struct ControlStack {
    inner: Vec<Block>,
    curr_height: u32,
}

impl ControlStack {
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

    fn add_block(&mut self, kind: BlockKind, blockty: &BlockType, types: &[FuncType]) {
        let (params, results) = Self::params_and_results_from_blockty(&blockty, types);

        let is_unreachable_traversing = self
            .inner
            .last()
            .map_or(false, |b| b.is_unreachable_traversing);

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

        // each block when ends leave the stack with height = recorded_height + arity (results or params in case of loops)
        // any params or condition is not part of the recorded height because its allowed to be consumed by the block.
        let recorded_height = match kind {
            BlockKind::Func => 0,
            BlockKind::Block { .. } => self.curr_height - params,
            BlockKind::Loop { .. } => self.curr_height - params,
            BlockKind::If { .. } => {
                let recorded_height = self.curr_height - params - 1; // top is the `if` condition and then params
                self.set_height(self.curr_height - 1); // if pops the condition

                recorded_height
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

    fn set_unreachable_traversing(&mut self) {
        let curr_block = self.get_curr_block_mut();
        curr_block.is_unreachable_traversing = true;
    }

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

    fn set_height(&mut self, height: u32) {
        if self.inner.is_empty() {
            self.curr_height = height;

            return;
        }

        // height is not changed by the instructions which are unreachable.
        // These instructions typically occur after unconditional br instructions.
        if self.get_curr_block().is_unreachable_traversing == true {
            return;
        }

        self.curr_height = height;
    }

    fn increase_height_by(&mut self, delta: u32) {
        self.set_height(self.curr_height + delta);
    }

    fn decrease_height_by(&mut self, delta: u32) {
        self.set_height(self.curr_height - delta);
    }

    fn pop(&mut self) -> Option<Block> {
        self.inner.pop()
    }
}

impl Instruction {
    pub(crate) fn emit_instruction_from_operator_reader(
        mut operator_reader: OperatorsReader<'_>,
        is_func: Option<(u32, u32)>, // arity of the function
        types: &[FuncType],
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

            let instruction = match operator {
                Operator::Unreachable => {
                    // all instructions after this is unreachable until the end of the current block
                    control_stack.set_unreachable_traversing();

                    Instruction::Unreachable
                }
                Operator::Nop => Instruction::Nop,
                Operator::Block { blockty } => {
                    control_stack.add_block(
                        BlockKind::Block {
                            index: instructions.len(),
                        },
                        &blockty,
                        types,
                    );

                    Instruction::Block {
                        blockty,
                        end_index: usize::MAX, // dummy value! will backpath when we see END for this block
                    }
                }
                Operator::Loop { blockty } => {
                    control_stack.add_block(
                        BlockKind::Loop {
                            index: instructions.len(),
                        },
                        &blockty,
                        types,
                    );

                    Instruction::Loop { blockty }
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

                    Instruction::If {
                        blockty,
                        else_index: None,
                        end_index: usize::MAX, // dummy value! will backpath when we see END for this `if`
                    }
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

                    // all the instructions after this till the `end` of the current block are unreachable!
                    control_stack.set_unreachable_traversing();

                    instr
                }
                Operator::BrIf { relative_depth } => {
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
                    control_stack.decrease_height_by(1); // condition is popped

                    instr
                }
                Operator::BrTable { targets: table } => {
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

                    Instruction::BrTable {
                        targets: br_targets,
                    }
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
                    // would see the height to be `recorded_heigh (at the if) + params` (condition is already popped)
                    control_stack.end_unreachable_traversing();
                    control_stack.set_height(recorded_height + params);

                    Instruction::Else {
                        if_end_index: usize::MAX,
                    } // dummy value! will backpatch when we see END for the `if` of this `else`
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

                    // if this block is not a `loop` then backpatch it's br instructions
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
                            _ => unreachable!(
                                "hitting this means TraceWasm has a bug recording the instructions"
                            ),
                        }
                    }

                    // backpath the respective blocks of this `end`
                    match block.kind {
                        BlockKind::Func | BlockKind::Loop { .. } => {} // no backpatching required
                        BlockKind::Block { index: block_index } => {
                            let Instruction::Block {
                                blockty: _blockty,
                                end_index,
                            } = &mut instructions[block_index]
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
                            let Instruction::If {
                                blockty: _blockty,
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

                    control_stack.set_height(recorded_height + results);

                    Instruction::End {
                        arity: results,
                        recorded_height,
                    }
                }
                _ => {
                    return Err(TraceWasmError::Unsupported(format!(
                        "instruction `{:?}`",
                        operator
                    )));
                }
            };

            instructions.push(instruction);
        }

        Ok(instructions)
    }
}
