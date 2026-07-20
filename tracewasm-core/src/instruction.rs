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
        end_index: usize,
    },
    If {
        blockty: BlockType,
        else_index: Option<usize>,
        end_index: usize,
    },
    Else,
    End,
    Br {
        target_index: u32,
    }, // target instruction index in the vector of instructions i.e. pc
}

enum BlockKind {
    Func,
    Block,
    Loop,
    If {
        index: usize,
        else_index: Option<usize>,
    },
}

struct Block {
    kind: BlockKind,
    attached_breaks: Vec<usize>,
}

impl Instruction {
    pub(crate) fn emit_instruction_from_operator_reader(
        mut operator_reader: OperatorsReader<'_>,
        is_func: bool,
    ) -> Result<Vec<Instruction>, anyhow::Error> {
        let mut instructions: Vec<Instruction> = vec![];
        let mut control_stack: Vec<Block> = vec![];

        if is_func {
            control_stack.push(Block {
                kind: BlockKind::Func,
                attached_breaks: vec![],
            });
        }

        while !operator_reader.eof() {
            let operator = operator_reader.read()?;

            let instruction = match operator {
                Operator::Unreachable => Instruction::Unreachable,
                Operator::Nop => Instruction::Nop,
                Operator::Block { blockty } => {
                    todo!()
                }
                Operator::Loop { blockty } => todo!(),
                Operator::If { blockty } => {
                    // record the index of this if inside instructions
                    // calls pop -> cond
                    // then pops the params (they stay on the stack just accounts for reduced recorded height)
                    // if cond is false then forward to else arm, so would need else + 1 index in instructions! NEEDED
                    // if there is no else then it should forward to the end of the if! NEEDED
                    let index = instructions.len();

                    control_stack.push(Block {
                        kind: BlockKind::If {
                            index,
                            else_index: None,
                        },
                        attached_breaks: vec![],
                    });

                    Instruction::If {
                        blockty,
                        else_index: None,
                        end_index: 0, // dummy value! will backpath when we see END for this `if`
                    }
                }
                Operator::Else => {
                    let index = instructions.len();
                    let curr_control_stack_len = control_stack.len();
                    let block = &mut control_stack[curr_control_stack_len - 1];

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

                    Instruction::Else
                }
                Operator::Br { relative_depth } => {
                    let block_index = control_stack.len() - 1 - relative_depth as usize;
                    let block = &mut control_stack[block_index];
                    let index = instructions.len();

                    block.attached_breaks.push(index);

                    Instruction::Br { target_index: 0 } // dummy value! will backpath when we see END for this block
                }
                Operator::End => {
                    if control_stack.len() == 0 && is_func {
                        return Ok(instructions);
                    }

                    let block = control_stack.pop().unwrap(); // validated already by wasmparser.
                    let attached_breaks = &block.attached_breaks;
                    let index = instructions.len();

                    for &br in attached_breaks {
                        let Instruction::Br { target_index } = &mut instructions[br] else {
                            unreachable!(
                                "hitting this means TraceWasm has a bug recording the instructions"
                            )
                        };

                        *target_index = index as u32; // backpatching the breaks with this end's index
                    }

                    // backpath the respective blocks of this `end`
                    match block.kind {
                        BlockKind::Func => todo!(),
                        BlockKind::Block => todo!(),
                        BlockKind::Loop => todo!(),
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
                        }
                    }

                    Instruction::End
                }
                _ => todo!(),
            };

            instructions.push(instruction);
        }

        Ok(instructions)
    }
}
