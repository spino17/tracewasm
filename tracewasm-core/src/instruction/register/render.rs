use crate::{
    instruction::register::{
        RegFrameLayout, RegInstruction, RegLoweredFuncBody, Registers, Slot, mnemonic,
    },
    module::FuncType,
};

impl RegInstruction {
    pub fn render(&self, frame: &RegFrameLayout, types: &[FuncType]) -> String {
        let ins = &frame.input_registers_arena;
        let outs = &frame.output_registers_arena;

        let sig1 = |i: &Registers<1, Slot>| i.registers(ins)[0].render();
        let list = |xs: &[Slot]| xs.iter().map(|x| x.render()).collect::<Vec<_>>().join(", ");

        let regs = |xs: &[u32]| {
            xs.iter()
                .map(|r| format!("r{r}"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        // Every pure value operator renders alike, so the arms below carry no body of
        // their own. They are split by arity because an or-pattern binds one type, and
        // then by family, so they scan in the order the enum declares them.
        let load_op = |kind, offset: u32, inputs: &[Slot], outputs: &[u32]| {
            format!(
                "{:<12} [{}]+{offset} -> {}",
                mnemonic(kind),
                inputs[0].render(),
                regs(outputs)
            )
        };

        let store_op = |kind, offset: u32, inputs: &[Slot]| {
            format!(
                "{:<12} [{}]+{offset} <- {}",
                mnemonic(kind),
                inputs[0].render(),
                inputs[1].render()
            )
        };

        let value_op = |kind, inputs: &[Slot], outputs: &[u32]| {
            format!(
                "{:<12} {} -> {}",
                mnemonic(kind),
                list(inputs),
                regs(outputs)
            )
        };

        match self {
            // Numeric instructions, in the order the enum declares them.
            //
            // Split by shape because an or-pattern binds one type, and the three
            // shapes bind different ones — but each body is a single call, so the
            // arms stay one line and the families stay in step with the enum.
            // i32 — loads
            RegInstruction::I32Load { offset, sig }
            | RegInstruction::I32Load8S { offset, sig }
            | RegInstruction::I32Load8U { offset, sig }
            | RegInstruction::I32Load16S { offset, sig }
            | RegInstruction::I32Load16U { offset, sig } => load_op(
                self.kind(),
                *offset,
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // i32 — stores
            RegInstruction::I32Store { offset, sig }
            | RegInstruction::I32Store8 { offset, sig }
            | RegInstruction::I32Store16 { offset, sig } => {
                store_op(self.kind(), *offset, sig.input.registers(ins))
            }

            // i32 — unary
            RegInstruction::I32Clz(sig)
            | RegInstruction::I32Ctz(sig)
            | RegInstruction::I32Eqz(sig)
            | RegInstruction::I32Extend16S(sig)
            | RegInstruction::I32Extend8S(sig)
            | RegInstruction::I32Popcnt(sig)
            | RegInstruction::I32ReinterpretF32(sig)
            | RegInstruction::I32TruncF32S(sig)
            | RegInstruction::I32TruncF32U(sig)
            | RegInstruction::I32TruncF64S(sig)
            | RegInstruction::I32TruncF64U(sig)
            | RegInstruction::I32TruncSatF32S(sig)
            | RegInstruction::I32TruncSatF32U(sig)
            | RegInstruction::I32TruncSatF64S(sig)
            | RegInstruction::I32TruncSatF64U(sig)
            | RegInstruction::I32WrapI64(sig) => value_op(
                self.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // i32 — binary
            RegInstruction::I32Add(sig)
            | RegInstruction::I32And(sig)
            | RegInstruction::I32DivS(sig)
            | RegInstruction::I32DivU(sig)
            | RegInstruction::I32Eq(sig)
            | RegInstruction::I32GeS(sig)
            | RegInstruction::I32GeU(sig)
            | RegInstruction::I32GtS(sig)
            | RegInstruction::I32GtU(sig)
            | RegInstruction::I32LeS(sig)
            | RegInstruction::I32LeU(sig)
            | RegInstruction::I32LtS(sig)
            | RegInstruction::I32LtU(sig)
            | RegInstruction::I32Mul(sig)
            | RegInstruction::I32Ne(sig)
            | RegInstruction::I32Or(sig)
            | RegInstruction::I32RemS(sig)
            | RegInstruction::I32RemU(sig)
            | RegInstruction::I32Rotl(sig)
            | RegInstruction::I32Rotr(sig)
            | RegInstruction::I32Shl(sig)
            | RegInstruction::I32ShrS(sig)
            | RegInstruction::I32ShrU(sig)
            | RegInstruction::I32Sub(sig)
            | RegInstruction::I32Xor(sig) => value_op(
                self.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // i64 — loads
            RegInstruction::I64Load { offset, sig }
            | RegInstruction::I64Load8S { offset, sig }
            | RegInstruction::I64Load8U { offset, sig }
            | RegInstruction::I64Load16S { offset, sig }
            | RegInstruction::I64Load16U { offset, sig }
            | RegInstruction::I64Load32S { offset, sig }
            | RegInstruction::I64Load32U { offset, sig } => load_op(
                self.kind(),
                *offset,
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // i64 — stores
            RegInstruction::I64Store { offset, sig }
            | RegInstruction::I64Store8 { offset, sig }
            | RegInstruction::I64Store16 { offset, sig }
            | RegInstruction::I64Store32 { offset, sig } => {
                store_op(self.kind(), *offset, sig.input.registers(ins))
            }

            // i64 — unary
            RegInstruction::I64Clz(sig)
            | RegInstruction::I64Ctz(sig)
            | RegInstruction::I64Eqz(sig)
            | RegInstruction::I64Extend16S(sig)
            | RegInstruction::I64Extend32S(sig)
            | RegInstruction::I64Extend8S(sig)
            | RegInstruction::I64ExtendI32S(sig)
            | RegInstruction::I64ExtendI32U(sig)
            | RegInstruction::I64Popcnt(sig)
            | RegInstruction::I64ReinterpretF64(sig)
            | RegInstruction::I64TruncF32S(sig)
            | RegInstruction::I64TruncF32U(sig)
            | RegInstruction::I64TruncF64S(sig)
            | RegInstruction::I64TruncF64U(sig)
            | RegInstruction::I64TruncSatF32S(sig)
            | RegInstruction::I64TruncSatF32U(sig)
            | RegInstruction::I64TruncSatF64S(sig)
            | RegInstruction::I64TruncSatF64U(sig) => value_op(
                self.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // i64 — binary
            RegInstruction::I64Add(sig)
            | RegInstruction::I64And(sig)
            | RegInstruction::I64DivS(sig)
            | RegInstruction::I64DivU(sig)
            | RegInstruction::I64Eq(sig)
            | RegInstruction::I64GeS(sig)
            | RegInstruction::I64GeU(sig)
            | RegInstruction::I64GtS(sig)
            | RegInstruction::I64GtU(sig)
            | RegInstruction::I64LeS(sig)
            | RegInstruction::I64LeU(sig)
            | RegInstruction::I64LtS(sig)
            | RegInstruction::I64LtU(sig)
            | RegInstruction::I64Mul(sig)
            | RegInstruction::I64Ne(sig)
            | RegInstruction::I64Or(sig)
            | RegInstruction::I64RemS(sig)
            | RegInstruction::I64RemU(sig)
            | RegInstruction::I64Rotl(sig)
            | RegInstruction::I64Rotr(sig)
            | RegInstruction::I64Shl(sig)
            | RegInstruction::I64ShrS(sig)
            | RegInstruction::I64ShrU(sig)
            | RegInstruction::I64Sub(sig)
            | RegInstruction::I64Xor(sig) => value_op(
                self.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // f32 — loads
            RegInstruction::F32Load { offset, sig } => load_op(
                self.kind(),
                *offset,
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // f32 — stores
            RegInstruction::F32Store { offset, sig } => {
                store_op(self.kind(), *offset, sig.input.registers(ins))
            }

            // f32 — unary
            RegInstruction::F32Abs(sig)
            | RegInstruction::F32Ceil(sig)
            | RegInstruction::F32ConvertI32S(sig)
            | RegInstruction::F32ConvertI32U(sig)
            | RegInstruction::F32ConvertI64S(sig)
            | RegInstruction::F32ConvertI64U(sig)
            | RegInstruction::F32DemoteF64(sig)
            | RegInstruction::F32Floor(sig)
            | RegInstruction::F32Nearest(sig)
            | RegInstruction::F32Neg(sig)
            | RegInstruction::F32ReinterpretI32(sig)
            | RegInstruction::F32Sqrt(sig)
            | RegInstruction::F32Trunc(sig) => value_op(
                self.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // f32 — binary
            RegInstruction::F32Add(sig)
            | RegInstruction::F32Copysign(sig)
            | RegInstruction::F32Div(sig)
            | RegInstruction::F32Eq(sig)
            | RegInstruction::F32Ge(sig)
            | RegInstruction::F32Gt(sig)
            | RegInstruction::F32Le(sig)
            | RegInstruction::F32Lt(sig)
            | RegInstruction::F32Max(sig)
            | RegInstruction::F32Min(sig)
            | RegInstruction::F32Mul(sig)
            | RegInstruction::F32Ne(sig)
            | RegInstruction::F32Sub(sig) => value_op(
                self.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // f64 — loads
            RegInstruction::F64Load { offset, sig } => load_op(
                self.kind(),
                *offset,
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // f64 — stores
            RegInstruction::F64Store { offset, sig } => {
                store_op(self.kind(), *offset, sig.input.registers(ins))
            }

            // f64 — unary
            RegInstruction::F64Abs(sig)
            | RegInstruction::F64Ceil(sig)
            | RegInstruction::F64ConvertI32S(sig)
            | RegInstruction::F64ConvertI32U(sig)
            | RegInstruction::F64ConvertI64S(sig)
            | RegInstruction::F64ConvertI64U(sig)
            | RegInstruction::F64Floor(sig)
            | RegInstruction::F64Nearest(sig)
            | RegInstruction::F64Neg(sig)
            | RegInstruction::F64PromoteF32(sig)
            | RegInstruction::F64ReinterpretI64(sig)
            | RegInstruction::F64Sqrt(sig)
            | RegInstruction::F64Trunc(sig) => value_op(
                self.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            // f64 — binary
            RegInstruction::F64Add(sig)
            | RegInstruction::F64Copysign(sig)
            | RegInstruction::F64Div(sig)
            | RegInstruction::F64Eq(sig)
            | RegInstruction::F64Ge(sig)
            | RegInstruction::F64Gt(sig)
            | RegInstruction::F64Le(sig)
            | RegInstruction::F64Lt(sig)
            | RegInstruction::F64Max(sig)
            | RegInstruction::F64Min(sig)
            | RegInstruction::F64Mul(sig)
            | RegInstruction::F64Ne(sig)
            | RegInstruction::F64Sub(sig) => value_op(
                self.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),

            RegInstruction::LocalSet { index, sig } => format!(
                "local.set    local{} <- {}",
                index.0,
                sig.input.registers(ins)[0].render()
            ),
            RegInstruction::LocalTee { index, sig } => format!(
                "local.tee    local{} <- {}",
                index.0,
                sig.input.registers(ins)[0].render()
            ),
            RegInstruction::GlobalSet { index, sig } => format!(
                "global.set   global{} <- {}",
                index.0,
                sig.input.registers(ins)[0].render()
            ),
            RegInstruction::LocalSpill { index, spill_index } => {
                format!("local.spill  local{} -> spill{spill_index}", index.0)
            }
            RegInstruction::GlobalSpill { index, spill_index } => {
                format!("global.spill global{} -> spill{spill_index}", index.0)
            }
            RegInstruction::RefIsNull(sig) => format!(
                "ref.is_null  {} -> {}",
                list(sig.input.registers(ins)),
                regs(sig.output.registers(outs))
            ),
            // Three operands and no result, so there is nothing to point an arrow
            // at. `memory.copy` reads destination, source, length; `memory.fill`
            // reads destination, byte, length.
            // The segment it reads from is an immediate, so it leads: the three
            // operands after it are destination, source offset, length.
            RegInstruction::MemoryInit {
                data_index,
                operands,
            } => format!(
                "{:<12} data{data_index} {}",
                mnemonic(self.kind()),
                list(operands.registers(ins))
            ),
            // No operands and no result — the segment it releases is the whole
            // instruction.
            RegInstruction::DataDrop(data_index) => {
                format!("{:<12} data{data_index}", mnemonic(self.kind()))
            }
            RegInstruction::MemoryCopy(input) | RegInstruction::MemoryFill(input) => {
                format!(
                    "{:<12} {}",
                    mnemonic(self.kind()),
                    list(input.registers(ins))
                )
            }
            // Kept out of the value-op groups above: it has the shape of a unary
            // operator but a side effect they do not, and the group comment there
            // says "pure".
            RegInstruction::MemoryGrow(sig) => value_op(
                self.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),
            // No operands to show: the only run it carries is its destination.
            RegInstruction::MemorySize(output) => {
                format!(
                    "{:<12} -> {}",
                    mnemonic(self.kind()),
                    regs(output.registers(outs))
                )
            }
            RegInstruction::Select(sig) => format!(
                "select       {} -> {}",
                list(sig.input.registers(ins)),
                regs(sig.output.registers(outs))
            ),
            RegInstruction::Move(sig) => format!(
                "move         {} -> {}",
                list(sig.input_registers(ins)),
                regs(sig.output_registers(outs))
            ),
            RegInstruction::If {
                cond,
                else_index,
                end_index,
            } => format!(
                "if           {} else={} end={}",
                sig1(&cond.input),
                else_index
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "-".into()),
                end_index
            ),
            RegInstruction::Else { end_index } => format!("else         end={end_index}"),
            RegInstruction::Br { target_index } => format!("br           -> {target_index}"),
            RegInstruction::BrIf {
                cond,
                mov,
                target_index,
            } => format!(
                "br_if        {} -> {target_index}{}",
                sig1(cond),
                if mov.is_empty() {
                    String::new()
                } else {
                    format!(
                        "  move {} -> {}",
                        list(mov.input_registers(ins)),
                        regs(mov.output_registers(outs))
                    )
                }
            ),
            RegInstruction::BrTable {
                index,
                targets_start,
                targets_len,
            } => {
                let arms = &frame.br_targets_arena
                    [*targets_start as usize..(targets_start + targets_len) as usize];

                let rendered: Vec<String> = arms
                    .iter()
                    .map(|a| {
                        if a.mov.is_empty() {
                            format!("->{}", a.target_index)
                        } else {
                            format!(
                                "->{} [{} -> {}]",
                                a.target_index,
                                list(a.mov.input_registers(ins)),
                                regs(a.mov.output_registers(outs))
                            )
                        }
                    })
                    .collect();

                format!("br_table     {} {}", sig1(index), rendered.join(" "))
            }
            RegInstruction::Return { target_index } => {
                format!("return       -> {target_index}")
            }
            RegInstruction::Call {
                func_index,
                caller_base,
            } => format!("call         f{} caller_base={caller_base}", func_index.0),
            RegInstruction::CallIndirect {
                ty_index,
                table_index,
                slot,
                operands,
                caller_base,
            } => {
                // Both runs are implicit: the arguments are the `params` operands
                // starting at `operands`, and their destinations are the same many
                // registers based at `caller_base`. Rendering them the way the
                // executor has to reconstruct them is the point — a test that read
                // them any other way would not notice the two disagreeing.
                let params = types[ty_index.0 as usize].params.len();
                let args = &ins[*operands as usize..*operands as usize + params];
                let dsts: Vec<u32> = (0..params as u32).map(|i| caller_base + i).collect();

                format!(
                    "call_indirect [{}] ty{} table{} caller_base={caller_base}{}",
                    sig1(slot),
                    ty_index.0,
                    table_index.0,
                    if params == 0 {
                        String::new()
                    } else {
                        format!("  move {} -> {}", list(args), regs(&dsts))
                    }
                )
            }
            RegInstruction::Unreachable => "unreachable".to_string(),
            RegInstruction::Loop => "loop".to_string(),
            RegInstruction::End => "end".to_string(),
        }
    }

    /// Renders a lowered body as one line per instruction, operands resolved against
    /// the arenas.
    ///
    /// Assertions compare against this rather than against the enum, so a failure shows
    /// the whole program and a reader can see what changed.
    pub fn render_body(body: &RegLoweredFuncBody, types: &[FuncType]) -> String {
        let (instructions, _, frame) = body;
        let mut out = String::new();

        for (pc, i) in instructions.iter().enumerate() {
            let line = i.render(frame, types);

            out.push_str(&format!("{pc:>3}  {line}\n"));
        }

        out.push_str(&format!(
            "     frame: {} registers, {} spills\n",
            frame.registers, frame.spills
        ));

        out
    }
}
