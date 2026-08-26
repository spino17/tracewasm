//! Human-readable rendering of a lowered register-machine body.
//!
//! Exists for tests and for reading a lowering by eye, so it prefers legibility over
//! fidelity to the encoding in two ways.
//!
//! **Operands are named, not numbered.** A [`Slot`] is an absolute frame index, so
//! `4` could be a local, a constant, a spill or a register depending on the region
//! sizes. [`Slot::render`] resolves that against the [`RegFrameLayout`] and prints
//! `local0`, `7`, `spill0` or `r1`.
//!
//! **Registers are numbered from the operand base**, not from the frame base. `r0` is
//! the frame's first operand register, which is the frame index
//! `locals_count + consts + spills` — the numbering the lowering reasons in. The same
//! offset is applied to a `caller_base`, so a call reads in the same units as the
//! registers around it.
//!
//! Both are conversions *out* of the executable form; nothing here runs at execution.

use crate::{
    instruction::register::{
        DynSignature, InputRegisters, MemoryOffset, RegFrameLayout, RegInstruction,
        RegLoweredFuncBody, Slot, interner::InternedId, mnemonic,
    },
    module::FuncType,
};

impl RegInstruction {
    /// Renders one instruction, with its operands resolved through `frame`.
    ///
    /// `types` is needed because a `call_indirect` stores only a type index, so how
    /// many of its arena operands are arguments is recoverable only from the module's
    /// type section — the same thing executing one has to do.
    pub fn render(&self, frame: &RegFrameLayout, types: &[FuncType]) -> String {
        let sig1 = |i: &InputRegisters<1>| i.registers[0].render(frame);
        let list = |xs: &[Slot]| {
            xs.iter()
                .map(|x| x.render(frame))
                .collect::<Vec<_>>()
                .join(", ")
        };

        // A destination is a frame index like any operand, so it is named through
        // the same region lookup rather than printed raw — otherwise a register
        // renders as its absolute index and reads as a different register.
        let regs = |start: u16, count: usize| {
            (0..count as u16)
                .map(|i| Slot(start + i).render(frame))
                .collect::<Vec<_>>()
                .join(", ")
        };

        // A move's destinations are the `len` registers based at its start, which is
        // the same run the executor writes — reconstructed here the way it does.
        let mov = |sig: &DynSignature| {
            format!(
                "{} -> {}",
                list(&sig.input),
                regs(sig.output_start, sig.input.len())
            )
        };

        // `caller_base` is a frame index too, and it is read alongside the `r0`,
        // `r1` … above, so it is shown in the same numbering rather than as the
        // absolute index the runtime uses.
        let operand_base = frame.locals_count + frame.consts.len() as u16 + frame.spills;
        let as_operand = |frame_index: u16| frame_index - operand_base;

        // A load or a store carries an id into `memory_offsets`, not the offset itself,
        // so it is resolved and shown as its value — the id is an artifact of keeping
        // the instruction eight bytes wide.
        let offset_of = |id: InternedId<MemoryOffset>| frame.memory_offsets.value(id).0;

        let load_op = |kind, id: InternedId<MemoryOffset>, inputs: &[Slot], output: u16| {
            format!(
                "{:<12} [{}]+{} -> {}",
                mnemonic(kind),
                inputs[0].render(frame),
                offset_of(id),
                regs(output, 1)
            )
        };

        let store_op = |kind, id: InternedId<MemoryOffset>, inputs: &[Slot]| {
            format!(
                "{:<12} [{}]+{} <- {}",
                mnemonic(kind),
                inputs[0].render(frame),
                offset_of(id),
                inputs[1].render(frame)
            )
        };

        // Every pure value operator renders alike, so the arms below carry no body of
        // their own. They are split by arity because an or-pattern binds one type, and
        // then by family, so they scan in the order the enum declares them.
        let value_op = |kind, inputs: &[Slot], output: u16| {
            format!(
                "{:<12} {} -> {}",
                mnemonic(kind),
                list(inputs),
                regs(output, 1)
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
            | RegInstruction::I32Load16U { offset, sig } => {
                load_op(self.kind(), *offset, &sig.input.registers, sig.output.start)
            }

            // i32 — stores
            RegInstruction::I32Store { offset, input }
            | RegInstruction::I32Store8 { offset, input }
            | RegInstruction::I32Store16 { offset, input } => {
                store_op(self.kind(), *offset, &input.registers)
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
            | RegInstruction::I32WrapI64(sig) => {
                value_op(self.kind(), &sig.input.registers, sig.output.start)
            }

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
            | RegInstruction::I32Xor(sig) => {
                value_op(self.kind(), &sig.input.registers, sig.output.start)
            }

            // i64 — loads
            RegInstruction::I64Load { offset, sig }
            | RegInstruction::I64Load8S { offset, sig }
            | RegInstruction::I64Load8U { offset, sig }
            | RegInstruction::I64Load16S { offset, sig }
            | RegInstruction::I64Load16U { offset, sig }
            | RegInstruction::I64Load32S { offset, sig }
            | RegInstruction::I64Load32U { offset, sig } => {
                load_op(self.kind(), *offset, &sig.input.registers, sig.output.start)
            }

            // i64 — stores
            RegInstruction::I64Store { offset, input }
            | RegInstruction::I64Store8 { offset, input }
            | RegInstruction::I64Store16 { offset, input }
            | RegInstruction::I64Store32 { offset, input } => {
                store_op(self.kind(), *offset, &input.registers)
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
            | RegInstruction::I64TruncSatF64U(sig) => {
                value_op(self.kind(), &sig.input.registers, sig.output.start)
            }

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
            | RegInstruction::I64Xor(sig) => {
                value_op(self.kind(), &sig.input.registers, sig.output.start)
            }

            // f32 — loads
            RegInstruction::F32Load { offset, sig } => {
                load_op(self.kind(), *offset, &sig.input.registers, sig.output.start)
            }

            // f32 — stores
            RegInstruction::F32Store { offset, input } => {
                store_op(self.kind(), *offset, &input.registers)
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
            | RegInstruction::F32Trunc(sig) => {
                value_op(self.kind(), &sig.input.registers, sig.output.start)
            }

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
            | RegInstruction::F32Sub(sig) => {
                value_op(self.kind(), &sig.input.registers, sig.output.start)
            }

            // f64 — loads
            RegInstruction::F64Load { offset, sig } => {
                load_op(self.kind(), *offset, &sig.input.registers, sig.output.start)
            }

            // f64 — stores
            RegInstruction::F64Store { offset, input } => {
                store_op(self.kind(), *offset, &input.registers)
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
            | RegInstruction::F64Trunc(sig) => {
                value_op(self.kind(), &sig.input.registers, sig.output.start)
            }

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
            | RegInstruction::F64Sub(sig) => {
                value_op(self.kind(), &sig.input.registers, sig.output.start)
            }

            RegInstruction::LocalSet { index, input } => format!(
                "local.set    local{} <- {}",
                index.0,
                input.registers[0].render(frame)
            ),
            RegInstruction::LocalTee { index, input } => format!(
                "local.tee    local{} <- {}",
                index.0,
                input.registers[0].render(frame)
            ),
            RegInstruction::GlobalGet { index, output } => format!(
                "global.get   global{} -> {}",
                index.0,
                regs(output.start, 1)
            ),
            RegInstruction::GlobalSet { index, input } => format!(
                "global.set   global{} <- {}",
                index.0,
                input.registers[0].render(frame)
            ),
            RegInstruction::LocalSpill { index, spill_index } => {
                format!("local.spill  local{} -> spill{spill_index}", index.0)
            }
            RegInstruction::RefIsNull(sig) => format!(
                "ref.is_null  {} -> {}",
                list(&sig.input.registers),
                regs(sig.output.start, 1)
            ),
            // The segment it reads from is an immediate, so it leads: the three
            // operands after it are destination, source offset, length. No result, so
            // there is nothing to point an arrow at.
            RegInstruction::MemoryInit(id) => {
                let entry = frame.memory_init_arena.get(*id);
                let data_index = entry.data_index;

                format!(
                    "{:<12} data{data_index} {}",
                    mnemonic(self.kind()),
                    list(&entry.operands.registers)
                )
            }
            // No operands and no result — the segment it releases is the whole
            // instruction.
            RegInstruction::DataDrop(data_index) => {
                format!("{:<12} data{data_index}", mnemonic(self.kind()))
            }
            // Three operands and no result, so there is nothing to point an arrow at.
            // `memory.copy` reads destination, source, length; `memory.fill` reads
            // destination, byte, length.
            RegInstruction::MemoryCopy(input) | RegInstruction::MemoryFill(input) => {
                format!("{:<12} {}", mnemonic(self.kind()), list(&input.registers))
            }
            // Kept out of the value-op groups above: it has the shape of a unary
            // operator but a side effect they do not, and the group comment there
            // says "pure".
            RegInstruction::MemoryGrow(sig) => {
                value_op(self.kind(), &sig.input.registers, sig.output.start)
            }
            // No operands to show: the only run it carries is its destination.
            RegInstruction::MemorySize(output) => {
                format!("{:<12} -> {}", mnemonic(self.kind()), regs(output.start, 1))
            }
            RegInstruction::Select(sig) => {
                let entry = frame.select_arena.get(*sig);

                format!(
                    "select       {} -> {}",
                    list(&entry.0.input.registers),
                    regs(entry.0.output.start, 1)
                )
            }
            RegInstruction::Move(id) => {
                format!("move         {}", mov(frame.dyn_signatures.get(*id)))
            }
            RegInstruction::If(id) => {
                let entry = frame.if_arena.get(*id);

                format!(
                    "if           {} else={} end={}",
                    sig1(&entry.cond),
                    entry
                        .else_index
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| "-".into()),
                    entry.end_index
                )
            }
            RegInstruction::Else { end_index } => format!("else         end={end_index}"),
            RegInstruction::Br { target_index } => format!("br           -> {target_index}"),
            RegInstruction::BrIf(id) => {
                let entry = frame.br_if_arena.get(*id);
                let target_index = entry.target_index;

                format!(
                    "br_if        {} -> {target_index}{}",
                    sig1(&entry.cond),
                    if entry.mov.is_empty() {
                        String::new()
                    } else {
                        format!("  move {}", mov(&entry.mov))
                    }
                )
            }
            RegInstruction::BrTable(id) => {
                let entry = frame.br_table_arena.get(*id);
                let arms = &entry.br_targets;

                let rendered: Vec<String> = arms
                    .iter()
                    .map(|a| {
                        if a.mov.is_empty() {
                            format!("->{}", a.target_index)
                        } else {
                            format!("->{} [{}]", a.target_index, mov(&a.mov))
                        }
                    })
                    .collect();

                format!("br_table     {} {}", sig1(&entry.index), rendered.join(" "))
            }
            RegInstruction::Return { target_index } => {
                format!("return       -> {target_index}")
            }
            RegInstruction::Call {
                func_index,
                caller_base,
            } => format!(
                "call         f{} caller_base={}",
                func_index.0,
                as_operand(*caller_base)
            ),
            RegInstruction::CallIndirect(id) => {
                let entry = frame.call_indirect_arena.get(*id);

                // Both runs are implicit: the arguments are the `params` operands
                // starting at `operands`, and their destinations are the same many
                // registers based at `caller_base`. Rendering them the way the
                // executor has to reconstruct them is the point — a test that read
                // them any other way would not notice the two disagreeing.
                let params = types[entry.ty_index.0 as usize].params.len();
                let args = &entry.operands.input[..params];

                format!(
                    "call_indirect [{}] ty{} table{} caller_base={}{}",
                    sig1(&entry.slot),
                    entry.ty_index.0,
                    entry.table_index.0,
                    as_operand(entry.caller_base),
                    if params == 0 {
                        String::new()
                    } else {
                        format!(
                            "  move {} -> {}",
                            list(args),
                            regs(entry.caller_base, params)
                        )
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

        // `registers` counts from the frame base, so it includes the locals; the
        // operand count is what a reader of the body above is looking for, since
        // that is what `r0`, `r1` … are numbered against.
        out.push_str(&format!(
            "     frame: {} registers, {} spills\n",
            frame.registers - frame.locals_count,
            frame.spills
        ));

        out
    }
}
