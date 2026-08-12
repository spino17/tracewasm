//! Tests for the register lowering pass.
//!
//! Two layers, because they catch different things.
//!
//! [`lower`] takes a `.wat` module, runs the real
//! [`RegInstruction::emit_instruction_for_func`] over its first function, and
//! returns the instruction stream plus its [`FrameLayout`]. That is the layer that
//! catches "the pass emits the wrong program", and it reads like the wasm it came
//! from, so a test is legible to someone who has never seen this file.
//!
//! Below it, tests drive [`SimulatedStack`] directly. Those assert on lowering-time
//! *state* — register indices, spill slots, lazy reference counts, the two heights —
//! which the emitted stream does not show. They also run today, whereas anything
//! going through [`lower`] needs `Operator::End`.
//!
//! When a `.wat` test can express the property, prefer it.

use super::*;
use crate::module::{FuncKind, TyIndex, ValType};
use wasmparser::Parser;

// ---------------------------------------------------------------------------
// wat-driven harness
// ---------------------------------------------------------------------------

/// Lowers the first function body of a `.wat` module.
///
/// Panics on malformed input, which in a test is what you want: the `.wat` is part
/// of the test, so a mistake in it is a test bug and should be loud.
fn lower(wat: &str) -> LoweredRegFuncBody {
    let bytes = wat::parse_str(wat).expect("invalid wat");

    let mut types: Vec<FuncType> = vec![];
    let mut func_tys: Vec<TyIndex> = vec![];
    let mut globals_count: u32 = 0;
    let mut body_bytes: Option<wasmparser::FunctionBody<'_>> = None;

    for payload in Parser::new(0).parse_all(&bytes) {
        match payload.expect("parse") {
            wasmparser::Payload::TypeSection(section) => {
                for ty in section.into_iter_err_on_gc_types() {
                    let ty = ty.expect("func type");

                    types.push(FuncType {
                        params: ty
                            .params()
                            .iter()
                            .map(|v| ValType::from_wasmparser(*v))
                            .collect(),
                        results: ty
                            .results()
                            .iter()
                            .map(|v| ValType::from_wasmparser(*v))
                            .collect(),
                    });
                }
            }
            wasmparser::Payload::FunctionSection(section) => {
                for ty_index in section {
                    func_tys.push(TyIndex(ty_index.expect("func ty index")));
                }
            }
            wasmparser::Payload::GlobalSection(section) => {
                globals_count += section.count();
            }
            wasmparser::Payload::CodeSectionEntry(body) => {
                if body_bytes.is_none() {
                    body_bytes = Some(body);
                }
            }
            _ => {}
        }
    }

    let body = body_bytes.expect("module has no function body");
    let ty = &types[func_tys[0].0 as usize];
    let (params, results) = (ty.params.len() as u32, ty.results.len() as u32);

    // Locals are run-length encoded in the body header; the pass needs the flat
    // count, params included, because it indexes the lazy origin table unchecked.
    let mut locals_count = params;
    for local in body.get_locals_reader().expect("locals") {
        let (count, _ty) = local.expect("local");
        locals_count += count;
    }

    let func_decls: Vec<FuncDecl> = func_tys
        .iter()
        .map(|ty| FuncDecl {
            kind: FuncKind::Local,
            ty: *ty,
        })
        .collect();

    RegInstruction::emit_instruction_for_func(
        body.get_operators_reader().expect("operators"),
        params,
        results,
        &types,
        &func_decls,
        locals_count,
        globals_count,
    )
    .expect("lowering failed")
}

/// Renders a lowered body as one line per instruction, operands resolved against
/// the arenas.
///
/// Assertions compare against this rather than against the enum, so a failure shows
/// the whole program and a reader can see what changed.
fn render(body: &LoweredRegFuncBody) -> String {
    let (instructions, frame) = body;
    let ins = &frame.input_registers_arena;
    let outs = &frame.output_registers_arena;

    let slot = |s: &Slot| match s {
        Slot::Const(Const::I32(v)) => format!("{v}"),
        Slot::Const(Const::I64(v)) => format!("{v}i64"),
        Slot::Const(Const::F32(v)) => format!("{v}f32"),
        Slot::Const(Const::F64(v)) => format!("{v}f64"),
        Slot::Local(n) => format!("local{n}"),
        Slot::Global(n) => format!("global{n}"),
        Slot::Spilled(s) => format!("spill{s}"),
        Slot::Register(r) => format!("r{r}"),
    };
    let sig1 = |i: &Registers<1, Slot>| slot(&i.registers(ins)[0]);
    let list = |xs: &[Slot]| xs.iter().map(slot).collect::<Vec<_>>().join(", ");
    let regs = |xs: &[u32]| xs.iter().map(|r| format!("r{r}")).collect::<Vec<_>>().join(", ");

    let mut out = String::new();
    for (pc, i) in instructions.iter().enumerate() {
        let line = match i {
            RegInstruction::I32Load { offset, sig } => format!(
                "i32.load     [{}]+{offset} -> {}",
                slot(&sig.input.registers(ins)[0]),
                regs(sig.output.registers(outs))
            ),
            RegInstruction::I32Store { offset, sig } => {
                let a = sig.input.registers(ins);
                format!("i32.store    [{}]+{offset} <- {}", slot(&a[0]), slot(&a[1]))
            }
            RegInstruction::LocalSet { index, sig } => format!(
                "local.set    local{} <- {}",
                index.0,
                slot(&sig.input.registers(ins)[0])
            ),
            RegInstruction::LocalTee { index, sig } => format!(
                "local.tee    local{} <- {}",
                index.0,
                slot(&sig.input.registers(ins)[0])
            ),
            RegInstruction::GlobalSet { index, sig } => format!(
                "global.set   global{} <- {}",
                index.0,
                slot(&sig.input.registers(ins)[0])
            ),
            RegInstruction::LocalSpill { index, spill_index } => {
                format!("local.spill  local{} -> spill{spill_index}", index.0)
            }
            RegInstruction::GlobalSpill { index, spill_index } => {
                format!("global.spill global{} -> spill{spill_index}", index.0)
            }
            RegInstruction::I32Add(sig) => format!(
                "i32.add      {} -> {}",
                list(sig.input.registers(ins)),
                regs(sig.output.registers(outs))
            ),
            RegInstruction::I32Eqz(sig) => format!(
                "i32.eqz      {} -> {}",
                list(sig.input.registers(ins)),
                regs(sig.output.registers(outs))
            ),
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
            } => format!(
                "br_table     {} arms={targets_start}..{}",
                sig1(index),
                targets_start + targets_len
            ),
        };

        out.push_str(&format!("{pc:>3}  {line}\n"));
    }

    out.push_str(&format!(
        "     frame: {} registers, {} spills\n",
        frame.registers, frame.spills
    ));

    out
}

/// Asserts a lowered body renders exactly as `expected`, ignoring leading
/// indentation so the expectation can be written inline.
fn assert_lowers_to(wat: &str, expected: &str) {
    let got = render(&lower(wat));
    let norm = |s: &str| {
        s.lines()
            .map(str::trim_end)
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim_start().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    };

    assert_eq!(
        norm(&got),
        norm(expected),
        "\n--- got ---\n{got}\n--- want ---\n{expected}\n"
    );
}

// ---------------------------------------------------------------------------
// SimulatedStack harness
// ---------------------------------------------------------------------------

/// A stack with the implicit function frame already pushed, as
/// `emit_instruction_for_func` sets up.
fn sim(locals: u32, globals: u32) -> SimulatedStack {
    let mut s = SimulatedStack::new(locals, globals);

    s.control_stack.stack.push(Block {
        kind: BlockKind::Func,
        recorded_height: 0,
        params: 0,
        results: 0,
        is_unreachable_traversing: false,
        has_inherited: false,
        attached_breaks: vec![],
    });

    s
}

fn func_ty(params: usize, results: usize) -> FuncType {
    FuncType {
        params: vec![ValType::I32; params].into_boxed_slice(),
        results: vec![ValType::I32; results].into_boxed_slice(),
    }
}

/// `(slot height, register index)` — the two heights the pass tracks separately.
fn heights(s: &SimulatedStack) -> (u32, usize) {
    (s.stack.height(), s.curr_register_index)
}

// ---------------------------------------------------------------------------
// lazy forwarding and spills
// ---------------------------------------------------------------------------

#[test]
fn borrows_of_one_local_share_a_single_spill() {
    let mut s = sim(2, 0);

    s.push_local(0);
    s.push_local(0);
    s.push_const(Const::I32(5));

    let spill = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills);
    let _set = s.registers_for::<1, 0>();
    let add = s.registers_for::<2, 1>();

    assert_eq!(spill, Some(0), "one spill covers both borrows");
    assert!(
        matches!(
            add.input.registers(&s.input_registers),
            [Slot::Spilled(0), Slot::Spilled(0)]
        ),
        "both operands redirect: {:?}",
        add.input.registers(&s.input_registers)
    );
}

#[test]
fn a_consumed_borrow_is_not_spilled() {
    let mut s = sim(2, 0);

    s.push_local(0);
    let _load = s.registers_for::<1, 1>(); // consumes the borrow
    s.push_const(Const::I32(5));

    assert_eq!(
        SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills),
        None,
        "nothing borrows local 0 any more"
    );
    assert_eq!(s.spills.allocation_len(), 0);
}

#[test]
fn successive_writes_produce_independent_snapshots() {
    let mut s = sim(2, 0);

    s.push_local(0);
    s.push_const(Const::I32(1));
    let first = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills);
    let _ = s.registers_for::<1, 0>();

    s.push_local(0);
    s.push_const(Const::I32(2));
    let second = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills);
    let _ = s.registers_for::<1, 0>();

    let add = s.registers_for::<2, 1>();

    assert_eq!((first, second), (Some(0), Some(1)), "distinct slots");
    assert!(
        matches!(
            add.input.registers(&s.input_registers),
            [Slot::Spilled(0), Slot::Spilled(1)]
        ),
        "each operand keeps its own snapshot: {:?}",
        add.input.registers(&s.input_registers)
    );
}

#[test]
fn dropping_the_last_borrow_releases_its_spill_slot() {
    let mut s = sim(2, 0);

    s.push_local(0);
    s.push_const(Const::I32(1));
    SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills);
    let _ = s.registers_for::<1, 0>();
    s.pop(); // drop the spilled borrow

    s.push_local(0);
    s.push_const(Const::I32(2));
    let reused = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills);

    assert_eq!(reused, Some(0), "the freed slot comes back");
    assert_eq!(s.spills.allocation_len(), 1, "peak live, not total spills");
}

#[test]
fn tee_spills_before_reading_the_top() {
    let mut s = sim(4, 0);

    s.push_local(3);
    let spill = SimulatedStack::set_lazy(3, &mut s.lazy_locals, &mut s.spills);
    let operand = s.tee();

    assert_eq!(spill, Some(0));
    assert!(
        matches!(operand, Slot::Spilled(0)),
        "tee must observe the redirect, got {operand:?}"
    );
    assert_eq!(s.stack.height(), 1, "tee peeks, it does not consume");
}

// ---------------------------------------------------------------------------
// block entry layout
// ---------------------------------------------------------------------------

#[test]
fn block_entry_layouts_hold_for_every_block_type() {
    let types = vec![func_ty(2, 1)];

    for variant in [BlockVariant::Block, BlockVariant::Loop, BlockVariant::If] {
        let is_if = matches!(variant, BlockVariant::If);

        for blockty in [
            BlockType::Empty,
            BlockType::Type(wasmparser::ValType::I32),
            BlockType::FuncType(0),
        ] {
            let mut s = sim(4, 0);

            s.push_local(0);
            let _ = s.registers_for::<1, 1>(); // a live register underneath

            let np = params_and_results_from_blockty(&blockty, &types).0;
            for _ in 0..np {
                s.push_const(Const::I32(9));
            }
            if is_if {
                s.push_const(Const::I32(1)); // condition
            }

            let (params, _) = s.add_block(variant_of(&variant), &blockty, &types, 0);
            let recorded = s.get_curr_block().recorded_height;

            if params != 0 {
                let _ = s.materialize_stack_slots_in_registers(params + u32::from(is_if));
            }
            if is_if {
                let _ = s.registers_for::<1, 0>();
            }

            assert_eq!(
                s.stack.height(),
                recorded + params,
                "entry layout must be recorded_height + params"
            );
            assert_eq!(s.curr_register_index, 1 + params as usize, "params sit above the live register");
        }
    }
}

fn variant_of(v: &BlockVariant) -> BlockVariant {
    match v {
        BlockVariant::If => BlockVariant::If,
        BlockVariant::Loop => BlockVariant::Loop,
        BlockVariant::Block => BlockVariant::Block,
        BlockVariant::Func => BlockVariant::Func,
    }
}

// ---------------------------------------------------------------------------
// branches
// ---------------------------------------------------------------------------

#[test]
fn branch_lowering_does_not_disturb_the_stack() {
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);
    let base = s.get_curr_block().recorded_height;
    s.push_local(0);
    let _ = s.registers_for::<1, 1>();
    s.push_local(1); // a live lazy borrow above the base

    let before = (heights(&s), s.lazy_locals.origin[1].is_some(), s.spills.allocation_len());
    let _mov = s.br_truncation_registers(base, 0);
    let after = (heights(&s), s.lazy_locals.origin[1].is_some(), s.spills.allocation_len());

    assert_eq!(before, after, "br must simulate, not unwind");
}

#[test]
fn branch_destinations_are_based_at_the_target_label() {
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);
    let outer = s.get_curr_block().recorded_height;
    s.push_local(0);
    let _ = s.registers_for::<1, 1>(); // r0, below the inner block
    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);
    let inner = s.get_curr_block().recorded_height;
    s.push_local(1);
    let _ = s.registers_for::<1, 1>(); // r1, the carried value

    let to_inner = s.br_truncation_registers(inner, 1);
    let to_outer = s.br_truncation_registers(outer, 1);

    assert_eq!(to_inner.output_registers(&s.output_registers), &[1]);
    assert_eq!(to_outer.output_registers(&s.output_registers), &[0]);
    assert_eq!(
        to_inner.input_registers(&s.input_registers).len(),
        1,
        "every arm reads the same operands"
    );
}

#[test]
fn branch_destinations_count_towards_the_frame() {
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);
    let base = s.get_curr_block().recorded_height;
    // operands that are not already registers: destinations sit above everything
    // allocated so far, so the frame has to grow for them
    s.push_const(Const::I32(1));
    s.push_const(Const::I32(2));

    let mov = s.br_truncation_registers(base, 2);
    let highest = mov
        .output_registers(&s.output_registers)
        .iter()
        .map(|r| r + 1)
        .max()
        .unwrap();

    assert!(
        highest <= s.max_registers,
        "frame sized {} but the move writes up to r{}",
        s.max_registers,
        highest - 1
    );
}

// ---------------------------------------------------------------------------
// unreachable tracking
// ---------------------------------------------------------------------------

/// Runs an operator sequence through the tracker, returning one entry per operator:
/// `true` when the lowering loop would process it.
fn reachability(ops: &[Operator<'static>]) -> Vec<bool> {
    let mut t = UnreachableTrackingControlStack::new();

    ops.iter()
        .map(|op| {
            let reachable = matches!(
                t.check_unreachablity(op),
                UnreachableCheckResult::Reachable
            );
            if reachable && matches!(op, Operator::Br { .. }) {
                t.set_unreachable();
            }
            reachable
        })
        .collect()
}

fn blk() -> Operator<'static> {
    Operator::Block {
        blockty: BlockType::Empty,
    }
}
fn iff() -> Operator<'static> {
    Operator::If {
        blockty: BlockType::Empty,
    }
}
fn lop() -> Operator<'static> {
    Operator::Loop {
        blockty: BlockType::Empty,
    }
}
fn br(d: u32) -> Operator<'static> {
    Operator::Br { relative_depth: d }
}
fn cst() -> Operator<'static> {
    Operator::I32Const { value: 1 }
}

#[test]
fn dead_code_is_skipped_until_the_enclosing_end() {
    //                  block  br    const  end
    let seq = [blk(), br(0), cst(), Operator::End];
    assert_eq!(reachability(&seq), [true, true, false, true]);
}

#[test]
fn an_else_after_a_dead_then_arm_is_live_again() {
    let seq = [iff(), br(0), Operator::Else, cst(), Operator::End];
    assert_eq!(reachability(&seq), [true, true, true, true, true]);
}

#[test]
fn an_if_opened_in_dead_code_keeps_both_arms_dead() {
    let seq = [
        blk(),
        br(0),
        iff(),
        cst(),
        Operator::Else,
        cst(),
        Operator::End,
        cst(),
        Operator::End,
    ];

    assert_eq!(
        reachability(&seq),
        [true, true, false, false, false, false, false, false, true],
        "the else must not resurrect an if that was born dead"
    );
}

#[test]
fn dead_nested_blocks_keep_the_control_stack_balanced() {
    let seq = [
        blk(),
        br(0),
        lop(),
        cst(),
        Operator::End,
        blk(),
        cst(),
        Operator::End,
        Operator::End,
    ];
    let live = reachability(&seq);

    // every construct the lowering loop opens is closed by an `end` it also processes
    let mut depth = 0i32;
    for (op, reachable) in seq.iter().zip(&live) {
        if !reachable {
            continue;
        }
        match op {
            Operator::Block { .. } | Operator::Loop { .. } | Operator::If { .. } => depth += 1,
            Operator::End => depth -= 1,
            _ => {}
        }
    }

    assert_eq!(depth, 0, "control stack left unbalanced: {live:?}");
}

// ---------------------------------------------------------------------------
// known gaps
// ---------------------------------------------------------------------------

/// A spill emitted inside one arm of a conditional is read on every path.
///
/// `local.get 0` borrows local 0 *below* the `if`, so the operand outlives the
/// construct. The `local.set 0` inside the then-arm rescues it with a
/// `LocalSpill`, but that instruction only executes when the condition is true —
/// while the consumer after `end` reads `Spilled(0)` unconditionally, because the
/// slot is resolved once at lowering time for all paths.
///
/// The fix is to spill live borrows at each point control can diverge (`if`,
/// `br_if`, `br_table`), so the spill dominates every reader. Deleting the
/// `#[ignore]` is the definition of done.
#[test]
#[ignore = "known gap: a spill inside a conditional arm does not dominate its readers"]
fn a_spill_inside_a_conditional_arm_must_dominate_its_readers() {
    let mut s = sim(2, 0);

    s.push_local(0); // the borrow, below the `if`
    s.push_local(1); // the condition
    s.add_block(BlockVariant::If, &BlockType::Empty, &[], 0);
    let _ = s.registers_for::<1, 0>();

    // then-arm: i32.const 5 ; local.set 0
    s.push_const(Const::I32(5));
    let spilled_in_arm = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills);
    let _ = s.registers_for::<1, 0>();

    // else-arm is empty; reset as the `Else` arm does
    let recorded = s.get_curr_block().recorded_height;
    let params = s.get_curr_block().params;
    s.pops_and_pushes(s.stack.height() - recorded, params);

    let consumer = s.pop();

    assert!(
        spilled_in_arm.is_some(),
        "the write does rescue the borrow, but only on its own path"
    );
    assert!(
        !matches!(consumer, Slot::Spilled(_)),
        "consumer reads {consumer:?}, which is only written when the condition is true"
    );
}

// ---------------------------------------------------------------------------
// end-to-end
//
// These go through `lower`, so they need `Operator::End`. Delete the `#[ignore]`s
// once it lands; the expectations are written against the design as documented and
// are the fastest way to find out whether it behaves as intended.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "blocked on Operator::End"]
fn straight_line_body_forwards_operands_without_copying() {
    assert_lowers_to(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             local.get 1
             i32.add))"#,
        "
          0  i32.add      local0, local1 -> r0
             frame: 1 registers, 0 spills
        ",
    );
}

#[test]
#[ignore = "blocked on Operator::End"]
fn overwriting_a_borrowed_local_spills_it_first() {
    assert_lowers_to(
        r#"(module (func (param i32) (result i32)
             local.get 0
             i32.const 5
             local.set 0
             local.get 0
             i32.add))"#,
        "
          0  local.spill  local0 -> spill0
          1  local.set    local0 <- 5
          2  i32.add      spill0, local0 -> r0
             frame: 1 registers, 1 spills
        ",
    );
}

#[test]
#[ignore = "blocked on Operator::End"]
fn a_block_with_no_params_emits_no_move() {
    assert_lowers_to(
        r#"(module (func (param i32) (result i32)
             (block (nop))
             local.get 0))"#,
        "
             frame: 0 registers, 0 spills
        ",
    );
}
