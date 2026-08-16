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
use crate::instruction::register::lazy::SpillIndex;
use crate::module::{FuncKind, TyIndex, ValType};
use std::ops::Range;
use wasmparser::Parser;

// ---------------------------------------------------------------------------
// rendering
//
// `SpillIndex` is deliberately opaque outside the pass — only `set_lazy` can mint
// one, and it has no `PartialEq`. Tests therefore compare *renderings* rather than
// values, which also keeps assertions reading like the emitted code.
// ---------------------------------------------------------------------------

/// Renders one operand: `5`, `local1`, `global0`, `spill0`, `r2`, `(0)ref`.
fn slot_str(s: &Slot) -> String {
    match s {
        Slot::Const(Const::I32(v)) => format!("{v}"),
        Slot::Const(Const::I64(v)) => format!("{v}i64"),
        Slot::Const(Const::F32(v)) => format!("{v}f32"),
        Slot::Const(Const::F64(v)) => format!("{v}f64"),
        Slot::Const(Const::Ref(v)) => {
            if let Some(func_index) = v {
                format!("({})ref", func_index.0)
            } else {
                "(null)ref".to_string()
            }
        }
        Slot::Local(n) => format!("local{n}"),
        Slot::Global(n) => format!("global{n}"),
        Slot::Spilled(i) => format!("spill{i}"),
        Slot::Register(r) => format!("r{r}"),
    }
}

/// Renders a run of operands.
fn slots(xs: &[Slot]) -> Vec<String> {
    xs.iter().map(slot_str).collect()
}

/// Renders what `set_lazy` returned: `Some("spill0")` when it rescued a borrow.
fn spilled_to(result: Option<SpillIndex>) -> Option<String> {
    result.map(|i| format!("spill{i}"))
}

// ---------------------------------------------------------------------------
// wat-driven harness
// ---------------------------------------------------------------------------

/// Lowers the first function body of a `.wat` module.
///
/// Panics on malformed input, which in a test is what you want: the `.wat` is part
/// of the test, so a mistake in it is a test bug and should be loud.
fn lower(wat: &str) -> LoweredRegFuncBody {
    lower_func(wat, 0)
}

/// Lowers the `n`th *defined* function of a `.wat` module, for tests whose subject
/// is a caller and so cannot be function 0.
///
/// Imported functions are not modelled: the index space here is the code section's,
/// so `call n` inside the wat must name a defined function.
fn lower_func(wat: &str, n: usize) -> LoweredRegFuncBody {
    lower_func_with_types(wat, n).0
}

/// [`lower_func`], also handing back the module's type section.
///
/// A `call_indirect` stores only a `ty_index`, so how many of the operands in its
/// arena run are arguments is recoverable only through the types. Rendering one
/// therefore needs exactly what executing one will need.
fn lower_func_with_types(wat: &str, n: usize) -> (LoweredRegFuncBody, Vec<FuncType>) {
    let bytes = wat::parse_str(wat).expect("invalid wat");

    // `wat::parse_str` assembles without type-checking, so an ill-typed body
    // reaches lowering intact — `i32.add` over an `f32` operand and all. The pass
    // trusts validation to have happened (as it has for a real module, in
    // `Module::compile`), so it would lower the nonsense without complaint and the
    // test would assert against it. Validating here is what makes a `.wat` in a
    // test self-checking, the same way `Module::compile` does for real input.
    wasmparser::Validator::new()
        .validate_all(&bytes)
        .expect("wat does not validate");

    let mut types: Vec<FuncType> = vec![];
    let mut func_tys: Vec<TyIndex> = vec![];
    let mut globals_count: u32 = 0;
    let mut bodies: Vec<wasmparser::FunctionBody<'_>> = vec![];

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
                bodies.push(body);
            }
            _ => {}
        }
    }

    let body = bodies.into_iter().nth(n).expect("no such function body");
    let ty = &types[func_tys[n].0 as usize];
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

    let body = RegInstruction::emit_instruction_for_func(
        body.get_operators_reader().expect("operators"),
        params,
        results,
        &types,
        &func_decls,
        locals_count,
        globals_count,
    )
    .expect("lowering failed");

    (body, types)
}

/// Renders a lowered body as one line per instruction, operands resolved against
/// the arenas.
///
/// Assertions compare against this rather than against the enum, so a failure shows
/// the whole program and a reader can see what changed.
fn render(body: &LoweredRegFuncBody, types: &[FuncType]) -> String {
    let (instructions, frame) = body;
    let ins = &frame.input_registers_arena;
    let outs = &frame.output_registers_arena;

    let slot = slot_str;
    let sig1 = |i: &Registers<1, Slot>| slot(&i.registers(ins)[0]);
    let list = |xs: &[Slot]| xs.iter().map(slot).collect::<Vec<_>>().join(", ");

    let regs = |xs: &[u32]| {
        xs.iter()
            .map(|r| format!("r{r}"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    // Every pure value operator renders alike, so the arms below carry no body of
    // their own. They are split by arity because an or-pattern binds one type, and
    // then by family, so they scan in the order the enum declares them.
    let value_op = |kind, inputs: &[Slot], outputs: &[u32]| {
        format!(
            "{:<12} {} -> {}",
            mnemonic(kind),
            list(inputs),
            regs(outputs)
        )
    };

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
            // Every pure value operator renders the same way, so they share two
            // arms — one per arity, since an or-pattern can only bind one type.
            // The mnemonic comes from the kind rather than a table; `mnemonic`
            // says why that is checked rather than merely convenient.
            // i32 - unary
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
                i.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),
            // i32 - binary
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
                i.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),
            // i64 - unary
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
                i.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),
            // i64 - binary
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
                i.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),
            // f32 - unary
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
                i.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),
            // f32 - binary
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
                i.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),
            // f64 - unary
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
                i.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),
            // f64 - binary
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
                i.kind(),
                sig.input.registers(ins),
                sig.output.registers(outs),
            ),
            RegInstruction::RefIsNull(sig) => format!(
                "ref.is_null  {} -> {}",
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
            RegInstruction::Return { target_index } => format!("return       -> {target_index}"),
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
            RegInstruction::End => "end".to_string(),
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
    assert_func_lowers_to(wat, 0, expected)
}

/// [`assert_lowers_to`] for a body that is not function 0 — a caller, typically.
fn assert_func_lowers_to(wat: &str, n: usize, expected: &str) {
    let (body, types) = lower_func_with_types(wat, n);
    let got = render(&body, &types);

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

    assert_eq!(
        spilled_to(spill),
        Some("spill0".into()),
        "one spill covers both borrows"
    );

    assert_eq!(
        slots(add.input.registers(&s.input_registers)),
        ["spill0", "spill0"],
        "both operands redirect"
    );
}

#[test]
fn a_consumed_borrow_is_not_spilled() {
    let mut s = sim(2, 0);

    s.push_local(0);

    let _load = s.registers_for::<1, 1>(); // consumes the borrow

    s.push_const(Const::I32(5));

    assert!(
        SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).is_none(),
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

    assert_eq!(
        (spilled_to(first), spilled_to(second)),
        (Some("spill0".into()), Some("spill1".into())),
        "distinct slots"
    );

    assert_eq!(
        slots(add.input.registers(&s.input_registers)),
        ["spill0", "spill1"],
        "each operand keeps its own snapshot"
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

    assert_eq!(
        spilled_to(reused),
        Some("spill0".into()),
        "the freed slot comes back"
    );

    assert_eq!(s.spills.allocation_len(), 1, "peak live, not total spills");
}

#[test]
fn tee_spills_before_reading_the_top() {
    let mut s = sim(4, 0);

    s.push_local(3);

    let spill = SimulatedStack::set_lazy(3, &mut s.lazy_locals, &mut s.spills);

    let operand = s.tee();

    assert_eq!(spilled_to(spill), Some("spill0".into()));

    assert_eq!(
        slot_str(&operand),
        "spill0",
        "tee must observe the redirect"
    );

    assert_eq!(s.stack.height(), 1, "tee peeks, it does not consume");
}

// ---------------------------------------------------------------------------
// reference immediates
//
// `ref.null` and `ref.func` push a value that is fully described by the operator
// itself, so they are [`Const`]s and behave like `i32.const`: nothing is emitted,
// nothing is allocated, and the value is read in place by whatever consumes it.
//
// A reference is `Option<FuncIndex>`, matching the runtime's `Val::Ref`, so the
// two agree on what a reference *is* without a conversion in between.
// ---------------------------------------------------------------------------

/// Two functions, so a `ref.func 0` has something to name. `elem declare` is what
/// makes referencing it legal without putting it in a table.
fn ref_module(body: &str) -> String {
    format!("(module (func) (elem declare func 0) {body})")
}

#[test]
fn a_reference_is_an_immediate_and_emits_nothing() {
    // the push itself costs no instruction; the value appears at its consumer
    assert_func_lowers_to(
        &ref_module("(func (result funcref) ref.func 0)"),
        1,
        "
          0  move         (0)ref -> r0
          1  end
             frame: 1 registers, 0 spills
        ",
    );

    assert_lowers_to(
        "(module (func (result funcref) ref.null func))",
        "
          0  move         (null)ref -> r0
          1  end
             frame: 1 registers, 0 spills
        ",
    );
}

/// Written straight into its destination, with no register in between — the whole
/// point of carrying it as an operand rather than materializing it.
#[test]
fn a_reference_is_stored_in_place() {
    assert_func_lowers_to(
        &format!(
            "(module (global (mut funcref) (ref.null func)) (func) (elem declare func 0) {})",
            "(func ref.func 0 global.set 0)"
        ),
        1,
        "
          0  global.set   global0 <- (0)ref
          1  end
             frame: 0 registers, 0 spills
        ",
    );

    assert_func_lowers_to(
        &ref_module("(func (local funcref) ref.func 0 local.set 0)"),
        1,
        "
          0  local.set    local0 <- (0)ref
          1  end
             frame: 0 registers, 0 spills
        ",
    );
}

/// The heap type is dropped, as it is in the stack pass: `Val::Ref(None)` is the
/// only null there is at execution, and validation has already established that
/// each null reached a slot willing to hold it. Nothing downstream can tell a null
/// `funcref` from a null `externref`, so nothing needs to.
#[test]
fn a_null_reference_is_the_same_immediate_whatever_its_heap_type() {
    let funcref = render(
        &lower("(module (func (result funcref) ref.null func))"),
        &[],
    );
    let externref = render(
        &lower("(module (func (result externref) ref.null extern))"),
        &[],
    );

    assert_eq!(funcref, externref);
    assert!(funcref.contains("(null)ref"), "{funcref}");
}

/// `ref.is_null` is the only consumer of a reference the pass has, and it reads
/// its operand wherever it already lives — as an immediate when the reference came
/// straight from `ref.func`/`ref.null`, out of a local otherwise.
#[test]
fn ref_is_null_reads_its_operand_in_place() {
    assert_func_lowers_to(
        &ref_module("(func (result i32) ref.func 0 ref.is_null)"),
        1,
        "
          0  ref.is_null  (0)ref -> r0
          1  move         r0 -> r0
          2  end
             frame: 1 registers, 0 spills
        ",
    );

    assert_lowers_to(
        "(module (func (param funcref) (result i32) local.get 0 ref.is_null))",
        "
          0  ref.is_null  local0 -> r0
          1  move         r0 -> r0
          2  end
             frame: 1 registers, 0 spills
        ",
    );
}

/// The result is an `i32` predicate, not a reference, so it feeds a branch with
/// nothing in between — the reason it follows the comparison convention.
#[test]
fn ref_is_null_feeds_a_branch_directly() {
    assert_lowers_to(
        r#"(module (func (param funcref) (result i32)
             block (result i32)
               i32.const 1
               local.get 0
               ref.is_null
               br_if 0
               drop
               i32.const 2
             end))"#,
        "
          0  ref.is_null  local0 -> r0
          1  br_if        r0 -> 3  move 1 -> r0
          2  move         2 -> r0
          3  end
          4  move         r0 -> r0
          5  end
             frame: 1 registers, 0 spills
        ",
    );
}

/// References take part in lazy forwarding like any other value: a `funcref`
/// global borrowed across a call is rescued, and `ref.is_null` reads the spill.
#[test]
fn a_reference_borrowed_across_a_call_is_rescued() {
    assert_func_lowers_to(
        r#"(module (global (mut funcref) (ref.null func)) (func)
             (func (result i32) global.get 0 call 0 ref.is_null))"#,
        1,
        "
          0  global.spill global0 -> spill0
          1  call         f0 caller_base=0
          2  ref.is_null  spill0 -> r0
          3  move         r0 -> r0
          4  end
             frame: 1 registers, 1 spills
        ",
    );
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

            assert_eq!(
                s.curr_register_index,
                1 + params as usize,
                "params sit above the live register"
            );
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

    let before = (
        heights(&s),
        s.lazy_locals.origin[1].is_some(),
        s.spills.allocation_len(),
    );

    let _mov = s.br_truncation_registers(base, 0);

    let after = (
        heights(&s),
        s.lazy_locals.origin[1].is_some(),
        s.spills.allocation_len(),
    );

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
            let reachable = matches!(t.check_unreachablity(op), UnreachableCheckResult::Reachable);
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

/// A frame just large enough to execute the instructions these tests emit, so a
/// claim about which path writes what is observed rather than argued.
///
/// Spill slots start poisoned, because the whole question is whether a slot is
/// written on every path that reads it.
struct Frame {
    locals: Vec<i64>,
    globals: Vec<i64>,
    spills: Vec<i64>,
}

const POISON: i64 = i64::MIN;

impl Frame {
    fn new(locals: &[i64], globals: &[i64]) -> Self {
        Frame {
            locals: locals.to_vec(),
            globals: globals.to_vec(),
            spills: vec![POISON; 8],
        }
    }

    fn read(&self, slot: &Slot) -> i64 {
        match slot {
            Slot::Const(Const::I32(v)) => *v as i64,
            Slot::Local(n) => self.locals[*n as usize],
            Slot::Global(n) => self.globals[*n as usize],
            Slot::Spilled(i) => self.spills[spill_slot(i)],
            Slot::Register(_) => unreachable!("these tests emit no register writes"),
            Slot::Const(_) => unreachable!("i32 constants only"),
        }
    }

    fn exec(&mut self, instruction: &RegInstruction, ins: &[Slot]) {
        match instruction {
            RegInstruction::LocalSpill { index, spill_index } => {
                self.spills[spill_slot(spill_index)] = self.locals[index.0 as usize];
            }
            RegInstruction::GlobalSpill { index, spill_index } => {
                self.spills[spill_slot(spill_index)] = self.globals[index.0 as usize];
            }
            RegInstruction::LocalSet { index, sig } => {
                self.locals[index.0 as usize] = self.read(&sig.input.registers(ins)[0]);
            }
            RegInstruction::GlobalSet { index, sig } => {
                self.globals[index.0 as usize] = self.read(&sig.input.registers(ins)[0]);
            }
            _ => unreachable!("these tests emit only spills and writes"),
        }
    }
}

/// `SpillIndex` is opaque outside the pass, so its slot number comes back through
/// `Display`.
fn spill_slot(index: &SpillIndex) -> usize {
    format!("{index}").parse().expect("spill index")
}

/// Executes `prog`, skipping the half-open range `skip` — the instructions a branch
/// jumps over on one of its paths.
fn run(prog: &[RegInstruction], ins: &[Slot], skip: Range<usize>, frame: &mut Frame) {
    for (pc, instruction) in prog.iter().enumerate() {
        if skip.contains(&pc) {
            continue;
        }

        frame.exec(instruction, ins);
    }
}

// ---------------------------------------------------------------------------
// a spill must dominate its readers, and run once
//
// A borrow resting on the operand stack across control flow is rescued by a spill.
// That rescue is only sound if the copy executes on *every* path that later reads
// it, and is not re-executed after the write it was rescuing from. `if`, `br_if`,
// `br_table` and `loop` are the four places that can break one of those, so each
// hoists live borrows before the construct.
// ---------------------------------------------------------------------------

/// `local.get 0 ; local.get 1 ; if ; i32.const 5 ; local.set 0 ; end ; <use>`
///
/// The borrow sits below the `if`, so it outlives the construct while the write
/// that invalidates it happens on one arm only.
#[test]
fn a_conditional_arm_cannot_own_the_spill() {
    let mut s = sim(2, 0);
    let mut prog: Vec<RegInstruction> = vec![];

    s.push_local(0); // the borrow
    s.push_local(1); // the condition

    RegInstruction::spill_live_locals(&mut s, &mut prog);
    RegInstruction::spill_live_globals(&mut s, &mut prog);

    assert!(
        !prog.is_empty(),
        "the borrow must be rescued above the split"
    );

    s.add_block(BlockVariant::If, &BlockType::Empty, &[], prog.len());

    let _cond = s.registers_for::<1, 0>();

    // then-arm
    let arm = prog.len()..{
        s.push_const(Const::I32(5));

        assert!(
            SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).is_none(),
            "the write finds nothing left to rescue"
        );

        let sig = s.registers_for::<1, 0>();

        prog.push(RegInstruction::LocalSet {
            index: LocalIndex(0),
            sig,
        });

        prog.len()
    };

    // else-arm is empty; reset as the `Else` arm does
    let recorded = s.get_curr_block().recorded_height;
    let params = s.get_curr_block().params;

    s.pops_and_pushes(s.stack.height() - recorded, params);

    let consumer = s.pop();

    for (tag, skip) in [("taken", 0..0), ("not taken", arm.clone())] {
        let mut frame = Frame::new(&[42, 1], &[]);

        run(&prog, &s.input_registers, skip, &mut frame);

        assert_eq!(
            frame.read(&consumer),
            42,
            "{tag}: <use> reads {}, which that path never wrote",
            slot_str(&consumer)
        );
    }
}

/// The same shape for a global, which uses the other arena.
#[test]
fn a_conditional_arm_cannot_own_a_global_spill() {
    let mut s = sim(1, 2);
    let mut prog: Vec<RegInstruction> = vec![];

    s.push_global(1);
    s.push_local(0); // the condition

    RegInstruction::spill_live_locals(&mut s, &mut prog);
    RegInstruction::spill_live_globals(&mut s, &mut prog);

    s.add_block(BlockVariant::If, &BlockType::Empty, &[], prog.len());

    let _cond = s.registers_for::<1, 0>();

    let arm = prog.len()..{
        s.push_const(Const::I32(5));

        assert!(SimulatedStack::set_lazy(1, &mut s.lazy_globals, &mut s.spills).is_none());

        let sig = s.registers_for::<1, 0>();

        prog.push(RegInstruction::GlobalSet {
            index: GlobalIndex(1),
            sig,
        });

        prog.len()
    };

    let recorded = s.get_curr_block().recorded_height;
    let params = s.get_curr_block().params;

    s.pops_and_pushes(s.stack.height() - recorded, params);

    let consumer = s.pop();

    for (tag, skip) in [("taken", 0..0), ("not taken", arm.clone())] {
        let mut frame = Frame::new(&[1], &[0, 42]);

        run(&prog, &s.input_registers, skip, &mut frame);

        assert_eq!(frame.read(&consumer), 42, "{tag}");
    }
}

/// `local.get 0 ; block ; br_if 0 ; i32.const 5 ; local.set 0 ; end ; <use>`
///
/// Here the *taken* path is the one that skips the write, so the polarity is the
/// opposite of the `if` case.
#[test]
fn a_taken_br_if_cannot_skip_the_spill() {
    let mut s = sim(2, 0);
    let mut prog: Vec<RegInstruction> = vec![];

    s.push_local(0); // the borrow
    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], prog.len());

    let base = s.get_curr_block().recorded_height;

    s.push_const(Const::I32(1)); // the condition

    RegInstruction::spill_live_locals(&mut s, &mut prog);
    RegInstruction::spill_live_globals(&mut s, &mut prog);

    let _cond = s.registers_for::<1, 0>();
    let _mov = s.br_truncation_registers(base, 0);

    // everything after the branch is skipped when it is taken
    let rest = prog.len()..{
        s.push_const(Const::I32(5));

        assert!(SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).is_none());

        let sig = s.registers_for::<1, 0>();

        prog.push(RegInstruction::LocalSet {
            index: LocalIndex(0),
            sig,
        });

        prog.len()
    };

    let consumer = s.pop();

    for (tag, skip) in [("not taken", 0..0), ("taken", rest.clone())] {
        let mut frame = Frame::new(&[42, 1], &[]);

        run(&prog, &s.input_registers, skip, &mut frame);

        assert_eq!(frame.read(&consumer), 42, "{tag}");
    }
}

/// `local.get 0 ; loop ; i32.const 5 ; local.set 0 ; br_if 0 ; end ; <use>`
///
/// Not a divergence but a repetition: a spill left inside the body would re-run on
/// the back-edge and capture the value the previous iteration wrote.
#[test]
fn a_loop_body_cannot_own_the_spill() {
    let mut s = sim(2, 0);
    let mut prog: Vec<RegInstruction> = vec![];

    s.push_local(0); // the borrow, below the loop

    RegInstruction::spill_live_locals(&mut s, &mut prog);
    RegInstruction::spill_live_globals(&mut s, &mut prog);

    let entry = 0..prog.len();

    assert!(
        !entry.is_empty(),
        "the borrow must be rescued above the header"
    );

    s.add_block(BlockVariant::Loop, &BlockType::Empty, &[], prog.len());

    // body
    let body = prog.len()..{
        s.push_const(Const::I32(5));

        assert!(
            SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).is_none(),
            "the write finds nothing left to rescue"
        );

        let sig = s.registers_for::<1, 0>();

        prog.push(RegInstruction::LocalSet {
            index: LocalIndex(0),
            sig,
        });

        // the back-edge branches too, and must add nothing
        let before = prog.len();

        RegInstruction::spill_live_locals(&mut s, &mut prog);

        assert_eq!(prog.len(), before, "nothing left to spill at the back-edge");

        prog.len()
    };

    let recorded = s.get_curr_block().recorded_height;

    s.pops_and_pushes(s.stack.height() - recorded, 0);

    let consumer = s.pop();

    for iterations in [1, 2, 5] {
        let mut frame = Frame::new(&[42, 1], &[]);

        for (pc, instruction) in prog.iter().enumerate() {
            if entry.contains(&pc) {
                frame.exec(instruction, &s.input_registers);
            }
        }

        for _ in 0..iterations {
            for (pc, instruction) in prog.iter().enumerate() {
                if body.contains(&pc) {
                    frame.exec(instruction, &s.input_registers);
                }
            }
        }

        assert_eq!(
            frame.read(&consumer),
            42,
            "{iterations} iteration(s): the entry spill must not be re-run"
        );
    }
}

/// Entering a `block` is unconditional and never repeats, so a spill inside it
/// already dominates its readers and runs once.
#[test]
fn a_block_does_not_hoist_spills() {
    let mut s = sim(2, 0);
    let mut prog: Vec<RegInstruction> = vec![];

    s.push_local(0);

    RegInstruction::spill_live_locals(&mut s, &mut prog); // the Block arm makes no such call

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], prog.len());

    // the call is shown only to contrast: the Block arm does not make it, and the
    // borrow would otherwise be spilled for nothing
    assert_eq!(prog.len(), 1, "this is what skipping the hoist avoids");
}

/// Nothing borrowed across the construct means the guard emits nothing at all,
/// which is the common case for anything rustc produces.
#[test]
fn hoisting_costs_nothing_when_no_borrow_is_live() {
    let mut s = sim(8, 4);
    let mut prog: Vec<RegInstruction> = vec![];

    s.push_local(0);

    let _ = s.registers_for::<1, 1>(); // consumed into a register

    s.push_const(Const::I32(1)); // a condition

    RegInstruction::spill_live_locals(&mut s, &mut prog);
    RegInstruction::spill_live_globals(&mut s, &mut prog);

    assert!(prog.is_empty(), "no live borrows, no instructions");
    assert_eq!(s.spills.allocation_len(), 0, "and no frame slots reserved");
}

// ---------------------------------------------------------------------------
// globals
//
// Globals run through the same machinery as locals but a separate arena, so the
// asymmetries are worth pinning rather than assuming.
// ---------------------------------------------------------------------------

#[test]
fn global_borrows_behave_like_local_ones() {
    let mut s = sim(2, 2);

    s.push_global(1);
    s.push_global(1);
    s.push_const(Const::I32(5));

    let spill = SimulatedStack::set_lazy(1, &mut s.lazy_globals, &mut s.spills);

    let _set = s.registers_for::<1, 0>();
    let add = s.registers_for::<2, 1>();

    assert_eq!(spilled_to(spill), Some("spill0".into()));

    assert_eq!(
        slots(add.input.registers(&s.input_registers)),
        ["spill0", "spill0"]
    );
}

#[test]
fn a_local_and_a_global_of_the_same_index_are_independent() {
    let mut s = sim(2, 2);

    s.push_local(0);
    s.push_global(0);
    s.push_const(Const::I32(7));

    // writing global 0 must not disturb the borrow of local 0
    let global_spill = SimulatedStack::set_lazy(0, &mut s.lazy_globals, &mut s.spills);

    let _set = s.registers_for::<1, 0>();

    assert_eq!(spilled_to(global_spill), Some("spill0".into()));

    assert!(
        s.lazy_locals.origin[0].is_some(),
        "local 0's borrow must survive a write to global 0"
    );

    assert_eq!(slot_str(&s.pop()), "spill0", "the global was rescued");
    assert_eq!(slot_str(&s.pop()), "local0", "the local still forwards");
}

#[test]
fn locals_and_globals_draw_from_one_spill_pool() {
    let mut s = sim(2, 2);

    s.push_local(0);
    s.push_global(0);

    let l = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills);
    let g = SimulatedStack::set_lazy(0, &mut s.lazy_globals, &mut s.spills);

    assert_eq!(
        (spilled_to(l), spilled_to(g)),
        (Some("spill0".into()), Some("spill1".into())),
        "distinct slots from the shared pool"
    );

    assert_eq!(s.spills.allocation_len(), 2);
}

#[test]
fn writing_one_local_leaves_other_borrows_alone() {
    let mut s = sim(4, 0);

    s.push_local(0);
    s.push_local(1);
    s.push_local(2);

    let spill = SimulatedStack::set_lazy(1, &mut s.lazy_locals, &mut s.spills);

    assert_eq!(spilled_to(spill), Some("spill0".into()));
    assert_eq!(slot_str(&s.pop()), "local2", "untouched above");
    assert_eq!(slot_str(&s.pop()), "spill0", "rescued");
    assert_eq!(slot_str(&s.pop()), "local0", "untouched below");
}

#[test]
fn writing_an_unborrowed_local_emits_nothing() {
    let mut s = sim(4, 0);

    s.push_local(0);

    assert!(SimulatedStack::set_lazy(3, &mut s.lazy_locals, &mut s.spills).is_none());
    assert_eq!(s.spills.allocation_len(), 0);
}

#[test]
fn three_borrows_share_one_entry() {
    let mut s = sim(2, 0);

    s.push_local(0);
    s.push_local(0);
    s.push_local(0);

    assert_eq!(
        spilled_to(SimulatedStack::set_lazy(
            0,
            &mut s.lazy_locals,
            &mut s.spills
        )),
        Some("spill0".into())
    );

    for _ in 0..3 {
        assert_eq!(slot_str(&s.pop()), "spill0", "all three redirect");
    }

    // the pool only reclaims once the last of them is gone
    assert_eq!(s.spills.allocation_len(), 1);
}

#[test]
fn tee_of_the_local_it_reads_round_trips_through_a_spill() {
    let mut s = sim(2, 0);

    s.push_local(0);

    let spill = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills);

    let operand = s.tee();

    // `local.tee 0` on a value read from local 0: correct, if redundant
    assert_eq!(spilled_to(spill), Some("spill0".into()));
    assert_eq!(slot_str(&operand), "spill0");
}

// ---------------------------------------------------------------------------
// drop
// ---------------------------------------------------------------------------

#[test]
fn drop_releases_a_register() {
    let mut s = sim(2, 0);

    s.push_local(0);

    let _ = s.registers_for::<1, 1>();

    assert_eq!(heights(&s), (1, 1));

    s.pop();

    assert_eq!(heights(&s), (0, 0), "the register comes back");
    assert_eq!(s.max_registers, 1, "but the peak is remembered");
}

#[test]
fn drop_releases_a_borrow_but_only_the_last_one() {
    let mut s = sim(2, 0);

    s.push_local(0);
    s.push_local(0);
    s.pop();

    assert!(
        SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).is_some(),
        "one borrow survives, so a write still has to rescue it"
    );

    let mut s = sim(2, 0);

    s.push_local(0);
    s.pop();

    assert!(
        SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).is_none(),
        "the sole borrow is gone"
    );
}

// ---------------------------------------------------------------------------
// operand ordering and register reuse
// ---------------------------------------------------------------------------

#[test]
fn operands_are_recorded_deepest_first() {
    let mut s = sim(4, 0);

    s.push_local(0); // a
    s.push_local(1); // b
    s.push_local(2); // condition, pushed last

    let select = s.registers_for::<3, 1>();

    assert_eq!(
        slots(select.input.registers(&s.input_registers)),
        ["local0", "local1", "local2"],
        "select reads a, b, cond in that order"
    );
}

#[test]
fn store_records_address_then_value() {
    let mut s = sim(4, 0);

    s.push_local(0); // address
    s.push_const(Const::I32(9)); // value

    let store = s.registers_for::<2, 0>();

    assert_eq!(
        slots(store.input.registers(&s.input_registers)),
        ["local0", "9"]
    );
}

#[test]
fn a_result_reuses_an_operands_register() {
    let mut s = sim(4, 0);

    s.push_local(0);

    let _ = s.registers_for::<1, 1>(); // r0

    s.push_local(1);

    let _ = s.registers_for::<1, 1>(); // r1
    let add = s.registers_for::<2, 1>();
    let inputs = add.input.registers(&s.input_registers);
    let output = add.output.registers(&s.output_registers)[0];

    assert_eq!(slots(inputs), ["r0", "r1"]);

    assert_eq!(
        output, 0,
        "the destination aliases an operand — an executor must read both first"
    );
}

#[test]
fn max_registers_tracks_the_peak_not_the_total() {
    // two loads, each consumed before the next
    let mut s = sim(4, 0);

    for l in 0..2 {
        s.push_local(l);

        let _ = s.registers_for::<1, 1>();
        let _ = s.registers_for::<1, 0>();
    }

    assert_eq!(s.max_registers, 1, "serial values reuse one register");

    // two loads live at once
    let mut s = sim(4, 0);

    for l in 0..2 {
        s.push_local(l);

        let _ = s.registers_for::<1, 1>();
    }

    assert_eq!(s.max_registers, 2);
}

#[test]
fn non_register_operands_occupy_stack_but_not_registers() {
    let mut s = sim(4, 2);

    s.push_const(Const::I32(1));
    s.push_local(0);
    s.push_global(0);

    assert_eq!(
        heights(&s),
        (3, 0),
        "three slots, zero registers — the two heights diverge"
    );
}

// ---------------------------------------------------------------------------
// block indices
// ---------------------------------------------------------------------------

#[test]
fn a_loop_back_edge_targets_the_first_body_instruction() {
    let types = vec![func_ty(1, 1)];

    // no params: no entry move, so the body starts where the loop was seen
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Loop, &BlockType::Empty, &types, 7);

    assert!(matches!(
        s.get_curr_block().kind,
        BlockKind::Loop { index: 7 }
    ));

    // with params: an entry move occupies index 7, body starts at 8
    let mut s = sim(4, 0);

    s.push_const(Const::I32(1));
    s.add_block(BlockVariant::Loop, &BlockType::FuncType(0), &types, 7);

    assert!(matches!(
        s.get_curr_block().kind,
        BlockKind::Loop { index: 8 }
    ));
}

#[test]
fn an_if_records_the_index_of_the_if_itself() {
    let types = vec![func_ty(1, 1)];
    let mut s = sim(4, 0);

    s.push_const(Const::I32(1)); // condition
    s.add_block(BlockVariant::If, &BlockType::Empty, &types, 7);

    assert!(matches!(
        s.get_curr_block().kind,
        BlockKind::If { index: 7, .. }
    ));

    let mut s = sim(4, 0);

    s.push_const(Const::I32(9)); // param
    s.push_const(Const::I32(1)); // condition
    s.add_block(BlockVariant::If, &BlockType::FuncType(0), &types, 7);

    assert!(matches!(
        s.get_curr_block().kind,
        BlockKind::If { index: 8, .. }
    ));
}

// ---------------------------------------------------------------------------
// branches, harder shapes
// ---------------------------------------------------------------------------

#[test]
fn a_branch_to_the_function_frame_unwinds_to_zero() {
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);
    s.push_local(0);

    let _ = s.registers_for::<1, 1>();

    let func_block = s.get_block(0);

    assert_eq!(func_block.recorded_height, 0);

    let mov = s.br_truncation_registers(0, 0);

    assert!(mov.is_empty());
    assert_eq!(heights(&s), (1, 1), "still a simulation");
}

#[test]
fn a_branch_carrying_several_values_lands_them_contiguously() {
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;

    s.push_local(0);

    let _ = s.registers_for::<1, 1>();

    s.push_const(Const::I32(2));
    s.push_local(1);

    let mov = s.br_truncation_registers(base, 3);

    assert_eq!(mov.output_registers(&s.output_registers), &[0, 1, 2]);
    assert_eq!(mov.input_registers(&s.input_registers).len(), 3);
}

#[test]
fn a_br_table_may_mix_loop_and_block_targets() {
    // loop's label type is its params; a block's is its results. Validation only
    // requires the types to match, so the arities agree while the bases differ.
    let types = vec![func_ty(1, 1)];
    let mut s = sim(4, 0);

    s.push_const(Const::I32(0)); // the loop's param
    s.add_block(BlockVariant::Loop, &BlockType::FuncType(0), &types, 0);

    let loop_block = *&s.get_curr_block().recorded_height;
    let loop_params = s.get_curr_block().params;

    let _ = s.materialize_stack_slots_in_registers(loop_params);

    s.add_block(
        BlockVariant::Block,
        &BlockType::Type(wasmparser::ValType::I32),
        &types,
        1,
    );

    let block_base = s.get_curr_block().recorded_height;
    let block_results = s.get_curr_block().results;

    s.push_local(0); // the value the branch carries

    let to_loop = s.br_truncation_registers(loop_block, loop_params);
    let to_block = s.br_truncation_registers(block_base, block_results);

    assert_eq!(loop_params, block_results, "arities agree");

    assert_ne!(
        to_loop.output_registers(&s.output_registers),
        to_block.output_registers(&s.output_registers),
        "but the destinations differ, which is why each arm carries its own move"
    );
}

#[test]
fn every_br_table_arm_sees_the_same_stack() {
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;

    s.push_local(0);

    let before = heights(&s);
    let arms: Vec<_> = (0..4).map(|_| s.br_truncation_registers(base, 1)).collect();

    assert_eq!(heights(&s), before, "four simulations, no mutation");

    let first = arms[0].output_registers(&s.output_registers).to_vec();

    for arm in &arms[1..] {
        assert_eq!(
            arm.output_registers(&s.output_registers),
            first.as_slice(),
            "identical targets must produce identical moves"
        );
    }
}

#[test]
fn a_branch_carrying_nothing_emits_no_move() {
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;

    s.push_local(0);

    let _ = s.registers_for::<1, 1>();

    assert!(
        s.br_truncation_registers(base, 0).is_empty(),
        "callers use this to skip emitting the move entirely"
    );
}

#[test]
fn br_if_consumes_only_its_condition() {
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;
    s.push_local(0);

    let _ = s.registers_for::<1, 1>();
    s.push_local(1);

    let live = (heights(&s), s.lazy_locals.origin[1].is_some());

    s.push_const(Const::I32(1)); // condition

    let _cond = s.registers_for::<1, 0>();
    let _mov = s.br_truncation_registers(base, 0);

    assert_eq!(
        (heights(&s), s.lazy_locals.origin[1].is_some()),
        live,
        "the fall-through path must see exactly what it saw before"
    );
}

// ---------------------------------------------------------------------------
// unreachable, harder shapes
// ---------------------------------------------------------------------------

#[test]
fn a_second_branch_in_dead_code_is_skipped() {
    let seq = [blk(), br(0), br(0), cst(), Operator::End];
    assert_eq!(reachability(&seq), [true, true, false, false, true]);
}

#[test]
fn an_if_without_an_else_in_dead_code_closes_cleanly() {
    let seq = [blk(), br(0), iff(), cst(), Operator::End, Operator::End];
    assert_eq!(reachability(&seq), [true, true, false, false, false, true]);
}

#[test]
fn dead_code_at_the_function_level_ends_at_the_final_end() {
    let seq = [br(0), cst(), Operator::End];
    assert_eq!(reachability(&seq), [true, false, true]);
}

#[test]
fn liveness_resumes_independently_in_each_sibling_block() {
    let seq = [
        blk(),
        br(0),
        cst(),
        Operator::End, // first block goes dead, recovers at its end
        cst(),         // live again
        blk(),
        br(0),
        cst(),
        Operator::End, // and the same again
        cst(),
    ];
    assert_eq!(
        reachability(&seq),
        [true, true, false, true, true, true, true, false, true, true]
    );
}

#[test]
fn deeply_nested_dead_constructs_unwind_in_order() {
    let seq = [
        blk(),
        br(0),
        blk(),
        lop(),
        iff(),
        Operator::Else,
        Operator::End, // closes if
        Operator::End, // closes loop
        Operator::End, // closes block
        cst(),
        Operator::End, // closes the live block -> reachable again
    ];
    let live = reachability(&seq);

    assert_eq!(
        live[live.len() - 1],
        true,
        "the live block's end must resume"
    );
    assert!(
        live[2..live.len() - 1].iter().all(|r| !r),
        "everything nested inside stays dead: {live:?}"
    );
}

// ---------------------------------------------------------------------------
// arenas travel with the body
//
// Every operand and every `br_table` arm is an index into a side table. If a table
// is not moved onto `FrameLayout`, the indices in the instruction list name freed
// storage and nothing can resolve them.
// ---------------------------------------------------------------------------

#[test]
fn br_table_arms_survive_lowering() {
    let mut s = sim(4, 0);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let outer = s.get_curr_block().recorded_height;

    s.push_local(0);

    let _ = s.registers_for::<1, 1>();

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let inner = s.get_curr_block().recorded_height;

    s.push_local(1);

    // two arms plus a default, as the BrTable arm records them
    for base in [inner, outer, inner] {
        let mov = s.br_truncation_registers(base, 1);

        s.br_targets.push(BrTarget {
            mov,
            target_index: u32::MAX,
        });
    }

    let frame = FrameLayout {
        registers: s.max_registers,
        spills: s.spills.allocation_len(),
        input_registers_arena: s.input_registers.into_boxed_slice(),
        output_registers_arena: s.output_registers.into_boxed_slice(),
        br_targets_arena: s.br_targets.into_boxed_slice(),
    };

    assert_eq!(
        frame.br_targets_arena.len(),
        3,
        "arms must ship with the body"
    );

    // and each one still resolves against the arenas it was recorded in
    let dests: Vec<&[u32]> = frame
        .br_targets_arena
        .iter()
        .map(|t| t.mov.output_registers(&frame.output_registers_arena))
        .collect();

    assert_eq!(dests, vec![&[1u32][..], &[0u32][..], &[1u32][..]]);
}

#[test]
fn a_body_without_a_br_table_carries_an_empty_arm_arena() {
    let mut s = sim(2, 0);

    s.push_local(0);

    let _ = s.registers_for::<1, 1>();

    let frame = FrameLayout {
        registers: s.max_registers,
        spills: s.spills.allocation_len(),
        input_registers_arena: s.input_registers.into_boxed_slice(),
        output_registers_arena: s.output_registers.into_boxed_slice(),
        br_targets_arena: s.br_targets.into_boxed_slice(),
    };

    assert!(frame.br_targets_arena.is_empty());
}

// ===========================================================================
// end-to-end
//
// Everything below goes through `lower`, so it exercises the real pass over real
// wasm: operand forwarding, spills, label layout, backpatching and the frame, all
// composed. Expectations are whole rendered programs wherever the output is small
// enough to read, because a diff of the program is the most legible failure there
// is.
// ===========================================================================

/// Position of the first instruction matching `pred`.
fn index_of(prog: &[RegInstruction], pred: impl Fn(&RegInstruction) -> bool) -> Option<usize> {
    prog.iter().position(pred)
}

/// [`index_of`] for the common case of looking for an instruction by kind rather
/// than by anything it carries.
fn index_of_kind(prog: &[RegInstruction], kind: RegInstructionKind) -> Option<usize> {
    index_of(prog, |instruction| instruction.kind() == kind)
}

// ---------------------------------------------------------------------------
// operand forwarding
// ---------------------------------------------------------------------------

#[test]
fn a_straight_line_body_copies_nothing() {
    assert_lowers_to(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             local.get 1
             i32.add))"#,
        "
          0  i32.add      local0, local1 -> r0
          1  move         r0 -> r0
          2  end
             frame: 1 registers, 0 spills
        ",
    );
}

#[test]
fn constants_and_globals_are_read_in_place() {
    assert_lowers_to(
        r#"(module
             (global i32 (i32.const 0))
             (func (param i32) (result i32)
               global.get 0
               i32.const 7
               i32.add))"#,
        "
          0  i32.add      global0, 7 -> r0
          1  move         r0 -> r0
          2  end
             frame: 1 registers, 0 spills
        ",
    );
}

#[test]
fn a_load_produces_the_only_register_a_body_needs() {
    assert_lowers_to(
        r#"(module (memory 1) (func (param i32) (result i32)
             local.get 0
             i32.load))"#,
        "
          0  i32.load     [local0]+0 -> r0
          1  move         r0 -> r0
          2  end
             frame: 1 registers, 0 spills
        ",
    );
}

// ---------------------------------------------------------------------------
// spilling a borrowed local
// ---------------------------------------------------------------------------

#[test]
fn writing_a_borrowed_local_rescues_it_first() {
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
          3  move         r0 -> r0
          4  end
             frame: 1 registers, 1 spills
        ",
    );
}

#[test]
fn an_unborrowed_write_rescues_nothing() {
    let (prog, frame) = lower(
        r#"(module (func (param i32)
             i32.const 5
             local.set 0))"#,
    );

    assert!(
        index_of_kind(&prog, RegInstructionKind::LocalSpill).is_none(),
        "nothing borrows local 0"
    );

    assert_eq!(frame.spills, 0);
}

// ---------------------------------------------------------------------------
// blocks and labels
// ---------------------------------------------------------------------------

#[test]
fn a_block_emits_nothing_of_its_own() {
    assert_lowers_to(
        r#"(module (func (param i32) (result i32)
             (block (nop))
             local.get 0))"#,
        "
          0  end
          1  move         local0 -> r0
          2  end
             frame: 1 registers, 0 spills
        ",
    );
}

#[test]
fn an_if_else_wires_both_arms_to_one_end() {
    assert_lowers_to(
        r#"(module (func (param i32) (result i32)
             local.get 0
             if (result i32) i32.const 1 else i32.const 2 end))"#,
        "
          0  local.spill  local0 -> spill0
          1  if           spill0 else=3 end=5
          2  move         1 -> r0
          3  else         end=5
          4  move         2 -> r0
          5  end
          6  move         r0 -> r0
          7  end
             frame: 1 registers, 1 spills
        ",
    );
}

#[test]
fn an_if_without_an_else_jumps_straight_to_the_end() {
    let (prog, _) = lower(
        r#"(module (func (param i32)
             local.get 0
             if i32.const 5 local.set 0 end))"#,
    );

    let at = index_of_kind(&prog, RegInstructionKind::If).unwrap();

    let (else_index, end_index) = match &prog[at] {
        RegInstruction::If {
            else_index,
            end_index,
            ..
        } => (*else_index, *end_index),
        _ => unreachable!(),
    };

    assert!(else_index.is_none(), "there is no else to jump to");

    assert!(
        matches!(prog[end_index as usize], RegInstruction::End),
        "end_index must name an End"
    );
}

#[test]
fn both_arms_of_an_if_materialise_results_into_the_same_registers() {
    let (prog, frame) = lower(
        r#"(module (func (param i32) (result i32)
             local.get 0
             if (result i32) i32.const 1 else i32.const 2 end))"#,
    );

    let dests: Vec<Vec<u32>> = prog
        .iter()
        .filter_map(|i| match i {
            RegInstruction::Move(sig) => {
                Some(sig.output_registers(&frame.output_registers_arena).to_vec())
            }
            _ => None,
        })
        .collect();

    assert!(
        dests.windows(2).all(|w| w[0] == w[1]),
        "every path must leave the result in the same register: {dests:?}"
    );
}

// ---------------------------------------------------------------------------
// branches
// ---------------------------------------------------------------------------

#[test]
fn a_branch_lands_past_the_fallthrough_move() {
    // the block's own `end` materialises results for the path that falls into it;
    // a branch carries its own, so it must jump past that move
    assert_lowers_to(
        r#"(module (func (result i32)
             block (result i32) i32.const 7 br 0 end))"#,
        "
          0  move         7 -> r0
          1  br           -> 3
          2  move         r0 -> r0
          3  end
          4  move         r0 -> r0
          5  end
             frame: 1 registers, 0 spills
        ",
    );
}

#[test]
fn an_outward_branch_targets_the_outer_label() {
    let (prog, _) = lower(
        r#"(module (func (param i32) (result i32)
             block (result i32)
               block (result i32) local.get 0 br 1 end
             end))"#,
    );

    let br = index_of_kind(&prog, RegInstructionKind::Br).unwrap();

    let RegInstruction::Br { target_index } = &prog[br] else {
        unreachable!()
    };

    let ends: Vec<usize> = prog
        .iter()
        .enumerate()
        .filter(|(_, i)| i.kind() == RegInstructionKind::End)
        .map(|(n, _)| n)
        .collect();

    assert_eq!(
        *target_index as usize, ends[1],
        "br 1 must reach the *second* end (the outer block's), not the inner one"
    );
}

#[test]
fn a_loop_back_edge_skips_the_entry_spill() {
    // the entry rescue must run once; the back-edge has to land after it
    assert_lowers_to(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             loop
               i32.const 5
               local.set 0
               local.get 1
               br_if 0
             end))"#,
        "
          0  local.spill  local0 -> spill0
          1  local.set    local0 <- 5
          2  local.spill  local1 -> spill1
          3  br_if        spill1 -> 1
          4  end
          5  move         spill0 -> r0
          6  end
             frame: 1 registers, 2 spills
        ",
    );
}

#[test]
fn a_branch_to_a_loop_carries_the_loops_params() {
    let (prog, frame) = lower(
        r#"(module (func (param i32) (result i32)
             i32.const 1
             loop (param i32) (result i32)
               drop i32.const 2 local.get 0 br_if 0
             end))"#,
    );

    let br = index_of_kind(&prog, RegInstructionKind::BrIf).unwrap();

    let RegInstruction::BrIf {
        mov, target_index, ..
    } = &prog[br]
    else {
        unreachable!()
    };

    assert_eq!(
        mov.input_registers(&frame.input_registers_arena).len(),
        1,
        "a back-edge transfers the loop's params, not its results"
    );

    assert!(
        (*target_index as usize) < br,
        "a back-edge jumps backwards: {target_index} should precede {br}"
    );
}

#[test]
fn br_table_arms_resolve_to_their_own_labels() {
    let (prog, frame) = lower(
        r#"(module (func (param i32)
             block block local.get 0 br_table 0 1 0 end end))"#,
    );

    let table = index_of_kind(&prog, RegInstructionKind::BrTable).unwrap();

    let RegInstruction::BrTable {
        targets_start,
        targets_len,
        ..
    } = &prog[table]
    else {
        unreachable!()
    };

    let arms =
        &frame.br_targets_arena[*targets_start as usize..(targets_start + targets_len) as usize];
    let targets: Vec<u32> = arms.iter().map(|a| a.target_index).collect();

    assert_eq!(targets.len(), 3, "two arms plus the default");

    assert_eq!(
        targets[0], targets[2],
        "arm 0 and the default both name depth 0"
    );

    assert_ne!(targets[0], targets[1], "depth 1 is a different label");

    for t in &targets {
        assert!(
            matches!(prog[*t as usize], RegInstruction::End),
            "every arm must land on an End"
        );
    }
}

#[test]
fn each_br_table_arm_carries_its_own_move() {
    let (prog, frame) = lower(
        r#"(module (func (param i32) (result i32)
             block (result i32) block (result i32)
               i32.const 9 local.get 0 br_table 0 1 0
             end end))"#,
    );

    let table = index_of_kind(&prog, RegInstructionKind::BrTable).unwrap();

    let RegInstruction::BrTable {
        targets_start,
        targets_len,
        ..
    } = &prog[table]
    else {
        unreachable!()
    };

    let arms =
        &frame.br_targets_arena[*targets_start as usize..(targets_start + targets_len) as usize];

    for arm in arms {
        assert_eq!(
            slots(arm.mov.input_registers(&frame.input_registers_arena)),
            ["9"],
            "every arm transfers the same value"
        );

        assert_eq!(
            arm.mov
                .output_registers(&frame.output_registers_arena)
                .len(),
            1
        );
    }
}

// ---------------------------------------------------------------------------
// the hoist happens at the call sites
//
// The unit tests above replay each arm by calling the helper directly, so they pin
// its behaviour but stay green if an arm stops calling it. These fail in that case.
// ---------------------------------------------------------------------------

#[test]
fn the_if_arm_hoists_the_spill_above_the_branch() {
    let body = lower(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             local.get 1
             if i32.const 5 local.set 0 end))"#,
    );

    let prog = &body.0;

    let spill =
        index_of_kind(prog, RegInstructionKind::LocalSpill).expect("the borrow must be rescued");
    let branch = index_of_kind(prog, RegInstructionKind::If).expect("the if");

    assert!(
        spill < branch,
        "spill at {spill} must precede the branch at {branch}:\n{}",
        render(&body, &[])
    );
}

#[test]
fn the_br_if_arm_hoists_the_spill_above_the_branch() {
    let (prog, _) = lower(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             block
               local.get 1
               br_if 0
               i32.const 5
               local.set 0
             end))"#,
    );

    let spill =
        index_of_kind(&prog, RegInstructionKind::LocalSpill).expect("the borrow must be rescued");
    let branch = index_of_kind(&prog, RegInstructionKind::BrIf).expect("the br_if");

    assert!(spill < branch, "spill at {spill}, branch at {branch}");
}

#[test]
fn the_loop_arm_hoists_the_spill_out_of_the_repeated_region() {
    let (prog, _) = lower(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             loop
               i32.const 5
               local.set 0
               local.get 1
               br_if 0
             end))"#,
    );

    let back_edge = index_of_kind(&prog, RegInstructionKind::BrIf).unwrap();

    let RegInstruction::BrIf { target_index, .. } = &prog[back_edge] else {
        unreachable!()
    };

    let rescue = prog
        .iter()
        .enumerate()
        .find(|(_, i)| matches!(i, RegInstruction::LocalSpill { index, .. } if index.0 == 0))
        .map(|(n, _)| n)
        .expect("local 0 must be rescued");

    assert!(
        rescue < *target_index as usize,
        "the rescue at {rescue} is inside the repeated region starting at {target_index}"
    );
}

#[test]
fn the_br_table_arm_hoists_the_spill_above_the_branch() {
    let (prog, _) = lower(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             block
               local.get 1
               br_table 0 0
               i32.const 5
               local.set 0
             end))"#,
    );

    let spill =
        index_of_kind(&prog, RegInstructionKind::LocalSpill).expect("the borrow must be rescued");
    let branch = index_of_kind(&prog, RegInstructionKind::BrTable).expect("the br_table");

    assert!(spill < branch, "spill at {spill}, branch at {branch}");
}

// ---------------------------------------------------------------------------
// return
//
// `return` is a `br` to the function frame: it transfers the results and lands on
// the body's final `End`, past that end's own fallthrough move.
// ---------------------------------------------------------------------------

#[test]
fn a_return_transfers_results_to_the_function_frame() {
    assert_lowers_to(
        r#"(module (func (param i32) (result i32)
             local.get 0
             return))"#,
        "
          0  move         local0 -> r0
          1  return       -> 3
          2  move         r0 -> r0
          3  end
             frame: 1 registers, 0 spills
        ",
    );
}

#[test]
fn a_return_with_no_results_transfers_nothing() {
    assert_lowers_to(
        r#"(module (func (param i32)
             return))"#,
        "
          0  return       -> 1
          1  end
             frame: 0 registers, 0 spills
        ",
    );
}

#[test]
fn a_return_lands_on_the_final_end_from_any_depth() {
    let (prog, _) = lower(
        r#"(module (func (param i32) (result i32)
             block block local.get 0 return end end
             i32.const 9))"#,
    );

    let at = index_of_kind(&prog, RegInstructionKind::Return).unwrap();

    let RegInstruction::Return { target_index } = &prog[at] else {
        unreachable!()
    };

    assert_eq!(
        *target_index as usize,
        prog.len() - 1,
        "a return from two blocks deep still reaches the body's last instruction"
    );

    assert!(matches!(prog[*target_index as usize], RegInstruction::End));
}

#[test]
fn a_conditional_return_shares_the_functions_result_register() {
    let (prog, frame) = lower(
        r#"(module (func (param i32) (result i32)
             local.get 0
             if (result i32) i32.const 1 return else i32.const 2 end))"#,
    );

    let dests: Vec<Vec<u32>> = prog
        .iter()
        .filter_map(|i| match i {
            RegInstruction::Move(sig) => {
                Some(sig.output_registers(&frame.output_registers_arena).to_vec())
            }
            _ => None,
        })
        .collect();

    assert!(
        dests.windows(2).all(|w| w[0] == w[1]),
        "returning and falling through must leave the result in one place: {dests:?}"
    );
}

/// The value a `return` carries is read *before* a write to the local it came
/// from, so the rescue above the branch is what keeps it reachable: the return
/// reads `spill0`, while `local0` itself now holds `5` on the taken path.
#[test]
fn a_return_reads_the_snapshot_the_hoist_preserved() {
    assert_lowers_to(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             local.get 1
             if
               i32.const 5
               local.set 0
             end
             return))"#,
        "
          0  local.spill  local0 -> spill0
          1  local.spill  local1 -> spill1
          2  if           spill1 else=- end=4
          3  local.set    local0 <- 5
          4  end
          5  move         spill0 -> r0
          6  return       -> 8
          7  move         r0 -> r0
          8  end
             frame: 1 registers, 2 spills
        ",
    );
}

#[test]
fn a_return_makes_the_rest_of_its_block_unreachable() {
    let (prog, _) = lower(
        r#"(module (func (param i32) (result i32)
             block (result i32)
               local.get 0
               return
               i32.const 7
               drop
               local.get 0
             end))"#,
    );

    // the operators after `return` are dropped, so nothing between the return and
    // the block's end survives except that end's own materialisation
    let at = index_of_kind(&prog, RegInstructionKind::Return).unwrap();

    let ends = prog
        .iter()
        .skip(at)
        .filter(|i| i.kind() == RegInstructionKind::End)
        .count();

    assert_eq!(ends, 2, "only the block's end and the body's end follow");
}

// ---------------------------------------------------------------------------
// calls
//
// `caller_base` is where the callee's frame is placed, expressed frame-relative
// like every other register index. The arguments are staged at
// `[caller_base, caller_base + params)` so they become the callee's locals in
// place, and its results come back to the same base — so the field has to be the
// *register* index below the arguments, not the simulated stack's slot height.
// ---------------------------------------------------------------------------

/// Two callers of the same function, differing only in what sits below the call.
fn caller_base_of(wat: &str) -> (u32, Vec<String>) {
    let (prog, frame) = lower_func(wat, 1);
    let at = index_of_kind(&prog, RegInstructionKind::Call).unwrap();

    let RegInstruction::Call { caller_base, .. } = &prog[at] else {
        unreachable!()
    };

    // the instruction that consumes the call's result
    let reader = prog[at + 1..]
        .iter()
        .find_map(|i| match i {
            RegInstruction::I32Add(sig) => {
                Some(slots(sig.input.registers(&frame.input_registers_arena)))
            }
            _ => None,
        })
        .expect("something must read the result");

    (*caller_base, reader)
}

#[test]
fn caller_base_is_a_register_index_not_a_slot_height() {
    const CALLEE: &str = "(func (param i32 i32) (result i32) local.get 0)";

    // a register below the call: the two heights agree, so this passes either way
    let (base, reader) = caller_base_of(&format!(
        r#"(module (memory 1) {CALLEE}
             (func (param i32) (result i32)
               local.get 0 i32.load
               local.get 0 i32.const 7 call 0
               i32.add))"#
    ));

    assert_eq!(base, 1);

    assert!(
        reader.contains(&"r1".to_string()),
        "result read at r1: {reader:?}"
    );

    // a const below the call occupies a stack slot but no register, so the slot
    // height would say 1 while the register base is 0
    let (base, reader) = caller_base_of(&format!(
        r#"(module {CALLEE}
             (func (param i32) (result i32)
               i32.const 100
               local.get 0 i32.const 7 call 0
               i32.add))"#
    ));

    assert_eq!(base, 0, "a const below the call consumes no register");

    assert!(
        reader.contains(&"r0".to_string()),
        "result read at r0: {reader:?}"
    );

    // two of them
    let (base, _) = caller_base_of(&format!(
        r#"(module {CALLEE}
             (func (param i32) (result i32)
               i32.const 100 i32.const 200
               local.get 0 i32.const 7 call 0
               i32.add i32.add))"#
    ));

    assert_eq!(base, 0);

    // a mixture: only the register counts
    let (base, reader) = caller_base_of(&format!(
        r#"(module (memory 1) {CALLEE}
             (func (param i32) (result i32)
               i32.const 100
               local.get 0 i32.load
               local.get 0
               local.get 0 i32.const 7 call 0
               i32.add i32.add i32.add))"#
    ));

    assert_eq!(base, 1, "const and local below consume no registers");
    assert!(reader.contains(&"r1".to_string()), "{reader:?}");
}

/// The arguments themselves may already be registers, and those must *not* count
/// towards the base — the base is what sits below them.
#[test]
fn registers_passed_as_arguments_do_not_raise_the_base() {
    const CALLEE: &str = "(func (param i32 i32) (result i32) local.get 0)";

    // one argument is a register, nothing below the call
    let (base, reader) = caller_base_of(&format!(
        r#"(module (memory 1) {CALLEE}
             (func (param i32) (result i32)
               local.get 0 i32.load
               i32.const 7
               call 0
               local.get 0
               i32.add))"#
    ));

    assert_eq!(
        base, 0,
        "the register argument sits *at* the base, not below it"
    );

    assert!(reader.contains(&"r0".to_string()), "{reader:?}");

    // a register below, and a register argument above it
    let (base, reader) = caller_base_of(&format!(
        r#"(module (memory 1) {CALLEE}
             (func (param i32) (result i32)
               local.get 0 i32.load
               local.get 0 i32.load
               i32.const 7
               call 0
               i32.add))"#
    ));

    assert_eq!(base, 1, "only the value below the arguments counts");
    assert!(reader.contains(&"r1".to_string()), "{reader:?}");

    // both arguments are registers
    let (base, _) = caller_base_of(&format!(
        r#"(module (memory 1) {CALLEE}
             (func (param i32) (result i32)
               local.get 0 i32.load
               local.get 0 i32.load
               call 0
               local.get 0
               i32.add))"#
    ));

    assert_eq!(base, 0);
}

#[test]
fn a_zero_param_call_still_reports_its_result_base() {
    const CALLEE: &str = "(func (result i32) i32.const 1)";

    let (base, reader) = caller_base_of(&format!(
        r#"(module {CALLEE}
             (func (result i32) i32.const 100 call 0 i32.add))"#
    ));

    assert_eq!(
        base, 0,
        "nothing is popped, so the base is the current index"
    );
    assert!(reader.contains(&"r0".to_string()), "{reader:?}");

    let (base, reader) = caller_base_of(&format!(
        r#"(module (memory 1) {CALLEE}
             (func (param i32) (result i32)
               local.get 0 i32.load call 0 i32.add))"#
    ));

    assert_eq!(base, 1);
    assert!(reader.contains(&"r1".to_string()), "{reader:?}");
}

#[test]
fn arguments_are_staged_at_the_caller_base() {
    let (prog, frame) = lower_func(
        r#"(module
             (func (param i32 i32) (result i32) local.get 0)
             (func (param i32) (result i32)
               local.get 0 i32.const 7 call 0))"#,
        1,
    );

    let at = index_of_kind(&prog, RegInstructionKind::Call).unwrap();

    let RegInstruction::Call { caller_base, .. } = &prog[at] else {
        unreachable!()
    };

    let RegInstruction::Move(sig) = &prog[at - 1] else {
        panic!("the arguments must be staged immediately before the call")
    };

    assert_eq!(
        sig.output_registers(&frame.output_registers_arena),
        &[*caller_base, caller_base + 1],
        "arguments land at [caller_base, caller_base + params)"
    );
}

/// A callee may write any global, so a borrow that outlives the call has to be
/// rescued first. Locals are safe — the callee has its own frame.
#[test]
fn a_global_read_across_a_call_is_rescued() {
    assert_func_lowers_to(
        r#"(module
             (global (mut i32) (i32.const 0))
             (func)
             (func (result i32)
               global.get 0
               call 0
               global.get 0
               i32.add))"#,
        1,
        "
          0  global.spill global0 -> spill0
          1  call         f0 caller_base=0
          2  i32.add      spill0, global0 -> r0
          3  move         r0 -> r0
          4  end
             frame: 1 registers, 1 spills
        ",
    );
}

#[test]
fn a_local_read_across_a_call_needs_no_rescue() {
    let (prog, _) = lower_func(
        r#"(module
             (func)
             (func (param i32) (result i32)
               local.get 0
               call 0
               local.get 0
               i32.add))"#,
        1,
    );

    assert!(
        index_of_kind(&prog, RegInstructionKind::LocalSpill).is_none(),
        "a callee cannot write the caller's locals"
    );
}

// ---------------------------------------------------------------------------
// call_indirect
//
// Everything `call` has to get right, plus one operand: the callee index sits
// *above* the arguments and is popped with them. That extra pop is what both
// heights have to account for — the slot height the call unwinds to, and the
// register index the callee's frame is based at.
//
// The instruction stores neither run explicitly. The arguments are the `params`
// operands starting at `operands`, `params` coming from the signature `ty_index`
// names; their destinations are the same many registers based at `caller_base`.
// The tests below read them back that way on purpose: reconstructing the two the
// way an executor must is what would show them disagreeing.
// ---------------------------------------------------------------------------

/// A signature to call through, plus the table to resolve against. Type 0 is the
/// call site's, so the tested function's own type lands at index 1.
fn indirect_module(ty: &str, body: &str) -> String {
    format!("(module (memory 1) (type (func {ty})) (table 4 funcref) {body})")
}

#[test]
fn call_indirect_stages_its_arguments_at_the_caller_base() {
    assert_lowers_to(
        &indirect_module(
            "(param i32 i32 i32) (result i32)",
            r#"(func (param i32) (result i32)
                 local.get 0 i32.load
                 local.get 0
                 i32.const 9
                 local.get 0 i32.load
                 call_indirect (type 0))"#,
        ),
        "
          0  i32.load     [local0]+0 -> r0
          1  i32.load     [local0]+0 -> r1
          2  call_indirect [r1] ty0 table0 caller_base=0  move r0, local0, 9 -> r0, r1, r2
          3  move         r0 -> r0
          4  end
             frame: 3 registers, 0 spills
        ",
    );
}

/// The index is the last thing pushed, so it is popped first and the arguments
/// are staged below it — a call that forgot it would leave one argument on the
/// simulated stack and shift every height after the call.
#[test]
fn the_callee_index_is_popped_with_the_arguments() {
    assert_lowers_to(
        &indirect_module(
            "(param i32) (result i32)",
            r#"(func (param i32) (result i32)
                 local.get 0 i32.load
                 local.get 0 i32.load
                 i32.const 3
                 call_indirect (type 0)
                 i32.add)"#,
        ),
        "
          0  i32.load     [local0]+0 -> r0
          1  i32.load     [local0]+0 -> r1
          2  call_indirect [3] ty0 table0 caller_base=1  move r1 -> r1
          3  i32.add      r0, r1 -> r0
          4  move         r0 -> r0
          5  end
             frame: 2 registers, 0 spills
        ",
    );
}

/// With no arguments the index is still popped, so the unwind is one slot deep
/// and not zero — the arithmetic must not run through `params - 1` on the way.
#[test]
fn a_zero_argument_call_indirect_still_pops_its_index() {
    assert_lowers_to(
        &indirect_module(
            "(result i32)",
            r#"(func (result i32)
                 i32.const 3
                 call_indirect (type 0))"#,
        ),
        "
          0  call_indirect [3] ty0 table0 caller_base=0
          1  move         r0 -> r0
          2  end
             frame: 1 registers, 0 spills
        ",
    );
}

/// `caller_base` is what sits below the *arguments*, so the window it counts over
/// spans the index as well as them. Counting one short reads the deepest argument
/// as if it were below the call, which shows up only when that argument holds a
/// register — the case both of these pin.
#[test]
fn caller_base_counts_past_the_index_to_the_deepest_argument() {
    let base_of = |wat: &str| {
        let (prog, _) = lower_func(wat, 0);
        let at = index_of_kind(&prog, RegInstructionKind::CallIndirect).unwrap();

        let RegInstruction::CallIndirect { caller_base, .. } = &prog[at] else {
            unreachable!()
        };

        *caller_base
    };

    // the only register is the argument itself: it sits *at* the base
    assert_eq!(
        base_of(&indirect_module(
            "(param i32) (result i32)",
            r#"(func (param i32) (result i32)
                 local.get 0 i32.load
                 i32.const 3
                 call_indirect (type 0))"#,
        )),
        0,
        "a register argument does not raise the base"
    );

    // one register below the call, one holding the argument
    assert_eq!(
        base_of(&indirect_module(
            "(param i32) (result i32)",
            r#"(func (param i32) (result i32)
                 local.get 0 i32.load
                 local.get 0 i32.load
                 i32.const 3
                 call_indirect (type 0)
                 i32.add)"#,
        )),
        1,
        "only the value below the arguments counts"
    );
}

/// Why the argument move is a field of the instruction rather than a `Move` ahead
/// of it: the staged arguments land at `[caller_base, caller_base + params)`, and
/// the callee index may be read from a register inside that range. Here `7` is
/// staged into `r0` while the index is read from `r0`.
///
/// An executor must therefore read the index *before* performing the move — the
/// same contract [`RegInstruction::BrIf`] has with its condition.
#[test]
fn staging_the_arguments_may_clobber_the_callee_index_register() {
    assert_lowers_to(
        &indirect_module(
            "(param i32) (result i32)",
            r#"(func (param i32) (result i32)
                 i32.const 7
                 local.get 0 i32.load
                 call_indirect (type 0))"#,
        ),
        "
          0  i32.load     [local0]+0 -> r0
          1  call_indirect [r0] ty0 table0 caller_base=0  move 7 -> r0
          2  move         r0 -> r0
          3  end
             frame: 1 registers, 0 spills
        ",
    );
}

/// Results come back to the caller base, however many there are.
#[test]
fn results_come_back_at_the_caller_base() {
    assert_lowers_to(
        &indirect_module(
            "(param i32) (result i32 i32)",
            r#"(func (param i32) (result i32)
                 local.get 0
                 i32.const 3
                 call_indirect (type 0)
                 i32.add)"#,
        ),
        "
          0  call_indirect [3] ty0 table0 caller_base=0  move local0 -> r0
          1  i32.add      r0, r1 -> r0
          2  move         r0 -> r0
          3  end
             frame: 2 registers, 0 spills
        ",
    );
}

/// The callee is not known until execution, so it may write any global: a borrow
/// that outlives the call is rescued exactly as it is for a direct one.
#[test]
fn a_global_read_across_a_call_indirect_is_rescued() {
    assert_lowers_to(
        &format!(
            "(module (global (mut i32) (i32.const 0)) (type (func (result i32))) (table 4 funcref)
               {})",
            r#"(func (result i32)
                 global.get 0
                 i32.const 3
                 call_indirect (type 0)
                 i32.add)"#
        ),
        "
          0  global.spill global0 -> spill0
          1  call_indirect [3] ty0 table0 caller_base=0
          2  i32.add      spill0, r0 -> r0
          3  move         r0 -> r0
          4  end
             frame: 1 registers, 1 spills
        ",
    );
}

/// A height that is off by one inside a block survives until the block's `end`
/// asserts on it, so a balanced block is the cheap end-to-end check.
#[test]
fn a_call_indirect_leaves_its_block_balanced() {
    assert_lowers_to(
        &indirect_module(
            "(param i32) (result i32)",
            r#"(func (param i32) (result i32)
                 block (result i32)
                   local.get 0
                   i32.const 3
                   call_indirect (type 0)
                 end)"#,
        ),
        "
          0  call_indirect [3] ty0 table0 caller_base=0  move local0 -> r0
          1  move         r0 -> r0
          2  end
          3  move         r0 -> r0
          4  end
             frame: 1 registers, 0 spills
        ",
    );
}

// ---------------------------------------------------------------------------
// unreachable
//
// It transfers control without naming a label, so it is lowered like `br` in
// every respect but the branch itself: no move, no attached break, no target —
// just the trap, the enclosing block reset to the layout its `end` expects, and
// everything up to that `end` marked dead.
//
// The reset is what these tests are about. The block's `end` still emits its own
// materialisation and still asserts on the height it arrives at, so an operand
// left behind here surfaces there rather than at the trap.
// ---------------------------------------------------------------------------

#[test]
fn an_unreachable_leaves_its_block_in_the_layout_its_end_expects() {
    assert_lowers_to(
        r#"(module (func (result i32)
             block (result i32)
               unreachable
             end))"#,
        "
          0  unreachable
          1  move         r0 -> r0
          2  end
          3  move         r0 -> r0
          4  end
             frame: 1 registers, 0 spills
        ",
    );
}

/// Operands the trap never consumes are discarded by the reset, registers and
/// stack-only slots alike — the block's `end` must see its own entry height plus
/// its results, not whatever was stranded above it.
#[test]
fn operands_live_at_an_unreachable_are_discarded() {
    // two constants, which occupy stack positions and no registers
    assert_lowers_to(
        r#"(module (func (result i32)
             block (result i32)
               i32.const 1
               i32.const 2
               unreachable
             end))"#,
        "
          0  unreachable
          1  move         r0 -> r0
          2  end
          3  move         r0 -> r0
          4  end
             frame: 1 registers, 0 spills
        ",
    );

    // and a register, which the reset has to give back before pushing the result
    assert_lowers_to(
        r#"(module (memory 1) (func (result i32)
             block (result i32)
               i32.const 0 i32.load
               unreachable
             end))"#,
        "
          0  i32.load     [0]+0 -> r0
          1  unreachable
          2  move         r0 -> r0
          3  end
          4  move         r0 -> r0
          5  end
             frame: 1 registers, 0 spills
        ",
    );
}

#[test]
fn code_after_an_unreachable_is_dropped() {
    let body = lower(
        r#"(module (func (result i32)
             block (result i32)
               unreachable
               i32.const 7
               i32.const 8
               i32.add
             end))"#,
    );

    assert!(
        index_of(&body.0, |i| i.kind() == RegInstructionKind::I32Add).is_none(),
        "the operators after the trap cannot execute:\n{}",
        render(&body, &[]),
    );
}

/// Only the arm that traps is dead. `else` closes a construct opened while
/// reachable, so lowering resumes there.
#[test]
fn an_unreachable_arm_does_not_kill_the_other_arm() {
    assert_lowers_to(
        r#"(module (func (param i32) (result i32)
             local.get 0
             if (result i32)
               unreachable
             else
               i32.const 1
             end))"#,
        "
          0  local.spill  local0 -> spill0
          1  if           spill0 else=4 end=6
          2  unreachable
          3  move         r0 -> r0
          4  else         end=6
          5  move         1 -> r0
          6  end
          7  move         r0 -> r0
          8  end
             frame: 1 registers, 1 spills
        ",
    );
}

/// The reset stops at the trapping block's own entry height: an operand the
/// *enclosing* block owns is still there afterwards, and still in its register.
#[test]
fn an_unreachable_block_does_not_disturb_the_enclosing_block() {
    assert_lowers_to(
        r#"(module (memory 1) (func (result i32)
             i32.const 0 i32.load
             block
               unreachable
             end
             drop
             i32.const 1))"#,
        "
          0  i32.load     [0]+0 -> r0
          1  unreachable
          2  end
          3  move         1 -> r0
          4  end
             frame: 1 registers, 0 spills
        ",
    );
}

/// The reset pops through the same path everything else does, so a borrow it
/// discards releases its spill slot — and one the enclosing block still holds
/// keeps it.
#[test]
fn a_spill_live_across_an_unreachable_survives_it() {
    assert_lowers_to(
        r#"(module (global (mut i32) (i32.const 0)) (func (result i32)
             global.get 0
             i32.const 5
             global.set 0
             block
               unreachable
             end
             i32.const 1
             i32.add))"#,
        "
          0  global.spill global0 -> spill0
          1  global.set   global0 <- 5
          2  unreachable
          3  end
          4  i32.add      spill0, 1 -> r0
          5  move         r0 -> r0
          6  end
             frame: 1 registers, 1 spills
        ",
    );
}

/// Results the trap can never produce are still allocated, because the `end`
/// below is lowered as if the block could fall into it. The registers are never
/// written at execution — the frame is one wider than it strictly needs, which is
/// the cheap side of the trade.
#[test]
fn a_trapping_block_still_reserves_its_results() {
    let (_, frame) = lower(
        r#"(module (func (result i32)
             block (result i32 i32)
               unreachable
             end
             drop))"#,
    );

    assert_eq!(frame.registers, 2, "both results are given registers");
}

// ---------------------------------------------------------------------------
// arity cross-check
//
// An operator lowered through `emit!` names only its variant, so the arity comes
// from the variant's own `Signature<I, O>` declaration and no arm restates it.
// That removes the one thing that used to disagree when a declaration was wrong:
// declaring `RefIsNull(Signature<2, 1>)` compiles with no error and no warning,
// and pops an operand that isn't there.
//
// So the declaration needs an outside opinion, and the only one that is neither
// this pass nor a number someone typed is wasmparser's validator: it type-checks
// the body against the spec, and `operand_stack_height` reads off what the spec
// says each operator did. Comparing that against what the pass recorded is what
// makes a wrong `Signature` a test failure instead of a silent miscompile.
//
// The `.wat` is self-checking too. Push one operand too few or too many and the
// module fails to validate — as a leftover value at the end of a body with no
// results, or as a missing one — so a case cannot quietly encode the wrong arity
// either.
// ---------------------------------------------------------------------------

/// The wasm mnemonic for a value-op kind: `I32TruncSatF32U` → `i32.trunc_sat_f32_u`.
///
/// Derived rather than tabulated, because 136 hand-written strings is 136 chances
/// to write `i32.extend_8_s` for `i32.extend8_s`. The rule is one line: start a
/// word at each capital, so digits stay attached to the word they follow, and the
/// first word is the type prefix.
///
/// It is not taken on trust — [`arity_case`] builds each operator's `.wat` from
/// this, and that `.wat` has to assemble, validate, and contain the operator the
/// kind is named after. A wrong mnemonic fails those rather than quietly
/// rendering an instruction under a name that does not exist.
fn mnemonic(kind: RegInstructionKind) -> String {
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

/// The type a value op's operands have, which its `.wat` case has to declare.
///
/// It is the source type the name carries when there is one — `i64.extend_i32_s`
/// takes an `i32`, `i32.trunc_sat_f32_u` an `f32` — and the type prefix otherwise.
/// `i32.extend8_s` has no source type: the `8` is a width, not an operand, which
/// is exactly the distinction the mnemonic's underscores encode.
fn operand_type(kind: RegInstructionKind) -> String {
    let mnemonic = mnemonic(kind);
    let (prefix, tail) = mnemonic.split_once('.').expect("a mnemonic has a prefix");

    tail.split('_')
        .find(|word| matches!(*word, "i32" | "i64" | "f32" | "f64"))
        .unwrap_or(prefix)
        .to_string()
}

/// A `.wat` applying one value op to `operands` operands of its own operand type,
/// dropping the result.
fn value_op_case(kind: RegInstructionKind, operands: usize) -> String {
    let gets = "local.get 0 ".repeat(operands);

    format!(
        "(module (func (param {}) {gets}{} drop))",
        operand_type(kind),
        mnemonic(kind)
    )
}

/// The `.wat` that pins one kind's arity, or `None` for a kind `emit!` does not
/// lower.
///
/// **Exhaustive on purpose, and visited in full.** [`RegInstructionKind`] is
/// derived from [`RegInstruction`], so adding a variant does two things at once:
/// this `match` stops compiling until the new kind is handled, and
/// [`RegInstructionKind::ALL`] grows to include it, so whatever this says about it
/// actually runs. A hand-written list of cases could only ever promise the first.
///
/// That matters more here than for most tables: with `emit!` the variant's own
/// `Signature<I, O>` is the only place its arity is written down, so an operator
/// that slipped past this would have nothing at all checking it.
///
/// A case applies the operator to exactly the operands it takes and drops the
/// result, so the body lowers to that one instruction and nothing else — no
/// trailing `Move`, since these bodies return nothing.
fn arity_case(kind: RegInstructionKind) -> Option<String> {
    match kind {
        // The pure value ops, whose case is derived from the kind: the mnemonic
        // and the operand type are both recoverable from the name, so 136
        // hand-written bodies would be 136 chances to mistype one.
        // i32 - unary
        RegInstructionKind::I32Clz
        | RegInstructionKind::I32Ctz
        | RegInstructionKind::I32Eqz
        | RegInstructionKind::I32Extend16S
        | RegInstructionKind::I32Extend8S
        | RegInstructionKind::I32Popcnt
        | RegInstructionKind::I32ReinterpretF32
        | RegInstructionKind::I32TruncF32S
        | RegInstructionKind::I32TruncF32U
        | RegInstructionKind::I32TruncF64S
        | RegInstructionKind::I32TruncF64U
        | RegInstructionKind::I32TruncSatF32S
        | RegInstructionKind::I32TruncSatF32U
        | RegInstructionKind::I32TruncSatF64S
        | RegInstructionKind::I32TruncSatF64U
        | RegInstructionKind::I32WrapI64 => Some(value_op_case(kind, 1)),

        // i32 - binary
        RegInstructionKind::I32Add
        | RegInstructionKind::I32And
        | RegInstructionKind::I32DivS
        | RegInstructionKind::I32DivU
        | RegInstructionKind::I32Eq
        | RegInstructionKind::I32GeS
        | RegInstructionKind::I32GeU
        | RegInstructionKind::I32GtS
        | RegInstructionKind::I32GtU
        | RegInstructionKind::I32LeS
        | RegInstructionKind::I32LeU
        | RegInstructionKind::I32LtS
        | RegInstructionKind::I32LtU
        | RegInstructionKind::I32Mul
        | RegInstructionKind::I32Ne
        | RegInstructionKind::I32Or
        | RegInstructionKind::I32RemS
        | RegInstructionKind::I32RemU
        | RegInstructionKind::I32Rotl
        | RegInstructionKind::I32Rotr
        | RegInstructionKind::I32Shl
        | RegInstructionKind::I32ShrS
        | RegInstructionKind::I32ShrU
        | RegInstructionKind::I32Sub
        | RegInstructionKind::I32Xor => Some(value_op_case(kind, 2)),

        // i64 - unary
        RegInstructionKind::I64Clz
        | RegInstructionKind::I64Ctz
        | RegInstructionKind::I64Eqz
        | RegInstructionKind::I64Extend16S
        | RegInstructionKind::I64Extend32S
        | RegInstructionKind::I64Extend8S
        | RegInstructionKind::I64ExtendI32S
        | RegInstructionKind::I64ExtendI32U
        | RegInstructionKind::I64Popcnt
        | RegInstructionKind::I64ReinterpretF64
        | RegInstructionKind::I64TruncF32S
        | RegInstructionKind::I64TruncF32U
        | RegInstructionKind::I64TruncF64S
        | RegInstructionKind::I64TruncF64U
        | RegInstructionKind::I64TruncSatF32S
        | RegInstructionKind::I64TruncSatF32U
        | RegInstructionKind::I64TruncSatF64S
        | RegInstructionKind::I64TruncSatF64U => Some(value_op_case(kind, 1)),

        // i64 - binary
        RegInstructionKind::I64Add
        | RegInstructionKind::I64And
        | RegInstructionKind::I64DivS
        | RegInstructionKind::I64DivU
        | RegInstructionKind::I64Eq
        | RegInstructionKind::I64GeS
        | RegInstructionKind::I64GeU
        | RegInstructionKind::I64GtS
        | RegInstructionKind::I64GtU
        | RegInstructionKind::I64LeS
        | RegInstructionKind::I64LeU
        | RegInstructionKind::I64LtS
        | RegInstructionKind::I64LtU
        | RegInstructionKind::I64Mul
        | RegInstructionKind::I64Ne
        | RegInstructionKind::I64Or
        | RegInstructionKind::I64RemS
        | RegInstructionKind::I64RemU
        | RegInstructionKind::I64Rotl
        | RegInstructionKind::I64Rotr
        | RegInstructionKind::I64Shl
        | RegInstructionKind::I64ShrS
        | RegInstructionKind::I64ShrU
        | RegInstructionKind::I64Sub
        | RegInstructionKind::I64Xor => Some(value_op_case(kind, 2)),

        // f32 - unary
        RegInstructionKind::F32Abs
        | RegInstructionKind::F32Ceil
        | RegInstructionKind::F32ConvertI32S
        | RegInstructionKind::F32ConvertI32U
        | RegInstructionKind::F32ConvertI64S
        | RegInstructionKind::F32ConvertI64U
        | RegInstructionKind::F32DemoteF64
        | RegInstructionKind::F32Floor
        | RegInstructionKind::F32Nearest
        | RegInstructionKind::F32Neg
        | RegInstructionKind::F32ReinterpretI32
        | RegInstructionKind::F32Sqrt
        | RegInstructionKind::F32Trunc => Some(value_op_case(kind, 1)),

        // f32 - binary
        RegInstructionKind::F32Add
        | RegInstructionKind::F32Copysign
        | RegInstructionKind::F32Div
        | RegInstructionKind::F32Eq
        | RegInstructionKind::F32Ge
        | RegInstructionKind::F32Gt
        | RegInstructionKind::F32Le
        | RegInstructionKind::F32Lt
        | RegInstructionKind::F32Max
        | RegInstructionKind::F32Min
        | RegInstructionKind::F32Mul
        | RegInstructionKind::F32Ne
        | RegInstructionKind::F32Sub => Some(value_op_case(kind, 2)),

        // f64 - unary
        RegInstructionKind::F64Abs
        | RegInstructionKind::F64Ceil
        | RegInstructionKind::F64ConvertI32S
        | RegInstructionKind::F64ConvertI32U
        | RegInstructionKind::F64ConvertI64S
        | RegInstructionKind::F64ConvertI64U
        | RegInstructionKind::F64Floor
        | RegInstructionKind::F64Nearest
        | RegInstructionKind::F64Neg
        | RegInstructionKind::F64PromoteF32
        | RegInstructionKind::F64ReinterpretI64
        | RegInstructionKind::F64Sqrt
        | RegInstructionKind::F64Trunc => Some(value_op_case(kind, 1)),

        // f64 - binary
        RegInstructionKind::F64Add
        | RegInstructionKind::F64Copysign
        | RegInstructionKind::F64Div
        | RegInstructionKind::F64Eq
        | RegInstructionKind::F64Ge
        | RegInstructionKind::F64Gt
        | RegInstructionKind::F64Le
        | RegInstructionKind::F64Lt
        | RegInstructionKind::F64Max
        | RegInstructionKind::F64Min
        | RegInstructionKind::F64Mul
        | RegInstructionKind::F64Ne
        | RegInstructionKind::F64Sub => Some(value_op_case(kind, 2)),

        // The rest carry something the derivation cannot supply: `select` takes
        // three operands rather than one or two, `ref.is_null` is not named
        // after a numeric type, and the memory ops need a memory to address.
        RegInstructionKind::Select => Some(
            "(module (func (param i32) local.get 0 local.get 0 local.get 0 select drop))".into(),
        ),
        RegInstructionKind::RefIsNull => {
            Some("(module (func (param funcref) local.get 0 ref.is_null drop))".into())
        }
        RegInstructionKind::I32Load => {
            Some("(module (memory 1) (func (param i32) local.get 0 i32.load drop))".into())
        }
        RegInstructionKind::I32Store => {
            Some("(module (memory 1) (func (param i32) local.get 0 local.get 0 i32.store))".into())
        }

        RegInstructionKind::LocalSet
        | RegInstructionKind::LocalTee
        | RegInstructionKind::GlobalSet
        | RegInstructionKind::LocalSpill
        | RegInstructionKind::GlobalSpill
        | RegInstructionKind::If
        | RegInstructionKind::Else
        | RegInstructionKind::Br
        | RegInstructionKind::BrIf
        | RegInstructionKind::BrTable
        | RegInstructionKind::Return
        | RegInstructionKind::Call
        | RegInstructionKind::CallIndirect
        | RegInstructionKind::Unreachable
        | RegInstructionKind::Move
        | RegInstructionKind::End => None,
    }
}

/// The net stack effect the *validator* attributes to each operator of a module's
/// first function, keyed by variant name.
///
/// This is the oracle. It is read from the validator that type-checked the body,
/// so it is the spec's arity rather than anything this crate believes.
fn validator_stack_deltas(bytes: &[u8]) -> Vec<(String, i64)> {
    let mut validator = wasmparser::Validator::new();
    let mut deltas = vec![];

    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload.expect("parse");

        if let wasmparser::ValidPayload::Func(to_validate, body) =
            validator.payload(&payload).expect("validate")
        {
            let mut func = to_validate.into_validator(Default::default());

            // the validator needs the body's local declarations before its
            // operators; the operators reader starts past them
            func.read_locals(&mut body.get_binary_reader())
                .expect("locals");

            let mut reader = body.get_operators_reader().expect("operators");

            while !reader.eof() {
                let (operator, offset) = reader.read_with_offset().expect("operator");
                let before = func.operand_stack_height() as i64;

                func.op(offset, &operator).expect("operator validates");

                let after = func.operand_stack_height() as i64;
                // `Debug` renders as `I32Add` or `I32Load { memarg: .. }`; the
                // variant name is everything up to the first space.
                let name = format!("{operator:?}");
                let name = name.split_whitespace().next().unwrap_or_default();

                deltas.push((name.to_string(), after - before));
            }

            // only the first function is measured
            break;
        }
    }

    deltas
}

#[test]
fn every_operator_pops_and_pushes_what_the_spec_says() {
    for &kind in RegInstructionKind::ALL {
        let Some(wat) = arity_case(kind) else {
            continue;
        };

        let bytes = wat::parse_str(&wat).expect("invalid wat");

        // the kind is named after the operator it lowers, which is what lets the
        // operator under test be picked out of the `local.get`s feeding it
        let operator = format!("{kind:?}");

        let expected = validator_stack_deltas(&bytes)
            .into_iter()
            .find(|(name, _)| *name == operator)
            .unwrap_or_else(|| panic!("{operator} does not occur in its own case:\n{wat}"))
            .1;

        let body = lower(&wat);
        let (prog, frame) = (&body.0, &body.1);

        // the case is built so the body is exactly this instruction and `end`,
        // which is what lets the arenas be read as that instruction's operands
        assert_eq!(
            prog.len(),
            2,
            "{operator}: case must lower to one instruction and `end`:\n{}",
            render(&body, &[])
        );

        let pops = frame.input_registers_arena.len() as i64;
        let pushes = frame.output_registers_arena.len() as i64;

        assert_eq!(
            pushes - pops,
            expected,
            "{operator}: lowered as {pops} -> {pushes}, but the spec says the net \
             stack effect is {expected}. The `Signature<I, O>` on the variant is \
             wrong — nothing else records an arity."
        );
    }
}

/// A case filed under the wrong kind would check some other operator's arity and
/// pass, so each case must lower to the kind it is filed under.
#[test]
fn every_case_lowers_to_the_kind_it_is_filed_under() {
    for &kind in RegInstructionKind::ALL {
        let Some(wat) = arity_case(kind) else {
            continue;
        };

        let (prog, _) = lower(&wat);

        assert_eq!(
            prog[0].kind(),
            kind,
            "the case filed under {kind:?} lowers to {:?}:\n{wat}",
            prog[0].kind()
        );

        assert!(
            matches!(prog.last(), Some(RegInstruction::End)),
            "{kind:?}: the body must end with `end`"
        );
    }
}

// ---------------------------------------------------------------------------
// frame layout
// ---------------------------------------------------------------------------

#[test]
fn the_frame_reports_peak_register_use() {
    let (_, frame) = lower(
        r#"(module (memory 1) (func (param i32) (result i32)
             local.get 0 i32.load
             local.get 0 i32.load
             i32.add))"#,
    );

    assert_eq!(frame.registers, 2, "two loads are live at once");
}

#[test]
fn the_frame_reports_peak_spill_use() {
    let (_, frame) = lower(
        r#"(module (func (param i32) (result i32)
             local.get 0 i32.const 1 local.set 0
             local.get 0 i32.const 2 local.set 0
             i32.add))"#,
    );

    assert_eq!(
        frame.spills, 2,
        "two snapshots of local 0 are live at the same time"
    );
}

#[test]
fn the_arenas_ship_with_the_body() {
    let (prog, frame) = lower(
        r#"(module (func (param i32)
             block block local.get 0 br_table 0 1 0 end end))"#,
    );

    assert!(!frame.input_registers_arena.is_empty(), "operands");
    assert!(!frame.br_targets_arena.is_empty(), "br_table arms");

    // and every index in the program resolves against them
    for instruction in &prog {
        if let RegInstruction::BrTable {
            targets_start,
            targets_len,
            ..
        } = instruction
        {
            assert!(
                (targets_start + targets_len) as usize <= frame.br_targets_arena.len(),
                "arm range runs past the arena"
            );
        }
    }
}
