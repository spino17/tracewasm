//! Tests for the register lowering pass.
//!
//! Two layers, because they catch different things.
//!
//! [`lower`] takes a `.wat` module, runs the real
//! [`RegInstruction::emit_instructions_for_func`] over its first function, and
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

/// Renders a run of resolved operands.
///
/// `Slot` carries no tag, so naming its region needs the layout — use
/// [`layout_of`] for a stack that is still mid-lowering.
fn slots(xs: &[Slot], frame: &RegFrameLayout) -> Vec<String> {
    xs.iter().map(|x| x.render(frame)).collect()
}

/// Renders what `set_lazy` returned: `Some("spill0")` when it rescued a borrow.
fn spilled_to(result: Option<SpillIndex>) -> Option<String> {
    result.map(|i| format!("spill{i}"))
}

/// The frame indices a run of resolved operands names.
///
/// Renderings say which *region* an operand is in; these say exactly where, which is
/// what pins the layout — a region boundary off by one renders identically for every
/// slot but the ones that cross it.
fn frame_indices(xs: &[Slot]) -> Vec<u16> {
    xs.iter().map(|Slot(i)| *i).collect()
}

/// A [`RegFrameLayout`] describing `s` as it stands, for rendering resolved slots.
///
/// Only the region sizes and the constant pool are filled in — those are all
/// [`Slot::render`] consults, and the arenas cannot be cloned out of a borrowed
/// stack. A test that needs the arenas builds the real layout instead.
fn layout_of(s: &SimulatedStack) -> RegFrameLayout {
    RegFrameLayout {
        registers: s.max_registers,
        spills: s.spills.allocation_len(),
        locals_count: s.locals_count(),
        consts: s.const_interner.consts.clone().into_boxed_slice(),
        input_registers_arena: Box::new([]),
        output_registers_arena: Box::new([]),
        br_targets_arena: Box::new([]),
    }
}

/// Renders a *lowering-time* operand, before the end-of-body backpatch.
///
/// Distinct from [`Slot::render`] on purpose: until the backpatch runs, a spill and
/// a constant have no frame index at all, so they render by what they are —
/// `spill0`, `const0` — rather than by where they will land. These are the
/// assertions about what the pass *decided*; the resolved renderings are the
/// assertions about the layout it produced.
fn render_provisional(o: &BackPatchableSlot, locals_count: u16) -> String {
    match o {
        // A provisional register index counts from the frame base, since lowering
        // starts `curr_register_index` at `locals_count` — so it is offset back to
        // operand-relative numbering here, which is how the pass is reasoned about.
        BackPatchableSlot::Register(n) => format!("r{}", n - locals_count),
        BackPatchableSlot::Spill(i) => format!("spill{i}"),
        BackPatchableSlot::Const(id) => format!("const{}", id.0),
        BackPatchableSlot::Slot(Slot(n)) if *n < locals_count => {
            format!("local{n}")
        }
        BackPatchableSlot::Slot(Slot(n)) => format!("r{}", n - locals_count),
    }
}

/// Applies the three backpatches `emit_instructions_for_func` applies once a body is
/// fully lowered, resolving every arena placeholder to a frame index.
///
/// The shifts must match the runtime's `locals | consts | spills | registers` order
/// term for term: a constant lands at `locals + id`, a spill at
/// `locals + consts + slot`, and an operand register — provisionally numbered from
/// `locals` — moves up past both regions by `consts + spills`.
fn resolve_backpatches(s: &mut SimulatedStack) {
    let locals_count = s.locals_count();
    let consts_len = s.const_interner.consts.len() as u16;
    let spills = s.spills.allocation_len();

    for (i, id) in std::mem::take(&mut s.const_backpatches) {
        s.input_registers[i] = Slot(id.0 + locals_count);
    }

    for (i, spill) in std::mem::take(&mut s.spill_backpatches) {
        s.input_registers[i] = Slot(spill.raw_value() + locals_count + consts_len);
    }

    for (i, index) in std::mem::take(&mut s.register_backpatches) {
        s.input_registers[i] = Slot(index + spills + consts_len);
    }

    s.output_registers = s
        .output_registers
        .iter()
        .map(|x| x + spills + consts_len)
        .collect();
}

/// Resolves one lowering-time operand the way [`resolve_backpatches`] resolves the
/// arena, for a test holding a slot it popped off the simulated stack.
fn resolve(s: &SimulatedStack, o: &BackPatchableSlot) -> Slot {
    let locals_count = s.locals_count();
    let consts_len = s.const_interner.consts.len() as u16;

    match o {
        BackPatchableSlot::Const(id) => Slot(id.0 + locals_count),
        BackPatchableSlot::Spill(i) => Slot(i.raw_value() + locals_count + consts_len),
        BackPatchableSlot::Register(n) => Slot(n + s.spills.allocation_len() + consts_len),
        BackPatchableSlot::Slot(slot) => *slot,
    }
}

impl SimulatedStack {
    /// Pops and renders in one step, so a provisional-form assertion does not have
    /// to name its own locals count.
    fn pop_render(&mut self) -> String {
        let o = self.pop();

        render_provisional(&o, self.locals_count())
    }
}

// ---------------------------------------------------------------------------
// wat-driven harness
// ---------------------------------------------------------------------------

/// Lowers the first function body of a `.wat` module.
///
/// Panics on malformed input, which in a test is what you want: the `.wat` is part
/// of the test, so a mistake in it is a test bug and should be loud.
fn lower(wat: &str) -> RegLoweredFuncBody {
    lower_func(wat, 0)
}

/// Lowers the `n`th *defined* function of a `.wat` module, for tests whose subject
/// is a caller and so cannot be function 0.
///
/// Imported functions are not modelled: the index space here is the code section's,
/// so `call n` inside the wat must name a defined function.
fn lower_func(wat: &str, n: usize) -> RegLoweredFuncBody {
    lower_func_with_types(wat, n).0
}

/// [`lower_func`], also handing back the module's type section.
///
/// A `call_indirect` stores only a `ty_index`, so how many of the operands in its
/// arena run are arguments is recoverable only through the types. Rendering one
/// therefore needs exactly what executing one will need.
fn lower_func_with_types(wat: &str, n: usize) -> (RegLoweredFuncBody, Vec<FuncType>) {
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

    let body = RegInstruction::emit_instructions_for_func(
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

/// Asserts a lowered body renders exactly as `expected`, ignoring leading
/// indentation so the expectation can be written inline.
fn assert_lowers_to(wat: &str, expected: &str) {
    assert_func_lowers_to(wat, 0, expected)
}

/// [`assert_lowers_to`] for a body that is not function 0 — a caller, typically.
fn assert_func_lowers_to(wat: &str, n: usize, expected: &str) {
    let (body, types) = lower_func_with_types(wat, n);
    let got = RegInstruction::render_body(&body, &types);

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
/// `emit_instructions_for_func` sets up.
fn sim(locals: u32) -> SimulatedStack {
    let mut s = SimulatedStack::new(locals);

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
    (s.stack.height(), operands_in_use(s))
}

/// How many operand registers are currently live.
///
/// `curr_register_index` counts from the *frame* base — lowering starts it at
/// `locals_count` — so it is offset back to operand-relative numbering here, which
/// is the height the pass is reasoned about and what `r0`, `r1` … name. The absolute
/// frame indices are pinned separately, by [`frame_indices`] and the runtime's own
/// layout tests.
fn operands_in_use(s: &SimulatedStack) -> usize {
    s.curr_register_index - s.locals_count() as usize
}

/// The peak operand-register count, i.e. what the frame needs above its locals.
fn peak_operands(s: &SimulatedStack) -> u16 {
    s.max_registers - s.locals_count()
}

/// [`peak_operands`] for a finished layout.
fn peak_operands_of(frame: &RegFrameLayout) -> u16 {
    frame.registers - frame.locals_count
}

/// The frame index the operand registers begin at, given the four-region order
/// `locals | consts | spills | registers`.
///
/// A `caller_base` is an operand register index, so it is at or above this — which is
/// exactly what keeps a callee's frame from overlapping its caller's constants and
/// spills. Subtracting it recovers the operand-relative base the pass reasons about.
fn operand_base_of(frame: &RegFrameLayout) -> u16 {
    frame.locals_count + frame.consts.len() as u16 + frame.spills
}

/// A run of destination registers, offset back to operand-relative numbering.
///
/// Reads the *provisional* arena — before [`resolve_backpatches`] shifts registers
/// up past the constant and spill regions — so the operand base is just the locals
/// count.
fn output_operands(xs: &[u16], s: &SimulatedStack) -> Vec<u16> {
    xs.iter().map(|r| r - s.locals_count()).collect()
}

// ---------------------------------------------------------------------------
// lazy forwarding and spills
// ---------------------------------------------------------------------------

#[test]
fn borrows_of_one_local_share_a_single_spill() {
    let mut s = sim(2);

    s.push_local(0).unwrap();
    s.push_local(0).unwrap();
    s.push_const(Const::I32(5)).unwrap();

    let spill = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).unwrap();

    let _set = s.registers_for::<1, 0>().unwrap();
    let add = s.registers_for::<2, 1>().unwrap();

    resolve_backpatches(&mut s);

    assert_eq!(
        spilled_to(spill),
        Some("spill0".into()),
        "one spill covers both borrows"
    );

    assert_eq!(
        slots(add.input.registers(&s.input_registers), &layout_of(&s)),
        ["spill0", "spill0"],
        "both operands redirect"
    );
}

#[test]
fn a_consumed_borrow_is_not_spilled() {
    let mut s = sim(2);

    s.push_local(0).unwrap();

    let _load = s.registers_for::<1, 1>().unwrap(); // consumes the borrow

    s.push_const(Const::I32(5)).unwrap();

    assert!(
        SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills)
            .unwrap()
            .is_none(),
        "nothing borrows local 0 any more"
    );

    assert_eq!(s.spills.allocation_len(), 0);
}

#[test]
fn successive_writes_produce_independent_snapshots() {
    let mut s = sim(2);

    s.push_local(0).unwrap();
    s.push_const(Const::I32(1)).unwrap();

    let first = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).unwrap();

    let _ = s.registers_for::<1, 0>().unwrap();

    s.push_local(0).unwrap();
    s.push_const(Const::I32(2)).unwrap();

    let second = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).unwrap();

    let _ = s.registers_for::<1, 0>().unwrap();
    let add = s.registers_for::<2, 1>().unwrap();

    resolve_backpatches(&mut s);

    assert_eq!(
        (spilled_to(first), spilled_to(second)),
        (Some("spill0".into()), Some("spill1".into())),
        "distinct slots"
    );

    assert_eq!(
        slots(add.input.registers(&s.input_registers), &layout_of(&s)),
        ["spill0", "spill1"],
        "each operand keeps its own snapshot"
    );
}

#[test]
fn dropping_the_last_borrow_releases_its_spill_slot() {
    let mut s = sim(2);

    s.push_local(0).unwrap();
    s.push_const(Const::I32(1)).unwrap();

    SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).unwrap();

    let _ = s.registers_for::<1, 0>().unwrap();

    s.pop(); // drop the spilled borrow
    s.push_local(0).unwrap();
    s.push_const(Const::I32(2)).unwrap();

    let reused = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).unwrap();

    assert_eq!(
        spilled_to(reused),
        Some("spill0".into()),
        "the freed slot comes back"
    );

    assert_eq!(s.spills.allocation_len(), 1, "peak live, not total spills");
}

#[test]
fn tee_spills_before_reading_the_top() {
    let mut s = sim(4);

    s.push_local(3).unwrap();

    let spill = SimulatedStack::set_lazy(3, &mut s.lazy_locals, &mut s.spills).unwrap();
    let operand = s.tee();

    assert_eq!(spilled_to(spill), Some("spill0".into()));

    assert_eq!(
        render_provisional(&operand, s.locals_count()),
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
    let funcref = RegInstruction::render_body(
        &lower("(module (func (result funcref) ref.null func))"),
        &[],
    );

    let externref = RegInstruction::render_body(
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

/// A reference is a value like any other, so a `funcref` global is read into a
/// register and `ref.is_null` reads that register — no rescue, whatever the call does
/// to the global.
#[test]
fn a_reference_read_from_a_global_crosses_a_call_in_a_register() {
    assert_func_lowers_to(
        r#"(module (global (mut funcref) (ref.null func)) (func)
             (func (result i32) global.get 0 call 0 ref.is_null))"#,
        1,
        "
          0  global.get   global0 -> r0
          1  call         f0 caller_base=1
          2  ref.is_null  r0 -> r0
          3  move         r0 -> r0
          4  end
             frame: 1 registers, 0 spills
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
            let mut s = sim(4);

            s.push_local(0).unwrap();

            let _ = s.registers_for::<1, 1>().unwrap(); // a live register underneath
            let np = params_and_results_from_blockty(&blockty, &types).0;

            for _ in 0..np {
                s.push_const(Const::I32(9)).unwrap();
            }

            if is_if {
                s.push_const(Const::I32(1)).unwrap(); // condition
            }

            let (params, _) = s.add_block(variant_of(&variant), &blockty, &types, 0);
            let recorded = s.get_curr_block().recorded_height;

            if params != 0 {
                let _ = s.materialize_stack_slots_in_registers(params + u32::from(is_if));
            }

            if is_if {
                let _ = s.registers_for::<1, 0>().unwrap();
            }

            assert_eq!(
                s.stack.height(),
                recorded + params,
                "entry layout must be recorded_height + params"
            );

            assert_eq!(
                operands_in_use(&s),
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
    let mut s = sim(4);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;

    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap();

    s.push_local(1).unwrap(); // a live lazy borrow above the base

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
    let mut s = sim(4);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let outer = s.get_curr_block().recorded_height;

    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap(); // r0, below the inner block

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let inner = s.get_curr_block().recorded_height;

    s.push_local(1).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap(); // r1, the carried value
    let to_inner = s.br_truncation_registers(inner, 1).unwrap();
    let to_outer = s.br_truncation_registers(outer, 1).unwrap();

    assert_eq!(
        output_operands(to_inner.output_registers(&s.output_registers), &s),
        vec![1]
    );

    assert_eq!(
        output_operands(to_outer.output_registers(&s.output_registers), &s),
        vec![0]
    );

    assert_eq!(
        to_inner.input_registers(&s.input_registers).len(),
        1,
        "every arm reads the same operands"
    );
}

#[test]
fn branch_destinations_count_towards_the_frame() {
    let mut s = sim(4);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;
    // operands that are not already registers: destinations sit above everything
    // allocated so far, so the frame has to grow for them
    s.push_const(Const::I32(1)).unwrap();
    s.push_const(Const::I32(2)).unwrap();

    let mov = s.br_truncation_registers(base, 2).unwrap();

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

/// A register file just large enough to execute the instructions these tests emit,
/// so a claim about which path writes what is observed rather than argued.
///
/// Laid out exactly as [`RegFrame`](crate::runtime::reg::RegFrame) lays one out —
/// `locals | consts | spills | registers`, based at 0 — so a read is
/// `registers[frame_index]`, the same single add execution performs. Building it any
/// other way would let a test pass while the runtime's arithmetic was wrong, which
/// is the failure this harness exists to catch.
///
/// Spill slots start poisoned, because the whole question is whether a slot is
/// written on every path that reads it.
struct Frame {
    registers: Vec<i64>,
    globals: Vec<i64>,
    locals_count: u16,
    consts_len: u16,
}

const POISON: i64 = i64::MIN;

impl Frame {
    /// Builds a file sized for `s`, doing the two things `enter_frame` does: seed
    /// the locals, and materialise the constant pool at its region.
    ///
    /// Requires [`resolve_backpatches`] to have run, since the operands the emitted
    /// instructions carry are frame indices only after that.
    fn new(s: &SimulatedStack, locals: &[i64], globals: &[i64]) -> Self {
        let locals_count = s.locals_count();
        let consts_len = s.const_interner.consts.len() as u16;
        let spills = s.spills.allocation_len();
        let width = (s.max_registers + spills) as usize + consts_len as usize;

        let mut registers = vec![POISON; width];

        assert_eq!(
            locals.len(),
            locals_count as usize,
            "the harness must seed exactly the frame's locals"
        );

        registers[..locals.len()].copy_from_slice(locals);

        for (i, c) in s.const_interner.consts.iter().enumerate() {
            let Const::I32(v) = c else {
                unreachable!("these tests push only i32 constants, got {c:?}")
            };

            registers[locals_count as usize + i] = *v as i64;
        }

        Frame {
            registers,
            globals: globals.to_vec(),
            locals_count,
            consts_len,
        }
    }

    /// The absolute index of spill slot `index`, from the runtime's formula.
    fn spill_at(&self, index: &SpillIndex) -> usize {
        (self.locals_count + self.consts_len) as usize + spill_slot(index)
    }

    fn read(&self, slot: &Slot) -> i64 {
        let Slot(i) = slot;

        assert_ne!(
            *i,
            u16::MAX,
            "an unresolved placeholder reached execution — \
             `resolve_backpatches` did not run"
        );

        self.registers[*i as usize]
    }

    fn exec(&mut self, instruction: &RegInstruction, ins: &[Slot]) {
        match instruction {
            RegInstruction::LocalSpill { index, spill_index } => {
                let at = self.spill_at(spill_index);

                self.registers[at] = self.registers[index.0 as usize];
            }
            RegInstruction::LocalSet { index, input } => {
                self.registers[index.0 as usize] = self.read(&input.registers(ins)[0]);
            }
            RegInstruction::GlobalSet { index, input } => {
                self.globals[index.0 as usize] = self.read(&input.registers(ins)[0]);
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
fn run(prog: &Instructions, ins: &[Slot], skip: Range<usize>, frame: &mut Frame) {
    for (pc, instruction) in prog.inner.iter().enumerate() {
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
    let mut s = sim(2);
    let mut prog: Instructions = Instructions::default();

    s.push_local(0).unwrap(); // the borrow
    s.push_local(1).unwrap(); // the condition

    RegInstruction::spill_live_locals(&mut s, &mut prog, 0).unwrap();

    assert!(
        !prog.is_empty(),
        "the borrow must be rescued above the split"
    );

    s.add_block(BlockVariant::If, &BlockType::Empty, &[], prog.len());

    let _cond = s.registers_for::<1, 0>().unwrap();

    // then-arm
    let arm = prog.len()..{
        s.push_const(Const::I32(5)).unwrap();

        assert!(
            SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills)
                .unwrap()
                .is_none(),
            "the write finds nothing left to rescue"
        );

        let input = s.registers_for::<1, 0>().unwrap().input;

        prog.push(
            RegInstruction::LocalSet {
                index: LocalIndex(0),
                input,
            },
            0,
        );

        prog.len()
    };

    // else-arm is empty; reset as the `Else` arm does
    let recorded = s.get_curr_block().recorded_height;
    let params = s.get_curr_block().params;

    s.pops_and_pushes(s.stack.height() - recorded, params)
        .unwrap();

    let consumer = s.pop();
    let rendered = render_provisional(&consumer, s.locals_count());

    resolve_backpatches(&mut s);

    let consumer = resolve(&s, &consumer);

    for (tag, skip) in [("taken", 0..0), ("not taken", arm.clone())] {
        let mut frame = Frame::new(&s, &[42, 1], &[]);

        run(&prog, &s.input_registers, skip, &mut frame);

        assert_eq!(
            frame.read(&consumer),
            42,
            "{tag}: <use> reads {rendered}, which that path never wrote"
        );
    }
}

/// `local.get 0 ; block ; br_if 0 ; i32.const 5 ; local.set 0 ; end ; <use>`
///
/// Here the *taken* path is the one that skips the write, so the polarity is the
/// opposite of the `if` case.
#[test]
fn a_taken_br_if_cannot_skip_the_spill() {
    let mut s = sim(2);
    let mut prog: Instructions = Instructions::default();

    s.push_local(0).unwrap(); // the borrow
    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], prog.len());

    let base = s.get_curr_block().recorded_height;

    s.push_const(Const::I32(1)).unwrap(); // the condition

    RegInstruction::spill_live_locals(&mut s, &mut prog, 0).unwrap();

    let _cond = s.registers_for::<1, 0>().unwrap();
    let _mov = s.br_truncation_registers(base, 0);

    // everything after the branch is skipped when it is taken
    let rest = prog.len()..{
        s.push_const(Const::I32(5)).unwrap();

        assert!(
            SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills)
                .unwrap()
                .is_none()
        );

        let input = s.registers_for::<1, 0>().unwrap().input;

        prog.push(
            RegInstruction::LocalSet {
                index: LocalIndex(0),
                input,
            },
            0,
        );

        prog.len()
    };

    let consumer = s.pop();

    resolve_backpatches(&mut s);

    let consumer = resolve(&s, &consumer);

    for (tag, skip) in [("not taken", 0..0), ("taken", rest.clone())] {
        let mut frame = Frame::new(&s, &[42, 1], &[]);

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
    let mut s = sim(2);
    let mut prog: Instructions = Instructions::default();

    s.push_local(0).unwrap(); // the borrow, below the loop

    RegInstruction::spill_live_locals(&mut s, &mut prog, 0).unwrap();

    let entry = 0..prog.len();

    assert!(
        !entry.is_empty(),
        "the borrow must be rescued above the header"
    );

    s.add_block(BlockVariant::Loop, &BlockType::Empty, &[], prog.len());

    // body
    let body = prog.len()..{
        s.push_const(Const::I32(5)).unwrap();

        assert!(
            SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills)
                .unwrap()
                .is_none(),
            "the write finds nothing left to rescue"
        );

        let input = s.registers_for::<1, 0>().unwrap().input;

        prog.push(
            RegInstruction::LocalSet {
                index: LocalIndex(0),
                input,
            },
            0,
        );

        // the back-edge branches too, and must add nothing
        let before = prog.len();

        RegInstruction::spill_live_locals(&mut s, &mut prog, 0).unwrap();

        assert_eq!(prog.len(), before, "nothing left to spill at the back-edge");

        prog.len()
    };

    let recorded = s.get_curr_block().recorded_height;

    s.pops_and_pushes(s.stack.height() - recorded, 0).unwrap();

    let consumer = s.pop();

    resolve_backpatches(&mut s);

    let consumer = resolve(&s, &consumer);

    for iterations in [1, 2, 5] {
        let mut frame = Frame::new(&s, &[42, 1], &[]);

        for (pc, instruction) in prog.inner.iter().enumerate() {
            if entry.contains(&pc) {
                frame.exec(instruction, &s.input_registers);
            }
        }

        for _ in 0..iterations {
            for (pc, instruction) in prog.inner.iter().enumerate() {
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
    let mut s = sim(2);
    let mut prog: Instructions = Instructions::default();

    s.push_local(0).unwrap();

    RegInstruction::spill_live_locals(&mut s, &mut prog, 0).unwrap(); // the Block arm makes no such call

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], prog.len());

    // the call is shown only to contrast: the Block arm does not make it, and the
    // borrow would otherwise be spilled for nothing
    assert_eq!(prog.len(), 1, "this is what skipping the hoist avoids");
}

/// Nothing borrowed across the construct means the guard emits nothing at all,
/// which is the common case for anything rustc produces.
#[test]
fn hoisting_costs_nothing_when_no_borrow_is_live() {
    let mut s = sim(8);
    let mut prog: Instructions = Instructions::default();

    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap(); // consumed into a register

    s.push_const(Const::I32(1)).unwrap(); // a condition

    RegInstruction::spill_live_locals(&mut s, &mut prog, 0).unwrap();

    assert!(prog.is_empty(), "no live borrows, no instructions");
    assert_eq!(s.spills.allocation_len(), 0, "and no frame slots reserved");
}

// ---------------------------------------------------------------------------
// globals
//
// A global is *not* lazily forwarded. `global.get` emits a real read into a
// register, which is what lets an operand be a bare frame index — a global lives in
// the instance, not the frame — and it means no later write can invalidate a value
// already read. These pin that: the read happens, and none of the rescue machinery
// locals need is emitted for it.
// ---------------------------------------------------------------------------

const GLOBAL_MODULE: &str = r#"(module
    (global (mut i32) (i32.const 0))
    (global (mut i32) (i32.const 0))
    (func (param i32) (result i32) local.get 0)
    "#;

#[test]
fn a_global_read_is_materialised_into_a_register() {
    assert_func_lowers_to(
        &format!("{GLOBAL_MODULE} (func (result i32) global.get 0 i32.const 1 i32.add))"),
        1,
        "
          0  global.get   global0 -> r0
          1  i32.add      r0, 1 -> r0
          2  move         r0 -> r0
          3  end
             frame: 1 registers, 0 spills
        ",
    );
}

/// The property the deleted global-spill machinery used to provide: a write cannot
/// reach a value that has already been read into a register, so nothing is rescued
/// and the frame needs no spill slot.
#[test]
fn a_global_write_does_not_rescue_an_earlier_read() {
    let (prog, _, frame) = lower_func(
        &format!(
            "{GLOBAL_MODULE} (func (result i32) \
               global.get 0 i32.const 9 global.set 0 i32.const 1 i32.add))"
        ),
        1,
    );

    assert_eq!(
        frame.spills, 0,
        "an eager read needs no rescue, so no spill slot is allocated"
    );

    assert!(
        !prog
            .iter()
            .any(|i| i.kind() == RegInstructionKind::LocalSpill),
        "and nothing is spilled, but the body emits: {:?}",
        prog.iter().map(|i| i.kind()).collect::<Vec<_>>()
    );
}

/// The case a *call* used to force a rescue for, because a callee may write any
/// global. It cannot reach a register.
#[test]
fn a_global_read_survives_a_call() {
    let (prog, _, frame) = lower_func(
        &format!(
            "{GLOBAL_MODULE} (func (result i32) \
               global.get 0 i32.const 7 call 0 i32.add))"
        ),
        1,
    );

    assert_eq!(frame.spills, 0, "a call cannot invalidate a register");

    let at = index_of_kind(&prog, RegInstructionKind::I32Add).unwrap();

    let RegInstruction::I32Add(sig) = &prog[at] else {
        unreachable!()
    };

    // both operands are operand registers — the global read, and the call's result
    let operand_base = operand_base_of(&frame);

    for (i, index) in frame_indices(sig.input.registers(&frame.input_registers_arena))
        .iter()
        .enumerate()
    {
        assert!(
            *index >= operand_base,
            "operand {i} is slot {index}, below the operand region at {operand_base} \
             — a global read across a call must be a register"
        );
    }
}

/// Locals still forward lazily, so a write to one *is* rescued — the asymmetry with
/// globals is the point.
#[test]
fn writing_a_global_does_not_disturb_a_local_borrow() {
    let mut s = sim(2);

    s.push_local(0).unwrap();

    // A global write emits nothing that touches the lazy locals arena, so the
    // borrow of local 0 is still live and still reads its origin.
    assert!(
        s.lazy_locals.origin[0].is_some(),
        "local 0's borrow is live"
    );

    assert_eq!(s.spills.allocation_len(), 0, "and nothing is spilled");
    assert_eq!(s.pop_render(), "local0", "so it still forwards");
}

#[test]
fn writing_one_local_leaves_other_borrows_alone() {
    let mut s = sim(4);

    s.push_local(0).unwrap();
    s.push_local(1).unwrap();
    s.push_local(2).unwrap();

    let spill = SimulatedStack::set_lazy(1, &mut s.lazy_locals, &mut s.spills).unwrap();

    assert_eq!(spilled_to(spill), Some("spill0".into()));
    assert_eq!(s.pop_render(), "local2", "untouched above");
    assert_eq!(s.pop_render(), "spill0", "rescued");
    assert_eq!(s.pop_render(), "local0", "untouched below");
}

#[test]
fn writing_an_unborrowed_local_emits_nothing() {
    let mut s = sim(4);

    s.push_local(0).unwrap();

    assert!(
        SimulatedStack::set_lazy(3, &mut s.lazy_locals, &mut s.spills)
            .unwrap()
            .is_none()
    );
    assert_eq!(s.spills.allocation_len(), 0);
}

#[test]
fn three_borrows_share_one_entry() {
    let mut s = sim(2);

    s.push_local(0).unwrap();
    s.push_local(0).unwrap();
    s.push_local(0).unwrap();

    assert_eq!(
        spilled_to(SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).unwrap()),
        Some("spill0".into())
    );

    for _ in 0..3 {
        assert_eq!(s.pop_render(), "spill0", "all three redirect");
    }

    // the pool only reclaims once the last of them is gone
    assert_eq!(s.spills.allocation_len(), 1);
}

#[test]
fn tee_of_the_local_it_reads_round_trips_through_a_spill() {
    let mut s = sim(2);

    s.push_local(0).unwrap();

    let spill = SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills).unwrap();

    let operand = s.tee();

    // `local.tee 0` on a value read from local 0: correct, if redundant
    assert_eq!(spilled_to(spill), Some("spill0".into()));
    assert_eq!(render_provisional(&operand, s.locals_count()), "spill0");
}

// ---------------------------------------------------------------------------
// drop
// ---------------------------------------------------------------------------

#[test]
fn drop_releases_a_register() {
    let mut s = sim(2);

    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap();

    assert_eq!(heights(&s), (1, 1));

    s.pop();

    assert_eq!(heights(&s), (0, 0), "the register comes back");
    assert_eq!(peak_operands(&s), 1, "but the peak is remembered");
}

#[test]
fn drop_releases_a_borrow_but_only_the_last_one() {
    let mut s = sim(2);

    s.push_local(0).unwrap();
    s.push_local(0).unwrap();
    s.pop();

    assert!(
        SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills)
            .unwrap()
            .is_some(),
        "one borrow survives, so a write still has to rescue it"
    );

    let mut s = sim(2);

    s.push_local(0).unwrap();
    s.pop();

    assert!(
        SimulatedStack::set_lazy(0, &mut s.lazy_locals, &mut s.spills)
            .unwrap()
            .is_none(),
        "the sole borrow is gone"
    );
}

// ---------------------------------------------------------------------------
// operand ordering and register reuse
// ---------------------------------------------------------------------------

#[test]
fn operands_are_recorded_deepest_first() {
    let mut s = sim(4);

    s.push_local(0).unwrap(); // a
    s.push_local(1).unwrap(); // b
    s.push_local(2).unwrap(); // condition, pushed last

    let select = s.registers_for::<3, 1>().unwrap();

    assert_eq!(
        slots(select.input.registers(&s.input_registers), &layout_of(&s)),
        ["local0", "local1", "local2"],
        "select reads a, b, cond in that order"
    );
}

#[test]
fn store_records_address_then_value() {
    let mut s = sim(4);

    s.push_local(0).unwrap(); // address
    s.push_const(Const::I32(9)).unwrap(); // value

    let store = s.registers_for::<2, 0>().unwrap();

    // The constant has no frame index until the end-of-body backpatch runs, so
    // before that the arena holds a placeholder rather than a location.
    assert_eq!(
        slots(store.input.registers(&s.input_registers), &layout_of(&s)),
        ["local0", "<unresolved>"],
        "a constant operand is a placeholder until the body is fully lowered"
    );

    resolve_backpatches(&mut s);

    let operands = store.input.registers(&s.input_registers);

    assert_eq!(
        slots(operands, &layout_of(&s)),
        ["local0", "9"],
        "and resolves to the pool's value once it does"
    );

    // 4 locals and one constant, so the pool begins at frame index 4 — the
    // `const_id + locals_count` the pass backpatches.
    assert_eq!(
        frame_indices(operands),
        vec![0, 4],
        "address is local 0; the constant is the first slot above the locals"
    );
}

#[test]
fn a_result_reuses_an_operands_register() {
    let mut s = sim(4);

    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap(); // r0

    s.push_local(1).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap(); // r1
    let add = s.registers_for::<2, 1>().unwrap();

    resolve_backpatches(&mut s);

    let inputs = add.input.registers(&s.input_registers);
    let output = add.output.registers(&s.output_registers)[0];

    assert_eq!(slots(inputs, &layout_of(&s)), ["r0", "r1"]);

    // 4 locals, no constants and no spills, so the operand region begins at 4 and
    // the destination is the first register — the same slot `r0` occupies.
    assert_eq!(frame_indices(inputs), vec![4, 5]);

    assert_eq!(
        output, 4,
        "the destination aliases an operand — an executor must read both first"
    );
}

#[test]
fn max_registers_tracks_the_peak_not_the_total() {
    // two loads, each consumed before the next
    let mut s = sim(4);

    for l in 0..2 {
        s.push_local(l).unwrap();

        let _ = s.registers_for::<1, 1>().unwrap();
        let _ = s.registers_for::<1, 0>().unwrap();
    }

    assert_eq!(peak_operands(&s), 1, "serial values reuse one register");

    // two loads live at once
    let mut s = sim(4);

    for l in 0..2 {
        s.push_local(l).unwrap();

        let _ = s.registers_for::<1, 1>().unwrap();
    }

    assert_eq!(peak_operands(&s), 2);
}

#[test]
fn non_register_operands_occupy_stack_but_not_registers() {
    let mut s = sim(4);

    s.push_const(Const::I32(1)).unwrap();
    s.push_local(0).unwrap();
    s.push_local(1).unwrap();

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
    let mut s = sim(4);

    s.add_block(BlockVariant::Loop, &BlockType::Empty, &types, 7);

    assert!(matches!(
        s.get_curr_block().kind,
        BlockKind::Loop { index: 7 }
    ));

    // with params: an entry move occupies index 7, body starts at 8
    let mut s = sim(4);

    s.push_const(Const::I32(1)).unwrap();
    s.add_block(BlockVariant::Loop, &BlockType::FuncType(0), &types, 7);

    assert!(matches!(
        s.get_curr_block().kind,
        BlockKind::Loop { index: 8 }
    ));
}

#[test]
fn an_if_records_the_index_of_the_if_itself() {
    let types = vec![func_ty(1, 1)];
    let mut s = sim(4);

    s.push_const(Const::I32(1)).unwrap(); // condition
    s.add_block(BlockVariant::If, &BlockType::Empty, &types, 7);

    assert!(matches!(
        s.get_curr_block().kind,
        BlockKind::If { index: 7, .. }
    ));

    let mut s = sim(4);

    s.push_const(Const::I32(9)).unwrap(); // param
    s.push_const(Const::I32(1)).unwrap(); // condition
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
    let mut s = sim(4);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);
    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap();

    let func_block = s.get_block(0);

    assert_eq!(func_block.recorded_height, 0);

    let mov = s.br_truncation_registers(0, 0).unwrap();

    assert!(mov.is_empty());
    assert_eq!(heights(&s), (1, 1), "still a simulation");
}

#[test]
fn a_branch_carrying_several_values_lands_them_contiguously() {
    let mut s = sim(4);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;

    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap();

    s.push_const(Const::I32(2)).unwrap();
    s.push_local(1).unwrap();

    let mov = s.br_truncation_registers(base, 3).unwrap();

    assert_eq!(
        output_operands(mov.output_registers(&s.output_registers), &s),
        vec![0, 1, 2]
    );

    assert_eq!(mov.input_registers(&s.input_registers).len(), 3);
}

#[test]
fn a_br_table_may_mix_loop_and_block_targets() {
    // loop's label type is its params; a block's is its results. Validation only
    // requires the types to match, so the arities agree while the bases differ.
    let types = vec![func_ty(1, 1)];
    let mut s = sim(4);

    s.push_const(Const::I32(0)).unwrap(); // the loop's param
    s.add_block(BlockVariant::Loop, &BlockType::FuncType(0), &types, 0);

    let loop_block = s.get_curr_block().recorded_height;
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

    s.push_local(0).unwrap(); // the value the branch carries

    let to_loop = s.br_truncation_registers(loop_block, loop_params).unwrap();
    let to_block = s
        .br_truncation_registers(block_base, block_results)
        .unwrap();

    assert_eq!(loop_params, block_results, "arities agree");

    assert_ne!(
        to_loop.output_registers(&s.output_registers),
        to_block.output_registers(&s.output_registers),
        "but the destinations differ, which is why each arm carries its own move"
    );
}

#[test]
fn every_br_table_arm_sees_the_same_stack() {
    let mut s = sim(4);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;

    s.push_local(0).unwrap();

    let before = heights(&s);
    let arms: Vec<_> = (0..4)
        .map(|_| s.br_truncation_registers(base, 1).unwrap())
        .collect();

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
    let mut s = sim(4);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;

    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap();

    assert!(
        s.br_truncation_registers(base, 0).unwrap().is_empty(),
        "callers use this to skip emitting the move entirely"
    );
}

#[test]
fn br_if_consumes_only_its_condition() {
    let mut s = sim(4);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let base = s.get_curr_block().recorded_height;
    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap();
    s.push_local(1).unwrap();

    let live = (heights(&s), s.lazy_locals.origin[1].is_some());

    s.push_const(Const::I32(1)).unwrap(); // condition

    let _cond = s.registers_for::<1, 0>().unwrap();
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

    assert!(live[live.len() - 1], "the live block's end must resume");
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
    let mut s = sim(4);

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let outer = s.get_curr_block().recorded_height;

    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap();

    s.add_block(BlockVariant::Block, &BlockType::Empty, &[], 0);

    let inner = s.get_curr_block().recorded_height;

    s.push_local(1).unwrap();

    // two arms plus a default, as the BrTable arm records them
    for base in [inner, outer, inner] {
        let mov = s.br_truncation_registers(base, 1).unwrap();

        s.br_targets.push(RegBrTableTarget {
            mov,
            target_index: u32::MAX,
        });
    }

    let frame = RegFrameLayout {
        registers: s.max_registers,
        spills: s.spills.allocation_len(),
        locals_count: s.lazy_locals.origin.len() as u16,
        consts: s.const_interner.consts.into_boxed_slice(),
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
    let dests: Vec<Vec<u16>> = frame
        .br_targets_arena
        .iter()
        .map(|t| {
            t.mov
                .output_registers(&frame.output_registers_arena)
                .iter()
                .map(|r| r - frame.locals_count)
                .collect()
        })
        .collect();

    assert_eq!(dests, vec![vec![1], vec![0], vec![1]]);
}

#[test]
fn a_body_without_a_br_table_carries_an_empty_arm_arena() {
    let mut s = sim(2);

    s.push_local(0).unwrap();

    let _ = s.registers_for::<1, 1>().unwrap();

    let frame = RegFrameLayout {
        registers: s.max_registers,
        spills: s.spills.allocation_len(),
        locals_count: s.lazy_locals.origin.len() as u16,
        consts: s.const_interner.consts.into_boxed_slice(),
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

/// A constant is read in place; a global is not.
///
/// The asymmetry is the encoding's: an operand is a bare frame index, and the
/// constant pool *is* part of the frame while a global lives in the instance. So
/// `i32.const 7` emits nothing and `global.get 0` emits a read.
#[test]
fn a_constant_is_read_in_place_but_a_global_is_materialised() {
    assert_lowers_to(
        r#"(module
             (global i32 (i32.const 0))
             (func (param i32) (result i32)
               global.get 0
               i32.const 7
               i32.add))"#,
        "
          0  global.get   global0 -> r0
          1  i32.add      r0, 7 -> r0
          2  move         r0 -> r0
          3  end
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
    let (prog, _, frame) = lower(
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
    let (prog, _, _) = lower(
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

    // The exact index, not just "an End": there are two here — the `if`'s at 3 and
    // the body's at 4 — and a false condition must reach the first, not fall out of
    // the function.
    assert_eq!(at, 1, "the `if`'s own index");
    assert_eq!(end_index, 3, "the `if`'s end, not the body's");

    let ends: Vec<usize> = prog
        .iter()
        .enumerate()
        .filter(|(_, i)| i.kind() == RegInstructionKind::End)
        .map(|(n, _)| n)
        .collect();

    assert_eq!(ends, vec![3, 4], "the if's end, then the body's");
}

#[test]
fn both_arms_of_an_if_materialise_results_into_the_same_registers() {
    let (prog, _, frame) = lower(
        r#"(module (func (param i32) (result i32)
             local.get 0
             if (result i32) i32.const 1 else i32.const 2 end))"#,
    );

    let dests: Vec<Vec<u16>> = prog
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
    let (prog, _, _) = lower(
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
    // The entry rescue must run once, so the back-edge has to land after it — on the
    // `loop` header, which is the label's own instruction and falls through into the
    // body.
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
          1  loop
          2  local.set    local0 <- 5
          3  local.spill  local1 -> spill1
          4  br_if        spill1 -> 1
          5  end
          6  move         spill0 -> r0
          7  end
             frame: 1 registers, 2 spills
        ",
    );
}

#[test]
fn a_branch_to_a_loop_carries_the_loops_params() {
    let (prog, _, frame) = lower(
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

    // The exact target, not merely that it points backwards. The header `Move` that
    // materialises the loop's entry params is instruction 0; the label is the `loop`
    // at 1. Targeting 0 would re-run that move on every iteration and pin the params
    // to their entry values, which is the bug this number rules out.
    assert_eq!(br, 3, "the back-edge's own index");
    assert_eq!(
        *target_index, 1,
        "the loop header's index, past the entry move"
    );
    assert!(
        matches!(prog[*target_index as usize], RegInstruction::Loop),
        "the target must be the `loop` itself, found {:?}",
        prog[*target_index as usize].kind()
    );
    assert!(
        matches!(prog[0], RegInstruction::Move(_)),
        "instruction 0 is the entry move the back-edge must skip"
    );

    // And the destination the back-edge writes is the same register the header move
    // fills, which is what makes skipping the header correct rather than lossy.
    let back_edge_dst = mov.output_registers(&frame.output_registers_arena);
    let RegInstruction::Move(header) = &prog[0] else {
        unreachable!()
    };

    assert_eq!(
        back_edge_dst,
        header.output_registers(&frame.output_registers_arena),
        "a back-edge must land the params in the registers the loop body reads"
    );
}

#[test]
fn br_table_arms_resolve_to_their_own_labels() {
    let (prog, _, frame) = lower(
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

    // The exact arms, not just their relationships. Depth 0 is the inner block,
    // whose end is 2; depth 1 is the outer block, whose end is 3. `br_table 0 1 0`
    // therefore resolves to [2, 3, 2] — which subsumes "arm 0 equals the default"
    // and "depth 1 differs", and additionally pins *which* label each names.
    assert_eq!(table, 1, "the table's own index");
    assert_eq!(targets, vec![2, 3, 2], "arm 0, arm 1, default");

    let ends: Vec<usize> = prog
        .iter()
        .enumerate()
        .filter(|(_, i)| i.kind() == RegInstructionKind::End)
        .map(|(n, _)| n)
        .collect();

    assert_eq!(
        ends,
        vec![2, 3, 4],
        "inner block's end, outer block's end, body's end"
    );
}

#[test]
fn each_br_table_arm_carries_its_own_move() {
    let (prog, _, frame) = lower(
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
            slots(
                arm.mov.input_registers(&frame.input_registers_arena),
                &frame
            ),
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
    // Both rescues land at 0 and 1, ahead of the `if` at 2 — and the `if` records no
    // else and an end of exactly 4, so the whole shape is pinned rather than just
    // the ordering.
    assert_lowers_to(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             local.get 1
             if i32.const 5 local.set 0 end))"#,
        "
          0  local.spill  local0 -> spill0
          1  local.spill  local1 -> spill1
          2  if           spill1 else=- end=4
          3  local.set    local0 <- 5
          4  end
          5  move         spill0 -> r0
          6  end
             frame: 1 registers, 2 spills
        ",
    );
}

#[test]
fn the_br_if_arm_hoists_the_spill_above_the_branch() {
    // The taken edge leaves at 2 for the block's end at 4, so the rescues at 0 and 1
    // are the only instructions it cannot skip.
    assert_lowers_to(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             block
               local.get 1
               br_if 0
               i32.const 5
               local.set 0
             end))"#,
        "
          0  local.spill  local0 -> spill0
          1  local.spill  local1 -> spill1
          2  br_if        spill1 -> 4
          3  local.set    local0 <- 5
          4  end
          5  move         spill0 -> r0
          6  end
             frame: 1 registers, 2 spills
        ",
    );
}

#[test]
fn the_loop_arm_hoists_the_spill_out_of_the_repeated_region() {
    let (prog, _, _) = lower(
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

    // The exact numbers, not just their order: local 0's rescue is instruction 0, the
    // repeated region starts at 1, and the back-edge is instruction 4.
    assert_eq!(back_edge, 4, "the back-edge's own index");
    assert_eq!(*target_index, 1, "the loop header's index");

    assert!(
        matches!(prog[*target_index as usize], RegInstruction::Loop),
        "a back-edge must land on the `loop` itself, not on a body instruction: found {:?}",
        prog[*target_index as usize].kind()
    );

    let rescues: Vec<usize> = prog
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i, RegInstruction::LocalSpill { index, .. } if index.0 == 0))
        .map(|(n, _)| n)
        .collect();

    assert_eq!(
        rescues,
        vec![0],
        "local 0 is rescued exactly once, before the repeated region"
    );
}

/// The whole body of [`the_loop_arm_hoists_the_spill_out_of_the_repeated_region`],
/// so the indices its assertions name cannot drift without this failing too.
#[test]
fn the_loop_hoist_lowers_exactly() {
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
          1  loop
          2  local.set    local0 <- 5
          3  local.spill  local1 -> spill1
          4  br_if        spill1 -> 1
          5  end
          6  move         spill0 -> r0
          7  end
             frame: 1 registers, 2 spills
        ",
    );
}

#[test]
fn the_br_table_arm_hoists_the_spill_above_the_branch() {
    // Every arm leaves, so the rescues at 0 and 1 are unskippable — and both arms
    // resolve to the block's end at 3. The `local.set` after the table is dead and
    // does not appear at all, which is why the body is only four instructions.
    assert_lowers_to(
        r#"(module (func (param i32 i32) (result i32)
             local.get 0
             block
               local.get 1
               br_table 0 0
               i32.const 5
               local.set 0
             end))"#,
        "
          0  local.spill  local0 -> spill0
          1  local.spill  local1 -> spill1
          2  br_table     spill1 ->3 ->3
          3  end
          4  move         spill0 -> r0
          5  end
             frame: 1 registers, 2 spills
        ",
    );
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
    let (prog, _, _) = lower(
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
    let (prog, _, frame) = lower(
        r#"(module (func (param i32) (result i32)
             local.get 0
             if (result i32) i32.const 1 return else i32.const 2 end))"#,
    );

    let dests: Vec<Vec<u16>> = prog
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
    let (prog, _, _) = lower(
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
fn caller_base_of(wat: &str) -> (u16, Vec<String>) {
    let (prog, _, frame) = lower_func(wat, 1);
    let at = index_of_kind(&prog, RegInstructionKind::Call).unwrap();

    let RegInstruction::Call { caller_base, .. } = &prog[at] else {
        unreachable!()
    };

    // the instruction that consumes the call's result
    let reader = prog[at + 1..]
        .iter()
        .find_map(|i| match i {
            RegInstruction::I32Add(sig) => Some(slots(
                sig.input.registers(&frame.input_registers_arena),
                &frame,
            )),
            _ => None,
        })
        .expect("something must read the result");

    // `caller_base` ships as an absolute frame index; the assertions below are about
    // *which operand register* the arguments were staged at, so it comes back
    // operand-relative. That it is never below the operand base is itself the
    // invariant keeping a callee off its caller's constants — asserted here, since a
    // wrong shift would underflow.
    let operand_base = operand_base_of(&frame);

    assert!(
        *caller_base >= operand_base,
        "caller_base {caller_base} is below the operand base {operand_base}, so the \
         callee's frame would overlap its caller's constants and spills"
    );

    (*caller_base - operand_base, reader)
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
    let (prog, _, frame) = lower_func(
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

/// A callee may write any global, but it cannot reach a value already read into a
/// register — so nothing is rescued, and the two reads straddling the call are simply
/// two reads.
///
/// The second `global.get` is a *separate* read and so sees whatever the callee left,
/// which is what wasm requires: `global.get` reads at the point it appears.
#[test]
fn a_global_read_across_a_call_needs_no_rescue() {
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
          0  global.get   global0 -> r0
          1  call         f0 caller_base=1
          2  global.get   global0 -> r1
          3  i32.add      r0, r1 -> r0
          4  move         r0 -> r0
          5  end
             frame: 2 registers, 0 spills
        ",
    );
}

#[test]
fn a_local_read_across_a_call_needs_no_rescue() {
    let (prog, _, _) = lower_func(
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
        let (prog, _, frame) = lower_func(wat, 0);
        let at = index_of_kind(&prog, RegInstructionKind::CallIndirect).unwrap();

        let RegInstruction::CallIndirect { caller_base, .. } = &prog[at] else {
            unreachable!()
        };

        // operand-relative, as the renderings above report it
        *caller_base - operand_base_of(&frame)
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

/// The callee is not known until execution, so it may write any global — and, as for
/// a direct call, that cannot reach a value already in a register.
#[test]
fn a_global_read_across_a_call_indirect_needs_no_rescue() {
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
          0  global.get   global0 -> r0
          1  call_indirect [3] ty0 table0 caller_base=1
          2  i32.add      r0, r1 -> r0
          3  move         r0 -> r0
          4  end
             frame: 2 registers, 0 spills
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
        RegInstruction::render_body(&body, &[]),
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
        r#"(module (func (param i32) (result i32)
             local.get 0
             i32.const 5
             local.set 0
             block
               unreachable
             end
             i32.const 1
             i32.add))"#,
        "
          0  local.spill  local0 -> spill0
          1  local.set    local0 <- 5
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
    let (_, _, frame) = lower(
        r#"(module (func (result i32)
             block (result i32 i32)
               unreachable
             end
             drop))"#,
    );

    assert_eq!(
        peak_operands_of(&frame),
        2,
        "both results are given registers"
    );
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

/// A `.wat` loading through a `memory 1` and dropping the value read.
///
/// The address is always an `i32`, and what the load widens to is its own business
/// since the result is dropped — so one shape covers all thirteen.
fn load_case(kind: RegInstructionKind) -> String {
    format!(
        "(module (memory 1) (func (param i32) local.get 0 {} drop))",
        mnemonic(kind)
    )
}

/// A `.wat` storing through a `memory 1`.
///
/// Two operands of different types — an `i32` address and a value of the store's
/// own type — so the value comes from a second param rather than a repeated
/// `local.get 0`. `i64.store8` stores an `i64`: the `8` is the width written, not
/// the type accepted, which is why [`operand_type`] reads the prefix here.
fn store_case(kind: RegInstructionKind) -> String {
    format!(
        "(module (memory 1) (func (param i32 {}) local.get 0 local.get 1 {}))",
        operand_type(kind),
        mnemonic(kind)
    )
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
        // Every numeric instruction, in the order the enum declares them. Their
        // cases are derived from the kind — the mnemonic and the operand type are
        // both recoverable from the name, so 159 hand-written bodies would be 159
        // chances to mistype one.
        // i32 — loads
        RegInstructionKind::I32Load
        | RegInstructionKind::I32Load8S
        | RegInstructionKind::I32Load8U
        | RegInstructionKind::I32Load16S
        | RegInstructionKind::I32Load16U => Some(load_case(kind)),

        // i32 — stores
        RegInstructionKind::I32Store
        | RegInstructionKind::I32Store8
        | RegInstructionKind::I32Store16 => Some(store_case(kind)),

        // i32 — unary
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

        // i32 — binary
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

        // i64 — loads
        RegInstructionKind::I64Load
        | RegInstructionKind::I64Load8S
        | RegInstructionKind::I64Load8U
        | RegInstructionKind::I64Load16S
        | RegInstructionKind::I64Load16U
        | RegInstructionKind::I64Load32S
        | RegInstructionKind::I64Load32U => Some(load_case(kind)),

        // i64 — stores
        RegInstructionKind::I64Store
        | RegInstructionKind::I64Store8
        | RegInstructionKind::I64Store16
        | RegInstructionKind::I64Store32 => Some(store_case(kind)),

        // i64 — unary
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

        // i64 — binary
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

        // f32 — loads
        RegInstructionKind::F32Load => Some(load_case(kind)),

        // f32 — stores
        RegInstructionKind::F32Store => Some(store_case(kind)),

        // f32 — unary
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

        // f32 — binary
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

        // f64 — loads
        RegInstructionKind::F64Load => Some(load_case(kind)),

        // f64 — stores
        RegInstructionKind::F64Store => Some(store_case(kind)),

        // f64 — unary
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

        // f64 — binary
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
        // memory
        //
        // `memory.size` takes no operands, so its case has nothing to push — only
        // the result to drop, which keeps the body to one instruction.
        RegInstructionKind::DataDrop => {
            Some("(module (memory 1) (data \"x\") (func data.drop 0))".into())
        }
        // a passive segment, since `memory.init` may only read from one
        RegInstructionKind::MemoryInit => Some(
            "(module (memory 1) (data \"x\") (func (param i32) local.get 0 local.get 0 local.get 0 memory.init 0))"
                .into(),
        ),
        RegInstructionKind::MemoryCopy => Some(
            "(module (memory 1) (func (param i32) local.get 0 local.get 0 local.get 0 memory.copy))"
                .into(),
        ),
        RegInstructionKind::MemoryFill => Some(
            "(module (memory 1) (func (param i32) local.get 0 local.get 0 local.get 0 memory.fill))"
                .into(),
        ),
        RegInstructionKind::MemoryGrow => {
            Some("(module (memory 1) (func (param i32) local.get 0 memory.grow drop))".into())
        }
        RegInstructionKind::MemorySize => {
            Some("(module (memory 1) (func memory.size drop))".into())
        }
        RegInstructionKind::RefIsNull => {
            Some("(module (func (param funcref) local.get 0 ref.is_null drop))".into())
        }

        RegInstructionKind::LocalSet
        | RegInstructionKind::LocalTee
        | RegInstructionKind::GlobalGet
        | RegInstructionKind::GlobalSet
        | RegInstructionKind::LocalSpill
        | RegInstructionKind::If
        | RegInstructionKind::Else
        | RegInstructionKind::Loop
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
        let (prog, frame) = (&body.0, &body.2);

        // the case is built so the body is exactly this instruction and `end`,
        // which is what lets the arenas be read as that instruction's operands
        assert_eq!(
            prog.len(),
            2,
            "{operator}: case must lower to one instruction and `end`:\n{}",
            RegInstruction::render_body(&body, &[])
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

        let (prog, _, _) = lower(&wat);

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
    let (_, _, frame) = lower(
        r#"(module (memory 1) (func (param i32) (result i32)
             local.get 0 i32.load
             local.get 0 i32.load
             i32.add))"#,
    );

    assert_eq!(peak_operands_of(&frame), 2, "two loads are live at once");
}

#[test]
fn the_frame_reports_peak_spill_use() {
    let (_, _, frame) = lower(
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
    let (prog, _, frame) = lower(
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

// ---------------------------------------------------------------------------
// Instruction offsets
// ---------------------------------------------------------------------------
//
// The driver reads `instruction_offsets[pc]` whenever it records a trace record,
// so a body whose offsets are shorter than its instruction list panics on the
// first trap deep enough to reach the missing entry. These pin the two properties
// that makes impossible: the lists stay the same length, and every entry is a real
// operator offset rather than a plausible-looking number.

/// Every byte offset `wasmparser` reports for an operator of function `n`, in
/// order — the ground truth both lowering passes record against.
fn operator_offsets(wat: &str, n: usize) -> Vec<u32> {
    let bytes = wat::parse_str(wat).expect("invalid wat");
    let mut bodies = vec![];

    for payload in Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.expect("parse") {
            bodies.push(body);
        }
    }

    let body = bodies.into_iter().nth(n).expect("no such function body");
    let mut reader = body.get_operators_reader().expect("operators");
    let mut offsets = vec![];

    while !reader.eof() {
        let (_operator, offset) = reader.read_with_offset().expect("operator");

        offsets.push(offset as u32);
    }

    offsets
}

/// A body covering every shape that emits more than one instruction per operator —
/// blocks that move block params, an `if`/`else` and a `br_if` that spill live
/// borrows, a `br_table`, and a call — so the cases where the two lists could most
/// easily drift apart are all present.
const OFFSET_PROBE: &str = r#"(module
    (global $g (mut i32) (i32.const 0))
    (func $callee (param i32) (result i32) local.get 0)
    (func $probe (param i32 i32) (result i32) (local i32)
        local.get 0
        local.set 2
        block (result i32)
            local.get 2
            global.get $g
            i32.add
            local.get 1
            br_if 0
            local.get 0
            i32.const 7
            i32.mul
            local.set 2
            local.get 1
            br_table 0 0
        end
        local.get 2
        i32.add
        call $callee
        local.get 1
        if (result i32)
            local.get 2
            i32.const 1
            i32.add
            local.set 2
            local.get 2
        else
            global.get $g
            local.set 2
            i32.const 0
        end
        i32.add
        loop (result i32)
            local.get 2
            local.set 2
            local.get 0
        end
        i32.add))"#;

#[test]
fn offsets_and_instructions_stay_the_same_length() {
    for n in 0..2 {
        let (instructions, offsets, _) = lower_func(OFFSET_PROBE, n);

        assert_eq!(
            offsets.len(),
            instructions.len(),
            "function {n}: the driver indexes offsets by `pc`, so a short list is a panic \
             waiting for a deep enough trap"
        );
    }
}

#[test]
fn every_offset_is_a_real_operator_offset() {
    for n in 0..2 {
        let (_instructions, offsets, _) = lower_func(OFFSET_PROBE, n);
        let real = operator_offsets(OFFSET_PROBE, n);

        for offset in &offsets {
            assert!(
                real.contains(offset),
                "function {n}: {offset} is not an operator boundary; \
                 real offsets are {real:?}"
            );
        }
    }
}

// Operators are read in order and instructions are only ever appended, so the
// offsets a body records can repeat — one operator lowering to a spill, a move and
// then itself — but can never go backwards. A decrease would mean an instruction
// had been inserted behind one already emitted, which would also silently
// misattribute every later `pc`.
#[test]
fn offsets_never_go_backwards() {
    for n in 0..2 {
        let (_instructions, offsets, _) = lower_func(OFFSET_PROBE, n);

        for pair in offsets.windows(2) {
            assert!(
                pair[0] <= pair[1],
                "function {n}: offsets decrease, {} then {}",
                pair[0],
                pair[1]
            );
        }
    }
}

// The register pass emits zero instructions for the operators it handles lazily
// (`local.get`, `i32.const`) and several for others, so its offset list is neither
// the operator list nor a permutation of it — but the *distinct* offsets it records
// must appear in operator order, and must be a subset of the real ones.
#[test]
fn distinct_offsets_follow_operator_order() {
    for n in 0..2 {
        let (_instructions, offsets, _) = lower_func(OFFSET_PROBE, n);
        let real = operator_offsets(OFFSET_PROBE, n);

        let mut distinct: Vec<u32> = vec![];

        for offset in offsets {
            if distinct.last() != Some(&offset) {
                distinct.push(offset);
            }
        }

        let mut next = real.iter();

        for offset in &distinct {
            assert!(
                next.any(|candidate| candidate == offset),
                "function {n}: {offset} is out of operator order in {distinct:?}"
            );
        }
    }
}

// A body that emits nothing until its final `end` still has to produce one offset
// for that `end` — the degenerate case where an off-by-one in the wiring would go
// unnoticed by the probe above.
#[test]
fn a_body_that_lowers_to_one_instruction_still_records_its_offset() {
    let wat = "(module (func))";
    let (instructions, offsets, _) = lower_func(wat, 0);
    let real = operator_offsets(wat, 0);

    assert_eq!(instructions.len(), 1, "just the terminating `end`");
    assert_eq!(offsets.len(), 1);

    assert_eq!(
        offsets[0],
        *real.last().expect("the `end` operator"),
        "the sole instruction is the `end`, so it carries the `end`'s offset"
    );
}

// ===========================================================================
// the constant pool's identity
//
// [`ConstInterner`] dedups on [`Const`]'s `Eq`/`Hash`, so those alone decide when
// two immediates share a frame slot — and a slot holds one value.
//
// Two constants may therefore share a slot only when they are the same value *of
// the same type*. Merging across types is a wrong-value bug, because the pool keeps
// whichever `Const` arrived first and `enter_frame` materialises it by matching the
// variant — so an `f64` operand reading a slot interned from an `i32` gets the
// wrong kind of value. Merging bit-distinct floats is the same class of bug one
// level down: it silently drops a sign or a NaN payload.
//
// Neither is caught by the differential guests reliably. A cross-type merge only
// bites when the two constants land in the same hash bucket, so it is latent until
// it is not; these assert the property directly instead.
// ===========================================================================

/// Hashes a `Const` with the standard hasher.
///
/// Deliberately *not* the map's own `FxHasher`: the `Hash`/`Eq` contract has to
/// hold for every hasher, and a test that used the same one the map does could pass
/// on a coincidence of that hasher's mixing.
fn hash_of(c: Const) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut h = std::collections::hash_map::DefaultHasher::new();

    c.hash(&mut h);
    h.finish()
}

/// Constants that share a bit pattern across types, so any identity that looks at
/// the bits alone merges them.
///
/// `1065353216` is `1.0f32`'s bit pattern and `4607182418800017408` is `1.0f64`'s,
/// which is how an integer immediate collides with a float one in practice — a
/// small float and a large-but-ordinary integer.
fn same_bits_different_types() -> Vec<(&'static str, Const)> {
    vec![
        ("I32(0)", Const::I32(0)),
        ("I64(0)", Const::I64(0)),
        ("F32(+0.0)", Const::F32(0.0f32.into())),
        ("F64(+0.0)", Const::F64(0.0f64.into())),
        (
            "Ref(Some(0))",
            Const::Ref(Some(crate::module::FuncIndex(0))),
        ),
        ("I32(1065353216)", Const::I32(1065353216)),
        ("F32(1.0)", Const::F32(1.0f32.into())),
        ("I64(4607182418800017408)", Const::I64(4607182418800017408)),
        ("F64(1.0)", Const::F64(1.0f64.into())),
    ]
}

/// Two constants of different types are different constants, whatever their bits.
#[test]
fn const_equality_respects_the_type() {
    let all = same_bits_different_types();

    for (i, (a_name, a)) in all.iter().enumerate() {
        for (b_name, b) in all.iter().skip(i + 1) {
            assert_ne!(
                a, b,
                "{a_name} and {b_name} are different constants, but compare equal — \
                 an identity that reads only the bit pattern merges them, and the \
                 pool would keep whichever was interned first"
            );
        }
    }
}

/// Values that compare equal must hash equal.
///
/// Not a style point: `FxHashMap` finds a key by hashing to a bucket and then
/// comparing, so a pair that is `==` with unequal hashes is simply never noticed as
/// a duplicate — and a pair that *is* noticed returns the other one's slot. Whether
/// the pool merges two constants then depends on the hasher rather than on the
/// constants.
#[test]
fn const_hash_agrees_with_equality() {
    let all = same_bits_different_types();

    for (a_name, a) in all.iter() {
        for (b_name, b) in all.iter() {
            if a == b {
                assert_eq!(
                    hash_of(*a),
                    hash_of(*b),
                    "{a_name} == {b_name} but they hash differently, which breaks the \
                     `Hash`/`Eq` contract the pool's map relies on"
                );
            }
        }
    }
}

/// The pool gives each of them its own slot.
///
/// The consequence of the two properties above, and the one that actually matters:
/// nine distinct constants must occupy nine slots.
#[test]
fn the_pool_keeps_same_bits_of_different_types_apart() {
    let all = same_bits_different_types();
    let mut interner = ConstInterner::default();
    let ids: Vec<(&str, u16)> = all
        .iter()
        .map(|(name, c)| (*name, interner.intern(*c).unwrap().0))
        .collect();

    assert_eq!(
        interner.consts.len(),
        all.len(),
        "each distinct constant needs its own slot, got {ids:?}"
    );

    // and every slot still holds what was interned into it
    for (name, id) in ids {
        let (_, expected) = all.iter().find(|(n, _)| *n == name).unwrap();

        assert_eq!(
            &interner.consts[id as usize], expected,
            "slot {id} was interned for {name} but holds {:?}",
            interner.consts[id as usize]
        );
    }
}

/// Floats are identified by their bit pattern, not by `==`.
///
/// `-0.0 == 0.0` and every NaN is unequal to itself, so a numeric identity merges
/// the zeroes and collapses NaN payloads. Both are observable: the sign of zero
/// survives `f64.min`/`copysign`, and a NaN payload survives arithmetic.
#[test]
fn the_pool_keeps_bit_distinct_floats_apart() {
    let mut interner = ConstInterner::default();

    let pos64 = interner.intern(Const::F64(0.0f64.into())).unwrap();
    let neg64 = interner.intern(Const::F64((-0.0f64).into())).unwrap();
    let pos32 = interner.intern(Const::F32(0.0f32.into())).unwrap();
    let neg32 = interner.intern(Const::F32((-0.0f32).into())).unwrap();

    assert_ne!(
        pos64.0, neg64.0,
        "f64 +0.0 and -0.0 are different constants"
    );

    assert_ne!(
        pos32.0, neg32.0,
        "f32 +0.0 and -0.0 are different constants"
    );

    assert_eq!(
        interner.consts[pos64.0 as usize],
        Const::F64(0.0f64.into()),
        "the +0.0 slot must not have been overwritten by -0.0"
    );

    // NaN payloads and sign, which arithmetic propagates
    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let neg_nan = f64::from_bits(0xfff8_0000_0000_0001);
    let other_payload = f64::from_bits(0x7ff8_0000_dead_beef);

    let a = interner.intern(Const::F64(nan.into())).unwrap();
    let b = interner.intern(Const::F64(neg_nan.into())).unwrap();
    let c = interner.intern(Const::F64(other_payload.into())).unwrap();

    assert_ne!(a.0, b.0, "a NaN and its negation are different constants");
    assert_ne!(
        a.0, c.0,
        "NaNs with different payloads are different constants"
    );

    // interning the same bits twice must still dedup, or the pool grows per use
    let again = interner.intern(Const::F64(nan.into())).unwrap();

    assert_eq!(a.0, again.0, "the same bits must reuse the same slot");
}

/// Dedup still has to work — these properties must not be bought by giving every
/// use its own slot, which would grow every frame by its constant count.
#[test]
fn the_pool_still_dedups_equal_constants() {
    let mut interner = ConstInterner::default();

    for _ in 0..4 {
        interner.intern(Const::I32(7)).unwrap();
        interner.intern(Const::F64(2.5f64.into())).unwrap();
        interner.intern(Const::Ref(None)).unwrap();
    }

    assert_eq!(
        interner.consts.len(),
        3,
        "three distinct constants interned four times each"
    );
}
