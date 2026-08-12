use crate::{
    error::TraceWasmError,
    instruction::{
        Block, BlockKind, params_and_results_from_blockty,
        register::lazy::{
            Global, GlobalSlot, LazyArena, LazyEntryDropResult, LazyLocation, LazySlot, Local,
            LocalSlot, SpillArena,
        },
    },
    module::{FuncDecl, FuncType, GlobalIndex, LocalIndex},
    vm::stack::Stack,
};
use std::marker::PhantomData;
use wasmparser::{BlockType, Operator, OperatorsReader};

pub mod lazy;

enum BlockVariant {
    If,
    Loop,
    Block,
    Func,
}

#[derive(Debug, Clone, Copy)]
pub enum Const {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

#[derive(Debug, Clone, Copy)]
pub enum Slot {
    Const(Const),
    Local(u32),
    Global(u32),
    Spilled(u32),
    Register(u32), // index into stack
}

impl Slot {
    fn is_register(&self) -> bool {
        matches!(self, Slot::Register(_))
    }
}

impl Default for Slot {
    fn default() -> Self {
        Slot::Const(Const::I32(0))
    }
}

#[derive(Clone, Copy)]
enum StackSlot {
    Const(Const),
    Register(u32),
    Local(LocalSlot),
    Global(GlobalSlot),
}

pub struct Registers<const L: usize, T> {
    start: u32,
    phantom: PhantomData<T>,
}

impl<const L: usize, T> Registers<L, T> {
    pub fn registers<'a>(&self, arena: &'a [T]) -> &'a [T; L] {
        let start = self.start as usize;

        arena[start..(start + L)].try_into().unwrap()
    }
}

pub struct Signature<const I: usize, const O: usize> {
    pub input: Registers<I, Slot>,
    pub output: Registers<O, u32>,
}

pub struct DynSignature {
    input: u32,
    output: u32,
    len: u32,
}

impl DynSignature {
    pub fn input_registers<'a>(&self, arena: &'a [Slot]) -> &'a [Slot] {
        let start = self.input as usize;

        &arena[start..(start + self.len as usize)]
    }

    pub fn output_registers<'a>(&self, arena: &'a [u32]) -> &'a [u32] {
        let start = self.output as usize;

        &arena[start..(start + self.len as usize)]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Where the operands recorded by [`SimulatedStack::pops_and_pushes_registers`]
/// landed in the flat arenas.
///
/// `#[must_use]` because dropping it means the entries just written are
/// unreferenced: the arenas ship inside [`FrameLayout`], so a discarded result is
/// dead weight in every compiled function. A caller that only needs the stack and
/// register bookkeeping wants [`SimulatedStack::pops_and_pushes`], which does the
/// same thing without touching the arenas.
#[must_use]
struct PopsPushesResult {
    input_start: u32,
    output_start: u32,
}

#[derive(Default)]
struct ControlStack {
    stack: Vec<Block>,
}

impl ControlStack {
    fn len(&self) -> usize {
        self.stack.len()
    }
}

struct UnreachableTrackingControlStack {
    blocks: Vec<BlockVariant>,
    unreachable: bool,
}

enum UnreachableCheckResult {
    Continue,
    Reachable,
}

impl UnreachableTrackingControlStack {
    fn new() -> Self {
        UnreachableTrackingControlStack {
            blocks: vec![],
            unreachable: false,
        }
    }

    fn set_unreachable(&mut self) {
        self.unreachable = true;
    }

    fn unset_unreachable(&mut self) {
        self.unreachable = false;
    }

    fn add_block(&mut self, block: BlockVariant) {
        self.blocks.push(block);
    }

    fn pop_block(&mut self) -> BlockVariant {
        self.blocks.pop().unwrap()
    }

    fn check_unreachablity(&mut self, operator: &Operator<'_>) -> UnreachableCheckResult {
        if !self.unreachable {
            return UnreachableCheckResult::Reachable;
        }

        if let Some(block) = Self::is_block(operator) {
            self.add_block(block);

            UnreachableCheckResult::Continue
        } else if Self::is_else(operator) {
            if self.is_empty() {
                self.unset_unreachable();

                UnreachableCheckResult::Reachable
            } else {
                debug_assert!(matches!(self.blocks.last().unwrap(), BlockVariant::If));

                UnreachableCheckResult::Continue
            }
        } else if Self::is_end(operator) {
            if self.is_empty() {
                self.unset_unreachable();

                UnreachableCheckResult::Reachable
            } else {
                self.pop_block();

                UnreachableCheckResult::Continue
            }
        } else {
            UnreachableCheckResult::Continue
        }
    }

    fn is_block(operator: &Operator<'_>) -> Option<BlockVariant> {
        match operator {
            Operator::Block { .. } => Some(BlockVariant::Block),
            Operator::If { .. } => Some(BlockVariant::If),
            Operator::Loop { .. } => Some(BlockVariant::Loop),
            _ => None,
        }
    }

    fn is_else(operator: &Operator<'_>) -> bool {
        matches!(operator, Operator::Else)
    }

    fn is_end(operator: &Operator<'_>) -> bool {
        matches!(operator, Operator::End)
    }

    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

struct BrTarget {
    mov: DynSignature,
    target_index: u32,
}

struct SimulatedStack {
    stack: Stack<StackSlot>,
    curr_register_index: usize,
    max_registers: u32,
    lazy_locals: LazyArena<Local>,
    lazy_globals: LazyArena<Global>,
    spills: SpillArena,
    input_registers: Vec<Slot>,
    output_registers: Vec<u32>,
    control_stack: ControlStack,
    br_targets: Vec<BrTarget>,
}

impl SimulatedStack {
    fn new(locals_count: u32, globals_count: u32) -> Self {
        SimulatedStack {
            stack: Stack::new_with_capacity(0),
            curr_register_index: 0,
            max_registers: 0,
            lazy_locals: LazyArena::new(locals_count),
            lazy_globals: LazyArena::new(globals_count),
            spills: SpillArena::default(),
            input_registers: vec![],
            output_registers: vec![],
            control_stack: ControlStack::default(),
            br_targets: vec![],
        }
    }

    fn advanced_register_index(&mut self) {
        self.curr_register_index += 1;

        if self.curr_register_index as u32 > self.max_registers {
            self.max_registers += 1;
        }
    }

    fn recede_register_index(&mut self) {
        self.curr_register_index -= 1;
    }

    fn add_block(
        &mut self,
        kind: BlockVariant,
        blockty: &BlockType,
        types: &[FuncType],
        instr_len: usize,
    ) -> (u32, u32) {
        let (params, results) = params_and_results_from_blockty(blockty, types);

        let kind = match kind {
            BlockVariant::Func => BlockKind::Func,
            BlockVariant::Block => BlockKind::Block,
            BlockVariant::If => BlockKind::If {
                index: if params != 0 {
                    instr_len + 1 // move instruction is emitted if params != 0 so the actuall instruction lands at `len + 1`
                } else {
                    instr_len
                } as u32,
                else_index: None,
            },
            BlockVariant::Loop => BlockKind::Loop {
                index: if params != 0 {
                    instr_len + 1 // see above.
                } else {
                    instr_len
                } as u32,
            },
        };

        let recorded_height = match kind {
            BlockKind::Func => 0,
            BlockKind::Block { .. } => self.stack.height() - params,
            BlockKind::Loop { .. } => self.stack.height() - params,
            BlockKind::If { .. } => {
                // top is the `if` condition and then params
                self.stack.height() - params - 1
            }
        };

        self.control_stack.stack.push(Block {
            kind,
            recorded_height,
            params,
            results,
            attached_breaks: vec![],

            // below two fields are not used in register lowering!
            // they are just placeholders
            is_unreachable_traversing: false,
            has_inherited: false,
        });

        (params, results)
    }

    fn get_curr_block(&self) -> &Block {
        debug_assert!(!self.control_stack.stack.is_empty());

        &self.control_stack.stack[self.control_stack.stack.len() - 1]
    }

    fn get_curr_block_mut(&mut self) -> &mut Block {
        debug_assert!(!self.control_stack.stack.is_empty());
        let len = self.control_stack.stack.len();

        &mut self.control_stack.stack[len - 1]
    }

    fn get_block(&self, index: usize) -> &Block {
        &self.control_stack.stack[index]
    }

    fn get_block_mut(&mut self, index: usize) -> &mut Block {
        &mut self.control_stack.stack[index]
    }

    fn pop_lazy<T>(
        slot: LazySlot<T>,
        arena: &mut LazyArena<T>,
        spills: &mut SpillArena,
    ) -> LazyLocation {
        let location = slot.location(&arena);

        if matches!(slot.decrease_ref_count(arena), LazyEntryDropResult::Dropped) {
            match location {
                LazyLocation::Original(local_index) => arena.origin[local_index as usize] = None,
                LazyLocation::Spilled(spill_index) => spills.free_slot(spill_index),
            }
        }

        location
    }

    fn push_lazy<T>(location: u32, arena: &mut LazyArena<T>) -> LazySlot<T> {
        let slot = match arena.origin[location as usize] {
            Some(slot) => {
                slot.advanced_ref_count(arena);

                slot
            }
            None => {
                let slot = arena.allocate(location);
                arena.origin[location as usize] = Some(slot);

                slot
            }
        };

        slot
    }

    fn pop(&mut self) -> Slot {
        let val = self.stack.pop();

        let slot = match val {
            StackSlot::Const(val) => Slot::Const(val),
            StackSlot::Register(val) => {
                self.recede_register_index();

                Slot::Register(val)
            }
            StackSlot::Local(slot) => {
                let location = Self::pop_lazy(slot, &mut self.lazy_locals, &mut self.spills);

                match location {
                    LazyLocation::Original(local_index) => Slot::Local(local_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
            StackSlot::Global(slot) => {
                let location = Self::pop_lazy(slot, &mut self.lazy_globals, &mut self.spills);

                match location {
                    LazyLocation::Original(global_index) => Slot::Global(global_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
        };

        slot
    }

    fn simulated_pop(&self, depth: u32) -> Slot {
        let val = *self.stack.peek_from_top(depth);

        let slot = match val {
            StackSlot::Const(val) => Slot::Const(val),
            StackSlot::Register(val) => Slot::Register(val),
            StackSlot::Local(slot) => {
                let location = slot.location(&self.lazy_locals);

                match location {
                    LazyLocation::Original(local_index) => Slot::Local(local_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
            StackSlot::Global(slot) => {
                let location = slot.location(&self.lazy_globals);

                match location {
                    LazyLocation::Original(global_index) => Slot::Global(global_index),
                    LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
                }
            }
        };

        slot
    }

    fn push(&mut self, val: Slot) {
        let slot = match val {
            Slot::Const(val) => StackSlot::Const(val),
            Slot::Register(val) => {
                self.advanced_register_index();

                StackSlot::Register(val)
            }
            Slot::Local(index) => {
                let slot = Self::push_lazy(index, &mut self.lazy_locals);

                StackSlot::Local(slot)
            }
            Slot::Global(index) => {
                let slot = Self::push_lazy(index, &mut self.lazy_globals);

                StackSlot::Global(slot)
            }
            Slot::Spilled(_) => unreachable!("spilled slots are never produced for push!"),
        };

        self.stack.push(slot);
    }

    fn tee(&self) -> Slot {
        let top_slot = &self.stack.top();

        match top_slot {
            StackSlot::Const(val) => Slot::Const(*val),
            StackSlot::Register(val) => Slot::Register(*val),
            StackSlot::Local(slot) => match slot.location(&self.lazy_locals) {
                LazyLocation::Original(local_index) => Slot::Local(local_index),
                LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
            },
            StackSlot::Global(slot) => match slot.location(&self.lazy_globals) {
                LazyLocation::Original(global_index) => Slot::Global(global_index),
                LazyLocation::Spilled(spill_index) => Slot::Spilled(spill_index),
            },
        }
    }

    fn push_const(&mut self, val: Const) {
        self.push(Slot::Const(val));
    }

    fn push_local(&mut self, index: u32) {
        self.push(Slot::Local(index));
    }

    fn push_global(&mut self, index: u32) {
        self.push(Slot::Global(index));
    }

    fn pops_and_pushes_registers(&mut self, pops: u32, pushes: u32) -> PopsPushesResult {
        let pops = pops as usize;
        let pushes = pushes as usize;
        let input_start = self.input_registers.len();
        let output_start = self.output_registers.len();

        self.input_registers
            .resize(input_start + pops, Slot::default());

        for i in 0..pops {
            self.input_registers[input_start + pops - 1 - i] = self.pop();
        }

        self.output_registers.resize(output_start + pushes, 0);

        for i in 0..pushes {
            self.output_registers[output_start + i] = self.curr_register_index as u32;
            let out = Slot::Register(self.curr_register_index as u32);

            self.push(out);
        }

        PopsPushesResult {
            input_start: input_start as u32,
            output_start: output_start as u32,
        }
    }

    fn pops_and_pushes(&mut self, pops: u32, pushes: u32) {
        let pops = pops as usize;
        let pushes = pushes as usize;

        for _ in 0..pops {
            self.pop();
        }

        for _ in 0..pushes {
            let out = Slot::Register(self.curr_register_index as u32);

            self.push(out);
        }
    }

    fn registers_for<const I: usize, const O: usize>(&mut self) -> Signature<I, O> {
        let result = self.pops_and_pushes_registers(I as u32, O as u32);

        Signature {
            input: Registers {
                start: result.input_start as u32,
                phantom: PhantomData,
            },
            output: Registers {
                start: result.output_start as u32,
                phantom: PhantomData,
            },
        }
    }

    fn materialize_stack_slots_in_registers(&mut self, depth: u32) -> DynSignature {
        let result = self.pops_and_pushes_registers(depth, depth);

        DynSignature {
            input: result.input_start,
            output: result.output_start,
            len: depth,
        }
    }

    fn br_truncation_registers(
        &mut self,
        base_height: u32,
        arity_to_preserve: u32,
    ) -> DynSignature {
        let input_start = self.input_registers.len();
        let output_start = self.output_registers.len();
        let arity_to_preserve = arity_to_preserve as usize;
        let curr_stack_height = self.stack.height();
        let popped_count = (curr_stack_height - base_height) as usize;
        let mut register_index = self.curr_register_index as u32;

        self.input_registers
            .resize(input_start + arity_to_preserve, Slot::default());

        self.output_registers
            .resize(output_start + arity_to_preserve, 0);

        for i in 0..popped_count {
            let slot = self.simulated_pop(i as u32);

            if slot.is_register() {
                register_index -= 1;
            }

            if i < arity_to_preserve {
                self.input_registers[input_start + arity_to_preserve - 1 - i] = slot;
            }
        }

        // output registers for the branch results
        for i in 0..arity_to_preserve {
            self.output_registers[output_start + i] = register_index;
            register_index += 1;
        }

        if register_index > self.max_registers {
            self.max_registers = register_index;
        }

        DynSignature {
            input: input_start as u32,
            output: output_start as u32,
            len: arity_to_preserve as u32,
        }
    }

    fn set_lazy<T>(
        location: u32,
        arena: &mut LazyArena<T>,
        spills: &mut SpillArena,
    ) -> Option<u32> {
        let Some(slot) = arena.origin[location as usize] else {
            return None;
        };

        let spill_index = spills.reserve_slot();

        slot.spill(spill_index, arena);
        arena.origin[location as usize] = None;

        Some(spill_index)
    }
}

/// The storage one lowered body needs, in slot counts.
///
/// Both fields are high-water marks over the whole body rather than counts at any
/// one point, so a frame sized to them never has to grow mid-execution.
///
/// **Operands only.** Locals are not counted here: like the stack pass's
/// [`max_height`](crate::instruction::stack), these are measured from the frame's
/// operand base, so a consumer laying out storage needs
/// `locals_len + registers + spills`.
pub struct FrameLayout {
    /// Operand registers, i.e. the peak `curr_register_index`.
    pub registers: u32,
    /// Spill slots holding locals and globals rescued from a later write by
    /// [`RegInstruction::LocalSpill`] / [`RegInstruction::GlobalSpill`].
    ///
    /// Zero for a body that never overwrites a lazily-forwarded local or global,
    /// which is the common case.
    pub spills: u32,
    pub input_registers_arena: Box<[Slot]>,
    pub output_registers_arena: Box<[u32]>,
}

/// The two outputs of lowering one function body into register form: the
/// instruction list, and the frame required to execute it.
pub type LoweredRegFuncBody = (Vec<RegInstruction>, FrameLayout);

pub enum RegInstruction {
    I32Load {
        offset: u32,
        sig: Signature<1, 1>,
    },
    GlobalSet {
        index: GlobalIndex,
        sig: Signature<1, 0>,
    },
    LocalSet {
        index: LocalIndex,
        sig: Signature<1, 0>,
    },
    LocalTee {
        index: LocalIndex,
        sig: Signature<1, 0>,
    },
    I32Store {
        offset: u32,
        sig: Signature<2, 0>,
    },
    LocalSpill {
        index: LocalIndex,
        spill_index: u32,
    },
    GlobalSpill {
        index: GlobalIndex,
        spill_index: u32,
    },
    If {
        cond: Signature<1, 0>,
        else_index: Option<u32>,
        end_index: u32,
    },
    Else {
        end_index: u32,
    },
    Br {
        target_index: u32,
    },
    BrIf {
        cond: Registers<1, Slot>,
        mov: DynSignature,
        target_index: u32,
    },
    BrTable {
        index: Registers<1, Slot>,
        targets_start: u32,
        targets_len: u32,
    },
    I32Add(Signature<2, 1>),
    I32Eqz(Signature<1, 1>),
    Select(Signature<3, 1>),
    /// Copies each input slot into the register named by the output at the same
    /// position, materializing block params and results so every path into a label
    /// leaves its values in the same registers.
    ///
    /// **Must be executed as a gather-then-scatter, not an in-place copy.**
    ///
    /// Two things rule out `copy_within` or any single-pass loop:
    ///
    /// * The inputs are not all registers. A [`Slot`] reads from a constant, the
    ///   locals array, the globals table, the spill area, or the register file, so
    ///   this gathers from five places into one contiguous destination range rather
    ///   than moving a block within one slice.
    ///
    /// * The register-sourced inputs *overlap* the destinations, and the two callers
    ///   need opposite copy directions — so there is no ordering that is correct for
    ///   both.
    ///
    /// `materialize_stack_slots_in_registers` pops `depth` slots and pushes `depth`
    /// registers from the same base, so a source's index never exceeds its own
    /// destination. Slots `[Const(5), Register(b)]` into `[b, b + 1]`:
    ///
    /// ```text
    /// ascending :  reg[b]   = 5           ;  reg[b+1] = reg[b]  -> reads 5   WRONG
    /// descending:  reg[b+1] = reg[b]  ok  ;  reg[b]   = 5       ok
    /// ```
    ///
    /// `br_truncation_registers` discards the slots between the target's base and the
    /// branch operands, which shifts the operands *down*, so there a source index can
    /// exceed its destination. Stack `[Register(b), Register(b + 1), Const]` above the
    /// base with `arity == 2` gives inputs `[Register(b + 1), Const]` into `[b, b + 1]`:
    ///
    /// ```text
    /// ascending :  reg[b]   = reg[b+1] ok ;  reg[b+1] = C       ok
    /// descending:  reg[b+1] = C           ;  reg[b]   = reg[b+1] -> reads C  WRONG
    /// ```
    ///
    /// Reading every input before writing any output is correct for both, and stays
    /// correct if the allocation pattern changes — neither direction rule is visible
    /// from the instruction itself, so relying on one is a trap for the next reader:
    ///
    /// ```text
    /// let mut tmp: SmallVec<[Value; 4]> = SmallVec::new();
    /// for slot in sig.input_registers(arena) { tmp.push(read(slot)); }
    /// for (i, &dst) in sig.output_registers(arena).iter().enumerate() { regs[dst] = tmp[i]; }
    /// ```
    ///
    /// The buffer costs nothing in practice: arities are the label's params or
    /// results, which are one or two values for anything rustc emits.
    Move(DynSignature),
}

// One `RegInstruction` per lowered operator, so this size is multiplied across every
// compiled module — the same budget, and the same reasoning, as `Instruction` in the
// stack pass.
//
// What holds it here is that operands live in the flat side tables rather than in the
// variant: a `Registers<I, O>` is a pair of `u32` starts (8 bytes) whatever `I` and
// `O` are, so the widest variant is `I32Load(u32, Registers<1, 1>)` at 12 bytes plus
// tag. Inlining the operands instead would put `Select(Registers<3, 1>)` alone at 56.
//
// The constraint this places on what comes next: an instruction whose arity is not a
// compile-time constant — `call`, `call_indirect`, the block param/result moves — must
// stay within the same 8-byte shape. Either derive both arities at execution from an
// index the variant already carries (as `CallIndirect` does with its `ty_index` in the
// stack pass), or store an explicit `len` and drop something else to pay for it.
const _: () = assert!(
    size_of::<RegInstruction>() <= 24,
    "RegInstruction grew past 24 bytes. Need to keep it compact."
);

impl RegInstruction {
    pub fn emit_instruction_for_func(
        mut operator_reader: OperatorsReader<'_>,
        params: u32,
        results: u32,
        types: &[FuncType],
        func_decls: &[FuncDecl],
        locals_count: u32,
        globals_count: u32,
    ) -> Result<LoweredRegFuncBody, TraceWasmError> {
        let mut instructions: Vec<RegInstruction> = vec![];
        let mut simulated_stack = SimulatedStack::new(locals_count, globals_count);
        let mut unreachable_tracking_stack = UnreachableTrackingControlStack::new();

        simulated_stack.control_stack.stack.push(Block {
            kind: BlockKind::Func,
            recorded_height: 0, // functions always have recorded height to be 0, so they leave stack with just its results
            params,
            results,
            is_unreachable_traversing: false,
            has_inherited: false,
            attached_breaks: vec![],
        });

        while !operator_reader.eof() {
            let (operator, offset) = operator_reader.read_with_offset()?;

            if !matches!(
                unreachable_tracking_stack.check_unreachablity(&operator),
                UnreachableCheckResult::Reachable
            ) {
                continue;
            }

            match operator {
                Operator::GlobalGet { global_index } => {
                    simulated_stack.push_global(global_index);
                }
                Operator::GlobalSet { global_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        global_index,
                        &mut simulated_stack.lazy_globals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::GlobalSpill {
                            index: GlobalIndex(global_index),
                            spill_index,
                        });
                    }

                    let registers = simulated_stack.registers_for::<1, 0>();

                    instructions.push(RegInstruction::GlobalSet {
                        index: GlobalIndex(global_index),
                        sig: registers,
                    });
                }
                Operator::LocalGet { local_index } => {
                    simulated_stack.push_local(local_index);
                }
                Operator::LocalSet { local_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        local_index,
                        &mut simulated_stack.lazy_locals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::LocalSpill {
                            index: LocalIndex(local_index),
                            spill_index,
                        });
                    }

                    let registers = simulated_stack.registers_for::<1, 0>();

                    instructions.push(RegInstruction::LocalSet {
                        index: LocalIndex(local_index),
                        sig: registers,
                    });
                }
                Operator::LocalTee { local_index } => {
                    if let Some(spill_index) = SimulatedStack::set_lazy(
                        local_index,
                        &mut simulated_stack.lazy_locals,
                        &mut simulated_stack.spills,
                    ) {
                        instructions.push(RegInstruction::LocalSpill {
                            index: LocalIndex(local_index),
                            spill_index,
                        });
                    }

                    let input_start = simulated_stack.input_registers.len();

                    simulated_stack.input_registers.push(simulated_stack.tee());

                    let registers = Signature {
                        input: Registers {
                            start: input_start as u32,
                            phantom: PhantomData,
                        },
                        output: Registers {
                            start: simulated_stack.output_registers.len() as u32,
                            phantom: PhantomData,
                        },
                    };

                    instructions.push(RegInstruction::LocalTee {
                        index: LocalIndex(local_index),
                        sig: registers,
                    });
                }
                Operator::I32Const { value } => {
                    simulated_stack.push_const(Const::I32(value));
                }
                Operator::I32Load { memarg } => {
                    let registers = simulated_stack.registers_for::<1, 1>();

                    instructions.push(RegInstruction::I32Load {
                        offset: memarg.offset as u32,
                        sig: registers,
                    });
                }
                Operator::I32Store { memarg } => {
                    let registers = simulated_stack.registers_for::<2, 0>();

                    instructions.push(RegInstruction::I32Store {
                        offset: memarg.offset as u32,
                        sig: registers,
                    });
                }
                Operator::I32Add => {
                    let registers = simulated_stack.registers_for::<2, 1>();

                    instructions.push(RegInstruction::I32Add(registers));
                }
                Operator::I32Eqz => {
                    let registers = simulated_stack.registers_for::<1, 1>();

                    instructions.push(RegInstruction::I32Eqz(registers));
                }
                Operator::Nop => {
                    continue;
                }
                Operator::Select => {
                    let registers = simulated_stack.registers_for::<3, 1>();

                    instructions.push(RegInstruction::Select(registers));
                }
                Operator::Drop => {
                    simulated_stack.pop();

                    continue;
                }
                Operator::Block { blockty } => {
                    let (block_params, _) = simulated_stack.add_block(
                        BlockVariant::Block,
                        &blockty,
                        types,
                        instructions.len(),
                    );

                    if block_params != 0 {
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(block_params);

                        instructions.push(RegInstruction::Move(move_registers));
                    }
                }
                Operator::Loop { blockty } => {
                    let (block_params, _) = simulated_stack.add_block(
                        BlockVariant::Loop,
                        &blockty,
                        types,
                        instructions.len(),
                    );

                    if block_params != 0 {
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(block_params);

                        instructions.push(RegInstruction::Move(move_registers));
                    }
                }
                Operator::If { blockty } => {
                    // the simulated stack would have layout like this at `if` instruction: [...other...][...params...][cond]
                    // to obtain recorded_height, we should pop params + 1 number of stack slots, and measure the `curr_register_index`
                    // which will be the recorded_height. So after the `end` of this `if` we should leave the stack
                    // at: recorded_height + results. We should materalize all the popped values by pushing it back on the stack
                    // making them materialized in registers. Same would be for results. So no matter what branch control flow takes
                    // the layout of frame in the start of the instruction and at the end of the instruction is same.

                    let (block_params, _) = simulated_stack.add_block(
                        BlockVariant::If,
                        &blockty,
                        types,
                        instructions.len(),
                    );

                    if block_params != 0 {
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(block_params + 1);

                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    instructions.push(RegInstruction::If {
                        cond: simulated_stack.registers_for::<1, 0>(),
                        else_index: None,
                        end_index: u32::MAX,
                    });
                }
                Operator::Else => {
                    let if_block = simulated_stack.get_curr_block_mut();
                    let recorded_height = if_block.recorded_height;
                    let block_params = if_block.params;
                    let block_results = if_block.results;

                    let BlockKind::If {
                        index: _,
                        else_index,
                    } = &mut if_block.kind
                    else {
                        unreachable!(
                            "hitting this means TraceWasm has a bug recording the instructions"
                        )
                    };

                    *else_index = Some(if block_results == 0 {
                        instructions.len() as u32
                    } else {
                        instructions.len() as u32 + 1 // mov instruction also emitted!
                    });

                    // materialize the results produced by the if arm.
                    // The else instruction is only reached by the if arm, if the condition is false
                    // then the pc is jumped directly to the first instruction of the else arm skipping
                    // the else instruction itself.
                    if block_results != 0 {
                        let move_registers =
                            simulated_stack.materialize_stack_slots_in_registers(block_results);

                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    // reset the frame layout with params on top for else instructions.
                    simulated_stack.pops_and_pushes(
                        simulated_stack.stack.height() - recorded_height,
                        block_params,
                    );

                    instructions.push(RegInstruction::Else {
                        end_index: u32::MAX,
                    });
                }
                Operator::Br { relative_depth } => {
                    let enclosing_block = simulated_stack.get_curr_block();
                    let enclosing_block_recorded_height = enclosing_block.recorded_height;
                    let enclosing_block_results = enclosing_block.results;

                    let block_index =
                        simulated_stack.control_stack.len() - 1 - relative_depth as usize;
                    let block = simulated_stack.get_block(block_index);
                    let params = block.params;
                    let results = block.results;
                    let recorded_height = block.recorded_height;

                    let (move_registers, target_index) =
                        if let Some(loop_index) = block.kind.is_loop() {
                            (
                                simulated_stack.br_truncation_registers(recorded_height, params),
                                loop_index,
                            )
                        } else {
                            let move_registers =
                                simulated_stack.br_truncation_registers(recorded_height, results);

                            simulated_stack
                                .get_block_mut(block_index)
                                .attached_breaks
                                .push((
                                    if move_registers.is_empty() {
                                        instructions.len() as u32
                                    } else {
                                        instructions.len() as u32 + 1
                                    },
                                    u32::MAX,
                                ));

                            (move_registers, u32::MAX)
                        };

                    if !move_registers.is_empty() {
                        instructions.push(RegInstruction::Move(move_registers));
                    }

                    instructions.push(RegInstruction::Br { target_index });

                    // set the layout correctly to the current enclosing block so that instructions
                    // after else or end would see correct layout as all the instructions between br and else/end
                    // are unreachable and stack is freezed.
                    simulated_stack.pops_and_pushes(
                        simulated_stack.stack.height() - enclosing_block_recorded_height,
                        enclosing_block_results,
                    );

                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::BrIf { relative_depth } => {
                    let block_index =
                        simulated_stack.control_stack.len() - 1 - relative_depth as usize;
                    let block = simulated_stack.get_block(block_index);
                    let params = block.params;
                    let results = block.results;
                    let recorded_height = block.recorded_height;
                    let block_kind = block.kind;

                    let cond = simulated_stack.registers_for::<1, 0>().input;

                    let (move_registers, target_index) =
                        if let Some(loop_index) = block_kind.is_loop() {
                            (
                                simulated_stack.br_truncation_registers(recorded_height, params),
                                loop_index,
                            )
                        } else {
                            let move_registers =
                                simulated_stack.br_truncation_registers(recorded_height, results);

                            simulated_stack
                                .get_block_mut(block_index)
                                .attached_breaks
                                .push((instructions.len() as u32, u32::MAX));

                            (move_registers, u32::MAX)
                        };

                    instructions.push(RegInstruction::BrIf {
                        cond,
                        mov: move_registers,
                        target_index,
                    });
                }
                Operator::BrTable { targets: table } => {
                    let enclosing_block = simulated_stack.get_curr_block();
                    let enclosing_block_recorded_height = enclosing_block.recorded_height;
                    let enclosing_block_results = enclosing_block.results;
                    let targets_start = simulated_stack.br_targets.len() as u32;
                    let mut targets_len = 0;

                    let targets = table.targets();
                    let mut targets = targets.collect::<Result<Vec<_>, _>>()?;

                    targets.push(table.default());

                    let table_index = simulated_stack.registers_for::<1, 0>().input; // targets index

                    for (i, &relative_depth) in targets.iter().enumerate() {
                        let block_index =
                            simulated_stack.control_stack.len() - 1 - relative_depth as usize;
                        let block = simulated_stack.get_block_mut(block_index);
                        let params = block.params;
                        let results = block.results;
                        let recorded_height = block.recorded_height;
                        let block_kind = block.kind;

                        let (move_registers, target_index) = if let Some(loop_index) =
                            block_kind.is_loop()
                        {
                            (
                                simulated_stack.br_truncation_registers(recorded_height, params),
                                loop_index,
                            )
                        } else {
                            let move_registers =
                                simulated_stack.br_truncation_registers(recorded_height, results);

                            simulated_stack
                                .get_block_mut(block_index)
                                .attached_breaks
                                .push((instructions.len() as u32, targets_start + i as u32));

                            (move_registers, u32::MAX)
                        };

                        let br_target = BrTarget {
                            mov: move_registers,
                            target_index,
                        };

                        simulated_stack.br_targets.push(br_target);

                        targets_len += 1;
                    }

                    instructions.push(RegInstruction::BrTable {
                        index: table_index,
                        targets_start,
                        targets_len,
                    });

                    // set the layout correctly to the current enclosing block so that instructions
                    // after else or end would see correct layout as all the instructions between br and else/end
                    // are unreachable and stack is freezed.
                    simulated_stack.pops_and_pushes(
                        simulated_stack.stack.height() - enclosing_block_recorded_height,
                        enclosing_block_results,
                    );

                    unreachable_tracking_stack.set_unreachable();
                }
                Operator::Return => {
                    todo!()
                }
                Operator::Call { function_index } => todo!(),
                Operator::CallIndirect {
                    type_index,
                    table_index,
                } => todo!(),
                Operator::End => {
                    // emit mov instruction for setting the layout correctly for the branch coming from
                    // just before this end.
                    //
                    // pop the control block, backpatch all the entries just like stack/mod.rs
                    //
                    // there are 3 ways to reach end.
                    // - from a br instruction -> no need for materializing the result registers, br already does that
                    // - from an instruction previous to it, for that it requires emitting mov before end so that it first
                    // executes mov instruction and then end
                    // - from a br resolved to a different block and instructions following it became unreachable! in this case
                    // too the registers are materialized by the reset function.
                    // Any branch which is not coming from just above always land directly to end instruction, they materialize
                    // the results.
                    todo!()
                }
                _ => {
                    return Err(TraceWasmError::Unsupported(format!(
                        "instruction `{:?}`",
                        operator
                    )));
                }
            }
        }

        // Both counts are already high-water marks — `max_registers` is maintained
        // by `advanced_register_index` and `allocation_len` only grows when no
        // freed spill slot can be reused — so they are read off directly here
        // rather than recomputed from the instruction list.
        let frame = FrameLayout {
            // Bounded by the operand-stack depth, which a function body's size in
            // the binary already bounds well below `u32::MAX`, so this cannot
            // truncate for any module that could be loaded at all.
            registers: simulated_stack.max_registers as u32,
            spills: simulated_stack.spills.allocation_len(),
            input_registers_arena: simulated_stack.input_registers.into_boxed_slice(),
            output_registers_arena: simulated_stack.output_registers.into_boxed_slice(),
        };

        Ok((instructions, frame))
    }
}
