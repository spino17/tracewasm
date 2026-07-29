//! Lowering of a WebAssembly operator stream into TraceWasm's flat instruction list.
//!
//! `Instruction::emit_instruction_for_func` (function bodies) and
//! `Instruction::emit_instruction_for_const_expr` (constant expressions) each
//! consume a [`wasmparser::OperatorsReader`] and produce a `Vec<Instruction>` in
//! which **structured control
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
    module::{FuncDecl, FuncIndex, FuncType, GlobalIndex, LocalIndex, TableIndex, ValType},
};
use wasmparser::{BlockType, Operator, OperatorsReader};

/// A lowered TraceWasm instruction.
///
/// `wasmparser` operators are translated into this owned form by
/// [`Instruction::emit_instruction_for_func`] (function bodies) and
/// [`Instruction::emit_instruction_for_const_expr`] (constant expressions); any
/// operator TraceWasm does not model is rejected as unsupported at lowering time.
/// Index fields (`end_index`, `else_index`, `target_index`, ...) are *absolute*
/// positions into the containing `Vec<Instruction>`, i.e. runtime program
/// counters.
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Traps unconditionally.
    Unreachable,
    /// Does nothing.
    Nop,
    /// `i32.const`: push an immediate `i32`.
    I32Const {
        /// The constant value pushed onto the stack.
        value: i32,
    },
    /// `i64.const`: push an immediate `i64`.
    I64Const {
        /// The constant value pushed onto the stack.
        value: i64,
    },
    /// `f32.const`: push an immediate `f32`, bit pattern preserved exactly.
    F32Const {
        /// The constant value pushed onto the stack.
        value: f32,
    },
    /// `f64.const`: push an immediate `f64`, bit pattern preserved exactly.
    F64Const {
        /// The constant value pushed onto the stack.
        value: f64,
    },
    /// `ref.null`: push a null reference.
    RefNull,
    /// `ref.func`: push a reference to the given function.
    RefFunc {
        /// Index of the function whose reference is pushed.
        function_index: FuncIndex,
    },
    /// `memory.size`: push the memory's current size in pages.
    MemorySize,
    /// `memory.grow`: pop a page delta and grow the memory, pushing the size
    /// *before* the growth.
    ///
    /// Does **not** trap when the request cannot be satisfied — it pushes `-1` and
    /// execution continues. The ceiling is the module's declared maximum, narrowed
    /// by the instance [`Config`](crate::instance::config::Config).
    MemoryGrow,
    /// `memory.copy`: pop `len`, `src`, `dest` and copy within linear memory.
    ///
    /// The ranges may overlap (`memmove` semantics). Traps if either runs past the
    /// end of memory, with nothing written.
    MemoryCopy,
    /// `memory.fill`: pop `len`, `value`, `dest` and set the range to the low byte
    /// of `value`. Traps if the range runs past the end of memory.
    MemoryFill,
    /// `memory.init`: pop `len`, `src`, `dest` and copy from a passive data
    /// segment into linear memory.
    ///
    /// Traps if the source range exceeds the segment or the destination exceeds
    /// memory. A segment already released by [`Self::DataDrop`] reads as empty, so
    /// a zero-length init still succeeds.
    MemoryInit {
        /// Index of the data segment to copy from.
        data_index: u32,
    },
    /// `data.drop`: release a passive data segment's bytes.
    ///
    /// The segment becomes empty rather than invalid, so a later
    /// [`Self::MemoryInit`] traps only if it asks for a non-empty range. Dropping
    /// twice is harmless.
    DataDrop {
        /// Index of the data segment to release.
        data_index: u32,
    },
    // Loads. Every variant pops an address and pushes one value read from
    // `address + offset` (little-endian); the narrow `*_u`/`*_s` forms read fewer
    // bytes than the result type and zero- / sign-extend to it. `offset` is the
    // static `memarg` byte offset added to the popped address; `align` is the
    // alignment hint (log2), which is validation-only — the interpreter ignores
    // it, since unaligned access is permitted.
    /// `i32.load`: load 4 bytes as the `i32` result.
    I32Load {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i32.load8_u`: load 1 byte, zero-extend to `i32`.
    I32Load8U {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i32.load8_s`: load 1 byte, sign-extend to `i32`.
    I32Load8S {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i32.load16_u`: load 2 bytes, zero-extend to `i32`.
    I32Load16U {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i32.load16_s`: load 2 bytes, sign-extend to `i32`.
    I32Load16S {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.load`: load 8 bytes as the `i64` result.
    I64Load {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.load8_u`: load 1 byte, zero-extend to `i64`.
    I64Load8U {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.load8_s`: load 1 byte, sign-extend to `i64`.
    I64Load8S {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.load16_u`: load 2 bytes, zero-extend to `i64`.
    I64Load16U {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.load16_s`: load 2 bytes, sign-extend to `i64`.
    I64Load16S {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.load32_u`: load 4 bytes, zero-extend to `i64`.
    I64Load32U {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.load32_s`: load 4 bytes, sign-extend to `i64`.
    I64Load32S {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `f32.load`: load 4 bytes as the `f32` result, preserving the exact bit
    /// pattern (no NaN canonicalization).
    F32Load {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `f64.load`: load 8 bytes as the `f64` result, preserving the exact bit
    /// pattern (no NaN canonicalization).
    F64Load {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    // Stores. Every variant pops the value then the address (the value is pushed
    // last, so it sits on top) and writes to `address + offset` little-endian.
    // `offset` and `align` carry the same meaning as for the loads above.
    /// `i32.store`: pop an `i32` value and an address, write the value's 4 bytes.
    I32Store {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i32.store8`: write the low 1 byte of the popped `i32` (wrapping).
    I32Store8 {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i32.store16`: write the low 2 bytes of the popped `i32` (wrapping).
    I32Store16 {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.store`: pop an `i64` value and an address, write the value's 8 bytes.
    I64Store {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.store8`: write the low 1 byte of the popped `i64` (wrapping).
    I64Store8 {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.store16`: write the low 2 bytes of the popped `i64` (wrapping).
    I64Store16 {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `i64.store32`: write the low 4 bytes of the popped `i64` (wrapping).
    I64Store32 {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `f32.store`: write the popped `f32`'s 4 bytes, preserving the exact bit
    /// pattern (no NaN canonicalization).
    F32Store {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    /// `f64.store`: write the popped `f64`'s 8 bytes, preserving the exact bit
    /// pattern (no NaN canonicalization).
    F64Store {
        /// Static byte offset added to the popped address.
        offset: u64,
        /// Alignment hint (log2); ignored at execution.
        align: u8,
    },
    // Unary bit-counting operators: each pops one value and pushes a count of the
    // same type as its operand (not an `i32`, unlike the comparisons). All three
    // are total — `clz`/`ctz` of `0` return the full operand width.
    /// `i32.clz`: count leading zero bits;
    /// `32` when the operand is `0`.
    I32Clz,
    /// `i32.ctz`: count trailing zero bits;
    /// `32` when the operand is `0`.
    I32Ctz,
    /// `i32.popcnt`: count set bits, over the two's-complement representation,
    /// so a negative operand counts its sign bits too.
    I32Popcnt,
    /// `i32.eqz`: `1` if the operand is zero, else `0`.
    ///
    /// Unlike the bit-counting operators above, the result is an `i32` even for
    /// the `i64` form — it is a comparison against zero, so it follows the
    /// comparison convention.
    I32Eqz,
    // The `extendN_s` operators below reinterpret the low bits of the operand as a
    // signed value of that narrower width, then widen back. Each keeps the
    // operand's type; none of them traps.
    /// `i32.extend8_s`: sign-extend the low 8 bits to `i32`.
    I32Extend8S,
    /// `i32.extend16_s`: sign-extend the low 16 bits to `i32`.
    I32Extend16S,
    /// `i32.wrap_i64`: narrow an `i64` to an `i32` by keeping its low 32 bits.
    ///
    /// The inverse direction of `i64.extend_i32_*`, and the one conversion that
    /// discards information rather than adding it. Out-of-range operands wrap
    /// rather than trapping, so `0x1_0000_0000` becomes `0`.
    I32WrapI64,
    // The `trunc` operators round their float operand toward zero and reinterpret
    // it as an integer. Unlike every other conversion here they can **trap**: a
    // NaN, an infinity, or a value that truncates outside the target's range has
    // no representation, so it fails rather than being clamped. (The `trunc_sat`
    // family is the clamping counterpart.)
    /// `i32.trunc_f32_u`: truncate an `f32` to an unsigned 32-bit integer.
    I32TruncF32U,
    /// `i32.trunc_f32_s`: truncate an `f32` to a signed 32-bit integer.
    I32TruncF32S,
    /// `i32.trunc_f64_u`: truncate an `f64` to an unsigned 32-bit integer.
    I32TruncF64U,
    /// `i32.trunc_f64_s`: truncate an `f64` to a signed 32-bit integer.
    I32TruncF64S,
    /// `i32.add`.
    I32Add,
    /// `i32.sub`.
    I32Sub,
    /// `i32.mul`.
    I32Mul,
    /// `i32.div_u`: unsigned division. Traps on a zero divisor.
    I32DivU,
    /// `i32.div_s`: signed division, truncating toward zero. Traps on a zero
    /// divisor **and** on overflow (`i32::MIN / -1`, whose quotient is not
    /// representable).
    I32DivS,
    /// `i32.rem_u`: unsigned remainder. Traps on a zero divisor.
    I32RemU,
    /// `i32.rem_s`: signed remainder, taking the sign of the dividend.
    ///
    /// Traps *only* on a zero divisor — unlike [`Self::I32DivS`] it does not trap
    /// on overflow: `i32::MIN % -1` is defined as `0`.
    I32RemS,
    /// `i32.and`: bitwise AND.
    I32And,
    /// `i32.or`: bitwise OR.
    I32Or,
    /// `i32.xor`: bitwise XOR.
    I32Xor,
    // Shifts and rotates take their count modulo the operand width, so a count of
    // 32 or more wraps rather than being an error — none of these can trap.
    /// `i32.shl`: shift left by `count mod 32`.
    I32Shl,
    /// `i32.shr_u`: logical shift right by `count mod 32`, shifting in zeros.
    I32ShrU,
    /// `i32.shr_s`: arithmetic shift right by `count mod 32`, replicating the sign
    /// bit.
    I32ShrS,
    /// `i32.rotl`: rotate left by `count mod 32`.
    I32Rotl,
    /// `i32.rotr`: rotate right by `count mod 32`.
    I32Rotr,
    // Comparisons. Each pops two operands and pushes an **`i32`** 0/1 — the result
    // is an `i32` even for the `i64` forms, so it can feed `br_if`/`select`
    // directly. `eq`/`ne` compare bit patterns, so they need no signed/unsigned
    // split; the ordered comparisons do, and the two disagree whenever an operand
    // has its high bit set (`-1` is the largest value unsigned).
    /// `i32.eq`: equality.
    I32Eq,
    /// `i32.ne`: inequality.
    I32Ne,
    /// `i32.lt_u`: unsigned less-than.
    I32LtU,
    /// `i32.lt_s`: signed less-than.
    I32LtS,
    /// `i32.gt_u`: unsigned greater-than.
    I32GtU,
    /// `i32.gt_s`: signed greater-than.
    I32GtS,
    /// `i32.le_u`: unsigned less-than-or-equal.
    I32LeU,
    /// `i32.le_s`: signed less-than-or-equal.
    I32LeS,
    /// `i32.ge_u`: unsigned greater-than-or-equal.
    I32GeU,
    /// `i32.ge_s`: signed greater-than-or-equal.
    I32GeS,
    /// `i64.clz`: count leading zero bits;
    /// `64` when the operand is `0`.
    I64Clz,
    /// `i64.ctz`: count trailing zero bits;
    /// `64` when the operand is `0`.
    I64Ctz,
    /// `i64.popcnt`: count set bits, over the two's-complement representation,
    /// so a negative operand counts its sign bits too.
    I64Popcnt,
    /// `i64.eqz`: `1` if the operand is zero, else `0`.
    ///
    /// Unlike the bit-counting operators above, the result is an `i32` even for
    /// the `i64` form — it is a comparison against zero, so it follows the
    /// comparison convention.
    I64Eqz,
    /// `i64.extend8_s`: sign-extend the low 8 bits to `i64`.
    I64Extend8S,
    /// `i64.extend16_s`: sign-extend the low 16 bits to `i64`.
    I64Extend16S,
    /// `i64.extend32_s`: sign-extend the low 32 bits to `i64`.
    I64Extend32S,
    /// `i64.extend_i32_u`: widen an `i32` to `i64` by **zero**-extending, so `-1`
    /// becomes `4294967295` rather than staying `-1`.
    I64ExtendI32U,
    /// `i64.extend_i32_s`: widen an `i32` to `i64` by **sign**-extending, so `-1`
    /// stays `-1`.
    I64ExtendI32S,
    /// `i64.trunc_f32_u`: truncate an `f32` to an unsigned 64-bit integer.
    ///
    /// Traps on unrepresentable operands, as the other `trunc` operators do.
    I64TruncF32U,
    /// `i64.trunc_f32_s`: truncate an `f32` to a signed 64-bit integer.
    I64TruncF32S,
    /// `i64.trunc_f64_u`: truncate an `f64` to an unsigned 64-bit integer.
    I64TruncF64U,
    /// `i64.trunc_f64_s`: truncate an `f64` to a signed 64-bit integer.
    I64TruncF64S,
    /// `i64.add`.
    I64Add,
    /// `i64.sub`.
    I64Sub,
    /// `i64.mul`.
    I64Mul,
    /// `i64.div_u`: unsigned division. Traps on a zero divisor.
    I64DivU,
    /// `i64.div_s`: signed division, truncating toward zero. Traps on a zero
    /// divisor **and** on overflow (`i64::MIN / -1`).
    I64DivS,
    /// `i64.rem_u`: unsigned remainder. Traps on a zero divisor.
    I64RemU,
    /// `i64.rem_s`: signed remainder. Traps *only* on a zero divisor;
    /// `i64::MIN % -1` is `0`, not an overflow. See [`Self::I32RemS`].
    I64RemS,
    /// `i64.and`: bitwise AND.
    I64And,
    /// `i64.or`: bitwise OR.
    I64Or,
    /// `i64.xor`: bitwise XOR.
    I64Xor,
    /// `i64.shl`: shift left by `count mod 64`.
    I64Shl,
    /// `i64.shr_u`: logical shift right by `count mod 64`, shifting in zeros.
    I64ShrU,
    /// `i64.shr_s`: arithmetic shift right by `count mod 64`, replicating the sign
    /// bit.
    I64ShrS,
    /// `i64.rotl`: rotate left by `count mod 64`.
    I64Rotl,
    /// `i64.rotr`: rotate right by `count mod 64`.
    I64Rotr,
    /// `i64.eq`: equality.
    I64Eq,
    /// `i64.ne`: inequality.
    I64Ne,
    /// `i64.lt_u`: unsigned less-than.
    I64LtU,
    /// `i64.lt_s`: signed less-than.
    I64LtS,
    /// `i64.gt_u`: unsigned greater-than.
    I64GtU,
    /// `i64.gt_s`: signed greater-than.
    I64GtS,
    /// `i64.le_u`: unsigned less-than-or-equal.
    I64LeU,
    /// `i64.le_s`: signed less-than-or-equal.
    I64LeS,
    /// `i64.ge_u`: unsigned greater-than-or-equal.
    I64GeU,
    /// `i64.ge_s`: signed greater-than-or-equal.
    I64GeS,
    // Float unary operators. All are total — none traps — and each preserves the
    // exact bit pattern where IEEE 754 says to, so a NaN operand yields a NaN
    // rather than being canonicalized.
    /// `f32.abs`: clear the sign bit. Applies to NaN and `-0.0` too.
    F32Abs,
    /// `f32.neg`: flip the sign bit. A sign flip, not `0 - x`, so `-(-0.0)`
    /// is `+0.0` and a NaN keeps its payload.
    F32Neg,
    /// `f32.ceil`: round up to an integral value.
    F32Ceil,
    /// `f32.floor`: round down to an integral value.
    F32Floor,
    /// `f32.trunc`: round toward zero.
    F32Trunc,
    /// `f32.sqrt`: square root; NaN for a negative operand, and `-0.0` for
    /// `-0.0`.
    F32Sqrt,
    /// `f32.nearest`: round to the nearest integral value, ties to **even**.
    ///
    /// Note this is *not* Rust's `round`, which breaks ties away from zero:
    /// `2.5` rounds to `2.0` here, but to `3.0` under `round`.
    F32Nearest,
    // Float arithmetic follows IEEE 754 exactly, which is what Rust's `f32`/`f64`
    // operators already provide. These never trap: overflow yields an infinity and
    // a NaN operand yields a NaN, whose payload the spec leaves nondeterministic.
    /// `f32.add`.
    F32Add,
    /// `f32.sub`.
    F32Sub,
    /// `f32.mul`.
    F32Mul,
    /// `f32.div`: unlike the integer divides this never traps — dividing by zero
    /// yields `±inf`, and `0.0 / 0.0` yields NaN.
    F32Div,
    // Float comparisons. Like the integer ones these push an **`i32`** 0/1, but the
    // ordering is IEEE 754 rather than two's complement: a NaN operand makes every
    // ordered comparison (and `eq`) false while making `ne` true, and `-0.0`
    // compares *equal* to `+0.0`. Rust's operators already have exactly these
    // semantics — unlike `min`/`max`, where they diverge from wasm.
    /// `f32.eq`: equality; false if either operand is NaN, true for `-0.0 == +0.0`.
    F32Eq,
    /// `f32.ne`: inequality; **true** if either operand is NaN.
    F32Ne,
    /// `f32.lt`: ordered less-than; false if either operand is NaN.
    F32Lt,
    /// `f32.gt`: ordered greater-than; false if either operand is NaN.
    F32Gt,
    /// `f32.le`: ordered less-than-or-equal; false if either operand is NaN.
    F32Le,
    /// `f32.ge`: ordered greater-than-or-equal; false if either operand is NaN.
    F32Ge,
    // `min`/`max` are the one place Rust's float methods do *not* match wasm:
    // `f32::min` returns the non-NaN operand where wasm returns NaN, and leaves
    // the `-0.0`/`+0.0` tie unspecified where wasm fixes it. Both are therefore
    // written out longhand in the interpreter rather than delegated.
    /// `f32.min`: the smaller operand.
    ///
    /// Returns NaN if *either* operand is NaN, and `-0.0` when the operands are
    /// `-0.0` and `+0.0` (which compare equal).
    F32Min,
    /// `f32.max`: the larger operand.
    ///
    /// Returns NaN if *either* operand is NaN, and `+0.0` for the
    /// `-0.0`/`+0.0` tie.
    F32Max,
    /// `f32.copysign`: the magnitude of the first operand with the sign of
    /// the second. A pure sign-bit transplant, so it is defined for NaN too and
    /// never traps.
    F32Copysign,
    /// `f64.abs`: clear the sign bit. Applies to NaN and `-0.0` too.
    F64Abs,
    /// `f64.neg`: flip the sign bit. A sign flip, not `0 - x`, so `-(-0.0)`
    /// is `+0.0` and a NaN keeps its payload.
    F64Neg,
    /// `f64.ceil`: round up to an integral value.
    F64Ceil,
    /// `f64.floor`: round down to an integral value.
    F64Floor,
    /// `f64.trunc`: round toward zero.
    F64Trunc,
    /// `f64.sqrt`: square root; NaN for a negative operand, and `-0.0` for
    /// `-0.0`.
    F64Sqrt,
    /// `f64.nearest`: round to the nearest integral value, ties to **even**.
    ///
    /// Note this is *not* Rust's `round`, which breaks ties away from zero:
    /// `2.5` rounds to `2.0` here, but to `3.0` under `round`.
    F64Nearest,
    /// `f64.add`.
    F64Add,
    /// `f64.sub`.
    F64Sub,
    /// `f64.mul`.
    F64Mul,
    /// `f64.div`: never traps; see [`Self::F32Div`].
    F64Div,
    /// `f64.eq`: equality; false if either operand is NaN, true for `-0.0 == +0.0`.
    F64Eq,
    /// `f64.ne`: inequality; **true** if either operand is NaN.
    F64Ne,
    /// `f64.lt`: ordered less-than; false if either operand is NaN.
    F64Lt,
    /// `f64.gt`: ordered greater-than; false if either operand is NaN.
    F64Gt,
    /// `f64.le`: ordered less-than-or-equal; false if either operand is NaN.
    F64Le,
    /// `f64.ge`: ordered greater-than-or-equal; false if either operand is NaN.
    F64Ge,
    /// `f64.min`: the smaller operand.
    ///
    /// Returns NaN if *either* operand is NaN, and `-0.0` when the operands are
    /// `-0.0` and `+0.0` (which compare equal).
    F64Min,
    /// `f64.max`: the larger operand.
    ///
    /// Returns NaN if *either* operand is NaN, and `+0.0` for the
    /// `-0.0`/`+0.0` tie.
    F64Max,
    /// `f64.copysign`: the magnitude of the first operand with the sign of
    /// the second. A pure sign-bit transplant, so it is defined for NaN too and
    /// never traps.
    F64Copysign,
    /// `local.get`: push the value of the local at `index`.
    LocalGet {
        /// Index of the local (params first, then declared locals).
        index: LocalIndex,
    },
    /// `local.set`: pop a value and store it into the local at `index`.
    LocalSet {
        /// Index of the local (params first, then declared locals).
        index: LocalIndex,
    },
    /// `local.tee`: store the top value into the local at `index`, leaving it on
    /// the stack.
    LocalTee {
        /// Index of the local (params first, then declared locals).
        index: LocalIndex,
    },
    /// `global.get`: push the value of the global at `index`.
    GlobalGet {
        /// Index into the module's global index space.
        index: GlobalIndex,
    },
    /// `global.set`: pop a value and store it into the (mutable) global at `index`.
    GlobalSet {
        /// Index into the module's global index space.
        index: GlobalIndex,
    },
    /// `call`: pop the callee's arguments and invoke it directly.
    ///
    /// An imported callee is dispatched to the host registry; a local one is
    /// interpreted recursively on the shared operand stack.
    Call {
        /// Index of the callee in the function index space.
        func_index: FuncIndex,
        /// Number of arguments the callee pops from the stack.
        params_count: u32,
    },
    /// `call_indirect`: pop a table index, resolve it to a function reference,
    /// and call it.
    ///
    /// Traps if the index is out of the table's bounds, the slot is null, or the
    /// callee's signature differs from the type recorded here — that last check is
    /// why the expected signature travels with the instruction.
    CallIndirect {
        /// The callee signature's parameter types.
        params: Box<[ValType]>,
        /// The callee signature's result types.
        results: Box<[ValType]>,
        /// Table holding the callee function references.
        table_index: TableIndex,
    },
    /// `drop`: discards the top
    Drop,
    /// `select`: pop cond, then b, then a; push cond != 0 ? a : b -> standard in LLVM
    Select,
    /// Opens a block. Purely a label: entering one does nothing at runtime, but a
    /// branch targeting it jumps forward to its `End`.
    Block {
        /// Absolute index of this block's matching `End`. Backpatched; a branch
        /// that targets this block jumps here.
        end_index: usize,
    },
    /// Opens a loop. Branches targeting a loop jump back to this instruction
    /// (the loop start), so no `end` index is needed.
    Loop,
    /// `if`: pop a condition and fall through when it is non-zero, otherwise jump
    /// to the `else` branch (or past the `end` when there is none).
    If {
        /// Absolute index of the matching `Else`, if one exists. Backpatched at
        /// `End`.
        else_index: Option<usize>,
        /// Absolute index of this `if`'s matching `End`. Backpatched.
        end_index: usize,
    },
    /// Reached only by falling out of a taken then-branch, which must skip the
    /// else-branch entirely — control jumps straight to the owning `if`'s `End`.
    ///
    /// A *false* condition never lands here: `If` jumps past this instruction to
    /// the first instruction of the else-branch.
    Else {
        /// Absolute index of the owning `if`'s `End`. When the then-branch falls
        /// through into `else`, control skips to this `End`. Backpatched.
        if_end_index: usize,
    },
    /// `br`: unconditional branch to an enclosing label, unwinding the stack to
    /// that label's height while preserving the top `arity` values.
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
    /// `br_if`: pop a condition and branch as [`Self::Br`] when it is non-zero;
    /// otherwise fall through.
    BrIf {
        /// See `Br::target_index`. Same target rules as `Br`.
        target_index: usize,
        /// Number of values transferred to the label; see `Br::arity`.
        arity: u32,
        /// Stack height the target label unwinds to; see `Block::recorded_height`.
        recorded_height: u32,
    },
    /// `br_table`: pop an index and branch to that arm, falling back to the
    /// default when it is out of range.
    ///
    /// The index is unsigned, so a negative value selects the default rather than
    /// wrapping to a valid arm.
    BrTable {
        /// One [`TargetBranch`] per explicit label, in order, followed by the
        /// default label as the final element.
        targets: Vec<TargetBranch>,
    },
    /// `return`: branch to the function's outermost label, leaving the frame's
    /// results on the stack.
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
    /// Closes a block, `if`, or the function body, resetting the stack to the
    /// label's height plus its results.
    ///
    /// Advancing past the function's final `End` is what ends the frame.
    End {
        /// Number of result values the just-closed block leaves on the stack.
        arity: u32,
        /// Height the stack unwinds to before the `arity` results are kept. See
        /// `Block::recorded_height`.
        recorded_height: u32,
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
    /// Number of values transferred to the label (loop params, else results).
    pub arity: u32,
    /// Stack height the target label unwinds to; see `Block::recorded_height`.
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

    /// Applies an operator's net stack effect as the single expression
    /// `curr_height = curr_height - pops + pushes`.
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
    PopPush {
        pops: u32,
        pushes: u32,
    },
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
    /// Shorthand for `PopPush { pops: 1, pushes: 1 }`: a load pops its address and
    /// pushes the value read, so the height is unchanged.
    Loads,
    /// Shorthand for `PopPush { pops: 2, pushes: 0 }`: a store pops the value and
    /// the address, pushing nothing.
    Stores,
    /// Shorthand for `PopPush { pops: 2, pushes: 1 }`: a binary operator pops both
    /// operands and pushes the single result. Covers the arithmetic, bitwise, and
    /// comparison operators — the comparisons included, since they push an `i32`
    /// boolean rather than nothing.
    BinaryOperator,
    UnaryOperator,
}

impl Instruction {
    fn check_memory_index(index: u32) -> Result<(), TraceWasmError> {
        if index != 0 {
            return Err(TraceWasmError::Unsupported(
                "more than one memory".to_string(),
            ));
        }

        Ok(())
    }

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
    ///
    /// Returns the lowered instructions alongside a parallel vector of source
    /// offsets: element `i` is the byte offset in the module binary of the
    /// operator that produced instruction `i`. The two are pushed together on
    /// every iteration, so they always have the same length and indexing.
    pub(crate) fn emit_instruction_for_func(
        mut operator_reader: OperatorsReader<'_>,
        params: u32,
        results: u32,
        types: &[FuncType],
        func_decls: &[FuncDecl],
    ) -> Result<(Vec<Instruction>, Vec<u32>), TraceWasmError> {
        let mut instructions: Vec<Instruction> = vec![];
        let mut instruction_offsets: Vec<u32> = vec![];
        let mut control_stack: ControlStack = ControlStack::default();

        control_stack.inner.push(Block {
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

            let (instruction, stack_effect): (Instruction, StackEffectResult) = match operator {
                Operator::Unreachable => {
                    // all instructions after this is unreachable until the end of the current block
                    control_stack.set_unreachable_traversing();

                    (Instruction::Unreachable, StackEffectResult::NoEffect)
                }
                Operator::Nop => (Instruction::Nop, StackEffectResult::NoEffect),
                // constants
                Operator::I32Const { value } => (
                    Instruction::I32Const { value },
                    StackEffectResult::PopPush { pops: 0, pushes: 1 },
                ),
                Operator::I64Const { value } => (
                    Instruction::I64Const { value },
                    StackEffectResult::PopPush { pops: 0, pushes: 1 },
                ),
                Operator::F32Const { value } => (
                    Instruction::F32Const {
                        value: f32::from_bits(value.bits()),
                    },
                    StackEffectResult::PopPush { pops: 0, pushes: 1 },
                ),
                Operator::F64Const { value } => (
                    Instruction::F64Const {
                        value: f64::from_bits(value.bits()),
                    },
                    StackEffectResult::PopPush { pops: 0, pushes: 1 },
                ),
                Operator::RefNull { hty: _hty } => (
                    Instruction::RefNull,
                    StackEffectResult::PopPush { pops: 0, pushes: 1 },
                ),
                Operator::RefFunc { function_index } => (
                    Instruction::RefFunc {
                        function_index: FuncIndex(function_index),
                    },
                    StackEffectResult::PopPush { pops: 0, pushes: 1 },
                ),
                // memory
                Operator::MemorySize { mem } => {
                    Self::check_memory_index(mem)?;

                    (
                        Instruction::MemorySize,
                        StackEffectResult::PopPush { pops: 0, pushes: 1 },
                    )
                }
                Operator::MemoryGrow { mem } => {
                    Self::check_memory_index(mem)?;

                    (
                        Instruction::MemoryGrow,
                        StackEffectResult::PopPush { pops: 1, pushes: 1 },
                    )
                }
                Operator::MemoryCopy { dst_mem, src_mem } => {
                    Self::check_memory_index(dst_mem)?;
                    Self::check_memory_index(src_mem)?;

                    (
                        Instruction::MemoryCopy,
                        StackEffectResult::PopPush { pops: 3, pushes: 0 },
                    )
                }
                Operator::MemoryFill { mem } => {
                    Self::check_memory_index(mem)?;

                    (
                        Instruction::MemoryFill,
                        StackEffectResult::PopPush { pops: 3, pushes: 0 },
                    )
                }
                Operator::MemoryInit { data_index, mem } => {
                    Self::check_memory_index(mem)?;

                    (
                        Instruction::MemoryInit { data_index },
                        StackEffectResult::PopPush { pops: 3, pushes: 0 },
                    )
                }
                Operator::DataDrop { data_index } => (
                    Instruction::DataDrop { data_index },
                    StackEffectResult::NoEffect,
                ),
                // loads
                Operator::I32Load { memarg } => (
                    Instruction::I32Load {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I32Load8U { memarg } => (
                    Instruction::I32Load8U {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I32Load8S { memarg } => (
                    Instruction::I32Load8S {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I32Load16U { memarg } => (
                    Instruction::I32Load16U {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I32Load16S { memarg } => (
                    Instruction::I32Load16S {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I64Load { memarg } => (
                    Instruction::I64Load {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I64Load8U { memarg } => (
                    Instruction::I64Load8U {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I64Load8S { memarg } => (
                    Instruction::I64Load8S {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I64Load16U { memarg } => (
                    Instruction::I64Load16U {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I64Load16S { memarg } => (
                    Instruction::I64Load16S {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I64Load32U { memarg } => (
                    Instruction::I64Load32U {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::I64Load32S { memarg } => (
                    Instruction::I64Load32S {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::F32Load { memarg } => (
                    Instruction::F32Load {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                Operator::F64Load { memarg } => (
                    Instruction::F64Load {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Loads,
                ),
                // stores
                Operator::I32Store { memarg } => (
                    Instruction::I32Store {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Stores,
                ),
                Operator::I32Store8 { memarg } => (
                    Instruction::I32Store8 {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Stores,
                ),
                Operator::I32Store16 { memarg } => (
                    Instruction::I32Store16 {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Stores,
                ),
                Operator::I64Store { memarg } => (
                    Instruction::I64Store {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Stores,
                ),
                Operator::I64Store8 { memarg } => (
                    Instruction::I64Store8 {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Stores,
                ),
                Operator::I64Store16 { memarg } => (
                    Instruction::I64Store16 {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Stores,
                ),
                Operator::I64Store32 { memarg } => (
                    Instruction::I64Store32 {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Stores,
                ),
                Operator::F32Store { memarg } => (
                    Instruction::F32Store {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Stores,
                ),
                Operator::F64Store { memarg } => (
                    Instruction::F64Store {
                        offset: memarg.offset,
                        align: memarg.align,
                    },
                    StackEffectResult::Stores,
                ),
                // i32 unary operations
                Operator::I32Clz => (Instruction::I32Clz, StackEffectResult::UnaryOperator),
                Operator::I32Ctz => (Instruction::I32Ctz, StackEffectResult::UnaryOperator),
                Operator::I32Popcnt => (Instruction::I32Popcnt, StackEffectResult::UnaryOperator),
                Operator::I32Eqz => (Instruction::I32Eqz, StackEffectResult::UnaryOperator),
                Operator::I32Extend8S => {
                    (Instruction::I32Extend8S, StackEffectResult::UnaryOperator)
                }
                Operator::I32Extend16S => {
                    (Instruction::I32Extend16S, StackEffectResult::UnaryOperator)
                }
                Operator::I32WrapI64 => (Instruction::I32WrapI64, StackEffectResult::UnaryOperator),
                Operator::I32TruncF32U => {
                    (Instruction::I32TruncF32U, StackEffectResult::UnaryOperator)
                }
                Operator::I32TruncF32S => {
                    (Instruction::I32TruncF32S, StackEffectResult::UnaryOperator)
                }
                Operator::I32TruncF64U => {
                    (Instruction::I32TruncF64U, StackEffectResult::UnaryOperator)
                }
                Operator::I32TruncF64S => {
                    (Instruction::I32TruncF64S, StackEffectResult::UnaryOperator)
                }
                // i32 binary operations
                Operator::I32Add => (Instruction::I32Add, StackEffectResult::BinaryOperator),
                Operator::I32Sub => (Instruction::I32Sub, StackEffectResult::BinaryOperator),
                Operator::I32Mul => (Instruction::I32Mul, StackEffectResult::BinaryOperator),
                Operator::I32DivU => (Instruction::I32DivU, StackEffectResult::BinaryOperator),
                Operator::I32DivS => (Instruction::I32DivS, StackEffectResult::BinaryOperator),
                Operator::I32RemU => (Instruction::I32RemU, StackEffectResult::BinaryOperator),
                Operator::I32RemS => (Instruction::I32RemS, StackEffectResult::BinaryOperator),
                Operator::I32And => (Instruction::I32And, StackEffectResult::BinaryOperator),
                Operator::I32Or => (Instruction::I32Or, StackEffectResult::BinaryOperator),
                Operator::I32Xor => (Instruction::I32Xor, StackEffectResult::BinaryOperator),
                Operator::I32Shl => (Instruction::I32Shl, StackEffectResult::BinaryOperator),
                Operator::I32ShrU => (Instruction::I32ShrU, StackEffectResult::BinaryOperator),
                Operator::I32ShrS => (Instruction::I32ShrS, StackEffectResult::BinaryOperator),
                Operator::I32Rotl => (Instruction::I32Rotl, StackEffectResult::BinaryOperator),
                Operator::I32Rotr => (Instruction::I32Rotr, StackEffectResult::BinaryOperator),
                Operator::I32Eq => (Instruction::I32Eq, StackEffectResult::BinaryOperator),
                Operator::I32Ne => (Instruction::I32Ne, StackEffectResult::BinaryOperator),
                Operator::I32LtU => (Instruction::I32LtU, StackEffectResult::BinaryOperator),
                Operator::I32LtS => (Instruction::I32LtS, StackEffectResult::BinaryOperator),
                Operator::I32GtU => (Instruction::I32GtU, StackEffectResult::BinaryOperator),
                Operator::I32GtS => (Instruction::I32GtS, StackEffectResult::BinaryOperator),
                Operator::I32LeU => (Instruction::I32LeU, StackEffectResult::BinaryOperator),
                Operator::I32LeS => (Instruction::I32LeS, StackEffectResult::BinaryOperator),
                Operator::I32GeU => (Instruction::I32GeU, StackEffectResult::BinaryOperator),
                Operator::I32GeS => (Instruction::I32GeS, StackEffectResult::BinaryOperator),
                // i64 unary operations
                Operator::I64Clz => (Instruction::I64Clz, StackEffectResult::UnaryOperator),
                Operator::I64Ctz => (Instruction::I64Ctz, StackEffectResult::UnaryOperator),
                Operator::I64Popcnt => (Instruction::I64Popcnt, StackEffectResult::UnaryOperator),
                Operator::I64Eqz => (Instruction::I64Eqz, StackEffectResult::UnaryOperator),
                Operator::I64Extend8S => {
                    (Instruction::I64Extend8S, StackEffectResult::UnaryOperator)
                }
                Operator::I64Extend16S => {
                    (Instruction::I64Extend16S, StackEffectResult::UnaryOperator)
                }
                Operator::I64Extend32S => {
                    (Instruction::I64Extend32S, StackEffectResult::UnaryOperator)
                }
                Operator::I64ExtendI32U => {
                    (Instruction::I64ExtendI32U, StackEffectResult::UnaryOperator)
                }
                Operator::I64ExtendI32S => {
                    (Instruction::I64ExtendI32S, StackEffectResult::UnaryOperator)
                }
                Operator::I64TruncF32U => {
                    (Instruction::I64TruncF32U, StackEffectResult::UnaryOperator)
                }
                Operator::I64TruncF32S => {
                    (Instruction::I64TruncF32S, StackEffectResult::UnaryOperator)
                }
                Operator::I64TruncF64U => {
                    (Instruction::I64TruncF64U, StackEffectResult::UnaryOperator)
                }
                Operator::I64TruncF64S => {
                    (Instruction::I64TruncF64S, StackEffectResult::UnaryOperator)
                }
                // i64 binary operations
                Operator::I64Add => (Instruction::I64Add, StackEffectResult::BinaryOperator),
                Operator::I64Sub => (Instruction::I64Sub, StackEffectResult::BinaryOperator),
                Operator::I64Mul => (Instruction::I64Mul, StackEffectResult::BinaryOperator),
                Operator::I64DivU => (Instruction::I64DivU, StackEffectResult::BinaryOperator),
                Operator::I64DivS => (Instruction::I64DivS, StackEffectResult::BinaryOperator),
                Operator::I64RemU => (Instruction::I64RemU, StackEffectResult::BinaryOperator),
                Operator::I64RemS => (Instruction::I64RemS, StackEffectResult::BinaryOperator),
                Operator::I64And => (Instruction::I64And, StackEffectResult::BinaryOperator),
                Operator::I64Or => (Instruction::I64Or, StackEffectResult::BinaryOperator),
                Operator::I64Xor => (Instruction::I64Xor, StackEffectResult::BinaryOperator),
                Operator::I64Shl => (Instruction::I64Shl, StackEffectResult::BinaryOperator),
                Operator::I64ShrU => (Instruction::I64ShrU, StackEffectResult::BinaryOperator),
                Operator::I64ShrS => (Instruction::I64ShrS, StackEffectResult::BinaryOperator),
                Operator::I64Rotl => (Instruction::I64Rotl, StackEffectResult::BinaryOperator),
                Operator::I64Rotr => (Instruction::I64Rotr, StackEffectResult::BinaryOperator),
                Operator::I64Eq => (Instruction::I64Eq, StackEffectResult::BinaryOperator),
                Operator::I64Ne => (Instruction::I64Ne, StackEffectResult::BinaryOperator),
                Operator::I64LtU => (Instruction::I64LtU, StackEffectResult::BinaryOperator),
                Operator::I64LtS => (Instruction::I64LtS, StackEffectResult::BinaryOperator),
                Operator::I64GtU => (Instruction::I64GtU, StackEffectResult::BinaryOperator),
                Operator::I64GtS => (Instruction::I64GtS, StackEffectResult::BinaryOperator),
                Operator::I64LeU => (Instruction::I64LeU, StackEffectResult::BinaryOperator),
                Operator::I64LeS => (Instruction::I64LeS, StackEffectResult::BinaryOperator),
                Operator::I64GeU => (Instruction::I64GeU, StackEffectResult::BinaryOperator),
                Operator::I64GeS => (Instruction::I64GeS, StackEffectResult::BinaryOperator),
                // f32 unary operations
                Operator::F32Abs => (Instruction::F32Abs, StackEffectResult::UnaryOperator),
                Operator::F32Neg => (Instruction::F32Neg, StackEffectResult::UnaryOperator),
                Operator::F32Ceil => (Instruction::F32Ceil, StackEffectResult::UnaryOperator),
                Operator::F32Floor => (Instruction::F32Floor, StackEffectResult::UnaryOperator),
                Operator::F32Trunc => (Instruction::F32Trunc, StackEffectResult::UnaryOperator),
                Operator::F32Sqrt => (Instruction::F32Sqrt, StackEffectResult::UnaryOperator),
                Operator::F32Nearest => (Instruction::F32Nearest, StackEffectResult::UnaryOperator),
                // f32 binary operations
                Operator::F32Add => (Instruction::F32Add, StackEffectResult::BinaryOperator),
                Operator::F32Sub => (Instruction::F32Sub, StackEffectResult::BinaryOperator),
                Operator::F32Mul => (Instruction::F32Mul, StackEffectResult::BinaryOperator),
                Operator::F32Div => (Instruction::F32Div, StackEffectResult::BinaryOperator),
                Operator::F32Eq => (Instruction::F32Eq, StackEffectResult::BinaryOperator),
                Operator::F32Ne => (Instruction::F32Ne, StackEffectResult::BinaryOperator),
                Operator::F32Lt => (Instruction::F32Lt, StackEffectResult::BinaryOperator),
                Operator::F32Gt => (Instruction::F32Gt, StackEffectResult::BinaryOperator),
                Operator::F32Le => (Instruction::F32Le, StackEffectResult::BinaryOperator),
                Operator::F32Ge => (Instruction::F32Ge, StackEffectResult::BinaryOperator),
                Operator::F32Min => (Instruction::F32Min, StackEffectResult::BinaryOperator),
                Operator::F32Max => (Instruction::F32Max, StackEffectResult::BinaryOperator),
                Operator::F32Copysign => {
                    (Instruction::F32Copysign, StackEffectResult::BinaryOperator)
                }
                // f64 unary operations
                Operator::F64Abs => (Instruction::F64Abs, StackEffectResult::UnaryOperator),
                Operator::F64Neg => (Instruction::F64Neg, StackEffectResult::UnaryOperator),
                Operator::F64Ceil => (Instruction::F64Ceil, StackEffectResult::UnaryOperator),
                Operator::F64Floor => (Instruction::F64Floor, StackEffectResult::UnaryOperator),
                Operator::F64Trunc => (Instruction::F64Trunc, StackEffectResult::UnaryOperator),
                Operator::F64Sqrt => (Instruction::F64Sqrt, StackEffectResult::UnaryOperator),
                Operator::F64Nearest => (Instruction::F64Nearest, StackEffectResult::UnaryOperator),
                // f64 binary operations
                Operator::F64Add => (Instruction::F64Add, StackEffectResult::BinaryOperator),
                Operator::F64Sub => (Instruction::F64Sub, StackEffectResult::BinaryOperator),
                Operator::F64Mul => (Instruction::F64Mul, StackEffectResult::BinaryOperator),
                Operator::F64Div => (Instruction::F64Div, StackEffectResult::BinaryOperator),
                Operator::F64Eq => (Instruction::F64Eq, StackEffectResult::BinaryOperator),
                Operator::F64Ne => (Instruction::F64Ne, StackEffectResult::BinaryOperator),
                Operator::F64Lt => (Instruction::F64Lt, StackEffectResult::BinaryOperator),
                Operator::F64Gt => (Instruction::F64Gt, StackEffectResult::BinaryOperator),
                Operator::F64Le => (Instruction::F64Le, StackEffectResult::BinaryOperator),
                Operator::F64Ge => (Instruction::F64Ge, StackEffectResult::BinaryOperator),
                Operator::F64Min => (Instruction::F64Min, StackEffectResult::BinaryOperator),
                Operator::F64Max => (Instruction::F64Max, StackEffectResult::BinaryOperator),
                Operator::F64Copysign => {
                    (Instruction::F64Copysign, StackEffectResult::BinaryOperator)
                }
                // locals
                Operator::LocalGet { local_index } => (
                    Instruction::LocalGet {
                        index: LocalIndex(local_index),
                    },
                    StackEffectResult::PopPush { pops: 0, pushes: 1 },
                ),
                Operator::LocalSet { local_index } => (
                    Instruction::LocalSet {
                        index: LocalIndex(local_index),
                    },
                    StackEffectResult::PopPush { pops: 1, pushes: 0 },
                ),
                Operator::LocalTee { local_index } => (
                    Instruction::LocalTee {
                        index: LocalIndex(local_index),
                    },
                    StackEffectResult::NoEffect,
                ),
                // globals
                Operator::GlobalGet { global_index } => (
                    Instruction::GlobalGet {
                        index: GlobalIndex(global_index),
                    },
                    StackEffectResult::PopPush { pops: 0, pushes: 1 },
                ),
                Operator::GlobalSet { global_index } => (
                    Instruction::GlobalSet {
                        index: GlobalIndex(global_index),
                    },
                    StackEffectResult::PopPush { pops: 1, pushes: 0 },
                ),
                // call
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
                Operator::CallIndirect {
                    type_index,
                    table_index,
                } => {
                    let func_ty = &types[type_index as usize];
                    let params = func_ty.params.clone();
                    let results = func_ty.results.clone();

                    let stack_effect = StackEffectResult::PopPush {
                        pops: 1 + params.len() as u32,
                        pushes: results.len() as u32,
                    };

                    (
                        Instruction::CallIndirect {
                            params,
                            results,
                            table_index: TableIndex(table_index),
                        },
                        stack_effect,
                    )
                }
                Operator::Select => (
                    Instruction::Select,
                    StackEffectResult::PopPush { pops: 3, pushes: 1 },
                ),
                Operator::Drop => (
                    Instruction::Drop,
                    StackEffectResult::PopPush { pops: 1, pushes: 0 },
                ),
                // blocks
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
                // branching
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
                // end
                Operator::End => {
                    let Some(block) = control_stack.pop() else {
                        unreachable!(
                            "unbalanced block calculation! getting this means block tracking logic of TraceWasm is incorrect"
                        )
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
                StackEffectResult::Stores => {
                    control_stack.apply_stack_effects_to_height(2, 0);
                }
                StackEffectResult::BinaryOperator => {
                    control_stack.apply_stack_effects_to_height(2, 1);
                }
                StackEffectResult::SetHeight(height) => control_stack.set_height(height),
                // Loads and unary operators each pop one value and push one, so
                // like the genuinely effect-free cases they leave the height alone.
                StackEffectResult::NoEffect
                | StackEffectResult::Unreachable
                | StackEffectResult::UnaryOperator
                | StackEffectResult::Loads => {}
            }

            // Offsets are bounded by the module's byte length, so this cannot
            // lose information for any module that could be loaded at all.
            debug_assert!(u32::try_from(offset).is_ok(), "module larger than 4 GiB");

            // Pushed together to keep the two vectors index-aligned.
            instruction_offsets.push(offset as u32);
            instructions.push(instruction);
        }

        Ok((instructions, instruction_offsets))
    }

    pub(crate) fn emit_instruction_for_const_expr(
        mut operator_reader: OperatorsReader<'_>,
    ) -> Result<Vec<Instruction>, TraceWasmError> {
        let mut instructions = vec![];

        while !operator_reader.eof() {
            let operator = operator_reader.read()?;

            let instr = match operator {
                Operator::I32Const { value } => Instruction::I32Const { value },
                Operator::I64Const { value } => Instruction::I64Const { value },
                Operator::F32Const { value } => Instruction::F32Const {
                    value: f32::from_bits(value.bits()),
                },
                Operator::F64Const { value } => Instruction::F64Const {
                    value: f64::from_bits(value.bits()),
                },
                Operator::GlobalGet { global_index } => Instruction::GlobalGet {
                    index: GlobalIndex(global_index),
                },
                Operator::RefNull { hty: _hty } => Instruction::RefNull,
                Operator::RefFunc { function_index } => Instruction::RefFunc {
                    function_index: FuncIndex(function_index),
                },
                Operator::I32Add => Instruction::I32Add,
                Operator::I32Sub => Instruction::I32Sub,
                Operator::I32Mul => Instruction::I32Mul,
                Operator::I64Add => Instruction::I64Add,
                Operator::I64Sub => Instruction::I64Sub,
                Operator::I64Mul => Instruction::I64Mul,
                Operator::End => break,
                _ => {
                    return Err(TraceWasmError::Unsupported(format!(
                        "operator `{:?}` in const expression stream",
                        operator
                    )));
                }
            };

            instructions.push(instr);
        }

        Ok(instructions)
    }
}
