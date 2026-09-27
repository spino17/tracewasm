use tracewasm_llvm::{cfg::context::Context, interner::TyId};

use crate::{
    module::FuncIndex,
    runtime::value::{TableVal, Val},
};
use std::marker::PhantomData;

pub struct RuntimeInstance {
    globals: Box<[GlobalVal]>,
    tables: Box<[TableEntry]>,
    table_entries: Box<[OptionalU32]>,
}

impl RuntimeInstance {
    pub fn new(globals: Box<[Val]>, tables: Box<[TableVal]>) -> Self {
        let gbls = globals
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<GlobalVal>>()
            .into_boxed_slice();

        let mut table_entries: Vec<OptionalU32> = vec![];
        let mut indices = vec![];
        let mut initial_index = 0;

        for table in tables {
            let table = table.table;
            let table_len = table.len();

            indices.push((initial_index, table_len));
            table_entries.extend(table.into_iter().map(OptionalU32::from));

            initial_index += table_len;
        }

        let mut table_entries = table_entries.into_boxed_slice();
        let tables_base_ptr = table_entries.as_mut_ptr();
        let mut tables: Vec<TableEntry> = vec![];

        for (index, len) in indices {
            let entry = TableEntry {
                table_ptr: unsafe { tables_base_ptr.add(index) },
                table_len: len as u32,
            };

            tables.push(entry);
        }

        RuntimeInstance {
            globals: gbls,
            tables: tables.into_boxed_slice(),
            table_entries,
        }
    }

    pub fn ctx(&mut self) -> RuntimeContext<'_> {
        RuntimeContext {
            globals_ptr: self.globals.as_mut_ptr(),
            globals_len: self.globals.len() as u32,
            tables_ptr: self.tables.as_ptr(),
            tables_len: self.tables.len() as u32,
            phantom: PhantomData,
        }
    }

    pub fn call<R>(&mut self, f: unsafe extern "C" fn(*mut RuntimeContext) -> R) -> R {
        let mut ctx = self.ctx();

        unsafe { f(&mut ctx) }
    }

    pub fn llvm_ty(ctx: &mut Context) -> TyId {
        todo!()
    }
}

#[repr(C)]
pub struct RuntimeContext<'a> {
    globals_ptr: *mut GlobalVal,
    globals_len: u32,
    tables_ptr: *const TableEntry,
    tables_len: u32,
    phantom: PhantomData<&'a RuntimeInstance>,
}

/// One global's slot: an untyped payload plus the tag saying how to read it.
///
/// Generated code never reads the tag. A `global.get i` knows the type from
/// `Module::globals[i]`, and validation guarantees that is the type the operand
/// has — so the tag is for the host, which asks for a global's value without
/// knowing what it is.
#[repr(C)]
pub struct GlobalVal {
    val: u64,
    tag: u8,
}

impl GlobalVal {
    /// The payload is an `i32`, in the low four bytes.
    pub const TAG_I32: u8 = 0;
    /// The payload is an `i64`.
    pub const TAG_I64: u8 = 1;
    /// The payload is an `f32` bit pattern, in the low four bytes.
    pub const TAG_F32: u8 = 2;
    /// The payload is an `f64` bit pattern.
    pub const TAG_F64: u8 = 3;
    /// The payload is a function index.
    pub const TAG_FUNC_REF: u8 = 4;
    /// A null function reference. The payload is not meaningful.
    pub const TAG_NULL_REF: u8 = 5;
}

impl From<Val> for GlobalVal {
    /// Flattens a tagged [`Val`] into the slot the lowered code reads.
    ///
    /// Two things here are load-bearing, and nothing downstream would catch either
    /// getting them wrong:
    ///
    /// The payload goes in the **low** bytes, which is where a `load i32` or
    /// `load float` at the slot's address finds it on a little-endian target. It is
    /// the only arrangement the generated code can read, since that code loads the
    /// global's declared type directly rather than an `i64` it then narrows.
    ///
    /// Floats are stored **by bit pattern**. `v as u64` would round the value to an
    /// integer; [`f32::to_bits`] keeps the encoding, so a NaN payload and the sign of
    /// a negative zero both survive.
    fn from(value: Val) -> Self {
        let (val, tag) = match value {
            // `as u32` before widening. A negative `i32` cast straight to `u64`
            // sign-extends, filling the upper four bytes; nothing reads those back
            // for an `i32` global, but leaving them set makes two equal globals
            // differ byte-for-byte, which is a poor thing for a slot to do.
            Val::I32(v) => (v as u32 as u64, Self::TAG_I32),
            Val::I64(v) => (v as u64, Self::TAG_I64),
            Val::F32(v) => (v.to_bits() as u64, Self::TAG_F32),
            Val::F64(v) => (v.to_bits(), Self::TAG_F64),
            // Null is a tag rather than a reserved index, so every `u32` remains a
            // usable function index.
            Val::Ref(Some(FuncIndex(index))) => (index as u64, Self::TAG_FUNC_REF),
            Val::Ref(None) => (0, Self::TAG_NULL_REF),
        };

        GlobalVal { val, tag }
    }
}

#[repr(C)]
pub struct TableEntry {
    table_ptr: *mut OptionalU32,
    table_len: u32,
}

/// One table slot: a nullable function index, flattened for `repr(C)`.
///
/// `Option<FuncIndex>` has no layout guarantee — the niche optimisation that makes
/// it `u32`-sized is an implementation detail Rust does not promise across versions,
/// and generated code cannot GEP into a promise like that. Spelling the
/// discriminant out costs four bytes of padding and makes the layout something both
/// sides can agree on.
#[repr(C)]
pub struct OptionalU32 {
    val: u32,
    tag: u8,
}

impl OptionalU32 {
    /// A null reference. `val` is not meaningful.
    pub const TAG_NULL: u8 = 0;
    /// A function reference; `val` is its index.
    pub const TAG_SOME: u8 = 1;
}

impl From<Option<FuncIndex>> for OptionalU32 {
    /// A null slot zeroes the payload as well as the tag, so a table's bytes depend
    /// only on its contents — two tables holding the same references compare equal
    /// whatever the slots held before.
    fn from(value: Option<FuncIndex>) -> Self {
        match value {
            Some(FuncIndex(index)) => OptionalU32 {
                val: index,
                tag: Self::TAG_SOME,
            },
            None => OptionalU32 {
                val: 0,
                tag: Self::TAG_NULL,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout the generated code will GEP into.
    ///
    /// Pinned here because `llvm_ty` builds a second description of these same
    /// bytes, and a field added on one side without the other corrupts every access
    /// silently. This half is the one `repr(C)` fixes; asserting it means the two
    /// can be compared rather than assumed equal.
    #[test]
    fn the_slot_layouts_are_what_the_llvm_side_must_match() {
        use std::mem::{align_of, offset_of, size_of};

        assert_eq!((size_of::<GlobalVal>(), align_of::<GlobalVal>()), (16, 8));
        assert_eq!(offset_of!(GlobalVal, val), 0);
        assert_eq!(offset_of!(GlobalVal, tag), 8);

        assert_eq!(
            (size_of::<OptionalU32>(), align_of::<OptionalU32>()),
            (8, 4)
        );
        assert_eq!(offset_of!(OptionalU32, val), 0);
        assert_eq!(offset_of!(OptionalU32, tag), 4);

        assert_eq!((size_of::<TableEntry>(), align_of::<TableEntry>()), (16, 8));
        assert_eq!(offset_of!(TableEntry, table_ptr), 0);
        assert_eq!(offset_of!(TableEntry, table_len), 8);

        assert_eq!(
            (size_of::<RuntimeContext>(), align_of::<RuntimeContext>()),
            (32, 8)
        );
        assert_eq!(offset_of!(RuntimeContext, globals_ptr), 0);
        assert_eq!(offset_of!(RuntimeContext, globals_len), 8);
        assert_eq!(offset_of!(RuntimeContext, tables_ptr), 16);
        assert_eq!(offset_of!(RuntimeContext, tables_len), 24);
    }

    /// Each variant's payload and tag, spelled out.
    #[test]
    fn a_val_encodes_into_its_slot() {
        let cases = [
            (Val::I32(7), 7u64, GlobalVal::TAG_I32),
            (Val::I64(7), 7, GlobalVal::TAG_I64),
            (Val::F32(1.0), 0x3f80_0000, GlobalVal::TAG_F32),
            (Val::F64(1.0), 0x3ff0_0000_0000_0000, GlobalVal::TAG_F64),
            (Val::Ref(Some(FuncIndex(3))), 3, GlobalVal::TAG_FUNC_REF),
            (Val::Ref(None), 0, GlobalVal::TAG_NULL_REF),
        ];

        for (val, expected_payload, expected_tag) in cases {
            let slot = GlobalVal::from(val);

            assert_eq!(slot.val, expected_payload);
            assert_eq!(slot.tag, expected_tag);
        }
    }

    /// A negative `i32` fills only the low four bytes.
    ///
    /// `v as u64` on an `i32` sign-extends, which would set the upper four. Nothing
    /// reads those back — a `load i32` takes the low four on a little-endian target —
    /// but it would make `-1` and a stale slot indistinguishable.
    #[test]
    fn a_negative_i32_does_not_sign_extend_into_the_upper_bytes() {
        let slot = GlobalVal::from(Val::I32(-1));

        assert_eq!(slot.val, 0x0000_0000_ffff_ffff);

        // Still `-1` when read back at the width the global was declared with, which
        // is the only way generated code reads it.
        assert_eq!(slot.val as u32 as i32, -1);
    }

    /// Floats are stored by bit pattern, so a NaN payload and a signed zero survive.
    ///
    /// `v as u64` would round to an integer instead: `f64::NAN as u64` is 0, and
    /// `-0.0 as u64` is 0, collapsing both onto the same slot as `0.0`.
    #[test]
    fn a_float_keeps_its_bit_pattern() {
        let nan = f64::from_bits(0x7ff8_0000_0000_dead);
        let slot = GlobalVal::from(Val::F64(nan));

        assert_eq!(slot.val, 0x7ff8_0000_0000_dead, "the NaN payload survives");

        let neg_zero = GlobalVal::from(Val::F64(-0.0));
        let pos_zero = GlobalVal::from(Val::F64(0.0));

        assert_eq!(neg_zero.val, 0x8000_0000_0000_0000);
        assert_eq!(pos_zero.val, 0);
        assert_ne!(
            neg_zero.val, pos_zero.val,
            "`-0.0` and `0.0` are different globals"
        );

        // The same for `f32`, whose payload sits in the low four bytes.
        let f32_nan = f32::from_bits(0x7fc0_beef);

        assert_eq!(GlobalVal::from(Val::F32(f32_nan)).val, 0x7fc0_beef);
        assert_eq!(GlobalVal::from(Val::F32(-0.0)).val, 0x8000_0000);
    }

    /// A table slot, both ways round.
    #[test]
    fn a_table_slot_encodes_its_nullability_in_the_tag() {
        let present = OptionalU32::from(Some(FuncIndex(9)));

        assert_eq!((present.val, present.tag), (9, OptionalU32::TAG_SOME));

        let null = OptionalU32::from(None);

        assert_eq!((null.val, null.tag), (0, OptionalU32::TAG_NULL));

        // Index 0 is an ordinary function, not a stand-in for null — the two differ
        // only in the tag, which is the point of having one.
        let zero = OptionalU32::from(Some(FuncIndex(0)));

        assert_eq!(zero.val, null.val);
        assert_ne!(zero.tag, null.tag);
    }

    /// `u32::MAX` is a usable index, not a sentinel.
    ///
    /// It would be the natural null marker if the tag were dropped, so this records
    /// that the current encoding does not reserve it.
    #[test]
    fn the_largest_index_is_an_ordinary_reference() {
        let slot = OptionalU32::from(Some(FuncIndex(u32::MAX)));

        assert_eq!((slot.val, slot.tag), (u32::MAX, OptionalU32::TAG_SOME));
    }
}
