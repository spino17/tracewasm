use tracewasm_llvm::{cfg::context::Context, interner::TyId};

use crate::{
    memory::mmap::MmapMemory,
    module::FuncIndex,
    runtime::value::{TableVal, Val},
};
use std::marker::PhantomData;

pub struct RuntimeInstance {
    globals: Box<[GlobalVal]>,
    memory: MmapMemory,
    tables: Box<[TableEntry]>,
    table_entries: Box<[OptionalU32]>,
}

impl RuntimeInstance {
    pub fn new(globals: Box<[Val]>, memory: MmapMemory, tables: Box<[TableVal]>) -> Self {
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
            memory,
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
}

#[repr(C)]
pub struct RuntimeContext<'a> {
    globals_ptr: *mut GlobalVal,
    globals_len: u32,
    tables_ptr: *const TableEntry,
    tables_len: u32,
    phantom: PhantomData<&'a RuntimeInstance>,
}

impl<'a> RuntimeContext<'a> {
    pub fn llvm_ty(ctx: &mut Context) -> TyId {
        let mut fields: Vec<TyId> = vec![];

        fields.push(ctx.ptr_ty());
        fields.push(ctx.i32_ty());
        fields.push(ctx.ptr_ty());
        fields.push(ctx.i32_ty());

        ctx.struct_ty(&fields, false).unwrap()
    }
}

/// One global's slot: eight bytes, untyped.
///
/// Nothing records *which* type a slot holds, because every reader already knows by
/// the time it looks. Generated code resolves the type from `Module::globals[i]`
/// when it emits the access and burns it into the instruction — by the time that
/// code runs there is no type left to consult, only a `load i64` or `load float` at
/// a fixed offset. The host reads the type from the same place, through the module
/// its instance holds. And encoding is handed a [`Val`], which carries its own.
///
/// So a tag would be written and never read. It is not free either: `u64` plus `u8`
/// rounds up to sixteen bytes, doubling the array and turning each access into a
/// two-field `getelementptr` rather than a scaled index.
#[repr(transparent)]
pub struct GlobalVal(u64);

impl GlobalVal {
    /// The function index reserved to mean a null reference.
    ///
    /// Wasm's function index space is `u32`, so this is in principle a real index —
    /// but a module needs four billion functions to reach it, and validation gives
    /// out long before. Reserving it is what lets a `funcref` global sit in eight
    /// bytes with no discriminant beside it.
    pub const NULL_FUNC_REF: u64 = u32::MAX as u64;

    /// The raw eight bytes, as generated code would load them.
    pub fn bits(&self) -> u64 {
        self.0
    }
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
        GlobalVal(match value {
            // `as u32` before widening. A negative `i32` cast straight to `u64`
            // sign-extends, filling the upper four bytes; nothing reads those back
            // for an `i32` global, but leaving them set makes two equal globals
            // differ byte-for-byte, which is a poor thing for a slot to do.
            Val::I32(v) => v as u32 as u64,
            Val::I64(v) => v as u64,
            Val::F32(v) => v.to_bits() as u64,
            Val::F64(v) => v.to_bits(),
            Val::Ref(Some(FuncIndex(index))) => {
                debug_assert!(
                    index as u64 != Self::NULL_FUNC_REF,
                    "function index {index} is the one reserved for a null reference"
                );

                index as u64
            }
            Val::Ref(None) => Self::NULL_FUNC_REF,
        })
    }
}

#[repr(C)]
pub struct TableEntry {
    table_ptr: *mut OptionalU32,
    table_len: u32,
}

impl TableEntry {
    pub fn llvm_ty(ctx: &mut Context) -> TyId {
        let mut fields: Vec<TyId> = vec![];

        fields.push(ctx.ptr_ty());
        fields.push(ctx.i32_ty());

        ctx.struct_ty(&fields, false).unwrap()
    }
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
    use tracewasm_llvm::cfg::module::{DataLayout, Triple};

    /// The layout the generated code will GEP into.
    ///
    /// Pinned here because `llvm_ty` builds a second description of these same
    /// bytes, and a field added on one side without the other corrupts every access
    /// silently. This half is the one `repr(C)` fixes; asserting it means the two
    /// can be compared rather than assumed equal.
    #[test]
    fn the_slot_layouts_are_what_the_llvm_side_must_match() {
        use std::mem::{align_of, offset_of, size_of};

        // Eight bytes and nothing beside them, so the generated access is a scaled
        // index off `globals_ptr` rather than a walk into a struct.
        assert_eq!((size_of::<GlobalVal>(), align_of::<GlobalVal>()), (8, 8));

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

    /// [`RuntimeContext::llvm_ty`] describes the same bytes as the `repr(C)` struct.
    ///
    /// The two are independent descriptions of one layout, so a field added to one
    /// and not the other corrupts every access with nothing to report it. The test
    /// above pins the Rust side against literals; this pins the LLVM side's field
    /// list, which is what determines its offsets.
    ///
    /// That the field list implies the *same* offsets was checked against LLVM
    /// itself — `getelementptr (%Ctx, ptr null, i32 0, i32 N)` on
    /// `{ ptr, i32, ptr, i32 }` gives size 32 and offsets 0, 8, 16, 24, matching
    /// `offset_of!` exactly. `PhantomData` is absent here on purpose: it is
    /// zero-sized, so it contributes nothing to the `repr(C)` layout either.
    #[test]
    fn the_llvm_type_lists_the_same_fields_as_the_rust_struct() {
        let mut ctx = Context::new(
            Triple::new(
                "arm64".to_string(),
                "apple".to_string(),
                "macosx".to_string(),
                None,
            ),
            DataLayout::default(),
        );

        let ty = RuntimeContext::llvm_ty(&mut ctx);

        assert_eq!(ctx.display(ty).to_string(), "{ ptr, i32, ptr, i32 }");
    }

    /// Each variant's payload and tag, spelled out.
    #[test]
    fn a_val_encodes_into_its_slot() {
        let cases = [
            (Val::I32(7), 7u64),
            (Val::I64(7), 7),
            (Val::F32(1.0), 0x3f80_0000),
            (Val::F64(1.0), 0x3ff0_0000_0000_0000),
            (Val::Ref(Some(FuncIndex(3))), 3),
            (Val::Ref(None), GlobalVal::NULL_FUNC_REF),
        ];

        for (val, expected) in cases {
            assert_eq!(GlobalVal::from(val).bits(), expected);
        }
    }

    /// A negative `i32` fills only the low four bytes.
    ///
    /// `v as u64` on an `i32` sign-extends, which would set the upper four. Nothing
    /// reads those back — a `load i32` takes the low four on a little-endian target —
    /// but it would make `-1` and a stale slot indistinguishable.
    #[test]
    fn a_negative_i32_does_not_sign_extend_into_the_upper_bytes() {
        let bits = GlobalVal::from(Val::I32(-1)).bits();

        assert_eq!(bits, 0x0000_0000_ffff_ffff);

        // Still `-1` when read back at the width the global was declared with, which
        // is the only way generated code reads it.
        assert_eq!(bits as u32 as i32, -1);
    }

    /// Floats are stored by bit pattern, so a NaN payload and a signed zero survive.
    ///
    /// `v as u64` would round to an integer instead: `f64::NAN as u64` is 0, and
    /// `-0.0 as u64` is 0, collapsing both onto the same slot as `0.0`.
    #[test]
    fn a_float_keeps_its_bit_pattern() {
        let nan = f64::from_bits(0x7ff8_0000_0000_dead);

        assert_eq!(
            GlobalVal::from(Val::F64(nan)).bits(),
            0x7ff8_0000_0000_dead,
            "the NaN payload survives"
        );

        let neg_zero = GlobalVal::from(Val::F64(-0.0)).bits();
        let pos_zero = GlobalVal::from(Val::F64(0.0)).bits();

        assert_eq!(neg_zero, 0x8000_0000_0000_0000);
        assert_eq!(pos_zero, 0);
        assert_ne!(neg_zero, pos_zero, "`-0.0` and `0.0` are different globals");

        // The same for `f32`, whose payload sits in the low four bytes.
        let f32_nan = f32::from_bits(0x7fc0_beef);

        assert_eq!(GlobalVal::from(Val::F32(f32_nan)).bits(), 0x7fc0_beef);
        assert_eq!(GlobalVal::from(Val::F32(-0.0)).bits(), 0x8000_0000);
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

    /// The two slot kinds disagree about `u32::MAX`, deliberately.
    ///
    /// A global has no room for a discriminant, so it spends the largest index on
    /// null. A table slot kept its tag and does not, since a table is a `u32` of
    /// payload either way and the fifth byte is padding it would pay for regardless.
    ///
    /// Pinned because the two are one field apart and reading either encoding with
    /// the other's rule gives a plausible wrong answer rather than a crash: a null
    /// global read as a table slot is function `4294967295`.
    #[test]
    fn only_a_global_reserves_the_largest_index_for_null() {
        let table_slot = OptionalU32::from(Some(FuncIndex(u32::MAX)));

        assert_eq!(
            (table_slot.val, table_slot.tag),
            (u32::MAX, OptionalU32::TAG_SOME),
            "a table slot spells null in the tag, so every index stays usable"
        );

        assert_eq!(
            GlobalVal::from(Val::Ref(None)).bits(),
            GlobalVal::NULL_FUNC_REF,
            "a global spells null as the reserved index"
        );
    }
}
