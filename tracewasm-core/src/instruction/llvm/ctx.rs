use crate::{
    module::FuncIndex,
    runtime::value::{TableVal, Val},
};

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

    pub fn ctx(&mut self) -> RuntimeContext {
        RuntimeContext {
            globals_ptr: self.globals.as_mut_ptr(),
            globals_len: self.globals.len() as u32,
            tables_ptr: self.tables.as_ptr(),
            tables_len: self.tables.len() as u32,
        }
    }

    pub fn call<R>(&mut self, f: unsafe extern "C" fn(*mut RuntimeContext) -> R) -> R {
        let mut ctx = self.ctx();

        unsafe { f(&mut ctx) }
    }
}

#[repr(C)]
pub struct RuntimeContext {
    globals_ptr: *mut GlobalVal,
    globals_len: u32,
    tables_ptr: *const TableEntry,
    tables_len: u32,
}

#[repr(C)]
pub struct GlobalVal {
    val: u64,
    tag: u8,
}

impl From<Val> for GlobalVal {
    fn from(value: Val) -> Self {
        todo!()
    }
}

#[repr(C)]
pub struct TableEntry {
    table_ptr: *mut OptionalU32,
    table_len: u32,
}

#[repr(C)]
pub struct OptionalU32 {
    val: u32,
    tag: u8,
}

impl From<Option<FuncIndex>> for OptionalU32 {
    fn from(value: Option<FuncIndex>) -> Self {
        todo!()
    }
}
