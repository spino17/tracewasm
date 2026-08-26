//! Per-instance resource limits.

/// Ceilings an [`Instance`](crate::instance::Instance) is created under, checked
/// at instantiation and during execution.
///
/// The fields are private so that a limit can only be set through its setter;
/// [`Module::instantiate`](crate::module::Module::instantiate) may narrow some of
/// them against what the module itself declares, so the value read back is the
/// effective one rather than the one supplied.
pub struct Config {
    /// Max memory size in pages: the ceiling for both the initial allocation and
    /// any later `memory.grow`.
    ///
    /// [`Module::instantiate`](crate::module::Module::instantiate) narrows this to
    /// the module's own declared maximum, so on an instantiated
    /// [`Instance`](crate::instance::Instance) it is the *effective* limit and may
    /// read lower than the value supplied.
    max_memory_size_in_pages: u32,
    /// Max number of elements in a table, checked when a table is materialized at
    /// instantiation.
    max_table_elements: u32,
    /// Max number of locals per function (including params).
    ///
    /// **Not currently enforced.** The setter and getter exist so that callers can
    /// carry the intent, but no code consults it; a module with more locals than
    /// this is accepted.
    max_locals_per_func: u32,
    /// Max depth of nested wasm calls before
    /// [`InstructionExecutionError::CallStackExhausted`](crate::error::InstructionExecutionError::CallStackExhausted).
    ///
    /// Each wasm frame costs a native frame, so this bounds guest recursion below
    /// the host stack's own limit, where an overflow would abort the process
    /// instead of unwinding.
    ///
    /// **The default is tied to the interpreter's native frame size.** Dispatch is
    /// inlined into the driver, so one frame holds the spill slots of every opcode
    /// arm: on aarch64 release that measures 608 bytes per nested call for the stack
    /// machine and 640 for the register machine, and the larger sets the bound. The
    /// default of 2000 keeps a full chain near 1.3 MiB, which fits the 2 MiB stack
    /// Rust gives a spawned thread — the smallest stack a host is likely to run on
    /// without having chosen one — with about a third to spare.
    ///
    /// A debug build is a different regime entirely, spending ~33 KB per frame, so
    /// this default is 50x too deep for one; see `MAX_TEST_RECURSION` in
    /// `tracewasm-test` for how the suites scale instead.
    ///
    /// Raising it is only safe if the host stack is larger to match: run the
    /// interpreter on a thread with an explicit `stack_size`, and size the limit
    /// against that. A limit the native stack cannot hold turns a clean trap back
    /// into a process abort, because the overflow arrives before the guard does.
    max_call_stack_depth: u32,
    /// Max number of function imports a module may declare.
    ///
    /// **Not currently enforced**, like [`Self::get_max_locals_per_func`]: the
    /// setter and getter carry the intent, but no code consults it, and a module
    /// declaring more imports than this instantiates normally.
    max_imported_funcs: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_memory_size_in_pages: 1000,
            max_table_elements: 10000,
            max_locals_per_func: 50000,
            max_call_stack_depth: 2000,
            max_imported_funcs: 1000,
        }
    }
}

impl Config {
    /// Sets `max_call_stack_depth`
    pub fn set_max_call_stack_depth(&mut self, depth: u32) {
        self.max_call_stack_depth = depth;
    }

    /// Returns `max_call_stack_depth`
    pub fn get_max_call_stack_depth(&self) -> u32 {
        self.max_call_stack_depth
    }

    /// Sets `max_memory_size_in_pages`
    pub fn set_max_memory_size_in_pages(&mut self, val: u32) {
        self.max_memory_size_in_pages = val;
    }

    /// Returns `max_memory_size_in_pages`
    pub fn get_max_memory_size_in_pages(&self) -> u32 {
        self.max_memory_size_in_pages
    }

    /// Sets `max_table_elements`
    pub fn set_max_table_elements(&mut self, val: u32) {
        self.max_table_elements = val;
    }

    /// Returns `max_table_elements`
    pub fn get_max_table_elements(&self) -> u32 {
        self.max_table_elements
    }

    /// Sets `max_locals_per_func`
    pub fn set_max_locals_per_func(&mut self, val: u32) {
        self.max_locals_per_func = val;
    }

    /// Returns `max_locals_per_func`
    pub fn get_max_locals_per_func(&self) -> u32 {
        self.max_locals_per_func
    }

    /// Returns `max_imported_funcs`
    pub fn get_max_imported_funcs(&self) -> u32 {
        self.max_imported_funcs
    }

    /// Sets `max_imported_funcs`
    pub fn set_max_imported_funcs(&mut self, val: u32) {
        self.max_imported_funcs = val;
    }
}
