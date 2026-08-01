/// Per-instance resource limits, enforced at instantiation and during execution.
pub struct Config {
    /// Max memory size in pages: the ceiling for both the initial allocation and
    /// any later `memory.grow`.
    ///
    /// [`Module::instantiate`](crate::module::Module::instantiate) narrows this to
    /// the module's own declared maximum, so on an instantiated
    /// [`Instance`](crate::instance::Instance) it is the *effective* limit and may
    /// read lower than the value supplied.
    max_memory_size_in_pages: u64,
    /// Max number of elements in a table.
    max_table_elements: u64,
    /// Max number of locals per function (including params).
    max_locals_per_func: u64,
    /// Max depth of nested wasm calls before
    /// [`TraceWasmError::CallStackExhausted`](crate::error::TraceWasmError::CallStackExhausted).
    ///
    /// Each wasm frame costs a native frame, so this bounds guest recursion below
    /// the host stack's own limit, where an overflow would abort the process
    /// instead of unwinding.
    ///
    /// **The default is tied to the interpreter's native frame size.** Dispatch is
    /// inlined into the driver loop, which puts every opcode arm's spill slots in
    /// one frame — about 5 KB per nested call. The default of 1200 keeps a full
    /// chain near 6 MB, inside a typical 8 MiB main thread with room to spare.
    ///
    /// Raising it is only safe if the host stack is larger to match: run the
    /// interpreter on a thread with an explicit `stack_size`, and size the limit
    /// against that. A limit the native stack cannot hold turns a clean trap back
    /// into a process abort, because the overflow arrives before the guard does.
    max_call_stack_depth: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_memory_size_in_pages: 1000,
            max_table_elements: 10000,
            max_locals_per_func: 50000,
            max_call_stack_depth: 1200,
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
    pub fn set_max_memory_size_in_pages(&mut self, val: u64) {
        self.max_memory_size_in_pages = val;
    }

    /// Returns `max_memory_size_in_pages`
    pub fn get_max_memory_size_in_pages(&self) -> u64 {
        self.max_memory_size_in_pages
    }

    /// Sets `max_table_elements`
    pub fn set_max_table_elements(&mut self, val: u64) {
        self.max_table_elements = val;
    }

    /// Returns `max_table_elements`
    pub fn get_max_table_elements(&self) -> u64 {
        self.max_table_elements
    }

    /// Sets `max_locals_per_func`
    pub fn set_max_locals_per_func(&mut self, val: u64) {
        self.max_locals_per_func = val;
    }

    /// Returns `max_locals_per_func`
    pub fn get_max_locals_per_func(&self) -> u64 {
        self.max_locals_per_func
    }
}
