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
    max_call_stack_depth: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_memory_size_in_pages: 1000,
            max_table_elements: 10000,
            max_locals_per_func: 50000,
            max_call_stack_depth: 2000,
        }
    }
}

impl Config {
    pub fn set_max_call_stack_depth(&mut self, depth: u32) {
        self.max_call_stack_depth = depth;
    }

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
