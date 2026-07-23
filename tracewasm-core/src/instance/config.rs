/// An Instance's Config.
pub struct Config {
    /// Max memory size which is used for capping initial pages or
    /// grow allocations beyond this value.
    max_memory_size_in_pages: u64,
    /// Max number of elements in a table.
    max_table_elements: u64,
    /// Max number of locals per function (including params).
    max_locals_per_func: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_memory_size_in_pages: 1000,
            max_table_elements: 10000,
            max_locals_per_func: 50000,
        }
    }
}

impl Config {
    /// Sets `max_memory_size_in_pages`
    pub fn set_max_memory_size_in_pages(&mut self, val: u64) {
        self.max_memory_size_in_pages = val;
    }

    /// Sets `max_table_elements`
    pub fn set_max_table_elements(&mut self, val: u64) {
        self.max_table_elements = val;
    }

    /// Sets `max_locals_per_func`
    pub fn set_max_locals_per_func(&mut self, val: u64) {
        self.max_locals_per_func = val;
    }
}
