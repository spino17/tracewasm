pub struct Memory {}

impl Default for Memory {
    fn default() -> Self {
        // should allocate starting size for memory according to WASM spec
        Memory {}
    }
}
