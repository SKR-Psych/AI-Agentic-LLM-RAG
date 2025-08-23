use std::collections::HashMap;

pub struct Memory {
    store: HashMap<String, String>,
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            store: HashMap::new(),
        }
    }

    pub fn store(&mut self, key: String, value: String) {
        self.store.insert(key, value);
    }

    pub fn recall(&self, key: &str) -> Option<&String> {
        self.store.get(key)
    }
}
