//! Base module trait implementation

/// Base module struct with common functionality
#[derive(Debug)]
pub struct BaseModule {
    pub training: bool,
}

impl BaseModule {
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Default for BaseModule {
    fn default() -> Self {
        Self::new()
    }
}
