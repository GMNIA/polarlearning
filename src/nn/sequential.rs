//! Sequential container for chaining layers

use super::Module;
use polars::prelude::*;
use anyhow::Result;

/// Sequential container that chains modules together
/// 
/// Similar to PyTorch's nn.Sequential
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Create a new empty sequential container
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
        }
    }
    
    /// Add a module to the end of the sequence
    pub fn add_module<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }
    
    /// Get the number of modules in the sequence
    pub fn len(&self) -> usize {
        self.modules.len()
    }
    
    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&mut self, input: &DataFrame) -> Result<DataFrame> {
        let mut output = input.clone();
        
        for module in &mut self.modules {
            output = module.forward(&output)?;
        }
        
        Ok(output)
    }
    
    fn backward(&mut self, grad_output: &DataFrame) -> Result<DataFrame> {
        let mut grad = grad_output.clone();
        
        // Backward pass through modules in reverse order
        for module in self.modules.iter_mut().rev() {
            grad = module.backward(&grad)?;
        }
        
        Ok(grad)
    }
    
    fn update_parameters(&mut self, learning_rate: f64) -> Result<()> {
        for module in &mut self.modules {
            module.update_parameters(learning_rate)?;
        }
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        for module in &mut self.modules {
            module.zero_grad();
        }
    }
    
    fn train(&mut self) {
        for module in &mut self.modules {
            module.train();
        }
    }
    
    fn eval(&mut self) {
        for module in &mut self.modules {
            module.eval();
        }
    }
}

// Builder pattern for convenient construction
impl Sequential {
    /// Create a sequential model from a vector of modules
    pub fn from_modules(modules: Vec<Box<dyn Module>>) -> Self {
        Self { modules }
    }
}

// Convenient constructors
impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear};
    use polars::prelude::*;
    
    #[test]
    fn test_sequential_construction() {
        let model = Sequential::new()
            .add_module(Linear::new(10, 20).unwrap())
            .add_module(Linear::new(20, 1).unwrap());
        
        assert_eq!(model.len(), 2);
        assert!(!model.is_empty());
    }
    
    #[test]
    fn test_empty_sequential() {
        let model = Sequential::new();
        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
    }
}
