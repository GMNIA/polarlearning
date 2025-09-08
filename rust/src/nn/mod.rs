//! Neural network layers and modules
//! 
//! This module provides PyTorch-style neural network components.

use polars::prelude::*;
use anyhow::Result;

/// Trainable parameter with data and gradients
#[derive(Debug, Clone)]
pub struct Parameter {
    pub data: DataFrame,
    pub grad: Option<DataFrame>,
}

impl Parameter {
    pub fn new(data: DataFrame) -> Self {
        Self {
            data,
            grad: None,
        }
    }
    
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }
}

/// Base trait for all neural network modules
pub trait Module {
    /// Forward pass
    fn forward(&mut self, input: &DataFrame) -> Result<DataFrame>;
    
    /// Backward pass
    fn backward(&mut self, grad_output: &DataFrame) -> Result<DataFrame>;
    
    /// Update parameters using gradients
    fn update_parameters(&mut self, learning_rate: f64) -> Result<()>;
    
    /// Zero all gradients
    fn zero_grad(&mut self);
    
    /// Set module to training mode
    fn train(&mut self);
    
    /// Set module to evaluation mode
    fn eval(&mut self);
}

pub mod module;
pub mod linear;
pub mod sequential;
pub mod activation;

pub use module::*;
pub use linear::*;
pub use sequential::*;
pub use activation::*;


