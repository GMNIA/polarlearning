//! Functional operations for neural networks
//! 
//! This module provides functional operations similar to PyTorch's torch.nn.functional,
//! including activation functions, loss functions, and other operations.

use polars::prelude::*;
use anyhow::Result;

pub mod activations;
pub mod linear;

pub use activations::*;
// Explicitly rename linear function to avoid ambiguity
pub use linear::linear as linear_transform;

/// Common functional operations
pub struct F;

impl F {
    /// Linear transformation: output = input @ weight + bias
    pub fn linear(input: &DataFrame, weight: &DataFrame, bias: Option<&DataFrame>) -> Result<DataFrame> {
        linear::linear(input, weight, bias)
    }
    
    /// Linear activation (identity function)
    pub fn linear_activation(input: &DataFrame) -> Result<DataFrame> {
        activations::linear(input)
    }
    
    /// ReLU activation function
    pub fn relu(input: &DataFrame) -> Result<DataFrame> {
        activations::relu(input)
    }
    
    /// Sigmoid activation function
    pub fn sigmoid(input: &DataFrame) -> Result<DataFrame> {
        activations::sigmoid(input)
    }
    
    /// Tanh activation function
    pub fn tanh(input: &DataFrame) -> Result<DataFrame> {
        activations::tanh(input)
    }
}
