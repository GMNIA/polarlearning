//! Loss functions for neural networks
//! 
//! This module provides various loss functions similar to PyTorch's torch.nn loss functions.

use polars::prelude::*;
use anyhow::Result;

pub mod mae;
pub mod cross_entropy;

pub use mae::*;
pub use cross_entropy::*;

/// Base trait for loss functions
pub trait Loss {
    /// Compute the forward pass (loss value)
    fn forward(&self, prediction: &DataFrame, target: &DataFrame) -> Result<f64>;
    
    /// Compute the backward pass (gradient with respect to prediction)
    fn backward(&self, prediction: &DataFrame, target: &DataFrame) -> Result<DataFrame>;
}

/// Reduction types for loss functions
#[derive(Debug, Clone, Copy)]
pub enum Reduction {
    /// No reduction, return full tensor
    None,
    /// Mean reduction
    Mean,
    /// Sum reduction
    Sum,
}
