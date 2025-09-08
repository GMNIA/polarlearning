//! Weight initialization strategies
//! 
//! This module provides various weight initialization methods commonly used in deep learning,
//! similar to PyTorch's torch.nn.init module.

use polars::prelude::*;
use anyhow::Result;

pub mod xavier;
pub mod kaiming;
pub mod normal;
pub mod uniform;

pub use xavier::*;
pub use kaiming::*;
pub use normal::*;
pub use uniform::*;

/// Trait for weight initialization strategies
pub trait Initializer {
    /// Initialize weights for a layer with given dimensions
    fn init_weights(&self, input_size: usize, output_size: usize) -> Result<DataFrame>;
    
    /// Initialize biases for a layer with given output size
    fn init_biases(&self, output_size: usize) -> Result<DataFrame>;
}

/// Common initialization utilities
pub struct InitUtils;

impl InitUtils {
    /// Create a DataFrame with specified shape filled with a constant value
    pub fn constant_fill(rows: usize, cols: usize, value: f64, col_prefix: &str) -> Result<DataFrame> {
        let mut columns = Vec::new();
        
        for j in 0..cols {
            let column_data = vec![value; rows];
            columns.push(Series::new(format!("{}_{}", col_prefix, j).into(), column_data));
        }
        
        Ok(DataFrame::new(columns)?)
    }
    
    /// Create a DataFrame with zeros
    pub fn zeros(rows: usize, cols: usize, col_prefix: &str) -> Result<DataFrame> {
        Self::constant_fill(rows, cols, 0.0, col_prefix)
    }
    
    /// Create a DataFrame with ones
    pub fn ones(rows: usize, cols: usize, col_prefix: &str) -> Result<DataFrame> {
        Self::constant_fill(rows, cols, 1.0, col_prefix)
    }
}
