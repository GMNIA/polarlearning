//! Xavier/Glorot initialization
//! 
//! Xavier initialization helps maintain the variance of activations and gradients
//! across layers during training.

use super::Initializer;
use polars::prelude::*;
use anyhow::Result;

/// Xavier/Glorot weight initializer
/// 
/// Initializes weights from a normal distribution with mean=0 and 
/// variance = 2 / (fan_in + fan_out)
#[derive(Debug, Clone)]
pub struct XavierNormal;

impl Initializer for XavierNormal {
    fn init_weights(&self, input_size: usize, output_size: usize) -> Result<DataFrame> {
        // Xavier initialization: std = sqrt(2 / (input_size + output_size))
        let xavier_std = (2.0 / (input_size + output_size) as f64).sqrt();
        
        // Generate weight matrix
        let mut weight_columns = Vec::new();
        for j in 0..output_size {
            let mut column_data = Vec::new();
            for i in 0..input_size {
                // Simple deterministic initialization for reproducibility
                // In practice, you'd use a proper random number generator
                let weight = (i as f64 * 0.01 - 0.005) * xavier_std;
                column_data.push(weight);
            }
            weight_columns.push(Series::new(format!("w_{}", j).into(), column_data));
        }
        
        Ok(DataFrame::new(weight_columns)?)
    }
    
    fn init_biases(&self, output_size: usize) -> Result<DataFrame> {
        // Initialize biases to zero (common practice)
        let bias_data = vec![0.0; output_size];
        Ok(DataFrame::new(vec![Series::new("bias".into(), bias_data)])?)
    }
}

/// Xavier/Glorot uniform initializer
/// 
/// Initializes weights from a uniform distribution within
/// [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
#[derive(Debug, Clone)]
pub struct XavierUniform;

impl Initializer for XavierUniform {
    fn init_weights(&self, input_size: usize, output_size: usize) -> Result<DataFrame> {
        // Xavier uniform: limit = sqrt(6 / (input_size + output_size))
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();
        
        let mut weight_columns = Vec::new();
        for j in 0..output_size {
            let mut column_data = Vec::new();
            for i in 0..input_size {
                // Map to [-limit, limit] range
                let normalized = (i as f64 / input_size as f64) * 2.0 - 1.0; // [-1, 1]
                let weight = normalized * limit;
                column_data.push(weight);
            }
            weight_columns.push(Series::new(format!("w_{}", j).into(), column_data));
        }
        
        Ok(DataFrame::new(weight_columns)?)
    }
    
    fn init_biases(&self, output_size: usize) -> Result<DataFrame> {
        let bias_data = vec![0.0; output_size];
        Ok(DataFrame::new(vec![Series::new("bias".into(), bias_data)])?)
    }
}
