//! Kaiming/He initialization
//! 
//! Kaiming initialization is designed for layers with ReLU activations.

use super::Initializer;
use polars::prelude::*;
use anyhow::Result;

/// Kaiming/He normal initializer
/// 
/// Initializes weights from a normal distribution with mean=0 and 
/// variance = 2 / fan_in (for ReLU activations)
#[derive(Debug, Clone)]
pub struct KaimingNormal;

impl Initializer for KaimingNormal {
    fn init_weights(&self, input_size: usize, output_size: usize) -> Result<DataFrame> {
        // Kaiming initialization: std = sqrt(2 / input_size)
        let kaiming_std = (2.0 / input_size as f64).sqrt();
        
        let mut weight_columns = Vec::new();
        for j in 0..output_size {
            let mut column_data = Vec::new();
            for i in 0..input_size {
                let weight = (i as f64 * 0.02 - 0.01) * kaiming_std;
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

/// Kaiming/He uniform initializer
#[derive(Debug, Clone)]
pub struct KaimingUniform;

impl Initializer for KaimingUniform {
    fn init_weights(&self, input_size: usize, output_size: usize) -> Result<DataFrame> {
        // Kaiming uniform: limit = sqrt(6 / input_size)
        let limit = (6.0 / input_size as f64).sqrt();
        
        let mut weight_columns = Vec::new();
        for j in 0..output_size {
            let mut column_data = Vec::new();
            for i in 0..input_size {
                let normalized = (i as f64 / input_size as f64) * 2.0 - 1.0;
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
