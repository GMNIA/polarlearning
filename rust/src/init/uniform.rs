//! Uniform distribution initialization

use super::Initializer;
use polars::prelude::*;
use anyhow::Result;

/// Uniform distribution initializer
#[derive(Debug, Clone)]
pub struct Uniform {
    pub low: f64,
    pub high: f64,
}

impl Uniform {
    pub fn new(low: f64, high: f64) -> Self {
        Self { low, high }
    }
}

impl Default for Uniform {
    fn default() -> Self {
        Self::new(-1.0, 1.0)
    }
}

impl Initializer for Uniform {
    fn init_weights(&self, input_size: usize, output_size: usize) -> Result<DataFrame> {
        let range = self.high - self.low;
        
        let mut weight_columns = Vec::new();
        for j in 0..output_size {
            let mut column_data = Vec::new();
            for i in 0..input_size {
                // Map to [low, high] range
                let normalized = i as f64 / input_size as f64; // [0, 1]
                let weight = self.low + normalized * range;
                column_data.push(weight);
            }
            weight_columns.push(Series::new(format!("w_{}", j).into(), column_data));
        }
        
        Ok(DataFrame::new(weight_columns)?)
    }
    
    fn init_biases(&self, output_size: usize) -> Result<DataFrame> {
        let bias_data = vec![self.low; output_size];
        Ok(DataFrame::new(vec![Series::new("bias".into(), bias_data)])?)
    }
}
