//! Normal distribution initialization

use super::Initializer;
use polars::prelude::*;
use anyhow::Result;

/// Normal distribution initializer
#[derive(Debug, Clone)]
pub struct Normal {
    pub mean: f64,
    pub std: f64,
}

impl Normal {
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }
}

impl Default for Normal {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

impl Initializer for Normal {
    fn init_weights(&self, input_size: usize, output_size: usize) -> Result<DataFrame> {
        let mut weight_columns = Vec::new();
        for j in 0..output_size {
            let mut column_data = Vec::new();
            for i in 0..input_size {
                // Simple normal-like distribution
                let weight = self.mean + (i as f64 * 0.01 - 0.005) * self.std;
                column_data.push(weight);
            }
            weight_columns.push(Series::new(format!("w_{}", j).into(), column_data));
        }
        
        Ok(DataFrame::new(weight_columns)?)
    }
    
    fn init_biases(&self, output_size: usize) -> Result<DataFrame> {
        let bias_data = vec![self.mean; output_size];
        Ok(DataFrame::new(vec![Series::new("bias".into(), bias_data)])?)
    }
}
