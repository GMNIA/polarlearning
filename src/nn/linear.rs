//! Linear (Dense/Fully Connected) layer

use super::{Module, Parameter, BaseModule};
use crate::init::{Initializer, XavierNormal};
use crate::functional;
use polars::prelude::*;
use anyhow::{Result, Context};

/// Linear (Dense/Fully Connected) layer
/// 
/// Applies a linear transformation: y = xW^T + b
#[derive(Debug)]
pub struct Linear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub input_size: usize,
    pub output_size: usize,
    pub base: BaseModule,
    
    // Cache for backward pass
    last_input: Option<DataFrame>,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(input_size: usize, output_size: usize) -> Result<Self> {
        Self::with_initializer(input_size, output_size, XavierNormal)
    }
    
    /// Create a new linear layer with custom initializer
    pub fn with_initializer<I: Initializer>(input_size: usize, output_size: usize, initializer: I) -> Result<Self> {
        let weight_data = initializer.init_weights(input_size, output_size)?;
        let bias_data = initializer.init_biases(output_size)?;
        
        Ok(Self {
            weight: Parameter::new(weight_data),
            bias: Some(Parameter::new(bias_data)),
            input_size,
            output_size,
            base: BaseModule::new(),
            last_input: None,
        })
    }
    
    /// Create a new linear layer without bias
    pub fn without_bias(input_size: usize, output_size: usize) -> Result<Self> {
        let initializer = XavierNormal;
        let weight_data = initializer.init_weights(input_size, output_size)?;
        
        Ok(Self {
            weight: Parameter::new(weight_data),
            bias: None,
            input_size,
            output_size,
            base: BaseModule::new(),
            last_input: None,
        })
    }
}

impl Module for Linear {
    fn forward(&mut self, input: &DataFrame) -> Result<DataFrame> {
        // Cache input for backward pass
        self.last_input = Some(input.clone());
        
        // Apply linear transformation
        let bias_df = self.bias.as_ref().map(|b| &b.data);
        functional::linear_transform(input, &self.weight.data, bias_df)
    }
    
    fn backward(&mut self, grad_output: &DataFrame) -> Result<DataFrame> {
        let input = self.last_input.as_ref()
            .context("No cached input for backward pass")?;
        
        let batch_size = input.height() as f64;
        
    // Ensure grad_output columns align by index, regardless of names
        let mut weight_grad_data = Vec::new();
        for i in 0..self.input_size {
            let mut grad_row = Vec::new();
            for j in 0..self.output_size {
                let mut grad_sum = 0.0;
                
                for batch_idx in 0..input.height() {
                    let input_val = input.get_row(batch_idx)?.0[i].extract::<f64>().unwrap_or(0.0);
                    let grad_val = grad_output.get_row(batch_idx)?.0[j].extract::<f64>().unwrap_or(0.0);
                    grad_sum += input_val * grad_val;
                }
                
                grad_row.push(grad_sum / batch_size);
            }
            weight_grad_data.push(grad_row);
        }
        
        // Store weight gradients
        let mut weight_grad_columns = Vec::new();
        for j in 0..self.output_size {
            let column_data: Vec<f64> = weight_grad_data.iter().map(|row| row[j]).collect();
            weight_grad_columns.push(Series::new(format!("grad_w_{}", j).into(), column_data));
        }
        self.weight.grad = Some(DataFrame::new(weight_grad_columns)?);
        
    // Compute bias gradients if bias exists
        if let Some(ref mut bias_param) = self.bias {
            let mut bias_grad_data = Vec::new();
            for j in 0..self.output_size {
                let mut grad_sum = 0.0;
                for batch_idx in 0..grad_output.height() {
                    let grad_val = grad_output.get_row(batch_idx)?.0[j].extract::<f64>().unwrap_or(0.0);
                    grad_sum += grad_val;
                }
                bias_grad_data.push(grad_sum / batch_size);
            }
            bias_param.grad = Some(DataFrame::new(vec![Series::new("grad_bias".into(), bias_grad_data)])?);
        }
        
    // Compute input gradients: grad_output @ weights.T
        let mut input_grad_data = Vec::new();
        for batch_idx in 0..grad_output.height() {
            let mut input_grad_row = Vec::new();
            
            for i in 0..self.input_size {
                let mut grad_sum = 0.0;
                
                for j in 0..self.output_size {
                    let grad_val = grad_output.get_row(batch_idx)?.0[j].extract::<f64>().unwrap_or(0.0);
                    let weight_val = self.weight.data.get_row(i)?.0[j].extract::<f64>().unwrap_or(0.0);
                    grad_sum += grad_val * weight_val;
                }
                
                input_grad_row.push(grad_sum);
            }
            input_grad_data.push(input_grad_row);
        }
        
        // Convert to DataFrame
        // IMPORTANT: Name gradient columns to match the input column names so that
        // upstream layers (e.g., ReLU) can align gradients by name during backward.
        let input_col_names: Vec<String> = input
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut input_grad_columns = Vec::new();
        for i in 0..self.input_size {
            let column_data: Vec<f64> = input_grad_data.iter().map(|row| row[i]).collect();
            let col_name = input_col_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("grad_in_{}", i));
            input_grad_columns.push(Series::new(col_name.into(), column_data));
        }
        
        Ok(DataFrame::new(input_grad_columns)?)
    }
    
    fn update_parameters(&mut self, learning_rate: f64) -> Result<()> {
        // Update weights
        if let Some(ref weight_grad) = self.weight.grad {
            let mut new_weights_data = Vec::new();
            for i in 0..self.input_size {
                let mut new_row = Vec::new();
                for j in 0..self.output_size {
                    let current_weight = self.weight.data.get_row(i)?.0[j].extract::<f64>().unwrap_or(0.0);
                    let gradient = weight_grad.get_row(i)?.0[j].extract::<f64>().unwrap_or(0.0);
                    let new_weight = current_weight - learning_rate * gradient;
                    new_row.push(new_weight);
                }
                new_weights_data.push(new_row);
            }
            
            let mut new_weight_columns = Vec::new();
            for j in 0..self.output_size {
                let column_data: Vec<f64> = new_weights_data.iter().map(|row| row[j]).collect();
                new_weight_columns.push(Series::new(format!("w_{}", j).into(), column_data));
            }
            self.weight.data = DataFrame::new(new_weight_columns)?;
        }
        
        // Update bias if it exists
        if let Some(ref mut bias_param) = self.bias {
            if let Some(ref bias_grad) = bias_param.grad {
                let mut new_bias_data = Vec::new();
                for _j in 0..self.output_size {
                    let current_bias = bias_param.data.get_row(0)?.0[0].extract::<f64>().unwrap_or(0.0);
                    let gradient = bias_grad.get_row(0)?.0[0].extract::<f64>().unwrap_or(0.0);
                    let new_bias = current_bias - learning_rate * gradient;
                    new_bias_data.push(new_bias);
                }
                bias_param.data = DataFrame::new(vec![Series::new("bias".into(), new_bias_data)])?;
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.zero_grad();
        }
    }
    
    fn train(&mut self) {
        self.base.training = true;
    }
    
    fn eval(&mut self) {
        self.base.training = false;
    }
}
