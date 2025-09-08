//! Activation layers

use super::{Module, BaseModule};
use crate::functional;
use polars::prelude::*;
use anyhow::Result;

/// ReLU activation layer
#[derive(Debug)]
pub struct ReLU {
    base: BaseModule,
    last_input: Option<DataFrame>,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            base: BaseModule::new(),
            last_input: None,
        }
    }
}

impl Module for ReLU {
    fn forward(&mut self, input: &DataFrame) -> Result<DataFrame> {
        self.last_input = Some(input.clone());
        functional::relu(input)
    }
    
    fn backward(&mut self, grad_output: &DataFrame) -> Result<DataFrame> {
        let input = self.last_input.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No cached input for backward pass"))?;
        
        functional::relu_backward(grad_output, input)
    }
    
    fn update_parameters(&mut self, _learning_rate: f64) -> Result<()> {
        // No parameters to update
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        // No gradients to zero
    }
    
    fn train(&mut self) {
        self.base.training = true;
    }
    
    fn eval(&mut self) {
        self.base.training = false;
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

/// Sigmoid activation layer
#[derive(Debug)]
pub struct Sigmoid {
    base: BaseModule,
    last_output: Option<DataFrame>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self {
            base: BaseModule::new(),
            last_output: None,
        }
    }
}

impl Module for Sigmoid {
    fn forward(&mut self, input: &DataFrame) -> Result<DataFrame> {
        let output = functional::sigmoid(input)?;
        self.last_output = Some(output.clone());
        Ok(output)
    }
    
    fn backward(&mut self, grad_output: &DataFrame) -> Result<DataFrame> {
        let output = self.last_output.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No cached output for backward pass"))?;
        
        functional::sigmoid_backward(grad_output, output)
    }
    
    fn update_parameters(&mut self, _learning_rate: f64) -> Result<()> {
        // No parameters to update
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        // No gradients to zero
    }
    
    fn train(&mut self) {
        self.base.training = true;
    }
    
    fn eval(&mut self) {
        self.base.training = false;
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

/// Tanh activation layer
#[derive(Debug)]
pub struct Tanh {
    base: BaseModule,
    last_output: Option<DataFrame>,
}

impl Tanh {
    pub fn new() -> Self {
        Self {
            base: BaseModule::new(),
            last_output: None,
        }
    }
}

impl Module for Tanh {
    fn forward(&mut self, input: &DataFrame) -> Result<DataFrame> {
        let output = functional::tanh(input)?;
        self.last_output = Some(output.clone());
        Ok(output)
    }
    
    fn backward(&mut self, grad_output: &DataFrame) -> Result<DataFrame> {
        let output = self.last_output.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No cached output for backward pass"))?;
        
        functional::tanh_backward(grad_output, output)
    }
    
    fn update_parameters(&mut self, _learning_rate: f64) -> Result<()> {
        // No parameters to update
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        // No gradients to zero
    }
    
    fn train(&mut self) {
        self.base.training = true;
    }
    
    fn eval(&mut self) {
        self.base.training = false;
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear (identity) activation layer
#[derive(Debug)]
pub struct Identity {
    base: BaseModule,
}

impl Identity {
    pub fn new() -> Self {
        Self {
            base: BaseModule::new(),
        }
    }
}

impl Module for Identity {
    fn forward(&mut self, input: &DataFrame) -> Result<DataFrame> {
        Ok(input.clone())
    }
    
    fn backward(&mut self, grad_output: &DataFrame) -> Result<DataFrame> {
        Ok(grad_output.clone())
    }
    
    fn update_parameters(&mut self, _learning_rate: f64) -> Result<()> {
        // No parameters to update
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        // No gradients to zero
    }
    
    fn train(&mut self) {
        self.base.training = true;
    }
    
    fn eval(&mut self) {
        self.base.training = false;
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new()
    }
}
