//! Optimizers module for updating neural network parameters

use crate::nn::Module;
use anyhow::Result;

/// Base trait for all optimizers
pub trait Optimizer {
    /// Update parameters of a module
    fn step(&mut self, module: &mut dyn Module) -> Result<()>;
    
    /// Zero gradients
    fn zero_grad(&mut self, module: &mut dyn Module);
}

/// Stochastic Gradient Descent optimizer
#[derive(Debug)]
pub struct SGD {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
        }
    }
    
    /// Create SGD optimizer with momentum
    pub fn with_momentum(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay: 0.0,
        }
    }
    
    /// Create SGD optimizer with weight decay
    pub fn with_weight_decay(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, module: &mut dyn Module) -> Result<()> {
        module.update_parameters(self.learning_rate)
    }
    
    fn zero_grad(&mut self, module: &mut dyn Module) {
        module.zero_grad();
    }
}

/// Adam optimizer (simplified version)
#[derive(Debug)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
}

impl Adam {
    /// Create a new Adam optimizer with default parameters
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
    
    /// Create Adam optimizer with custom parameters
    pub fn with_params(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, module: &mut dyn Module) -> Result<()> {
        // Simplified Adam - just use the learning rate for now
        // Full Adam implementation would require maintaining momentum buffers
        module.update_parameters(self.learning_rate)
    }
    
    fn zero_grad(&mut self, module: &mut dyn Module) {
        module.zero_grad();
    }
}
