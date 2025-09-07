//! Activation functions
//! 
//! Provides various activation functions for neural networks.

use polars::prelude::*;
use anyhow::Result;

/// Linear activation (identity function): f(x) = x
pub fn linear(input: &DataFrame) -> Result<DataFrame> {
    // Linear activation is just the identity function
    Ok(input.clone())
}

/// ReLU backward pass (derivative)
/// Returns the gradient: 1 if input > 0, 0 otherwise
pub fn relu_backward(grad_output: &DataFrame, input: &DataFrame) -> Result<DataFrame> {
    let mut result_columns = Vec::new();
    
    for (col_idx, col_name) in input.get_column_names().iter().enumerate() {
        let input_col = input.column(col_name)?;
        // Prefer matching by name; if missing, fall back to index alignment
        let grad_col = match grad_output.column(col_name) {
            Ok(c) => c,
            Err(_) => grad_output.get_columns().get(col_idx)
                .ok_or_else(|| anyhow::anyhow!(format!("Gradient column not found for '{}' and index {}", col_name, col_idx)))?
        };
        
        // Simple element-wise backward pass for ReLU
        let mut backward_data = Vec::new();
        for i in 0..input_col.len() {
            let input_val = input_col.get(i).unwrap().extract::<f64>().unwrap_or(0.0);
            let grad_val = grad_col.get(i).unwrap().extract::<f64>().unwrap_or(0.0);
            backward_data.push(if input_val > 0.0 { grad_val } else { 0.0 });
        }
        
    let backward_col = Series::new(col_name.as_str().into(), backward_data);
        
        result_columns.push(backward_col);
    }
    
    Ok(DataFrame::new(result_columns)?)
}

/// Sigmoid backward pass (derivative)
/// Uses the fact that sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
pub fn sigmoid_backward(grad_output: &DataFrame, sigmoid_output: &DataFrame) -> Result<DataFrame> {
    let mut result_columns = Vec::new();
    
    for (col_idx, col_name) in sigmoid_output.get_column_names().iter().enumerate() {
        let sigmoid_col = sigmoid_output.column(col_name)?;
        let grad_col = match grad_output.column(col_name) {
            Ok(c) => c,
            Err(_) => grad_output.get_columns().get(col_idx)
                .ok_or_else(|| anyhow::anyhow!(format!("Gradient column not found for '{}' and index {}", col_name, col_idx)))?
        };
        
        // Simple element-wise backward pass for Sigmoid
        let mut backward_data = Vec::new();
        for i in 0..sigmoid_col.len() {
            let sig_val = sigmoid_col.get(i).unwrap().extract::<f64>().unwrap_or(0.0);
            let grad_val = grad_col.get(i).unwrap().extract::<f64>().unwrap_or(0.0);
            backward_data.push(grad_val * sig_val * (1.0 - sig_val));
        }
        
        let backward_col = Series::new(col_name.as_str().into(), backward_data);
        
        result_columns.push(backward_col);
    }
    
    Ok(DataFrame::new(result_columns)?)
}

/// Tanh backward pass (derivative)
/// Uses the fact that tanh'(x) = 1 - tanh²(x)
pub fn tanh_backward(grad_output: &DataFrame, tanh_output: &DataFrame) -> Result<DataFrame> {
    let mut result_columns = Vec::new();
    
    for (col_idx, col_name) in tanh_output.get_column_names().iter().enumerate() {
        let tanh_col = tanh_output.column(col_name)?;
        let grad_col = match grad_output.column(col_name) {
            Ok(c) => c,
            Err(_) => grad_output.get_columns().get(col_idx)
                .ok_or_else(|| anyhow::anyhow!(format!("Gradient column not found for '{}' and index {}", col_name, col_idx)))?
        };
        
        // Simple element-wise backward pass for Tanh
        let mut backward_data = Vec::new();
        for i in 0..tanh_col.len() {
            let tanh_val = tanh_col.get(i).unwrap().extract::<f64>().unwrap_or(0.0);
            let grad_val = grad_col.get(i).unwrap().extract::<f64>().unwrap_or(0.0);
            backward_data.push(grad_val * (1.0 - tanh_val * tanh_val));
        }
        
        let backward_col = Series::new(col_name.as_str().into(), backward_data);
        
        result_columns.push(backward_col);
    }
    
    Ok(DataFrame::new(result_columns)?)
}
pub fn relu(input: &DataFrame) -> Result<DataFrame> {
    let mut output_columns = Vec::new();
    
    for (col_idx, column) in input.get_columns().iter().enumerate() {
        let column_name = column.name().clone();
        let mut activated_data = Vec::new();
        
        for row_idx in 0..input.height() {
            let value = input.get_row(row_idx)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
            activated_data.push(value.max(0.0));
        }
        
        output_columns.push(Series::new(column_name, activated_data));
    }
    
    Ok(DataFrame::new(output_columns)?)
}

/// Sigmoid activation: f(x) = 1 / (1 + exp(-x))
pub fn sigmoid(input: &DataFrame) -> Result<DataFrame> {
    let mut output_columns = Vec::new();
    
    for (col_idx, column) in input.get_columns().iter().enumerate() {
        let column_name = column.name().clone();
        let mut activated_data = Vec::new();
        
        for row_idx in 0..input.height() {
            let value = input.get_row(row_idx)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
            activated_data.push(1.0 / (1.0 + (-value).exp()));
        }
        
        output_columns.push(Series::new(column_name, activated_data));
    }
    
    Ok(DataFrame::new(output_columns)?)
}

/// Tanh activation: f(x) = tanh(x)
pub fn tanh(input: &DataFrame) -> Result<DataFrame> {
    let mut output_columns = Vec::new();
    
    for (col_idx, column) in input.get_columns().iter().enumerate() {
        let column_name = column.name().clone();
        let mut activated_data = Vec::new();
        
        for row_idx in 0..input.height() {
            let value = input.get_row(row_idx)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
            activated_data.push(value.tanh());
        }
        
        output_columns.push(Series::new(column_name, activated_data));
    }
    
    Ok(DataFrame::new(output_columns)?)
}

/// Compute derivative of ReLU for backpropagation
pub fn relu_derivative(input: &DataFrame) -> Result<DataFrame> {
    let mut output_columns = Vec::new();
    
    for (col_idx, column) in input.get_columns().iter().enumerate() {
        let column_name = column.name().clone();
        let mut derivative_data = Vec::new();
        
        for row_idx in 0..input.height() {
            let value = input.get_row(row_idx)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
            derivative_data.push(if value > 0.0 { 1.0 } else { 0.0 });
        }
        
        output_columns.push(Series::new(column_name, derivative_data));
    }
    
    Ok(DataFrame::new(output_columns)?)
}

/// Compute derivative of linear activation (always 1)
pub fn linear_derivative(input: &DataFrame) -> Result<DataFrame> {
    let mut output_columns = Vec::new();
    
    for column in input.get_columns().iter() {
        let column_name = column.name().clone();
        let derivative_data = vec![1.0; input.height()];
        output_columns.push(Series::new(column_name, derivative_data));
    }
    
    Ok(DataFrame::new(output_columns)?)
}
