//! Linear operations
//! 
//! Core linear algebra operations for neural networks.

use polars::prelude::*;
use anyhow::Result;

/// Linear transformation: output = input @ weight + bias
/// 
/// Args:
///   input: Input tensor [batch_size, input_features]
///   weight: Weight matrix [input_features, output_features]
///   bias: Optional bias vector [output_features]
/// 
/// Returns:
///   Output tensor [batch_size, output_features]
pub fn linear(input: &DataFrame, weight: &DataFrame, bias: Option<&DataFrame>) -> Result<DataFrame> {
    let batch_size = input.height();
    let input_features = input.width();
    let output_features = weight.width();
    
    // Matrix multiplication: input @ weight
    let mut output_data = Vec::new();
    for row_idx in 0..batch_size {
        let mut output_row = Vec::new();
        
        for col_idx in 0..output_features {
            let mut sum = 0.0;
            
            // Dot product of input row with weight column
            for input_col in 0..input_features {
                let input_val = input.get_row(row_idx)?.0[input_col].extract::<f64>().unwrap_or(0.0);
                let weight_val = weight.get_row(input_col)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
                sum += input_val * weight_val;
            }
            
            // Add bias if provided
            if let Some(bias_df) = bias {
                let bias_val = bias_df.get_row(0)?.0[0].extract::<f64>().unwrap_or(0.0);
                sum += bias_val;
            }
            
            output_row.push(sum);
        }
        output_data.push(output_row);
    }
    
    // Convert to DataFrame
    let mut output_columns = Vec::new();
    for j in 0..output_features {
        let column_data: Vec<f64> = output_data.iter().map(|row| row[j]).collect();
        output_columns.push(Series::new(format!("out_{}", j).into(), column_data));
    }
    
    Ok(DataFrame::new(output_columns)?)
}

/// Matrix multiplication for backpropagation
pub fn matmul(a: &DataFrame, b: &DataFrame) -> Result<DataFrame> {
    let a_rows = a.height();
    let a_cols = a.width();
    let b_cols = b.width();
    
    let mut result_data = Vec::new();
    for i in 0..a_rows {
        let mut result_row = Vec::new();
        for j in 0..b_cols {
            let mut sum = 0.0;
            for k in 0..a_cols {
                let a_val = a.get_row(i)?.0[k].extract::<f64>().unwrap_or(0.0);
                let b_val = b.get_row(k)?.0[j].extract::<f64>().unwrap_or(0.0);
                sum += a_val * b_val;
            }
            result_row.push(sum);
        }
        result_data.push(result_row);
    }
    
    let mut result_columns = Vec::new();
    for j in 0..b_cols {
        let column_data: Vec<f64> = result_data.iter().map(|row| row[j]).collect();
        result_columns.push(Series::new(format!("col_{}", j).into(), column_data));
    }
    
    Ok(DataFrame::new(result_columns)?)
}

/// Transpose a DataFrame (swap rows and columns)
pub fn transpose(input: &DataFrame) -> Result<DataFrame> {
    let rows = input.height();
    let cols = input.width();
    
    let mut transposed_columns = Vec::new();
    for row_idx in 0..rows {
        let mut row_data = Vec::new();
        for col_idx in 0..cols {
            let value = input.get_row(row_idx)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
            row_data.push(value);
        }
        transposed_columns.push(Series::new(format!("row_{}", row_idx).into(), row_data));
    }
    
    Ok(DataFrame::new(transposed_columns)?)
}
