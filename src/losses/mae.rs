//! Mean Absolute Error loss

use super::{Loss, Reduction};
use polars::prelude::*;
use anyhow::Result;

/// Mean Absolute Error loss function
/// 
/// L(y, ŷ) = |y - ŷ|
#[derive(Debug, Clone)]
pub struct MAELoss {
    pub reduction: Reduction,
}

impl MAELoss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Default for MAELoss {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

impl Loss for MAELoss {
    fn forward(&self, prediction: &DataFrame, target: &DataFrame) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for row_idx in 0..prediction.height() {
            for col_idx in 0..prediction.width() {
                let pred_val = prediction.get_row(row_idx)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
                let target_val = target.get_row(row_idx)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
                total_loss += (pred_val - target_val).abs();
                count += 1;
            }
        }
        
        match self.reduction {
            Reduction::Mean => Ok(total_loss / count as f64),
            Reduction::Sum => Ok(total_loss),
            Reduction::None => Ok(total_loss),
        }
    }
    
    fn backward(&self, prediction: &DataFrame, target: &DataFrame) -> Result<DataFrame> {
        let batch_size = prediction.height() as f64;
        let mut grad_columns = Vec::new();
        
        for col_idx in 0..prediction.width() {
            let column_name = prediction.get_columns()[col_idx].name().clone();
            let mut grad_data = Vec::new();
            
            for row_idx in 0..prediction.height() {
                let pred_val = prediction.get_row(row_idx)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
                let target_val = target.get_row(row_idx)?.0[col_idx].extract::<f64>().unwrap_or(0.0);
                
                let sign = if pred_val > target_val { 1.0 } else { -1.0 };
                let grad = match self.reduction {
                    Reduction::Mean => sign / batch_size,
                    Reduction::Sum => sign,
                    Reduction::None => sign,
                };
                
                grad_data.push(grad);
            }
            
            grad_columns.push(Series::new(column_name, grad_data));
        }
        
        Ok(DataFrame::new(grad_columns)?)
    }
}
