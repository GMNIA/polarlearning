//! Data preprocessing transforms
//! 
//! PyTorch-style data transformations for preprocessing datasets

use polars::prelude::*;
use anyhow::{Result, Context};
use std::path::Path;

/// Standard scaler for normalizing features
/// Applies z-score normalization: (x - mean) / std
#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub means: Option<Vec<f64>>,
    pub stds: Option<Vec<f64>>,
    pub feature_names: Option<Vec<String>>,
}

impl StandardScaler {
    /// Create a new StandardScaler
    pub fn new() -> Self {
        Self {
            means: None,
            stds: None,
            feature_names: None,
        }
    }
    
    /// Fit the scaler to the training data
    pub fn fit(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<()> {
        let mut means = Vec::new();
        let mut stds = Vec::new();
        
        for col_name in feature_columns {
            let col = df.column(col_name)
                .context(format!("Column '{}' not found", col_name))?;
            
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(1).unwrap_or(1.0);
            
            means.push(mean);
            stds.push(if std > 1e-8 { std } else { 1.0 }); // Avoid division by zero
        }
        
        self.means = Some(means);
        self.stds = Some(stds);
        self.feature_names = Some(feature_columns.to_vec());
        
        println!("StandardScaler fitted to {} features", feature_columns.len());
        Ok(())
    }
    
    /// Transform data using fitted parameters
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let means = self.means.as_ref()
            .context("Scaler not fitted. Call fit() first.")?;
        let stds = self.stds.as_ref()
            .context("Scaler not fitted. Call fit() first.")?;
        let feature_names = self.feature_names.as_ref()
            .context("Scaler not fitted. Call fit() first.")?;
        
        let mut result_df = df.clone();
        
        for (i, col_name) in feature_names.iter().enumerate() {
            let col = df.column(col_name)
                .context(format!("Column '{}' not found", col_name))?;
            
            let mean = means[i];
            let std = stds[i];
            
            let normalized_col = col.f64()?
                .apply(|val| val.map(|v| (v - mean) / std))
                .with_name(col_name.into());
            
            result_df.with_column(normalized_col);
        }
        
        Ok(result_df)
    }
    
    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<DataFrame> {
        self.fit(df, feature_columns)?;
        self.transform(df)
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// MinMax scaler for normalizing features to [0, 1] range
/// Applies min-max normalization: (x - min) / (max - min)
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    pub mins: Option<Vec<f64>>,
    pub maxs: Option<Vec<f64>>,
    pub feature_names: Option<Vec<String>>,
}

impl MinMaxScaler {
    /// Create a new MinMaxScaler
    pub fn new() -> Self {
        Self {
            mins: None,
            maxs: None,
            feature_names: None,
        }
    }
    
    /// Fit the scaler to the training data
    pub fn fit(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<()> {
        let mut mins = Vec::new();
        let mut maxs = Vec::new();
        
        for col_name in feature_columns {
            let col = df.column(col_name)
                .context(format!("Column '{}' not found", col_name))?;
            
            let min_val = col.min::<f64>().unwrap_or(Some(0.0)).unwrap_or(0.0);
            let max_val = col.max::<f64>().unwrap_or(Some(1.0)).unwrap_or(1.0);
            
            mins.push(min_val);
            maxs.push(max_val);
        }
        
        self.mins = Some(mins);
        self.maxs = Some(maxs);
        self.feature_names = Some(feature_columns.to_vec());
        
        println!("MinMaxScaler fitted to {} features", feature_columns.len());
        Ok(())
    }
    
    /// Transform data using fitted parameters
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mins = self.mins.as_ref()
            .context("Scaler not fitted. Call fit() first.")?;
        let maxs = self.maxs.as_ref()
            .context("Scaler not fitted. Call fit() first.")?;
        let feature_names = self.feature_names.as_ref()
            .context("Scaler not fitted. Call fit() first.")?;
        
        let mut result_df = df.clone();
        
        for (i, col_name) in feature_names.iter().enumerate() {
            let col = df.column(col_name)
                .context(format!("Column '{}' not found", col_name))?;
            
            let min_val = mins[i];
            let max_val = maxs[i];
            let range = max_val - min_val;
            
            let normalized_col = if range > 1e-8 {
                col.f64()?
                    .apply(|val| val.map(|v| (v - min_val) / range))
                    .with_name(col_name.into())
            } else {
                // If range is zero, set all values to 0.5
                col.f64()?
                    .apply(|_| Some(0.5))
                    .with_name(col_name.into())
            };
            
            result_df.with_column(normalized_col);
        }
        
        Ok(result_df)
    }
    
    /// Inverse transform normalized values back to original scale
    
    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<DataFrame> {
        self.fit(df, feature_columns)?;
        self.transform(df)
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Data preprocessing pipeline
pub struct DataProcessor {
    pub feature_scaler: Box<dyn ScalerTrait>,
    pub target_scaler: Option<Box<dyn ScalerTrait>>,
}

pub trait ScalerTrait {
    fn fit(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<()>;
    fn transform(&self, df: &DataFrame) -> Result<DataFrame>;
    fn fit_transform(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<DataFrame>;
}

impl ScalerTrait for StandardScaler {
    fn fit(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<()> {
        StandardScaler::fit(self, df, feature_columns)
    }
    
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        StandardScaler::transform(self, df)
    }
    
    fn fit_transform(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<DataFrame> {
        StandardScaler::fit_transform(self, df, feature_columns)
    }
}

impl ScalerTrait for MinMaxScaler {
    fn fit(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<()> {
        MinMaxScaler::fit(self, df, feature_columns)
    }
    
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        MinMaxScaler::transform(self, df)
    }
    
    fn fit_transform(&mut self, df: &DataFrame, feature_columns: &[String]) -> Result<DataFrame> {
        MinMaxScaler::fit_transform(self, df, feature_columns)
    }
}

impl DataProcessor {
    /// Create a new DataProcessor with StandardScaler
    pub fn new() -> Self {
        Self {
            feature_scaler: Box::new(StandardScaler::new()),
            target_scaler: None,
        }
    }
    
    /// Create a new DataProcessor with MinMaxScaler
    pub fn with_minmax() -> Self {
        Self {
            feature_scaler: Box::new(MinMaxScaler::new()),
            target_scaler: None,
        }
    }
    
    /// Add target scaling (useful for regression)
    pub fn with_target_scaling(mut self, scaler: Box<dyn ScalerTrait>) -> Self {
        self.target_scaler = Some(scaler);
        self
    }
    
    /// Process raw data and save to processed directory
    pub fn process_and_save(
        &mut self,
        raw_data_path: &Path,
        processed_data_path: &Path,
        feature_columns: &[String],
        target_column: &str,
    ) -> Result<()> {
        // Load raw data (with headers enabled since CSV has proper column names)
        let df = LazyCsvReader::new(raw_data_path)
            .with_has_header(true)
            .finish()?
            .collect()?;
        
        println!("Loaded raw data: {} rows, {} columns", df.height(), df.width());
        
        // Separate features and target
        let features = df.select(feature_columns)?;
        let target = df.select([target_column])?;
        
        // Scale features
        let scaled_features = self.feature_scaler.fit_transform(&features, feature_columns)?;
        
        // Scale target if scaler is provided
        let scaled_target = if let Some(ref mut target_scaler) = self.target_scaler {
            target_scaler.fit_transform(&target, &[target_column.to_string()])?
        } else {
            target
        };
        
        // Combine scaled features and target
        let processed_df = scaled_features.hstack(scaled_target.get_columns())?;
        
        // Create processed directory if it doesn't exist
        if let Some(parent) = processed_data_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Save processed data
        let mut file = std::fs::File::create(processed_data_path)?;
        CsvWriter::new(&mut file)
            .finish(&mut processed_df.clone())?;
        
        println!("Processed data saved to: {}", processed_data_path.display());
        println!("Features scaled: {:?}", feature_columns);
        if self.target_scaler.is_some() {
            println!("Target scaled: {}", target_column);
        }
        
        Ok(())
    }
}
