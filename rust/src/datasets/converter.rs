//! Dataset conversion utilities for the polarlearning project.
//!
//! Handles conversion from raw dataset files to Polars-friendly formats
//! (CSV + Parquet), with intelligent caching to avoid redundant work.

use polars::prelude::*;
use std::path::Path;
use anyhow::Result;
use std::fs;
use crate::datasets::california_housing::CaliforniaHousingProcessor;

/// Verbosity levels for dataset conversion operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// Silent mode - no output
    Silent = 0,
    /// Normal mode - basic progress and results
    Normal = 1,
    /// Verbose mode - detailed progress with timing
    Verbose = 2,
    // TODO: Add more verbosity levels:
    // TODO: Debug = 3,    // Debug information and intermediate steps
    // TODO: Trace = 4,    // Full trace with data samples
}

impl Default for Verbosity {
    /// Provide the default (Silent) verbosity level.
    fn default() -> Self {
        Verbosity::Silent
    }
}


impl From<u8> for Verbosity {
    /// Convert a numeric level into a `Verbosity` enum value.
    fn from(level: u8) -> Self {
        match level {
            0 => Verbosity::Silent,
            1 => Verbosity::Normal,
            2 => Verbosity::Verbose,
            _ => Verbosity::Normal, // Default to Normal for unknown levels
        }
    }
}

/// Data converter that handles conversion from raw datasets to Polars format
pub struct DatasetConverter;

impl DatasetConverter {
    /// Convert a dataset to Polars format if needed.
    ///
    /// This function checks if the required input files exist and whether
    /// converted artifacts already exist. If not, it performs the conversion
    /// and writes CSV + Parquet outputs.
    pub async fn convert_if_needed(
        dataset_name: &str,
        data_file_path: &str,
        domain_file_path: &str,
        verbosity: Verbosity,
    ) -> Result<String> {
        // --- Output base directory ---
        let output_path = "model-input-data/raw";

        // --- Validate input file existence ---
        if !Path::new(data_file_path).exists() {
            return Err(anyhow::anyhow!("Data file not found: {}", data_file_path));
        }
        if !Path::new(domain_file_path).exists() {
            return Err(anyhow::anyhow!("Domain file not found: {}", domain_file_path));
        }

        // --- Ensure output directory exists ---
        fs::create_dir_all(output_path)?;

        // --- Pre-compute expected target file paths ---
        let csv_file = format!("{}/{}.csv", output_path, dataset_name);
        let parquet_file = format!("{}/{}.parquet", output_path, dataset_name);

        // --- Fast path: outputs already present ---
        if Path::new(&csv_file).exists() && Path::new(&parquet_file).exists() {
            if verbosity != Verbosity::Silent {
                println!("✅ Polar format already exists for {} - skipping conversion", dataset_name);
                println!("   CSV: {}", csv_file);
                println!("   Parquet: {}", parquet_file);
            }
            
            // TODO: Auto-copy to processed directory even when skipping conversion
            // This ensures the training pipeline always has access to processed data
            let processed_dir = "model-input-data/processed";
            fs::create_dir_all(processed_dir)?;
            
            let processed_csv = format!("{}/{}.csv", processed_dir, dataset_name);
            if !Path::new(&processed_csv).exists() {
                if verbosity != Verbosity::Silent {
                    println!("📋 Copying to processed directory for training pipeline...");
                }
                fs::copy(&csv_file, &processed_csv)?;
                if verbosity != Verbosity::Silent {
                    println!("✅ Processed data ready at: {}", processed_csv);
                }
            }
            
            return Ok(output_path.to_string());
        }

        // --- Perform actual conversion dispatch ---
        if verbosity != Verbosity::Silent {
            println!("🔄 Converting {} to Polars format...", dataset_name);
        }
        
        // --- Generic CSV conversion (no hardcoded dataset names) ---
        Self::convert_csv_dataset(data_file_path, output_path, dataset_name, verbosity).await?;

        Ok(output_path.to_string())
    }


    /// Convert California Housing dataset to Polars format (CSV + Parquet) and print basic stats.
    /// Convert any CSV dataset to Polars format (CSV + Parquet) and print basic stats.
    ///
    /// This is a generic converter that assumes the input is a comma-separated file
    /// without headers. Column names are preserved as-is from the input file.
    async fn convert_csv_dataset(
        data_file_path: &str,
        output_path: &str,
        dataset_name: &str,
        verbosity: Verbosity,
    ) -> Result<()> {
        // --- Announce conversion start ---
        if verbosity != Verbosity::Silent {
            println!("📊 Converting {} dataset to Polars format...", dataset_name);
        }

        // --- Target file paths ---
        let csv_file = format!("{}/{}.csv", output_path, dataset_name);
        let parquet_file = format!("{}/{}.parquet", output_path, dataset_name);

        // --- Read raw CSV data with generic column handling ---
        let mut df = LazyCsvReader::new(data_file_path)
            .with_has_header(false)
            .with_separator(b',')
            .finish()?
            .collect()?;

        // --- Apply dataset-specific column transformations ---
        if dataset_name == "CaliforniaHousing" {
            println!("🔧 Applying California Housing column transformations...");
            df = CaliforniaHousingProcessor::transform_columns(df.lazy())?;
            println!("✅ Column transformations applied successfully");
            
            if verbosity != Verbosity::Silent {
                println!("📋 Transformed columns: {:?}", df.get_column_names());
            }
        }

        // --- Optional preview output ---
        if verbosity != Verbosity::Silent {
            println!("📈 Dataset shape: {} rows × {} columns", df.height(), df.width());
            println!("📋 First few rows:");
            println!("{}", df.head(Some(5)));
        }

        // --- Persist as CSV ---
        let mut csv_file_handle = std::fs::File::create(&csv_file)?;
        CsvWriter::new(&mut csv_file_handle)
            .include_header(true)
            .finish(&mut df.clone())?;
        if verbosity != Verbosity::Silent {
            println!("💾 Saved CSV to: {}", csv_file);
        }

        // --- Persist as Parquet ---
        let parquet_file_handle = std::fs::File::create(&parquet_file)?;
        ParquetWriter::new(parquet_file_handle)
            .finish(&mut df.clone())?;
        if verbosity != Verbosity::Silent {
            println!("💾 Saved Parquet to: {}", parquet_file);
        }

        // --- Basic statistics summary ---
        if verbosity != Verbosity::Silent {
            let row_count = df.height();
            let column_count = df.width();
            println!("\n📊 Dataset Statistics:");
            println!("   Rows: {}", row_count);
            println!("   Columns: {}", column_count);
        }

        // TODO: Auto-copy converted files to processed directory for training pipeline
        // This is a temporary solution until we implement proper data preprocessing
        let processed_dir = "model-input-data/processed";
        fs::create_dir_all(processed_dir)?;
        
        let processed_csv = format!("{}/{}.csv", processed_dir, dataset_name);
        if verbosity != Verbosity::Silent {
            println!("📋 Copying to processed directory for training pipeline...");
        }
        
        fs::copy(&csv_file, &processed_csv)?;
        
        if verbosity != Verbosity::Silent {
            println!("✅ Processed data ready at: {}", processed_csv);
        }

        Ok(())
    }
}
