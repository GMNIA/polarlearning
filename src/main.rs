//! CLI entry point for the polarlearning data preparation tool.
//!
//! Converts raw datasets into Polars-friendly formats (CSV + Parquet) if
//! they are not already present under `model-input-data/raw/`.

mod dataset_converter;
mod california_housing;

use anyhow::Result;
use dataset_converter::{DatasetConverter, Verbosity};
use california_housing::CaliforniaHousingProcessor;
use polars::prelude::*;


/// Program entry point: ensures the California Housing dataset is converted
/// into Polars formats (CSV + Parquet) and reports the resulting location.
#[tokio::main]
async fn main() -> Result<()> {
    let verbosity = Verbosity::Normal;
    
    // --- Define input paths for the California Housing dataset ---
    let data_file_path = "datasets/CaliforniaHousing/cal_housing.data";
    let domain_file_path = "datasets/CaliforniaHousing/cal_housing.domain";

    // --- Perform generic conversion ---
    let output_path = DatasetConverter::convert_if_needed(
        "CaliforniaHousing",
        data_file_path,
        domain_file_path,
        verbosity,
    ).await?;

    // --- Apply California Housing specific transformations ---
    let csv_file = format!("{}/CaliforniaHousing.csv", output_path);
    let parquet_file = format!("{}/CaliforniaHousing.parquet", output_path);
    
    // --- Check if we need to apply transformations ---
    if std::path::Path::new(&csv_file).exists() {
        if verbosity != Verbosity::Silent {
            println!("🔄 Applying California Housing specific transformations...");
        }
        
        // --- Load the generic CSV ---
        let df = LazyCsvReader::new(&csv_file)
            .with_has_header(true)
            .finish()?;
            
        // --- Apply domain-specific transformations ---
        let transformed_df = CaliforniaHousingProcessor::transform_columns(df)?;
        
        // --- Save transformed version ---
        let mut csv_file_handle = std::fs::File::create(&csv_file)?;
        CsvWriter::new(&mut csv_file_handle)
            .include_header(true)
            .finish(&mut transformed_df.clone())?;
            
        let parquet_file_handle = std::fs::File::create(&parquet_file)?;
        ParquetWriter::new(parquet_file_handle)
            .finish(&mut transformed_df.clone())?;
            
        // --- Print domain-specific statistics ---
        CaliforniaHousingProcessor::print_statistics(&transformed_df, verbosity)?;
    }

    // --- Report final output location ---
    if verbosity != Verbosity::Silent {
        println!("✅ Dataset available in Polars format at: {}", output_path);
    }

    Ok(())
}
