use polars::prelude::*;
use std::path::Path;
use anyhow::Result;
use std::fs;

/// Verbosity levels for dataset conversion operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// Silent mode - no output
    Silent = 0,
    /// Normal mode - basic progress and results
    Normal = 1,
    // TODO: Add more verbosity levels:
    // TODO: Verbose = 2,  // Detailed progress with timing
    // TODO: Debug = 3,    // Debug information and intermediate steps
    // TODO: Trace = 4,    // Full trace with data samples
}

impl Default for Verbosity {
    fn default() -> Self {
        Verbosity::Silent
    }
}

impl From<u8> for Verbosity {
    fn from(level: u8) -> Self {
        match level {
            0 => Verbosity::Silent,
            1 => Verbosity::Normal,
            // TODO: Handle additional levels when implemented
            _ => Verbosity::Normal, // Default to Normal for unknown levels
        }
    }
}

/// Data converter that handles conversion from raw datasets to Polars format
pub struct DatasetConverter;

impl DatasetConverter {
    /// Convert a dataset to Polars format if needed
    /// 
    /// This function checks if the dataset folder contains the data files and
    /// if the corresponding Polars format already exists in model-input-data/raw.
    /// If not, it performs the conversion from raw data to CSV and Parquet formats.
    pub async fn convert_if_needed(
        dataset_name: &str, 
        data_file_path: &str, 
        domain_file_path: &str,
        verbosity: Verbosity
    ) -> Result<String> {
        let output_path = "model-input-data/raw";
        
        // Check if input files exist
        if !Path::new(data_file_path).exists() {
            return Err(anyhow::anyhow!("Data file not found: {}", data_file_path));
        }
        
        if !Path::new(domain_file_path).exists() {
            return Err(anyhow::anyhow!("Domain file not found: {}", domain_file_path));
        }
        
        // Create output directory
        fs::create_dir_all(output_path)?;
        
        // Check if polar format already exists
        let csv_file = format!("{}/{}.csv", output_path, dataset_name);
        let parquet_file = format!("{}/{}.parquet", output_path, dataset_name);
        
        if Path::new(&csv_file).exists() && Path::new(&parquet_file).exists() {
            if verbosity != Verbosity::Silent {
                println!("✅ Polar format already exists for {} - skipping conversion", dataset_name);
                println!("   CSV: {}", csv_file);
                println!("   Parquet: {}", parquet_file);
            }
            return Ok(output_path.to_string());
        }
        
        if verbosity != Verbosity::Silent {
            println!("🔄 Converting {} to Polars format...", dataset_name);
        }
        
        // Perform conversion based on dataset name
        match dataset_name {
            "CaliforniaHousing" => {
                Self::convert_california_housing(data_file_path, output_path, dataset_name, verbosity).await?;
            }
            _ => {
                return Err(anyhow::anyhow!("Unknown dataset: {}", dataset_name));
            }
        }
        
        Ok(output_path.to_string())
    }
    
    /// Convert California Housing dataset to Polars format
    async fn convert_california_housing(
        data_file_path: &str, 
        output_path: &str, 
        dataset_name: &str,
        verbosity: Verbosity
    ) -> Result<()> {
        if verbosity != Verbosity::Silent {
            println!("📊 Converting California Housing dataset to Polars format...");
        }
        
        let csv_file = format!("{}/{}.csv", output_path, dataset_name);
        let parquet_file = format!("{}/{}.parquet", output_path, dataset_name);
        
        // Read the raw data file using LazyCsvReader
        // Column order: Longitude, Latitude, HousingMedianAge, TotalRooms, TotalBedrooms, Population, Households, MedianIncome, MedianHouseValue
        let df = LazyCsvReader::new(data_file_path)
            .with_has_header(false)
            .with_separator(b',')
            .finish()?
            .with_columns([
                col("column_1").alias("longitude"),
                col("column_2").alias("latitude"), 
                col("column_3").alias("housing_median_age"),
                col("column_4").alias("total_rooms"),
                col("column_5").alias("total_bedrooms"),
                col("column_6").alias("population"),
                col("column_7").alias("households"),
                col("column_8").alias("median_income"),
                col("column_9").alias("median_house_value"),
            ])
            .select([
                col("longitude"),
                col("latitude"),
                col("housing_median_age"),
                col("total_rooms"),
                col("total_bedrooms"),
                col("population"),
                col("households"),
                col("median_income"),
                col("median_house_value"),
            ])
            .collect()?;
        
        if verbosity != Verbosity::Silent {
            println!("📈 Dataset shape: {} rows × {} columns", df.height(), df.width());
            println!("📋 First few rows:");
            println!("{}", df.head(Some(5)));
        }
        
        // Save as CSV with proper headers
        let mut csv_file_handle = std::fs::File::create(&csv_file)?;
        CsvWriter::new(&mut csv_file_handle)
            .include_header(true)
            .finish(&mut df.clone())?;
        
        if verbosity != Verbosity::Silent {
            println!("💾 Saved CSV to: {}", csv_file);
        }
        
        // Save as Parquet (more efficient for Polars)
        let parquet_file_handle = std::fs::File::create(&parquet_file)?;
        ParquetWriter::new(parquet_file_handle)
            .finish(&mut df.clone())?;
        
        if verbosity != Verbosity::Silent {
            println!("💾 Saved Parquet to: {}", parquet_file);
        }
        
        // Show some basic statistics
        if verbosity != Verbosity::Silent {
            let row_count = df.height();
            let stats = df.lazy()
                .select([
                    lit(row_count as i64).alias("row_count"),
                    col("median_house_value").mean().alias("mean_price"),
                    col("median_house_value").median().alias("median_price"),
                    col("median_income").mean().alias("mean_income"),
                    col("total_rooms").sum().alias("total_rooms_sum"),
                ])
                .collect()?;
            
            println!("\n📊 Dataset Statistics:");
            println!("{}", stats);
        }
        
        Ok(())
    }
}
