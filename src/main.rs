mod dataset_converter;

use anyhow::Result;
use dataset_converter::{DatasetConverter, Verbosity};

#[tokio::main]
async fn main() -> Result<()> {
    // Define paths for California Housing dataset
    let data_file_path = "datasets/CaliforniaHousing/cal_housing.data";
    let domain_file_path = "datasets/CaliforniaHousing/cal_housing.domain";
    
    // Process California Housing dataset
    let output_path = DatasetConverter::convert_if_needed(
        "CaliforniaHousing", 
        data_file_path, 
        domain_file_path,
        Verbosity::Normal
    ).await?;
    
    println!("✅ Dataset available in Polars format at: {}", output_path);
    
    Ok(())
}
