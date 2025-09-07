//! California Housing dataset-specific transformation utilities.
//!
//! Contains column mappings and transformations specific to the California Housing
//! dataset from the StatLib repository. This module handles the domain-specific
//! knowledge while keeping the core converter generic.

use polars::prelude::*;
use anyhow::Result;
use crate::datasets::converter::Verbosity;

/// California Housing dataset column transformations and processing.
pub struct CaliforniaHousingProcessor;


impl CaliforniaHousingProcessor {
    /// Apply California Housing specific column transformations to a raw DataFrame.
    ///
    /// Maps generic column_1, column_2, etc. to proper feature names.
    pub fn transform_columns(df: LazyFrame) -> Result<DataFrame> {
        // The raw data file has no headers, so Polars generates column_1, column_2, etc.
        // We need to rename them to the actual California Housing feature names:
        // longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        // population, households, median_income, median_house_value
        
        let transformed_df = df
            .rename([
                "column_1", "column_2", "column_3", "column_4", "column_5",
                "column_6", "column_7", "column_8", "column_9"
            ], [
                "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                "population", "households", "median_income", "median_house_value"
            ])
            .collect()?;

        Ok(transformed_df)
    }


    /// Print California Housing specific statistics.
    ///
    /// Displays domain-relevant statistics like price metrics and housing characteristics.
    pub fn print_statistics(df: &DataFrame, verbosity: Verbosity) -> Result<()> {
        // --- Only print if verbosity allows ---
        if verbosity == Verbosity::Silent {
            return Ok(());
        }

        // --- Calculate domain-specific statistics ---
        let row_count = df.height();
        let stats = df.clone().lazy()
            .select([
                lit(row_count as i64).alias("row_count"),
                col("median_house_value").mean().alias("mean_price"),
                col("median_house_value").median().alias("median_price"),
                col("median_income").mean().alias("mean_income"),
                col("total_rooms").sum().alias("total_rooms_sum"),
            ])
            .collect()?;

        println!("\n📊 California Housing Dataset Statistics:");
        println!("{}", stats);

        Ok(())
    }
}
