//! California Housing dataset-specific transformation utilities.
//!
//! Contains column mappings and transformations specific to the California Housing
//! dataset from the StatLib repository. This module handles the domain-specific
//! knowledge while keeping the core converter generic.

use polars::prelude::*;
use anyhow::Result;
use crate::dataset_converter::Verbosity;

/// California Housing dataset column transformations and processing.
pub struct CaliforniaHousingProcessor;


impl CaliforniaHousingProcessor {
    /// Apply California Housing specific column transformations to a raw DataFrame.
    ///
    /// Transforms generic column names (column_1, column_2, etc.) into meaningful
    /// domain-specific names based on the known schema of the California Housing dataset.
    pub fn transform_columns(df: LazyFrame) -> Result<DataFrame> {
        // --- Apply semantic column names for California Housing dataset ---
        // Column order: Longitude, Latitude, HousingMedianAge, TotalRooms, TotalBedrooms,
        // Population, Households, MedianIncome, MedianHouseValue
        let transformed_df = df
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
