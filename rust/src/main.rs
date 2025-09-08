use polars::prelude::*;
use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::fs;
use std::env;

// Import our modular components
pub mod init;
pub mod functional;
pub mod losses;
pub mod nn;
pub mod optim;
pub mod transforms;
pub mod datasets;
pub mod persistence;

use init::XavierNormal;
use losses::{MAELoss, Loss};
use nn::{Module, Sequential, Linear, ReLU};
use optim::{SGD, Optimizer};
use transforms::DataProcessor;
use datasets::{DatasetConverter, Verbosity, CaliforniaHousingProcessor};
use persistence::{
    ModelPersistence, TrainingTracker, TrainingConfig, 
    ScalingParameters, EvaluationMetrics
};

/// Load and preprocess the California Housing dataset
fn load_california_housing() -> Result<(DataFrame, DataFrame, DataFrame, DataFrame)> {
    let processed_path = Path::new("model-input-data/processed/CaliforniaHousing.csv");
    
    let df = LazyCsvReader::new(processed_path)
        .with_has_header(true)
        .finish()?
        .collect()?;
    
    println!("Loaded processed dataset: {} rows, {} columns", df.height(), df.width());
    
    // Extract features (first 8 columns) and target (last column)
    let feature_cols: Vec<String> = df.get_column_names()[..8].iter().map(|s| s.to_string()).collect();
    let target_col = df.get_column_names()[8];
    
    let features = df.select(&feature_cols)?;
    let target = df.select([target_col.as_str()])?;
    
    // Split into train/test (80/20) - data is already scaled
    let n_samples = df.height();
    let train_size = (n_samples as f64 * 0.8) as usize;
    
    let x_train = features.slice(0, train_size);
    let y_train = target.slice(0, train_size);
    let x_test = features.slice(train_size as i64, n_samples - train_size);
    let y_test = target.slice(train_size as i64, n_samples - train_size);
    
    println!("Training set: {} samples, Test set: {} samples", x_train.height(), x_test.height());
    
    Ok((x_train, y_train, x_test, y_test))
}

/// Create a neural network using the modular architecture
fn create_model() -> Result<Sequential> {
    let model = Sequential::new()
        .add_module(Linear::with_initializer(8, 64, XavierNormal)?)
        .add_module(ReLU::new())
        .add_module(Linear::with_initializer(64, 32, XavierNormal)?)
        .add_module(ReLU::new())
        .add_module(Linear::with_initializer(32, 1, XavierNormal)?);
    
    println!("Created neural network with {} layers", model.len());
    Ok(model)
}

/// Training loop with comprehensive tracking
fn train_model_with_tracking(
    model: &mut Sequential,
    x_train: &DataFrame,
    y_train: &DataFrame,
    x_test: &DataFrame,
    y_test: &DataFrame,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
) -> Result<TrainingTracker> {
    let mut optimizer = SGD::new(learning_rate);
    let criterion = MAELoss::default();
    let mut tracker = TrainingTracker::new();
    let debug_train = std::env::var("DEBUG_TRAIN").ok().map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false);
    
    let n_samples = x_train.height();
    let n_batches = (n_samples + batch_size - 1) / batch_size;
    
    println!("Training for {} epochs with batch size {}", epochs, batch_size);
    
    // Ensure model is in training mode
    model.train();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        if debug_train { println!("[DEBUG] Epoch {}/{} start: {} batches", epoch + 1, epochs, n_batches); }
        
        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, n_samples);
            let current_batch_size = end_idx - start_idx;
            
            // Get batch
            let x_batch = x_train.slice(start_idx as i64, current_batch_size);
            let y_batch = y_train.slice(start_idx as i64, current_batch_size);
            
            // Zero gradients
            optimizer.zero_grad(model);
            
            // Forward pass
            if debug_train { println!("[DEBUG] Epoch {}, batch {}/{}: forward...", epoch + 1, batch_idx + 1, n_batches); }
            let t0 = std::time::Instant::now();
            let predictions = model.forward(&x_batch)?;
            if debug_train { println!("[DEBUG] forward done in {:.3?}", t0.elapsed()); }
            
            // Compute loss
            if debug_train { println!("[DEBUG] computing loss..."); }
            let t1 = std::time::Instant::now();
            let loss_value = criterion.forward(&predictions, &y_batch)?;
            if debug_train { println!("[DEBUG] loss computed in {:.3?} -> {:.6}", t1.elapsed(), loss_value); }
            epoch_loss += loss_value;
            
            // Track batch loss
            tracker.add_batch_loss(loss_value);
            
            // Backward pass
            if debug_train { println!("[DEBUG] backward... (loss backward) "); }
            let t2 = std::time::Instant::now();
            let grad_loss = criterion.backward(&predictions, &y_batch)?;
            if debug_train { println!("[DEBUG] loss backward done in {:.3?}", t2.elapsed()); }

            if debug_train { println!("[DEBUG] model backward..."); }
            let t3 = std::time::Instant::now();
            model.backward(&grad_loss)?;
            if debug_train { println!("[DEBUG] model backward done in {:.3?}", t3.elapsed()); }
            
            // Update parameters
            if debug_train { println!("[DEBUG] optimizer step..."); }
            let t4 = std::time::Instant::now();
            optimizer.step(model)?;
            if debug_train { println!("[DEBUG] optimizer step done in {:.3?}", t4.elapsed()); }
        }
        
        if n_batches > 0 { epoch_loss /= n_batches as f64; }
        
        // Track epoch metrics
        tracker.add_epoch(epoch_loss, learning_rate);
        
        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("Epoch {}/{}, Loss: {:.6}", epoch + 1, epochs, epoch_loss);
        }
    }
    
    // Get final predictions for tracking
    model.eval();
    let final_predictions = model.forward(x_test)?;
    tracker.set_predictions(final_predictions, y_test.clone());
    
    Ok(tracker)
}

/// Evaluate the model and return metrics
fn evaluate_model(
    model: &mut Sequential,
    x_test: &DataFrame,
    y_test: &DataFrame,
) -> Result<EvaluationMetrics> {
    model.eval();
    
    let predictions = model.forward(x_test)?;
    
    let criterion = MAELoss::default();
    let test_mse = criterion.forward(&predictions, y_test)?;
    let test_rmse = test_mse.sqrt();
    
    println!("\nTest Results:");
    println!("  MSE: {:.6}", test_mse);
    println!("  RMSE: {:.6}", test_rmse);
    
    // Show sample predictions
    println!("\nSample predictions:");
    for i in 0..std::cmp::min(5, predictions.height()) {
        let pred = predictions.get_row(i)?.0[0].extract::<f64>().unwrap_or(0.0);
        let actual = y_test.get_row(i)?.0[0].extract::<f64>().unwrap_or(0.0);
        println!("  Predicted: {:.3}, Actual: {:.3}", pred, actual);
    }
    
    Ok(EvaluationMetrics {
        train_mse: 0.0, // Would be calculated during training
        test_mse,
        train_rmse: 0.0, // Would be calculated during training
        test_rmse,
        train_mae: None,
        test_mae: None,
        r2_score: None,
        final_predictions_saved: true,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 PolarLearning Neural Network Training");
    println!("======================================");
    
    println!("DEBUG: Starting main function");
    
    // Dataset preparation
    println!("\n📊 Preparing dataset...");
    // Resolve dataset location: prefer CWD datasets/, else ../datasets/, else env var DATASETS_DIR
    let datasets_dir = std::env::var("DATASETS_DIR").ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            let here = PathBuf::from("datasets/CaliforniaHousing");
            if here.exists() { Some(PathBuf::from("datasets")) } else { None }
        })
        .or_else(|| {
            let parent = PathBuf::from("../datasets/CaliforniaHousing");
            if parent.exists() { Some(PathBuf::from("../datasets")) } else { None }
        })
        .unwrap_or_else(|| PathBuf::from("datasets"));

    let data_file_path = datasets_dir.join("CaliforniaHousing/cal_housing.data");
    let domain_file_path = datasets_dir.join("CaliforniaHousing/cal_housing.domain");
    
    println!("DEBUG: About to call convert_if_needed");
    
    // Convert dataset if needed
    let output_path = DatasetConverter::convert_if_needed(
        "CaliforniaHousing",
        data_file_path.to_string_lossy().as_ref(),
        domain_file_path.to_string_lossy().as_ref(),
        Verbosity::Normal,
    ).await?;
    
    println!("DEBUG: convert_if_needed completed, output_path: {}", output_path);
    
    // Load the dataset (already has proper column names from conversion)
    let parquet_file = format!("{}/CaliforniaHousing.parquet", output_path);
    println!("DEBUG: About to load parquet file: {}", parquet_file);
    let df = LazyFrame::scan_parquet(parquet_file, ScanArgsParquet::default())?
        .collect()
        .context("Failed to load dataset")?;
    
    // Process raw data and save to processed directory
    println!("\n🔄 Processing raw data with StandardScaler...");
    let mut processor = DataProcessor::new(); // Uses StandardScaler by default
    
    let feature_columns = vec![
        "longitude".to_string(), "latitude".to_string(), "housing_median_age".to_string(),
        "total_rooms".to_string(), "total_bedrooms".to_string(), "population".to_string(),
        "households".to_string(), "median_income".to_string()
    ];
    
    processor.process_and_save(
        Path::new("model-input-data/raw/CaliforniaHousing.csv"),
        Path::new("model-input-data/processed/CaliforniaHousing.csv"),
        &feature_columns,
        "median_house_value",
    )?;
    
    // Load and preprocess data
    println!("\n📊 Loading processed dataset...");
    let (x_train, y_train, x_test, y_test) = load_california_housing()
        .context("Failed to load California Housing dataset")?;
    
    // Create model
    println!("\n🧠 Creating neural network...");
    let mut model = create_model()
        .context("Failed to create model")?;
    
    // Setup training configuration (configurable via env vars)
    // Defaults: epochs=1, batch_size defaults to ~1/3 of training set size (rounded up)
    let epochs = env::var("EPOCHS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
    // Compute default batch as ceil(n/3), with a minimum of 1
    let n_train = x_train.height();
    let default_batch = std::cmp::max(1, (n_train + 9) / 10);
    let batch_size = env::var("BATCH_SIZE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default_batch);
    println!("Training configuration: epochs={}, batch_size={}", epochs, batch_size);

    let training_config = TrainingConfig {
        batch_size,
        epochs,
        train_test_split: 0.8,
        random_seed: Some(42),
        loss_function: "MSE".to_string(),
        early_stopping_patience: None,
        early_stopping_threshold: None,
    };
    
    // Setup scaling parameters
    let scaling_params = ScalingParameters {
        feature_scaling_type: "StandardScaler".to_string(),
        target_scaling_type: Some("StandardScaler".to_string()),
        feature_columns: feature_columns.clone(),
        target_column: "median_house_value".to_string(),
        scaling_saved: true,
    };
    
    // Train model with comprehensive tracking
    println!("\n🎯 Training model...");
    let tracker = train_model_with_tracking(
        &mut model, 
        &x_train, 
        &y_train, 
        &x_test, 
        &y_test,
    training_config.epochs, 
    training_config.batch_size, 
        0.01
    ).context("Failed to train model")?;
    
    // Evaluate model
    println!("\n🧪 Evaluating model...");
    let _evaluation = evaluate_model(&mut model, &x_test, &y_test)
        .context("Failed to evaluate model")?;
    
    // Save complete model state
    println!("\n💾 Saving complete model state...");
    let optimizer = SGD::new(0.01);
    let timestamp = ModelPersistence::save_complete_model(
        &model,
        &tracker,
        &optimizer,
        &training_config,
        &scaling_params,
        Path::new("models"),
    ).context("Failed to save model")?;
    
    println!("\n✅ Training complete!");
    println!("🎉 Model and all training data saved with timestamp: {}", timestamp);
    println!("📁 Check 'models/' and 'outputs/' folders for all saved files");
    
    Ok(())
}
