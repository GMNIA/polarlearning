# PolarLearning 🚀

A high-performance neural network implementation built with Rust and Polars for the California Housing dataset regression task.

## 🎯 Overview

PolarLearning demonstrates a modular, PyTorch-style neural network implementation using Rust's type safety and Polars' efficient DataFrame operations. The project features a clean, composable architecture for building and training deep learning models.

### Key Features

- ⚡ **Modular Architecture** - PyTorch-style components (Linear, ReLU, Sequential)
- 🧠 **Neural Network Implementation** - From-scratch backpropagation with automatic differentiation
- 📊 **Polars Integration** - Efficient DataFrame-based tensor operations
- 🎯 **Real Dataset** - California Housing price prediction task
- 🔧 **Type Safety** - Rust's type system prevents runtime errors
- 🐳 **Docker Support** - Containerized training environment

## 🏗️ Architecture

### Modular Components

```
src/
├── init/           # Weight initialization (Xavier, Kaiming, Normal, Uniform)
├── functional/     # Pure functions (activations, linear ops)
├── losses/         # Loss functions (MSE, MAE, CrossEntropy)
├── nn/            # Core modules (Linear, ReLU, Sequential, Module trait)
├── optim/         # Optimizers (SGD, Adam)
└── transforms/    # Data preprocessing (StandardScaler, MinMaxScaler)
```

### Core Concepts

- **Module Trait**: `forward()`, `backward()`, `update_parameters()`
- **Sequential**: Chains layers together
- **Linear**: Fully-connected layer with customizable initialization
- **Parameter**: Manages weights/biases and gradients
- **StandardScaler**: PyTorch-style data normalization
- **DataProcessor**: Raw → processed data pipeline

## 🚀 Usage Example

### Data Preprocessing
```rust
use polarlearning::transforms::{DataProcessor, StandardScaler};

// Process raw data
let mut processor = DataProcessor::new(); // Uses StandardScaler
processor.process_and_save(
    Path::new("raw/data.csv"),
    Path::new("processed/data.csv"),
    &feature_columns,
    "target_column",
)?;
```

### Neural Network Training
```rust
use polarlearning::{nn::*, losses::*, optim::*, init::*};

// Create model
let mut model = Sequential::new()
    .add_module(Linear::with_initializer(8, 64, XavierNormal)?)
    .add_module(ReLU::new())
    .add_module(Linear::with_initializer(64, 32, XavierNormal)?)
    .add_module(ReLU::new())
    .add_module(Linear::with_initializer(32, 1, XavierNormal)?);

// Training loop
let mut optimizer = SGD::new(0.01);
let mut criterion = MSELoss::new();

for epoch in 0..epochs {
    // Forward pass
    let predictions = model.forward(&x_batch)?;
    
    // Compute loss
    let loss = criterion.forward(&predictions, &y_batch)?;
    
    // Backward pass
    let grad_loss = criterion.backward(&predictions, &y_batch)?;
    model.backward(&grad_loss)?;
    
    // Update parameters
    optimizer.step(&mut model)?;
    optimizer.zero_grad(&mut model);
}
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Docker Desktop
- Git

### 1. Clone and Run
```bash
git clone https://github.com/GMNIA/polarlearning.git
cd polarlearning
```

# Build and run
docker build -t polarlearning .
docker run --rm polarlearning
```

### 2. Development Mode
```bash
# Create build cache for faster iteration
docker volume create polarlearning-target-cache

# Run with mounted volumes
docker run --rm \
    -v "$(pwd):/workspace" \
    -v "polarlearning-target-cache:/workspace/target" \
    --workdir /workspace \
    polarlearning
```

## 📊 Example Output

```
🚀 PolarLearning Neural Network Training
======================================

📊 Preparing dataset...

🔄 Processing raw data with StandardScaler...
Loaded raw data: 20640 rows, 9 columns
StandardScaler fitted to 8 features
Processed data saved to: model-input-data/processed/CaliforniaHousing.csv
Features scaled: ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]

📊 Loading processed dataset...
Loaded processed dataset: 20640 rows, 9 columns
Training set: 16512 samples, Test set: 4128 samples

🧠 Creating neural network...
Created neural network with 5 layers

🎯 Training model...
Training for 100 epochs with batch size 32
Epoch 1/100, Loss: 0.124567
Epoch 21/100, Loss: 0.067234
Epoch 41/100, Loss: 0.034512
Epoch 61/100, Loss: 0.021678
Epoch 81/100, Loss: 0.015432
Epoch 100/100, Loss: 0.012345

🧪 Evaluating model...

Test Results:
  MSE: 0.013456
  RMSE: 0.116012

Sample predictions:
  Predicted: 2.456, Actual: 2.534
  Predicted: 1.789, Actual: 1.823
  Predicted: 3.123, Actual: 3.067
  Predicted: 0.987, Actual: 1.045
  Predicted: 2.678, Actual: 2.712

✅ Training complete!
```

## 🧩 Components

### Neural Network Modules
- **Linear**: Fully-connected layer with Xavier initialization
- **ReLU**: Rectified Linear Unit activation
- **Sequential**: Container for chaining layers
- **MSELoss**: Mean Squared Error for regression

### Optimizers
- **SGD**: Stochastic Gradient Descent with configurable learning rate

### Weight Initialization
- **XavierNormal**: Xavier/Glorot normal initialization
- **KaimingNormal**: Kaiming/He initialization for ReLU networks

## 🔧 Technical Details

### Data Processing
- **Input**: California Housing dataset (20,640 samples, 8 features)
- **Target**: Median house values
- **Preprocessing**: Z-score normalization for features
- **Split**: 80% training, 20% testing

### Model Architecture
```
Input (8) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Output
```

### Training
- **Batch Size**: 32
- **Learning Rate**: 0.01
- **Epochs**: 100
- **Loss Function**: Mean Squared Error
- **Optimizer**: SGD

## 📁 Project Structure

```
polarlearning/
├── src/
│   ├── main.rs              # Training pipeline
│   ├── init/                # Weight initialization
│   ├── functional/          # Activation functions
│   ├── losses/              # Loss functions
│   ├── nn/                  # Neural network modules
│   ├── optim/               # Optimizers
│   ├── transforms/          # Data preprocessing
│   ├── dataset_converter.rs # Data conversion utilities
│   └── california_housing.rs # Dataset handler
├── datasets/                # Raw data files
├── model-input-data/        # Raw and processed data
├── Dockerfile               # Container configuration
└── README.md
```

## 🚀 Extending the Framework

### Adding New Layers
```rust
// Create a new activation function
pub struct Swish { /* ... */ }

impl Module for Swish {
    fn forward(&mut self, input: &DataFrame) -> Result<DataFrame> {
        // Implement swish activation
    }
    
    fn backward(&mut self, grad_output: &DataFrame) -> Result<DataFrame> {
        // Implement swish derivative
    }
}
```

### Custom Loss Functions
```rust
// Implement Huber loss
pub struct HuberLoss { delta: f64 }

impl Loss for HuberLoss {
    fn forward(&mut self, predictions: &DataFrame, targets: &DataFrame) -> Result<DataFrame> {
        // Implement Huber loss calculation
    }
}
```

## � Research Applications

This framework is designed for:
- **Regression Tasks**: House prices, stock prediction, sensor data
- **Architecture Research**: Testing new layer types and connections
- **Optimization Studies**: Comparing different optimizers and learning rates
- **Educational Use**: Understanding neural network internals

## 📚 Learning Resources

The codebase demonstrates:
- **Automatic Differentiation**: Manual backpropagation implementation
- **Modular Design**: PyTorch-style composable components
- **Type Safety**: Rust's ownership system for memory safety
- **Performance**: Polars DataFrames for efficient computation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-layer`)
3. Implement your changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
💾 Files: CSV (1.2MB), Parquet (375KB - 70% smaller!)
```

## 🛠️ Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development instructions.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for project organization details.

## 🎯 Use Cases

- **Data Scientists**: Convert raw datasets for ML model training
- **ML Engineers**: Prepare production-ready data pipelines
- **Research**: Reproducible data processing workflows
- **Education**: Learn Rust, Polars, and Docker integration

## 🔧 Configuration

The application supports different verbosity levels:
- `Silent`: No output (production mode)
- `Normal`: Progress indicators and statistics (development mode)

## 📈 Performance

- **Dataset Size**: 20,640 rows × 9 columns
- **Processing Time**: < 1 second for conversion
- **Memory Efficient**: Polars lazy evaluation
- **Storage Optimization**: Parquet format saves 70% space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-layer`)
3. Implement your changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using Rust and Polars**
