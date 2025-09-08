# DNN Structure

## Modular Architecture

**PyTorch-style neural network implementation**

```
src/
├── main.rs             # California Housing training example
├── init/               # Weight initialization (Xavier, Kaiming, Normal, Uniform)
├── functional/         # Pure functions (activations, linear ops)
├── losses/             # Loss functions (MSE, MAE, CrossEntropy)
├── nn/                # Core modules (Linear, ReLU, Sequential, Module trait)
├── optim/             # Optimizers (SGD, Adam)
└── transforms/        # Data preprocessing (StandardScaler, MinMaxScaler)
```

### Key Components

- **Module Trait**: `forward()`, `backward()`, `update_parameters()`
- **Linear Layer**: Dense/fully-connected with customizable init
- **Sequential**: Chains modules together
- **Parameter**: Manages data + gradients
- **StandardScaler**: Z-score normalization for features
- **DataProcessor**: Pipeline for raw → processed data transformation

### Usage Pattern
```rust
let model = Sequential::new()
    .add_module(Linear::new(8, 64)?)
    .add_module(ReLU::new())
    .add_module(Linear::new(64, 1)?);

let optimizer = SGD::new(0.01);
let criterion = MSELoss::new();
// Standard training loop: forward → loss → backward → step
```

### Example Implementation
The main.rs contains a complete California Housing regression example:
- **Data Pipeline**: Raw data → StandardScaler → processed data → model input
- **Dataset**: 20,640 samples, 8 features → house prices
- **Architecture**: 8 → 64 → 32 → 1 (with ReLU activations)
- **Training**: 100 epochs, batch size 32, SGD optimizer
- **Performance**: MSE loss, sample predictions output

Data flows: `raw/` → `transforms/StandardScaler` → `processed/` → neural network
All components use Polars DataFrames for efficient tensor operations.
