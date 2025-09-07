# Project Structure

```
polarlearning/
├── Cargo.toml
├── Dockerfile
├── build_fast.ps1
├── dev_shell.ps1
├── DNN_STRUCTURE.md
├── datasets/
│   └── CaliforniaHousing/
├── model-input-data/
│   ├── processed/
│   └── raw/
├── src/
│   ├── main.rs                    # Training pipeline
│   ├── dataset_converter.rs       # Data conversion utilities
│   ├── california_housing.rs      # Dataset handler
│   ├── init/                      # Weight initialization
│   ├── functional/                # Pure functions (activations, linear)
│   ├── losses/                    # Loss functions
│   ├── nn/                        # Neural network modules
│   ├── optim/                     # Optimizers
│   └── transforms/                # Data preprocessing (StandardScaler, MinMaxScaler)
└── target/
```


