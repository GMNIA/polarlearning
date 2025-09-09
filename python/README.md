# PolarLearning (Python)

Python implementation using PyTorch, Polars for data processing, and Docker containerization.

## Features
- PyTorch neural networks with standard components
- Polars + scikit-learn for data preprocessing
- MAE loss + SGD optimizer  
- California Housing regression task

## Quick Start
```bash
./build_fast.sh       # Build & run with caching
./dev_shell_fast.sh   # Interactive development
pytest               # Run tests
```

Architecture: 8→64→32→1 with ReLU activation, matching the Rust version for comparison.
