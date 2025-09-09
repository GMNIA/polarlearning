# PolarLearning

A dual-implementation neural network project with **Rust** and **Python** versions that solve the same California Housing regression task.

## Structure

- `datasets/` — Shared California Housing dataset
- `rust/` — Rust implementation using Polars + custom NN layers
- `python/` — Python implementation using PyTorch + scikit-learn

## Parallel Goals

Both projects:
- Load the same California Housing dataset
- Apply identical preprocessing (StandardScaler)
- Train neural networks with the same architecture (8→64→32→1 with ReLU)
- Use MAE loss and SGD optimizer
- Save models and evaluate test performance

## Quick Start

```bash
# Rust version
cd rust && ./build_fast.sh

# Python version  
cd python && ./build_fast.sh
```

Each implementation is fully containerized with Docker for reproducible results.
