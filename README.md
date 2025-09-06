# PolarLearning 🚀

A high-performance data processing and machine learning pipeline built with Rust and Polars, designed for efficient dataset conversion and preparation.

## 🎯 Overview

PolarLearning is a Rust-based data processing tool that converts raw datasets into ML-ready formats (CSV and Parquet) using the powerful Polars library. The project demonstrates modern data engineering practices with Docker containerization and efficient caching strategies.

### Key Features

- ⚡ **High-Performance Data Processing** - Built with Polars for lightning-fast DataFrame operations
- 🐳 **Docker-First Approach** - Separate development and production environments
- 📊 **Smart Dataset Conversion** - Automatic detection of existing conversions to avoid redundant work
- 🔧 **CUDA Support** - GPU-accelerated processing capabilities
- 📈 **Statistical Analysis** - Built-in dataset statistics and validation
- 🔄 **Flexible Verbosity** - Silent and normal modes for different use cases

## 🏃‍♂️ Quick Start

### Prerequisites
- Docker Desktop
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/GMNIA/polarlearning.git
cd polarlearning

# Make scripts executable
chmod +x build_fast.sh dev_shell.sh
```

### 2. Development Mode (Fast Iteration)
```bash
# Create persistent build cache
docker volume create polarlearning-target-cache

# Build the development image
docker build -t polarlearning-fast .

# Run with mounted volumes for development
docker run --rm \
    -v "$(pwd):/workspace" \
    -v "polarlearning-target-cache:/workspace/target" \
    --workdir /workspace \
    polarlearning-fast
```

### 3. Production Mode (Self-Contained)
```bash
# Build and run in production mode
docker build -t polarlearning-prod .
docker run --rm polarlearning-prod
```

## 📁 What It Does

The application processes the California Housing dataset:

1. **Input**: Raw data files from `datasets/CaliforniaHousing/`
2. **Processing**: 
   - Converts generic column names to meaningful ones
   - Applies data transformations
   - Generates statistical summaries
3. **Output**: 
   - Clean CSV file (`model-input-data/raw/CaliforniaHousing.csv`)
   - Optimized Parquet file (`model-input-data/raw/CaliforniaHousing.parquet`)

### Sample Output
```
📊 Dataset shape: 20640 rows × 9 columns
📋 Columns: longitude, latitude, housing_median_age, total_rooms, 
           total_bedrooms, population, households, median_income, median_house_value
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
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Polars](https://pola.rs/) - Amazing DataFrame library for Rust
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) - GPU acceleration support
- California Housing Dataset from StatLib repository

---

**Built with ❤️ using Rust and Polars**
