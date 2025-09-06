# Development Guide 🛠️

This guide covers development workflows, build processes, and deployment strategies for PolarLearning.

## 🔄 Development Modes

### 🧪 Development Mode (Volume Mounted)
**Purpose**: Fast iteration, debugging, and experimentation

**Characteristics**:
- Source code mounted from host
- Build cache persisted between runs
- Output files accessible on host machine
- Real-time code changes

**Commands**:
```bash
# Setup (one-time)
docker volume create polarlearning-target-cache

# Build development image
docker build -t polarlearning-dev .

# Run with mounted volumes
docker run --rm \
    -v "$(pwd):/workspace" \
    -v "polarlearning-target-cache:/workspace/target" \
    --workdir /workspace \
    polarlearning-dev
```

**File Locations**:
- **Source**: `./src/` (editable on host)
- **Executable**: `./target/release/polarlearning` (accessible on host)
- **Output**: `./model-input-data/raw/` (accessible on host)
- **Cache**: Docker volume (persistent between runs)

### 🚀 Production Mode (Self-Contained)
**Purpose**: Deployment, distribution, and isolated execution

**Characteristics**:
- Everything bundled in container
- No external dependencies
- Consistent runtime environment
- Smaller attack surface

**Commands**:
```bash
# Build production image
docker build -t polarlearning-prod .

# Run production container
docker run --rm polarlearning-prod

# Deploy to registry
docker tag polarlearning-prod myregistry/polarlearning:v1.0
docker push myregistry/polarlearning:v1.0
```

**File Locations**:
- **Source**: Inside container only
- **Executable**: `/workspace/target/release/polarlearning` (container only)
- **Output**: Inside container (lost after run)
- **Cache**: None (fresh build each time)

## 🐚 Development Scripts

### `build_fast.sh` - Development Build Script
```bash
#!/bin/bash
# Fast build script with persistent Docker volumes for Rust build cache

echo "🚀 Fast Polar Learning Build Script"

# Create named volume for Rust build cache if it doesn't exist
echo "📦 Creating/ensuring Docker volume for build cache..."
docker volume create polarlearning-target-cache > /dev/null 2>&1

# Build the Docker image (this will cache dependencies)
echo "🔨 Building Docker image with cached dependencies..."
docker build -t polarlearning-fast .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Build complete! Running with persistent cache..."

# Run the container with mounted volumes for fast rebuilds
docker run --rm \
    -v "$(pwd):/workspace" \
    -v "polarlearning-target-cache:/workspace/target" \
    --workdir /workspace \
    polarlearning-fast

echo "🎉 Execution complete!"
```

**Usage**: `./build_fast.sh`

### `dev_shell.sh` - Interactive Development Shell
```bash
#!/bin/bash
# Interactive development shell with mounted volumes

echo "🔧 Starting PolarLearning Development Shell..."

# Ensure volume exists
echo "📦 Creating build cache volume..."
docker volume create polarlearning-target-cache > /dev/null 2>&1

# Build development image
echo "🔨 Building development image..."
docker build -t polarlearning-dev .

# Start interactive shell
echo "🐚 Starting interactive shell..."
docker run -it --rm \
    -v "$(pwd):/workspace" \
    -v "polarlearning-target-cache:/workspace/target" \
    --workdir /workspace \
    polarlearning-dev bash

echo "👋 Development shell exited"
```

**Usage**: `./dev_shell.sh`

## 🔧 Build Process Deep Dive

### Dockerfile Stages
```dockerfile
# Stage 1: Base Environment Setup
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    curl ca-certificates pkg-config build-essential \
    clang llvm libssl-dev python3 git

# Stage 2: Rust Toolchain Installation  
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default stable && rustup component add clippy rustfmt

# Stage 3: Dependency Caching (Optimization)
WORKDIR /workspace
COPY Cargo.toml Cargo.lock* ./
RUN mkdir -p src && echo "fn main() {}" > src/main.rs
RUN cargo build --release  # Cache dependencies
RUN rm src/main.rs && rm -rf src

# Stage 4: Source Code & Final Build
COPY . .
RUN cargo build --release  # Only app code compiles

# Stage 5: Runtime Configuration
ENV RUST_LOG=info
CMD ["./target/release/polarlearning"]
```

### Build Optimization Strategies

#### 1. **Dependency Caching**
- Pre-compile dependencies with dummy `main.rs`
- Only recompile application code on changes
- Reduces build time from ~5 minutes to ~30 seconds

#### 2. **Volume Mounting**
- Persistent `target/` directory across builds
- Incremental compilation benefits
- Shared build artifacts between containers

#### 3. **Multi-Stage Builds** (Future Enhancement)
```dockerfile
# Development stage
FROM base as development
COPY . .
RUN cargo build

# Production stage  
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 as production
COPY --from=development /workspace/target/release/polarlearning .
CMD ["./polarlearning"]
```

## 🚀 Deployment Strategies

### Local Development
```bash
# Quick iteration cycle
1. Edit code in VS Code
2. Run: docker build -t polar-dev .
3. Run: docker run --rm -v $(pwd):/workspace polar-dev
4. Check results in ./model-input-data/raw/
```

### CI/CD Pipeline
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t polarlearning-test .
      - name: Run tests
        run: docker run --rm polarlearning-test cargo test
      
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Build production image
        run: docker build -t polarlearning:${{ github.sha }} .
      - name: Push to registry
        run: docker push polarlearning:${{ github.sha }}
```

### Cloud Deployment

#### AWS ECS
```json
{
  "family": "polarlearning",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "polarlearning",
      "image": "myregistry/polarlearning:v1.0",
      "memory": 2048,
      "cpu": 1024,
      "essential": true
    }
  ]
}
```

#### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: polarlearning
spec:
  replicas: 3
  selector:
    matchLabels:
      app: polarlearning
  template:
    metadata:
      labels:
        app: polarlearning
    spec:
      containers:
      - name: polarlearning
        image: polarlearning:v1.0
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## 🔍 Debugging & Troubleshooting

### Common Issues

#### 1. **Script Permissions**
```bash
# Make scripts executable
chmod +x build_fast.sh dev_shell.sh
```

#### 2. **Docker Volume Permissions**
```bash
# Linux: Permission denied errors
docker run --rm -v $(pwd):/workspace --user $(id -u):$(id -g) polarlearning-dev
```

#### 3. **Build Cache Issues**
```bash
# Clear Docker build cache
docker builder prune -f

# Remove and recreate volume
docker volume rm polarlearning-target-cache
docker volume create polarlearning-target-cache
```

#### 4. **CUDA Warnings**
```
WARNING: The NVIDIA Driver was not detected.
```
This is normal if you don't have NVIDIA GPU drivers installed. The application will run on CPU.

### Debug Mode
```bash
# Run with debug output
docker run --rm -e RUST_LOG=debug polarlearning-dev

# Interactive debugging
docker run -it --rm polarlearning-dev bash
# Inside container:
cargo run
```

## 📈 Performance Monitoring

### Build Times
- **First build**: ~5-8 minutes (dependency compilation)
- **Incremental builds**: ~30 seconds (code changes only)
- **Cached builds**: ~5 seconds (no changes)

### Runtime Performance
- **Dataset size**: 20,640 rows × 9 columns
- **Processing time**: < 1 second
- **Memory usage**: ~100MB peak
- **Output size**: CSV 1.2MB → Parquet 375KB (70% reduction)

---

This development guide provides everything needed to efficiently work with the PolarLearning project across different environments and deployment scenarios.
