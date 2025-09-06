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
