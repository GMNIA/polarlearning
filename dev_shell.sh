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
