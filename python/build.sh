#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Building python/polarlearning image"
# Build with caching; CUDA runtime base already includes torch
docker build -t polarlearning-py .

echo "✅ Build complete"
