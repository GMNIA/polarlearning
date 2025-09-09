#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Fast build (Python) with cached deps"

# Ensure shared caches/volumes exist
echo "📦 Ensuring Docker volume for pip cache..."
docker volume create polarlearning-py-pip-cache >/dev/null 2>&1 || true

echo "🔨 Building Docker image (layer-cached) ..."
docker build -t polarlearning-py .

echo "✅ Build complete! Running with mounted caches and datasets..."
PARENT_DIR="$(cd .. && pwd)"
docker run --rm \
  -v "$(pwd):/workspace" \
  -v "${PARENT_DIR}/datasets:/workspace/datasets" \
  -v "polarlearning-py-pip-cache:/root/.cache/pip" \
  --workdir /workspace \
  -e DATASETS_DIR=/workspace/datasets \
  polarlearning-py

echo "🎉 Execution complete"
