#!/usr/bin/env bash
set -euo pipefail

# Mount parent datasets dir so paths like datasets/CaliforniaHousing work
PARENT_DIR="$(cd .. && pwd)"

echo "🛠️  Starting dev container..."
docker run -it --rm \
  -v "$(pwd):/workspace" \
  -v "${PARENT_DIR}/datasets:/workspace/datasets" \
  -v "$(pwd)/models:/workspace/models" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  -v "$(pwd)/model-input-data:/workspace/model-input-data" \
  --workdir /workspace \
  polarlearning-py bash
