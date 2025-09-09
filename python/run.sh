#!/usr/bin/env bash
set -euo pipefail

PARENT_DIR="$(cd .. && pwd)"

echo "▶️  Running training..."
docker run --rm \
  -v "$(pwd):/workspace" \
  -v "${PARENT_DIR}/datasets:/workspace/datasets" \
  --workdir /workspace \
  -e DATASETS_DIR=/workspace/datasets \
  polarlearning-py
