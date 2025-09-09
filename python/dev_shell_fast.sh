#!/usr/bin/env bash
set -euo pipefail

echo "🛠️  Opening Python dev shell with cached pip and mounted datasets..."

docker volume create polarlearning-py-pip-cache >/dev/null 2>&1 || true
PARENT_DIR="$(cd .. && pwd)"

docker run -it --rm \
  -v "$(pwd):/workspace" \
  -v "${PARENT_DIR}/datasets:/workspace/datasets" \
  -v "polarlearning-py-pip-cache:/root/.cache/pip" \
  --workdir /workspace \
  -e DATASETS_DIR=/workspace/datasets \
  polarlearning-py bash

echo "👋 Development session ended."
