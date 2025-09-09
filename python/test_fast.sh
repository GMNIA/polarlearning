#!/usr/bin/env bash
set -euo pipefail

echo "🧪 Fast test (Docker) — tests/test_long_epochs.py"

# Ensure Docker volume for pip cache exists
docker volume create polarlearning-py-pip-cache >/dev/null 2>&1 || true

# Build image (cached layers)
echo "🔨 Building Docker image (cached) ..."
docker build -t polarlearning-py . >/dev/null

# Resolve paths
PARENT_DIR="$(cd .. && pwd)"

# Run pytest inside container with mounted project and datasets
echo "▶️  Running tests inside container..."
docker run --rm \
  -v "$(pwd):/workspace" \
  -v "${PARENT_DIR}/datasets:/workspace/datasets" \
  -v "polarlearning-py-pip-cache:/root/.cache/pip" \
  --workdir /workspace \
  -e DATASETS_DIR=/workspace/datasets \
  polarlearning-py \
  pytest -q tests/test_long_epochs.py

rc=$?
if [ $rc -eq 0 ]; then
  echo "✅ tests/test_long_epochs.py passed"
else
  echo "❌ tests/test_long_epochs.py failed (exit $rc)"
fi
exit $rc
