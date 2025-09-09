#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Hyperparameter tuning for California Housing regression"
echo "=========================================================="

# Build image first
docker build -t polarlearning-py . >/dev/null 2>&1

# Function to run with specific hyperparameters
run_experiment() {
    local epochs=$1
    local lr=$2
    local patience=$3
    local description=$4
    
    echo ""
    echo "🧪 Experiment: $description"
    echo "   EPOCHS=$epochs, LR=$lr, PATIENCE=$patience"
    
    docker run --rm \
        -v "$(pwd):/workspace" \
        -v "$(cd .. && pwd)/datasets:/workspace/datasets" \
        --workdir /workspace \
        -e DATASETS_DIR=/workspace/datasets \
        -e EPOCHS=$epochs \
        -e LR=$lr \
        -e PATIENCE=$patience \
        polarlearning-py python src/main.py | grep -E "(🎯|MAE|R²|Training Summary)" || true
}

# Current baseline (200 epochs, lr=1e-3)
run_experiment 200 0.001 20 "Baseline (current settings)"

# Higher learning rate
run_experiment 200 0.01 20 "Higher LR (10x)"

# Lower learning rate with more epochs
run_experiment 300 0.0005 30 "Lower LR, more epochs"

# Adaptive learning rate (already using ReduceLROnPlateau)
run_experiment 150 0.005 15 "Medium LR, shorter patience"

echo ""
echo "✅ Hyperparameter tuning complete!"
