#!/bin/bash

echo "🚀 Fast Verbosity Testing (with cached builds)"
echo "=============================================="

echo ""
echo "=== Testing SILENT mode ==="
# Update to use Silent
sed -i 's/Verbosity::Normal/Verbosity::Silent/' src/main.rs
rm -rf model-input-data/raw/*
# Fast incremental build (deps already cached)
cargo build --release
./target/release/polarlearning

echo ""
echo "=== Testing NORMAL mode ==="
# Update to use Normal  
sed -i 's/Verbosity::Silent/Verbosity::Normal/' src/main.rs
rm -rf model-input-data/raw/*
# Fast incremental build (deps already cached)
cargo build --release
./target/release/polarlearning

echo ""
echo "✅ Verbosity testing complete!"
