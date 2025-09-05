#!/usr/bin/env pwsh
# Fast build script with persistent Docker volumes for Rust build cache

Write-Host "🚀 Fast Polar Learning Build Script" -ForegroundColor Green

# Create named volume for Rust build cache if it doesn't exist
Write-Host "📦 Creating/ensuring Docker volume for build cache..." -ForegroundColor Yellow
docker volume create polarlearning-target-cache | Out-Null

# Build the Docker image (this will cache dependencies)
Write-Host "🔨 Building Docker image with cached dependencies..." -ForegroundColor Yellow
docker build -t polarlearning-fast .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build complete! Running with persistent cache..." -ForegroundColor Green

# Run the container with mounted volumes for fast rebuilds
docker run --rm `
    -v "${PWD}:/workspace" `
    -v "polarlearning-target-cache:/workspace/target" `
    --workdir /workspace `
    polarlearning-fast

Write-Host "🎉 Execution complete!" -ForegroundColor Green
