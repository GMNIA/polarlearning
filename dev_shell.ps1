#!/usr/bin/env pwsh
# Development shell with persistent build cache

Write-Host "🛠️  Opening development shell with persistent build cache..." -ForegroundColor Green

# Ensure volume exists
docker volume create polarlearning-target-cache | Out-Null

# Start interactive development container
docker run -it --rm `
    -v "${PWD}:/workspace" `
    -v "polarlearning-target-cache:/workspace/target" `
    --workdir /workspace `
    polarlearning-fast /bin/bash

Write-Host "👋 Development session ended." -ForegroundColor Green
