# Development (Python)

This mirrors the Rust fast-build approach with Docker caching and mounted datasets.

## Scripts

- build.sh — standard build (Dockerfile)
- run.sh — run container and execute training (main.py)
- dev.sh — interactive shell (basic)
- build_fast.sh — fast build & run with pip cache volume and datasets mount
- dev_shell_fast.sh — interactive dev shell with pip cache and datasets mount

## Datasets

Parent ../datasets is mounted into the container as /workspace/datasets.
You can override via DATASETS_DIR.

## Tips

- Set EPOCHS and LR env vars for quick experiments.
- Use pytest inside dev_shell_fast.sh to run tests: pytest.
