# Python training env with PyTorch + Polars
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /workspace

# Copy dependency manifests first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Default envs
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLBACKEND=Agg

# Entrypoint
CMD ["python", "src/main.py"]
