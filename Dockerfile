# CUDA-enabled Rust dev+run image (simple and reliable)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ----- System deps for Rust, Polars, and general builds -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates pkg-config build-essential clang llvm \
    libssl-dev python3 git && \
    rm -rf /var/lib/apt/lists/*

# ----- Install Rust toolchain -----
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default stable && rustup component add clippy rustfmt

# Workdir is your project root (same folder as this Dockerfile)
WORKDIR /app

# ----- Cache dependencies first (faster rebuilds) -----
COPY Cargo.toml ./
# create a dummy file so cargo can resolve deps without your sources yet
RUN mkdir -p src && echo "fn main() {}" > src/main.rs && cargo fetch

# ----- Bring in your actual project sources (from this path) -----
COPY . .

# ----- Build once at image build time (optimized) -----
RUN cargo build --release

# Helpful defaults
ENV RUST_LOG=info

# ----- Run your app on container start -----
# This will use the already-built binary; if sources changed and you rebuild the image,
# it will recompile during docker build, not at runtime.
CMD ["bash", "-lc", "cargo run --release"]
