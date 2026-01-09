# Justfile for Pulsing

# Default target
default: dev

# =============================================================================
# Development
# =============================================================================

# Install all packages in development mode
dev:
    @echo "Building core..."
    maturin develop
    @echo "Building benchmarks..."
    maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
    @echo "Ready to code!"

# Build release wheels
build:
    maturin build --release
    maturin build --release --manifest-path crates/pulsing-bench-py/Cargo.toml

# =============================================================================
# Testing & QA
# =============================================================================

# Run all tests
test: test-rust test-python

# Run Rust tests
test-rust:
    cargo test --workspace --exclude pulsing-bench-py --exclude pulsing-py

# Run Python tests
test-python:
    pytest tests/python

# Format all code (Rust + Python)
fmt:
    cargo fmt
    ruff format .

# Lint all code
lint:
    cargo clippy --workspace -- -D warnings
    ruff check .

# =============================================================================
# Maintenance
# =============================================================================

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/
    rm -rf **/*.so
    rm -rf **/*.pyd
