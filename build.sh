#!/bin/bash
set -e

# Build core module
maturin develop

# Build benchmark module
maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
