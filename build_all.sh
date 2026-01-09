#!/bin/bash
# Build script for Pulsing with separate core and bench modules
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Building Pulsing modules...${NC}"

# Build core module (_core.so ~7.6MB)
echo -e "${GREEN}[1/2] Building _core module (Actor System)...${NC}"
maturin build --release

# Build bench module (_bench.so ~15MB)
echo -e "${GREEN}[2/2] Building _bench module (Benchmark Tools)...${NC}"
maturin build --release --manifest-path crates/pulsing-bench-py/Cargo.toml

echo -e "${BLUE}Build complete!${NC}"
echo ""
echo "Wheels are in: target/wheels/"
ls -lh target/wheels/*.whl
echo ""
echo "Install with:"
echo "  pip install target/wheels/pulsing-*.whl"
echo "  pip install target/wheels/pulsing_bench-*.whl  # Optional: only if you need benchmark tools"
echo ""
echo "For development, use:"
echo "  maturin develop"
echo "  maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml  # Optional"
