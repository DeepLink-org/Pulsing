#!/bin/bash
# Complete demo script for Pulsing Inspect CLI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    pkill -f "demo_service.py" 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Trap to cleanup on exit
trap cleanup EXIT INT TERM

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Check if pulsing command is available
if ! command -v pulsing &> /dev/null; then
    echo -e "${RED}Error: pulsing command not found. Please install pulsing first.${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pulsing Inspect CLI Demo${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Start nodes
echo -e "${GREEN}Step 1: Starting demo service nodes...${NC}"

echo "  Starting node 1 (seed) on port 8000..."
echo "  $ python3 examples/inspect/demo_service.py --port 8000"
python3 examples/inspect/demo_service.py --port 8000 > /tmp/pulsing_node1.log 2>&1 &
NODE1_PID=$!
echo "    ✓ Node 1 started (PID: $NODE1_PID)"

sleep 3

echo "  Starting node 2 on port 8001..."
echo "  $ python3 examples/inspect/demo_service.py --port 8001 --seed 127.0.0.1:8000"
python3 examples/inspect/demo_service.py --port 8001 --seed 127.0.0.1:8000 > /tmp/pulsing_node2.log 2>&1 &
NODE2_PID=$!
echo "    ✓ Node 2 started (PID: $NODE2_PID)"

sleep 3

echo "  Starting node 3 on port 8002..."
echo "  $ python3 examples/inspect/demo_service.py --port 8002 --seed 127.0.0.1:8000"
python3 examples/inspect/demo_service.py --port 8002 --seed 127.0.0.1:8000 > /tmp/pulsing_node3.log 2>&1 &
NODE3_PID=$!
echo "    ✓ Node 3 started (PID: $NODE3_PID)"

echo ""
echo -e "${GREEN}✓ All nodes started${NC}"
echo ""

# Wait for cluster to stabilize
echo -e "${YELLOW}Waiting for cluster to stabilize...${NC}"
sleep 5

# Step 2: Inspect cluster
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 2: Inspect Cluster${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}$ pulsing inspect cluster --seeds 127.0.0.1:8000${NC}"
pulsing inspect cluster --seeds 127.0.0.1:8000
echo ""

# Wait a bit
sleep 2

# Step 3: Inspect actors
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 3: Inspect Actors${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}All actors:${NC}"
echo -e "${YELLOW}$ pulsing inspect actors --seeds 127.0.0.1:8000${NC}"
pulsing inspect actors --seeds 127.0.0.1:8000
echo ""

echo -e "${YELLOW}Top 5 actors:${NC}"
echo -e "${YELLOW}$ pulsing inspect actors --seeds 127.0.0.1:8000 --top 5${NC}"
pulsing inspect actors --seeds 127.0.0.1:8000 --top 5
echo ""

echo -e "${YELLOW}Filtered (worker):${NC}"
echo -e "${YELLOW}$ pulsing inspect actors --seeds 127.0.0.1:8000 --filter worker${NC}"
pulsing inspect actors --seeds 127.0.0.1:8000 --filter worker
echo ""

# Step 4: Inspect metrics
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 4: Inspect Metrics${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Key metrics summary:${NC}"
echo -e "${YELLOW}$ pulsing inspect metrics --seeds 127.0.0.1:8000 --raw False${NC}"
pulsing inspect metrics --seeds 127.0.0.1:8000 --raw False
echo ""

# Step 5: Watch mode (short demo)
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 5: Watch Mode (3 rounds)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Watching cluster changes (3 rounds, 2s interval)...${NC}"
echo -e "${YELLOW}$ pulsing inspect watch --seeds 127.0.0.1:8000 --kind cluster --interval 2.0 --max_rounds 3${NC}"
echo ""
pulsing inspect watch --seeds 127.0.0.1:8000 --kind cluster --interval 2.0 --max_rounds 3
echo ""

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Demo Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Nodes are still running. You can:"
echo "  - Run more inspect commands manually"
echo "  - Check logs: /tmp/pulsing_node*.log"
echo "  - Press Ctrl+C to stop all nodes"
echo ""
echo "Try these commands:"
echo "  pulsing inspect cluster --seeds 127.0.0.1:8000"
echo "  pulsing inspect actors --seeds 127.0.0.1:8000 --top 10"
echo "  pulsing inspect metrics --seeds 127.0.0.1:8000"
echo "  pulsing inspect watch --seeds 127.0.0.1:8000 --kind all"
echo ""

# Keep running until interrupted
echo -e "${YELLOW}Press Ctrl+C to stop all nodes and exit...${NC}"
wait
