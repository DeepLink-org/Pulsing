#!/bin/bash
#
# 运行 Queue & Topic 基线吞吐 Benchmark（单节点）
#
# Usage:
#   ./benchmarks/run_baseline_throughput.sh
#   DURATION=15 ./benchmarks/run_baseline_throughput.sh
#   python benchmarks/baseline_throughput.py --topic-only --topic-subscribers 3
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

DURATION=${DURATION:-10}
OUTPUT=${OUTPUT:-}

echo "=========================================="
echo "Baseline Throughput (Queue + Topic)"
echo "=========================================="
echo "Duration: ${DURATION}s per benchmark"
echo "=========================================="

if [ -n "$OUTPUT" ]; then
  python benchmarks/baseline_throughput.py --duration "$DURATION" --output "$OUTPUT"
else
  python benchmarks/baseline_throughput.py --duration "$DURATION"
fi
