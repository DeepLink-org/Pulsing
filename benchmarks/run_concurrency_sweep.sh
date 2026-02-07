#!/bin/bash
#
# 多并发扫描：不同 生产者/消费者 组合下的吞吐
#
# Usage:
#   ./benchmarks/run_concurrency_sweep.sh
#   DURATION=5 ./benchmarks/run_concurrency_sweep.sh
#   python benchmarks/concurrency_sweep.py --producers 1 2 4 --consumers 1 2 4 --output sweep.json
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

DURATION=${DURATION:-8}
OUTPUT=${OUTPUT:-}

echo "=========================================="
echo "Concurrency Sweep (P producers, C consumers)"
echo "=========================================="
echo "Duration per (P,C): ${DURATION}s"
echo "Producers/Consumers: 1,2,4,8 (default)"
echo "=========================================="

if [ -n "$OUTPUT" ]; then
  python benchmarks/concurrency_sweep.py --duration "$DURATION" --output "$OUTPUT"
else
  python benchmarks/concurrency_sweep.py --duration "$DURATION"
fi
