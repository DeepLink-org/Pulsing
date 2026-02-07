#!/bin/bash
#
# 多进程压测（multiprocessing，不依赖 torchrun）
#
# Usage:
#   ./benchmarks/run_stress_multiprocessing.sh
#   NPROCS=8 DURATION=30 ./benchmarks/run_stress_multiprocessing.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

NPROCS=${NPROCS:-4}
DURATION=${DURATION:-20}
OUTPUT=${OUTPUT:-}

echo "=========================================="
echo "Multiprocessing Stress (Queue + Topic)"
echo "=========================================="
echo "Processes: $NPROCS"
echo "Duration:  ${DURATION}s"
echo "=========================================="

if [ -n "$OUTPUT" ]; then
  python benchmarks/stress_multiprocessing.py --nprocs "$NPROCS" --duration "$DURATION" --output "$OUTPUT"
else
  python benchmarks/stress_multiprocessing.py --nprocs "$NPROCS" --duration "$DURATION"
fi
