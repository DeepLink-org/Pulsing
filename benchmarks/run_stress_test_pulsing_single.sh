#!/bin/bash
# Pulsing 压测启动脚本 - 单进程版本（与 Ray 单进程版本等价对比）

set -e

DURATION=${DURATION:-30}
RATE=${RATE:-100}
NUM_WORKERS=${NUM_WORKERS:-50}

echo "=========================================="
echo "Pulsing Stress Test (Single Process Mode)"
echo "=========================================="
echo "Duration: ${DURATION}s"
echo "Rate: ${RATE} req/s"
echo "Workers per type: ${NUM_WORKERS}"
echo "Total Workers: $((NUM_WORKERS * 5))"
echo "=========================================="
echo ""

python benchmarks/large_scale_stress_test_pulsing_single.py \
    --duration $DURATION \
    --rate $RATE \
    --num-workers $NUM_WORKERS

echo ""
echo "=========================================="
echo "Stress test completed!"
echo "=========================================="
echo "Log: benchmark_logs/stress_test_pulsing_single.log"
echo "Stats: benchmark_logs/stress_test_stats_pulsing_single.json"
echo "=========================================="
