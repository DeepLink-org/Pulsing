#!/bin/bash
# Ray 压测启动脚本 - 单进程版本（正确的 Ray 使用方式）
#
# Ray 设计为单 driver 进程 + 多 Actor，不应使用 torchrun 多进程模式。

set -e

DURATION=${DURATION:-30}
RATE=${RATE:-100}
NUM_WORKERS=${NUM_WORKERS:-50}

echo "=========================================="
echo "Ray Stress Test (Single Process Mode)"
echo "=========================================="
echo "Duration: ${DURATION}s"
echo "Rate: ${RATE} req/s"
echo "Workers per type: ${NUM_WORKERS}"
echo "Total Workers: $((NUM_WORKERS * 5))"
echo "=========================================="
echo ""

python benchmarks/large_scale_stress_test_ray_single.py \
    --duration $DURATION \
    --rate $RATE \
    --num-workers $NUM_WORKERS

echo ""
echo "=========================================="
echo "Stress test completed!"
echo "=========================================="
echo "Log: benchmark_logs/stress_test_ray_single.log"
echo "Stats: benchmark_logs/stress_test_stats_ray_single.json"
echo "=========================================="
