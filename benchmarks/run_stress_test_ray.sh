#!/bin/bash
# 大规模压测启动脚本 - Ray版本

set -e

# 默认参数
DURATION=${DURATION:-30}
RATE=${RATE:-100}
MASTER_PORT=${MASTER_PORT:-29500}

# 解析参数
ARGS=""
if [ -n "$RAY_ADDRESS" ]; then
    ARGS="--address $RAY_ADDRESS"
fi

echo "=========================================="
echo "Large Scale Stress Test (Ray Version)"
echo "=========================================="
echo "Duration: ${DURATION}s"
echo "Rate: ${RATE} req/s"
echo "Processes: 10"
echo "Workers: 5 types (echo, compute, stream, batch, stateful)"
echo "Master Port: ${MASTER_PORT}"
echo "=========================================="
echo ""

# 检查端口是否被占用
if command -v lsof >/dev/null 2>&1; then
    if lsof -Pi :$MASTER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Warning: Port $MASTER_PORT is already in use!"
        echo "   You can set MASTER_PORT environment variable to use a different port:"
        echo "   MASTER_PORT=29501 ./benchmarks/run_stress_test_ray.sh"
        echo ""
        echo "   Or kill the process using the port:"
        echo "   lsof -ti :$MASTER_PORT | xargs kill -9"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# 使用torchrun启动10个进程
torchrun \
    --nproc_per_node=10 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=$MASTER_PORT \
    benchmarks/large_scale_stress_test_ray.py \
    --duration $DURATION \
    --rate $RATE \
    --stabilize-timeout 5 \
    $ARGS

echo ""
echo "=========================================="
echo "Stress test completed!"
echo "=========================================="
echo "Logs: benchmark_logs/stress_test_ray_rank_*.log"
echo "Stats: benchmark_logs/stress_test_stats_ray_rank_*.json"
echo "=========================================="
