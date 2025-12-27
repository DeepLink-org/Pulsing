#!/bin/bash
# 大规模压测启动脚本

set -e

# 默认参数
DURATION=${DURATION:-30}
RATE=${RATE:-100}
SEED_NODES=${SEED_NODES:-""}

# 解析参数
ARGS=""
if [ -n "$SEED_NODES" ]; then
    ARGS="--seed-nodes $SEED_NODES"
fi

echo "=========================================="
echo "Large Scale Stress Test"
echo "=========================================="
echo "Duration: ${DURATION}s"
echo "Rate: ${RATE} req/s"
echo "Processes: 10"
echo "Workers: 5 types (echo, compute, stream, batch, stateful)"
echo "=========================================="
echo ""

# 使用torchrun启动10个进程
# 注意：大规模压测时，增加稳定等待时间以避免节点被误判为失败
# 如果端口被占用，可以设置 MASTER_PORT 环境变量使用不同端口
MASTER_PORT=${MASTER_PORT:-29500}

# 检查端口是否被占用
if command -v lsof >/dev/null 2>&1; then
    if lsof -Pi :$MASTER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Warning: Port $MASTER_PORT is already in use!"
        echo "   You can set MASTER_PORT environment variable to use a different port:"
        echo "   MASTER_PORT=29501 ./benchmarks/run_stress_test.sh"
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

torchrun \
    --nproc_per_node=10 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=$MASTER_PORT \
    benchmarks/large_scale_stress_test.py \
    --duration $DURATION \
    --rate $RATE \
    --stabilize-timeout 5 \
    $ARGS

echo ""
echo "=========================================="
echo "Stress test completed!"
echo "=========================================="
echo "Logs: stress_test_logs/stress_test_rank_*.log"
echo "Stats: stress_test_logs/stress_test_stats_rank_*.json"
echo ""
echo "View logs:"
echo "  ./benchmarks/view_logs.sh all      - Show all logs"
echo "  ./benchmarks/view_logs.sh errors   - Show errors"
echo "  ./benchmarks/view_logs.sh summary  - Show summary"
echo "  ./benchmarks/view_logs.sh 0        - Show rank 0 log"
echo "=========================================="
