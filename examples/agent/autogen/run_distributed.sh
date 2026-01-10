#!/bin/bash
# 分布式 Agent 启动脚本
# Usage:
#   ./run_distributed.sh              # torchrun 模式
#   ./run_distributed.sh --manual     # 手动模式

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

NPROC=3
MODE="torchrun"
PULSING_BASE_PORT=19000

while [[ $# -gt 0 ]]; do
    case $1 in
        --manual)  MODE="manual"; shift ;;
        --nproc)   NPROC="$2"; shift 2 ;;
        --port)    PULSING_BASE_PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--manual] [--nproc N] [--port PORT]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "Mode: $MODE | Processes: $NPROC | Port: $PULSING_BASE_PORT"
echo ""

if [ "$MODE" = "torchrun" ]; then
    PULSING_BASE_PORT=$PULSING_BASE_PORT torchrun \
        --nproc_per_node=$NPROC \
        --master_addr=127.0.0.1 \
        --master_port=29500 \
        distributed.py
else
    cleanup() { kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; }
    trap cleanup EXIT INT TERM

    python distributed.py writer &
    sleep 1
    python distributed.py editor &
    sleep 1
    python distributed.py manager
    sleep 1
fi

echo "Done!"
