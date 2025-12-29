#!/bin/bash
#
# 分布式内存队列压测脚本
#
# 使用 torchrun 启动 10 个进程，每个进程同时作为生产者和消费者
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 默认参数
NPROC=${NPROC:-10}
DURATION=${DURATION:-30}
NUM_BUCKETS=${NUM_BUCKETS:-16}
BATCH_SIZE=${BATCH_SIZE:-100}
RECORD_SIZE=${RECORD_SIZE:-100}

echo "=========================================="
echo "Distributed Queue Benchmark"
echo "=========================================="
echo "Processes: ${NPROC}"
echo "Duration: ${DURATION}s"
echo "Buckets: ${NUM_BUCKETS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Record size: ${RECORD_SIZE} bytes"
echo "=========================================="
echo ""

# 清理旧数据
rm -rf ./queue_benchmark_data
rm -rf ./benchmark_logs/queue_benchmark_*.log
rm -rf ./benchmark_logs/queue_benchmark_*.json

# 运行压测
torchrun --nproc_per_node="$NPROC" benchmarks/queue_benchmark.py \
    --duration "$DURATION" \
    --num-buckets "$NUM_BUCKETS" \
    --batch-size "$BATCH_SIZE" \
    --record-size "$RECORD_SIZE" \
    --log-dir benchmark_logs

echo ""
echo "=========================================="
echo "Aggregating results..."
echo "=========================================="

# 汇总结果
python3 -c "
import json
import os
import glob

log_dir = 'benchmark_logs'
result_files = glob.glob(os.path.join(log_dir, 'queue_benchmark_rank_*.json'))

if not result_files:
    print('No result files found')
    exit(1)

total_writes = 0
total_reads = 0
total_write_latency = 0
total_read_latency = 0
successful_writes = 0
successful_reads = 0
duration = 0

for f in result_files:
    with open(f) as fp:
        data = json.load(fp)
        r = data['results']
        total_writes += r['writes']['records_written']
        total_reads += r['reads']['records_read']
        total_write_latency += r['writes']['avg_latency_ms'] * r['writes']['successful']
        total_read_latency += r['reads']['avg_latency_ms'] * r['reads']['successful']
        successful_writes += r['writes']['successful']
        successful_reads += r['reads']['successful']
        duration = max(duration, r['duration_s'])

avg_write_latency = total_write_latency / successful_writes if successful_writes > 0 else 0
avg_read_latency = total_read_latency / successful_reads if successful_reads > 0 else 0

print(f'Total processes: {len(result_files)}')
print(f'Duration: {duration:.1f}s')
print()
print('=== Writes ===')
print(f'  Total records: {total_writes}')
print(f'  Throughput: {total_writes / duration:.0f} records/s')
print(f'  Avg latency: {avg_write_latency:.2f}ms')
print()
print('=== Reads ===')
print(f'  Total records: {total_reads}')
print(f'  Throughput: {total_reads / duration:.0f} records/s')
print(f'  Avg latency: {avg_read_latency:.2f}ms')
"

# 清理数据
rm -rf ./queue_benchmark_data

echo ""
echo "Benchmark finished! Results in benchmark_logs/"
