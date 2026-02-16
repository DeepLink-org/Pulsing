#!/usr/bin/env python3
"""
Distributed In-Memory Queue Benchmark Script - Multi-Process Version

Uses torchrun to start multiple processes, each process acts as both producer and consumer.
Uses rank/world_size for distributed consumption, testing cross-node queue performance.

Usage:
    torchrun --nproc_per_node=10 benchmarks/queue_benchmark.py \
        --duration 30 \
        --num-buckets 16

Arguments:
    --duration: Benchmark duration (seconds), default 30
    --num-buckets: Number of queue buckets, default 16
    --batch-size: Write batch size, default 100
    --record-size: Record data size (bytes), default 100
    --rate-limit: Writes per second per process (None=unlimited)
"""

import argparse
import asyncio
import json
import os
import random
import string
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import pulsing as pul
from pulsing.core import SystemConfig
from pulsing.streaming import read_queue, write_queue


# ============================================================================
# Benchmark Statistics
# ============================================================================


@dataclass
class QueueBenchmarkStats:
    """Queue benchmark statistics"""

    total_writes: int = 0
    total_reads: int = 0
    successful_writes: int = 0
    successful_reads: int = 0
    failed_writes: int = 0
    failed_reads: int = 0
    total_write_latency_ms: float = 0.0
    total_read_latency_ms: float = 0.0
    write_latencies: list[float] = field(default_factory=list)
    read_latencies: list[float] = field(default_factory=list)
    bytes_written: int = 0
    records_written: int = 0
    records_read: int = 0
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_write(
        self,
        success: bool,
        latency_ms: float,
        records: int = 1,
        bytes_size: int = 0,
        error: str | None = None,
    ):
        self.total_writes += 1
        if success:
            self.successful_writes += 1
            self.total_write_latency_ms += latency_ms
            self.write_latencies.append(latency_ms)
            self.records_written += records
            self.bytes_written += bytes_size
        else:
            self.failed_writes += 1
            if error:
                self.errors[error[:50]] += 1

    def add_read(
        self,
        success: bool,
        latency_ms: float,
        records: int = 0,
        error: str | None = None,
    ):
        self.total_reads += 1
        if success:
            self.successful_reads += 1
            self.total_read_latency_ms += latency_ms
            self.read_latencies.append(latency_ms)
            self.records_read += records
        else:
            self.failed_reads += 1
            if error:
                self.errors[error[:50]] += 1

    def get_summary(self, duration: float) -> dict:
        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        write_throughput = self.records_written / duration if duration > 0 else 0
        read_throughput = self.records_read / duration if duration > 0 else 0
        avg_write_latency = (
            self.total_write_latency_ms / self.successful_writes
            if self.successful_writes > 0
            else 0.0
        )
        avg_read_latency = (
            self.total_read_latency_ms / self.successful_reads
            if self.successful_reads > 0
            else 0.0
        )

        return {
            "duration_s": duration,
            "writes": {
                "total_ops": self.total_writes,
                "successful": self.successful_writes,
                "failed": self.failed_writes,
                "success_rate": (
                    self.successful_writes / self.total_writes * 100
                    if self.total_writes > 0
                    else 0.0
                ),
                "records_written": self.records_written,
                "throughput_records_per_s": write_throughput,
                "bytes_written": self.bytes_written,
                "throughput_mb_per_s": self.bytes_written / duration / 1024 / 1024
                if duration > 0
                else 0,
                "avg_latency_ms": avg_write_latency,
                "p50_latency_ms": percentile(self.write_latencies, 50),
                "p95_latency_ms": percentile(self.write_latencies, 95),
                "p99_latency_ms": percentile(self.write_latencies, 99),
            },
            "reads": {
                "total_ops": self.total_reads,
                "successful": self.successful_reads,
                "failed": self.failed_reads,
                "success_rate": (
                    self.successful_reads / self.total_reads * 100
                    if self.total_reads > 0
                    else 0.0
                ),
                "records_read": self.records_read,
                "throughput_records_per_s": read_throughput,
                "avg_latency_ms": avg_read_latency,
                "p50_latency_ms": percentile(self.read_latencies, 50),
                "p95_latency_ms": percentile(self.read_latencies, 95),
                "p99_latency_ms": percentile(self.read_latencies, 99),
            },
            "errors": dict(self.errors),
        }


# ============================================================================
# Data Generation
# ============================================================================


def generate_record(record_size: int = 100) -> dict:
    """Generate test record"""
    return {
        "id": f"user_{random.randint(0, 10000)}",
        "timestamp": time.time(),
        "value": random.randint(0, 1000000),
        "data": "".join(random.choices(string.ascii_letters, k=record_size)),
    }


def estimate_record_size(record: dict) -> int:
    """Estimate record size"""
    return len(json.dumps(record))


# ============================================================================
# Producer
# ============================================================================


async def producer_loop(
    writer,
    stats: QueueBenchmarkStats,
    end_time: float,
    record_size: int,
    rate_limit: float | None,
):
    """Producer loop"""
    interval = 1.0 / rate_limit if rate_limit else 0

    while time.time() < end_time:
        record = generate_record(record_size)
        record_bytes = estimate_record_size(record)
        start = time.time()

        try:
            await writer.put(record)
            latency_ms = (time.time() - start) * 1000
            stats.add_write(True, latency_ms, records=1, bytes_size=record_bytes)
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            stats.add_write(False, latency_ms, error=str(e))

        if interval > 0:
            await asyncio.sleep(interval)


# ============================================================================
# Consumer
# ============================================================================


async def consumer_loop(
    reader,
    stats: QueueBenchmarkStats,
    end_time: float,
    read_batch_size: int,
):
    """Consumer loop"""
    while time.time() < end_time:
        start = time.time()

        try:
            records = await reader.get(limit=read_batch_size, wait=True, timeout=1.0)
            latency_ms = (time.time() - start) * 1000

            if records:
                stats.add_read(True, latency_ms, records=len(records))
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            stats.add_read(False, latency_ms, error=str(e))


# ============================================================================
# Main Function
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Distributed In-Memory Queue Benchmark (Multi-Process)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Benchmark duration (seconds)"
    )
    parser.add_argument(
        "--num-buckets", type=int, default=16, help="Number of queue buckets"
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Write batch size")
    parser.add_argument(
        "--read-batch-size", type=int, default=100, help="Read batch size"
    )
    parser.add_argument(
        "--record-size", type=int, default=100, help="Record data size (bytes)"
    )
    parser.add_argument(
        "--rate-limit", type=float, default=None, help="Writes per second per process"
    )
    parser.add_argument("--port", type=int, default=0, help="Base port (0=auto)")
    parser.add_argument(
        "--seed-nodes", type=str, nargs="+", default=[], help="Seed nodes"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="./queue_benchmark_data",
        help="Storage path",
    )
    parser.add_argument(
        "--log-dir", type=str, default="benchmark_logs", help="Log directory"
    )
    parser.add_argument(
        "--stabilize-timeout",
        type=float,
        default=5.0,
        help="Cluster stabilization wait time",
    )

    args = parser.parse_args()

    # Get torchrun information from environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", local_rank))

    # Create log directory
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.storage_path, exist_ok=True)

    # Log file
    log_file = os.path.join(log_dir, f"queue_benchmark_rank_{rank}.log")

    class TeeOutput:
        def __init__(self, file_path, original):
            self.file = open(file_path, "w", encoding="utf-8")
            self.original = original

        def write(self, text):
            try:
                self.file.write(text)
                self.file.flush()
            except Exception:
                pass
            try:
                self.original.write(text)
                self.original.flush()
            except Exception:
                pass

        def flush(self):
            try:
                self.file.flush()
            except Exception:
                pass
            try:
                self.original.flush()
            except Exception:
                pass

        def close(self):
            try:
                self.file.close()
            except Exception:
                pass

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee = TeeOutput(log_file, original_stdout)
    sys.stdout = tee
    sys.stderr = tee

    print(f"\n{'=' * 60}")
    print(f"Queue Benchmark - Process {rank}/{world_size}")
    print(f"{'=' * 60}")
    print(f"Duration: {args.duration}s")
    print(f"Buckets: {args.num_buckets}, Batch size: {args.batch_size}")
    print(f"Record size: {args.record_size} bytes")
    print(f"Log file: {log_file}")
    print(f"{'=' * 60}\n")

    # Configure system
    if args.port == 0:
        base_port = 9000
        port = base_port + rank
    else:
        port = args.port + rank

    addr = f"0.0.0.0:{port}"

    # Add seed nodes
    seeds = None
    if args.seed_nodes:
        seeds = args.seed_nodes
    elif rank > 0:
        prev_port = 9000 + (rank - 1)
        seeds = [f"127.0.0.1:{prev_port}"]

    # Create system
    system = await pul.actor_system(addr=addr, seeds=seeds)
    print(f"[Process {rank}] ActorSystem started at {system.addr}")

    # Wait for cluster to stabilize
    print(
        f"[Process {rank}] Waiting for cluster to stabilize ({args.stabilize_timeout}s)..."
    )
    await asyncio.sleep(args.stabilize_timeout)

    # Get cluster members
    members = await system.members()
    print(f"[Process {rank}] Cluster members: {len(members)}")
    for member in members:
        print(
            f"  - Node {member['node_id']}: {member['addr']} [{member.get('status', 'unknown')}]"
        )

    topic = "benchmark_queue"
    stats = QueueBenchmarkStats()

    # Create writer (each process can write)
    writer = await write_queue(
        system,
        topic=topic,
        bucket_column="id",
        num_buckets=args.num_buckets,
        batch_size=args.batch_size,
        storage_path=args.storage_path,
    )
    print(f"[Process {rank}] Writer created")

    # Create reader (use rank/world_size for distributed consumption)
    reader = await read_queue(
        system,
        topic=topic,
        rank=rank,
        world_size=world_size,
        num_buckets=args.num_buckets,
        storage_path=args.storage_path,
    )
    assigned_buckets = [i for i in range(args.num_buckets) if i % world_size == rank]
    print(f"[Process {rank}] Reader created, assigned buckets: {assigned_buckets}")

    # Wait for all processes to be ready
    await asyncio.sleep(2.0)

    # Start producer and consumer
    print(f"\n[Process {rank}] Starting benchmark...")
    start_time = time.time()
    end_time = start_time + args.duration

    producer_task = asyncio.create_task(
        producer_loop(writer, stats, end_time, args.record_size, args.rate_limit)
    )
    consumer_task = asyncio.create_task(
        consumer_loop(reader, stats, end_time, args.read_batch_size)
    )

    # Progress reporting
    report_interval = 10.0

    async def report_progress():
        while time.time() < end_time:
            await asyncio.sleep(report_interval)
            elapsed = time.time() - start_time
            summary = stats.get_summary(elapsed)
            print(f"\n[Process {rank}] Progress ({elapsed:.0f}s):")
            print(
                f"  Writes: {summary['writes']['records_written']} records "
                f"({summary['writes']['throughput_records_per_s']:.0f}/s)"
            )
            print(
                f"  Reads: {summary['reads']['records_read']} records "
                f"({summary['reads']['throughput_records_per_s']:.0f}/s)"
            )
            print(
                f"  Write latency: avg={summary['writes']['avg_latency_ms']:.2f}ms, "
                f"p99={summary['writes']['p99_latency_ms']:.2f}ms"
            )

    report_task = asyncio.create_task(report_progress())

    # Wait for completion
    try:
        await asyncio.gather(producer_task, consumer_task)
    except KeyboardInterrupt:
        print(f"\n[Process {rank}] Interrupted by user")

    report_task.cancel()
    try:
        await report_task
    except asyncio.CancelledError:
        pass

    # Final flush
    try:
        await writer.flush()
    except Exception:
        pass

    # Final statistics
    actual_duration = time.time() - start_time
    summary = stats.get_summary(actual_duration)

    print(f"\n{'=' * 60}")
    print(f"Final Statistics - Process {rank}")
    print(f"{'=' * 60}")
    print(json.dumps(summary, indent=2))

    # Save results
    result_file = os.path.join(log_dir, f"queue_benchmark_rank_{rank}.json")
    with open(result_file, "w") as f:
        json.dump(
            {
                "rank": rank,
                "world_size": world_size,
                "config": {
                    "duration": args.duration,
                    "num_buckets": args.num_buckets,
                    "batch_size": args.batch_size,
                    "record_size": args.record_size,
                    "rate_limit": args.rate_limit,
                    "assigned_buckets": assigned_buckets,
                },
                "results": summary,
            },
            f,
            indent=2,
        )
    print(f"\n[Process {rank}] Results saved to {result_file}")

    # Cleanup
    await system.shutdown()
    print(f"[Process {rank}] Benchmark complete")

    # Restore stdout/stderr
    if hasattr(sys.stdout, "close"):
        try:
            sys.stdout.close()
        except Exception:
            pass
    sys.stdout = original_stdout
    sys.stderr = original_stderr


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback

        print(f"\n[FATAL ERROR] Process failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
