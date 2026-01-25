#!/usr/bin/env python3
"""
Ray Stress Test Script - Single Process Version (Correct Ray Usage with Generators)

Ray is designed as a single driver process + multiple Actors, should not use torchrun multi-process mode.
This script creates multiple Actors within a single process, simulating equivalent load to Pulsing.

This version uses Ray Generators for streaming, providing fair comparison with Pulsing's streaming.

Usage:
    python benchmarks/large_scale_stress_test_ray_single.py \
        --duration 300 \
        --rate 100 \
        --num-workers 50
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import ray
except ImportError:
    print("Error: Ray is not installed. Please install it with: pip install ray")
    sys.exit(1)


# ============================================================================
# Stress Test Statistics
# ============================================================================


@dataclass
class StressTestStats:
    """Stress test statistics"""

    total_requests: int = 0
    total_streams: int = 0
    successful_requests: int = 0
    successful_streams: int = 0
    failed_requests: int = 0
    failed_streams: int = 0
    total_latency_ms: float = 0.0
    total_stream_latency_ms: float = 0.0
    request_latencies: list[float] = field(default_factory=list)
    stream_latencies: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_request(self, success: bool, latency_ms: float, error: str | None = None):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.total_latency_ms += latency_ms
            self.request_latencies.append(latency_ms)
        else:
            self.failed_requests += 1
            if error:
                self.errors[error] += 1

    def add_stream(self, success: bool, latency_ms: float, error: str | None = None):
        self.total_streams += 1
        if success:
            self.successful_streams += 1
            self.total_stream_latency_ms += latency_ms
            self.stream_latencies.append(latency_ms)
        else:
            self.failed_streams += 1
            if error:
                self.errors[error] += 1

    def get_summary(self) -> dict:
        avg_latency = (
            self.total_latency_ms / self.successful_requests
            if self.successful_requests > 0
            else 0.0
        )
        avg_stream_latency = (
            self.total_stream_latency_ms / self.successful_streams
            if self.successful_streams > 0
            else 0.0
        )

        request_latencies_sorted = sorted(self.request_latencies)
        stream_latencies_sorted = sorted(self.stream_latencies)

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": (
                    self.successful_requests / self.total_requests * 100
                    if self.total_requests > 0
                    else 0.0
                ),
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": percentile(request_latencies_sorted, 50),
                "p95_latency_ms": percentile(request_latencies_sorted, 95),
                "p99_latency_ms": percentile(request_latencies_sorted, 99),
            },
            "streams": {
                "total": self.total_streams,
                "successful": self.successful_streams,
                "failed": self.failed_streams,
                "success_rate": (
                    self.successful_streams / self.total_streams * 100
                    if self.total_streams > 0
                    else 0.0
                ),
                "avg_latency_ms": avg_stream_latency,
                "p50_latency_ms": percentile(stream_latencies_sorted, 50),
                "p95_latency_ms": percentile(stream_latencies_sorted, 95),
                "p99_latency_ms": percentile(stream_latencies_sorted, 99),
            },
            "errors": dict(self.errors),
        }


# ============================================================================
# Ray Actor Definitions
# ============================================================================


@ray.remote
class EchoWorker:
    """Echo Worker - Simple echo"""

    async def echo(self, text: str) -> dict:
        return {"echo": text}


@ray.remote
class ComputeWorker:
    """Compute Worker - Compute intensive"""

    async def compute(self, n: int) -> dict:
        result = sum(i * i for i in range(n))
        return {"result": result}


@ray.remote
class StreamWorker:
    """Stream Worker - Streamed response using Ray Generators"""

    async def generate_stream(self, count: int, delay: float):
        """Generate stream using yield (Ray Generator)"""
        for i in range(count):
            await asyncio.sleep(delay)
            yield {
                "index": i,
                "value": f"item_{i}",
                "timestamp": time.time(),
            }


@ray.remote
class BatchWorker:
    """Batch Worker - Batch processing"""

    def __init__(self):
        self.batch = []
        self.batch_size = 10

    async def batch_add(self, item: int) -> dict:
        self.batch.append(item)
        if len(self.batch) >= self.batch_size:
            result = sum(self.batch)
            self.batch = []
            return {"sum": result}
        return {"count": len(self.batch)}


@ray.remote
class StatefulWorker:
    """Stateful Worker - Stateful processing"""

    def __init__(self):
        self.state = {}
        self.counter = 0

    async def set_state(self, key: str, value: int) -> dict:
        self.state[key] = value
        self.counter += 1
        return {"counter": self.counter}

    async def get_state(self, key: str) -> dict:
        value = self.state.get(key)
        return {"key": key, "value": value}


WORKER_CLASSES = {
    "echo": EchoWorker,
    "compute": ComputeWorker,
    "stream": StreamWorker,
    "batch": BatchWorker,
    "stateful": StatefulWorker,
}


# ============================================================================
# Stress Test Client
# ============================================================================


class StressTestClient:
    """Stress test client"""

    def __init__(
        self,
        workers: dict[str, list],  # {worker_type: [actor_handles]}
        stats: StressTestStats,
        rate: float = 100.0,
    ):
        self.workers = workers
        self.stats = stats
        self.rate = rate
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self.running = True

    async def send_single_request(self) -> bool:
        """Send a single request"""
        # Randomly select worker type
        worker_types = ["echo", "compute", "batch", "stateful"]
        worker_type = random.choice(worker_types)

        if worker_type not in self.workers or not self.workers[worker_type]:
            return False

        # Randomly select a worker of this type
        worker = random.choice(self.workers[worker_type])
        start_time = time.time()

        try:
            if worker_type == "echo":
                await worker.echo.remote(f"echo_{random.randint(1, 1000)}")
            elif worker_type == "compute":
                await worker.compute.remote(random.randint(100, 10000))
            elif worker_type == "batch":
                await worker.batch_add.remote(random.randint(1, 100))
            elif worker_type == "stateful":
                if random.random() < 0.5:
                    await worker.set_state.remote(
                        f"key_{random.randint(1, 100)}", random.randint(1, 1000)
                    )
                else:
                    await worker.get_state.remote(f"key_{random.randint(1, 100)}")

            latency_ms = (time.time() - start_time) * 1000
            self.stats.add_request(True, latency_ms)
            return True

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.stats.add_request(False, latency_ms, str(e)[:100])
            return False

    async def send_stream_request(self) -> bool:
        """Send a stream request using Ray Generators (async for)"""
        if "stream" not in self.workers or not self.workers["stream"]:
            return False

        worker = random.choice(self.workers["stream"])
        start_time = time.time()

        try:
            count = random.randint(5, 15)
            delay = 0.01

            # Use async for to stream results from Ray Generator
            # This is the correct way to consume Ray Generators in asyncio
            chunk_count = 0
            async for ref in worker.generate_stream.remote(count, delay):
                # await the ObjectRef to get the actual value
                item = await ref
                chunk_count += 1
                # Process item if needed (currently just counting)

            latency_ms = (time.time() - start_time) * 1000
            self.stats.add_stream(True, latency_ms)
            return True

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.stats.add_stream(False, latency_ms, str(e)[:100])
            return False

    async def run_stress_test(self, duration: float):
        """Run stress test"""
        end_time = time.time() + duration
        report_interval = 10.0

        print(f"[StressTest] Starting stress test for {duration}s at {self.rate} req/s")

        async def worker_loop():
            while self.running and time.time() < end_time:
                # 70% single, 30% stream
                if random.random() < 0.7:
                    await self.send_single_request()
                else:
                    await self.send_stream_request()

                if self.interval > 0:
                    await asyncio.sleep(self.interval)

        # Start multiple concurrent workers
        num_workers = max(1, int(self.rate / 10))
        tasks = [asyncio.create_task(worker_loop()) for _ in range(num_workers)]

        # Periodic reporting
        async def report_loop():
            while self.running and time.time() < end_time:
                await asyncio.sleep(report_interval)
                summary = self.stats.get_summary()
                print("\n[StressTest] Progress Report:")
                print(
                    f"  Requests: {summary['requests']['total']} "
                    f"(success: {summary['requests']['successful']}, "
                    f"failed: {summary['requests']['failed']})"
                )
                print(
                    f"  Streams: {summary['streams']['total']} "
                    f"(success: {summary['streams']['successful']}, "
                    f"failed: {summary['streams']['failed']})"
                )
                if summary["requests"]["successful"] > 0:
                    print(
                        f"  Avg Latency: {summary['requests']['avg_latency_ms']:.2f}ms"
                    )

        report_task = asyncio.create_task(report_loop())

        await asyncio.gather(*tasks, report_task)

        self.running = False
        print("\n[StressTest] Stress test completed")


# ============================================================================
# Main Function
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Ray Stress Test Script - Single Process Version"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Stress test duration (seconds)"
    )
    parser.add_argument("--rate", type=float, default=100.0, help="Requests per second")
    parser.add_argument(
        "--num-workers", type=int, default=50, help="Number of workers per type"
    )
    parser.add_argument(
        "--num-cpus", type=int, default=None, help="Number of CPUs for Ray to use"
    )
    parser.add_argument("--address", type=str, default=None, help="Ray cluster address")
    parser.add_argument(
        "--log-dir", type=str, default="benchmark_logs", help="Log directory"
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Ray Stress Test (Single Process Mode)")
    print(f"Duration: {args.duration}s, Rate: {args.rate} req/s")
    print(f"Workers per type: {args.num_workers}")
    print(f"{'=' * 60}\n")

    # Initialize Ray
    if not ray.is_initialized():
        if args.address:
            ray.init(address=args.address)
            print(f"Connected to Ray cluster at {args.address}")
        else:
            num_cpus = args.num_cpus or os.cpu_count()
            ray.init(num_cpus=num_cpus, include_dashboard=False)
            print(f"Ray initialized with {num_cpus} CPUs")

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    # Create Workers (simulate 10 nodes × 5 types = 50 Workers)
    workers = {}
    for worker_type, worker_class in WORKER_CLASSES.items():
        workers[worker_type] = []
        for i in range(args.num_workers):
            try:
                worker = worker_class.options(name=f"{worker_type}_{i}").remote()
                workers[worker_type].append(worker)
            except Exception as e:
                print(f"Failed to create {worker_type}_{i}: {e}")
        print(f"Created {len(workers[worker_type])} {worker_type} workers")

    # Wait for all Workers to be ready
    print("Waiting for workers to be ready...")
    await asyncio.sleep(2.0)

    # Create stress test client
    stats = StressTestStats()
    client = StressTestClient(workers, stats, rate=args.rate)

    # Run stress test
    try:
        await client.run_stress_test(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        client.running = False

    # Print final statistics
    print(f"\n{'=' * 60}")
    print("Final Statistics (Ray Single Process)")
    print(f"{'=' * 60}")
    summary = stats.get_summary()
    print(json.dumps(summary, indent=2))

    # Save statistics
    stats_file = os.path.join(log_dir, "stress_test_stats_ray_single.json")
    with open(stats_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStatistics saved to {stats_file}")

    # Shutdown Ray
    ray.shutdown()
    print("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
