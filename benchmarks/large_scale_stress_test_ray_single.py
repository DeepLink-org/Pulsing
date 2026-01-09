#!/usr/bin/env python3
"""
Ray 压测脚本 - 单进程版本（正确的 Ray 使用方式）

Ray 设计为单 driver 进程 + 多 Actor，不应使用 torchrun 多进程模式。
本脚本在单进程内创建多个 Actor，模拟与 Pulsing 等价的负载。

使用方法:
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
# 压测统计
# ============================================================================


@dataclass
class StressTestStats:
    """压测统计信息"""

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
# Ray Actor 定义
# ============================================================================


@ray.remote
class EchoWorker:
    """Echo Worker - 简单回显"""

    async def echo(self, text: str) -> dict:
        return {"echo": text}


@ray.remote
class ComputeWorker:
    """Compute Worker - 计算密集型"""

    async def compute(self, n: int) -> dict:
        result = sum(i * i for i in range(n))
        return {"result": result}


@ray.remote
class StreamWorker:
    """Stream Worker - 流式响应"""

    async def generate_stream(self, count: int, delay: float) -> list[dict]:
        result = []
        for i in range(count):
            result.append(
                {
                    "index": i,
                    "value": f"item_{i}",
                    "timestamp": time.time(),
                }
            )
            await asyncio.sleep(delay)
        return result


@ray.remote
class BatchWorker:
    """Batch Worker - 批量处理"""

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
    """Stateful Worker - 有状态处理"""

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
# 压测客户端
# ============================================================================


class StressTestClient:
    """压测客户端"""

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
        """发送单个请求"""
        # 随机选择 worker 类型
        worker_types = ["echo", "compute", "batch", "stateful"]
        worker_type = random.choice(worker_types)

        if worker_type not in self.workers or not self.workers[worker_type]:
            return False

        # 随机选择该类型的一个 worker
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
        """发送流式请求"""
        if "stream" not in self.workers or not self.workers["stream"]:
            return False

        worker = random.choice(self.workers["stream"])
        start_time = time.time()

        try:
            count = random.randint(5, 20)
            delay = random.uniform(0.01, 0.05)

            stream_items = await worker.generate_stream.remote(count, delay)

            chunk_count = 0
            for _ in stream_items:
                chunk_count += 1

            latency_ms = (time.time() - start_time) * 1000
            self.stats.add_stream(True, latency_ms)
            return True

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.stats.add_stream(False, latency_ms, str(e)[:100])
            return False

    async def run_stress_test(self, duration: float):
        """运行压测"""
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

        # 启动多个并发 worker
        num_workers = max(1, int(self.rate / 10))
        tasks = [asyncio.create_task(worker_loop()) for _ in range(num_workers)]

        # 定期报告
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
# 主函数
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="Ray压测脚本 - 单进程版本")
    parser.add_argument("--duration", type=float, default=30.0, help="压测时长（秒）")
    parser.add_argument("--rate", type=float, default=100.0, help="每秒请求数")
    parser.add_argument(
        "--num-workers", type=int, default=50, help="每种类型的Worker数量"
    )
    parser.add_argument("--num-cpus", type=int, default=None, help="Ray使用的CPU数量")
    parser.add_argument("--address", type=str, default=None, help="Ray集群地址")
    parser.add_argument(
        "--log-dir", type=str, default="benchmark_logs", help="日志目录"
    )

    args = parser.parse_args()

    # 设置日志
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Ray Stress Test (Single Process Mode)")
    print(f"Duration: {args.duration}s, Rate: {args.rate} req/s")
    print(f"Workers per type: {args.num_workers}")
    print(f"{'=' * 60}\n")

    # 初始化 Ray
    if not ray.is_initialized():
        if args.address:
            ray.init(address=args.address)
            print(f"Connected to Ray cluster at {args.address}")
        else:
            num_cpus = args.num_cpus or os.cpu_count()
            ray.init(num_cpus=num_cpus, include_dashboard=False)
            print(f"Ray initialized with {num_cpus} CPUs")

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    # 创建 Workers（模拟 10 个节点 × 5 种类型 = 50 个 Worker）
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

    # 等待所有 Worker 就绪
    print("Waiting for workers to be ready...")
    await asyncio.sleep(2.0)

    # 创建压测客户端
    stats = StressTestStats()
    client = StressTestClient(workers, stats, rate=args.rate)

    # 运行压测
    try:
        await client.run_stress_test(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        client.running = False

    # 打印最终统计
    print(f"\n{'=' * 60}")
    print("Final Statistics (Ray Single Process)")
    print(f"{'=' * 60}")
    summary = stats.get_summary()
    print(json.dumps(summary, indent=2))

    # 保存统计
    stats_file = os.path.join(log_dir, "stress_test_stats_ray_single.json")
    with open(stats_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStatistics saved to {stats_file}")

    # 关闭 Ray
    ray.shutdown()
    print("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
