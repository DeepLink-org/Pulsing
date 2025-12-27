#!/usr/bin/env python3
"""
Pulsing 压测脚本 - 单进程版本（与 Ray 单进程版本等价对比）

本脚本在单进程内创建多个 Actor，与 Ray 单进程版本保持等价。

使用方法:
    python benchmarks/large_scale_stress_test_pulsing_single.py \
        --duration 300 \
        --rate 100 \
        --num-workers 50
"""

import argparse
import asyncio
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

from pulsing.actor import (
    Actor,
    ActorRef,
    Message,
    StreamMessage,
    SystemConfig,
    create_actor_system,
)

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
# Pulsing Actor 定义
# ============================================================================


class EchoWorker(Actor):
    """Echo Worker - 简单回显"""

    async def receive(self, msg: Message):
        if msg.msg_type == "Echo":
            data = msg.to_json()
            return Message.from_json("EchoResponse", {"echo": data.get("text", "")})
        return Message.empty()


class ComputeWorker(Actor):
    """Compute Worker - 计算密集型"""

    async def receive(self, msg: Message):
        if msg.msg_type == "Compute":
            data = msg.to_json()
            n = data.get("n", 1000)
            result = sum(i * i for i in range(n))
            return Message.from_json("ComputeResponse", {"result": result})
        return Message.empty()


class StreamWorker(Actor):
    """Stream Worker - 流式响应"""

    async def receive(self, msg: Message):
        if msg.msg_type == "GenerateStream":
            data = msg.to_json()
            count = data.get("count", 10)
            delay = data.get("delay", 0.01)

            stream_msg, writer = StreamMessage.create("StreamItem")

            async def produce():
                try:
                    for i in range(count):
                        await writer.write_json(
                            {
                                "index": i,
                                "value": f"item_{i}",
                                "timestamp": time.time(),
                            }
                        )
                        await asyncio.sleep(delay)
                    await writer.close()
                except Exception as e:
                    await writer.error(str(e))

            asyncio.create_task(produce())
            return stream_msg
        return Message.empty()


class BatchWorker(Actor):
    """Batch Worker - 批量处理"""

    def __init__(self):
        self.batch = []
        self.batch_size = 10

    async def receive(self, msg: Message):
        if msg.msg_type == "BatchAdd":
            data = msg.to_json()
            self.batch.append(data.get("item"))

            if len(self.batch) >= self.batch_size:
                result = sum(self.batch)
                self.batch = []
                return Message.from_json("BatchResult", {"sum": result})
            return Message.from_json("BatchAck", {"count": len(self.batch)})
        return Message.empty()


class StatefulWorker(Actor):
    """Stateful Worker - 有状态处理"""

    def __init__(self):
        self.state = {}
        self.counter = 0

    async def receive(self, msg: Message):
        if msg.msg_type == "SetState":
            data = msg.to_json()
            key = data.get("key")
            value = data.get("value")
            self.state[key] = value
            self.counter += 1
            return Message.from_json("StateSet", {"counter": self.counter})

        elif msg.msg_type == "GetState":
            data = msg.to_json()
            key = data.get("key")
            value = self.state.get(key)
            return Message.from_json("StateValue", {"key": key, "value": value})

        return Message.empty()


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
        workers: dict[str, list[ActorRef]],
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
        worker_types = ["echo", "compute", "batch", "stateful"]
        worker_type = random.choice(worker_types)

        if worker_type not in self.workers or not self.workers[worker_type]:
            return False

        worker = random.choice(self.workers[worker_type])
        start_time = time.time()

        try:
            if worker_type == "echo":
                msg = Message.from_json(
                    "Echo", {"text": f"echo_{random.randint(1, 1000)}"}
                )
            elif worker_type == "compute":
                msg = Message.from_json("Compute", {"n": random.randint(100, 10000)})
            elif worker_type == "batch":
                msg = Message.from_json("BatchAdd", {"item": random.randint(1, 100)})
            elif worker_type == "stateful":
                if random.random() < 0.5:
                    msg = Message.from_json(
                        "SetState",
                        {
                            "key": f"key_{random.randint(1, 100)}",
                            "value": random.randint(1, 1000),
                        },
                    )
                else:
                    msg = Message.from_json(
                        "GetState", {"key": f"key_{random.randint(1, 100)}"}
                    )

            await worker.ask(msg)

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
            payload = {
                "count": random.randint(5, 20),
                "delay": random.uniform(0.01, 0.05),
            }

            msg = Message.from_json("GenerateStream", payload)
            response = await worker.ask(msg)
            reader = response.stream_reader()

            chunk_count = 0
            async for _chunk_bytes in reader:
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
                if random.random() < 0.7:
                    await self.send_single_request()
                else:
                    await self.send_stream_request()

                if self.interval > 0:
                    await asyncio.sleep(self.interval)

        num_workers = max(1, int(self.rate / 10))
        tasks = [asyncio.create_task(worker_loop()) for _ in range(num_workers)]

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
    parser = argparse.ArgumentParser(description="Pulsing压测脚本 - 单进程版本")
    parser.add_argument("--duration", type=float, default=30.0, help="压测时长（秒）")
    parser.add_argument("--rate", type=float, default=100.0, help="每秒请求数")
    parser.add_argument(
        "--num-workers", type=int, default=50, help="每种类型的Worker数量"
    )
    parser.add_argument("--port", type=int, default=8000, help="Pulsing端口")
    parser.add_argument(
        "--log-dir", type=str, default="stress_test_logs", help="日志目录"
    )

    args = parser.parse_args()

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Pulsing Stress Test (Single Process Mode)")
    print(f"Duration: {args.duration}s, Rate: {args.rate} req/s")
    print(f"Workers per type: {args.num_workers}")
    print(f"{'='*60}\n")

    # 初始化 Pulsing
    config = SystemConfig.with_addr(f"0.0.0.0:{args.port}")
    system = await create_actor_system(config)
    print(f"Pulsing ActorSystem started at {system.addr}")

    # 创建 Workers
    workers = {}
    for worker_type, worker_class in WORKER_CLASSES.items():
        workers[worker_type] = []
        for i in range(args.num_workers):
            try:
                worker_ref = await system.spawn(f"{worker_type}_{i}", worker_class())
                workers[worker_type].append(worker_ref)
            except Exception as e:
                print(f"Failed to create {worker_type}_{i}: {e}")
        print(f"Created {len(workers[worker_type])} {worker_type} workers")

    # 等待就绪
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
    print(f"\n{'='*60}")
    print("Final Statistics (Pulsing Single Process)")
    print(f"{'='*60}")
    summary = stats.get_summary()
    print(json.dumps(summary, indent=2))

    # 保存统计
    stats_file = os.path.join(log_dir, "stress_test_stats_pulsing_single.json")
    with open(stats_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStatistics saved to {stats_file}")

    print("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
