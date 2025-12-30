#!/usr/bin/env python3
"""
Pulsing 压测脚本 - 单进程版本

使用方法:
    python benchmarks/large_scale_stress_test_pulsing_single.py \
        --duration 300 --rate 100 --num-workers 50
"""

import argparse
import asyncio
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

from pulsing.actor import Actor, StreamMessage, SystemConfig, create_actor_system


# ============================================================================
# 统计
# ============================================================================


@dataclass
class Stats:
    total: int = 0
    success: int = 0
    failed: int = 0
    latencies: list = field(default_factory=list)
    errors: dict = field(default_factory=lambda: defaultdict(int))

    def add(self, ok: bool, latency_ms: float, error: str = None):
        self.total += 1
        if ok:
            self.success += 1
            self.latencies.append(latency_ms)
        else:
            self.failed += 1
            if error:
                self.errors[error[:50]] += 1

    def summary(self):
        lat = sorted(self.latencies)
        pct = lambda p: lat[int(len(lat) * p / 100)] if lat else 0
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "success_rate": self.success / self.total * 100 if self.total else 0,
            "avg_ms": sum(lat) / len(lat) if lat else 0,
            "p50_ms": pct(50),
            "p95_ms": pct(95),
            "p99_ms": pct(99),
            "errors": dict(self.errors),
        }


# ============================================================================
# Workers - 使用简化的消息格式
# ============================================================================


class EchoWorker(Actor):
    async def receive(self, msg):
        if isinstance(msg, dict) and msg.get("type") == "echo":
            return {"echo": msg.get("text", "")}


class ComputeWorker(Actor):
    async def receive(self, msg):
        if isinstance(msg, dict) and msg.get("type") == "compute":
            n = msg.get("n", 1000)
            return {"result": sum(i * i for i in range(n))}


class StreamWorker(Actor):
    async def receive(self, msg):
        if isinstance(msg, dict) and msg.get("type") == "stream":
            count = msg.get("count", 10)
            delay = msg.get("delay", 0.01)

            stream_msg, writer = StreamMessage.create("items")

            async def produce():
                try:
                    for i in range(count):
                        await writer.write_json({"index": i, "ts": time.time()})
                        await asyncio.sleep(delay)
                    await writer.close()
                except Exception:
                    pass  # Stream closed during shutdown, ignore

            asyncio.create_task(produce())
            return stream_msg


class BatchWorker(Actor):
    def __init__(self):
        self.batch = []

    async def receive(self, msg):
        if isinstance(msg, dict) and msg.get("type") == "batch":
            self.batch.append(msg.get("item", 0))
            if len(self.batch) >= 10:
                result = sum(self.batch)
                self.batch = []
                return {"sum": result}
            return {"count": len(self.batch)}


class StatefulWorker(Actor):
    def __init__(self):
        self.state = {}

    async def receive(self, msg):
        if isinstance(msg, dict):
            if msg.get("type") == "set":
                self.state[msg["key"]] = msg["value"]
                return {"ok": True}
            if msg.get("type") == "get":
                return {"value": self.state.get(msg["key"])}


WORKERS = {
    "echo": EchoWorker,
    "compute": ComputeWorker,
    "stream": StreamWorker,
    "batch": BatchWorker,
    "stateful": StatefulWorker,
}


# ============================================================================
# 压测
# ============================================================================


async def run_benchmark(workers, stats_req, stats_stream, duration, rate):
    end_time = time.time() + duration
    interval = 1.0 / rate if rate > 0 else 0

    async def worker_loop():
        while time.time() < end_time:
            # 70% single, 30% stream
            if random.random() < 0.7:
                await send_request(workers, stats_req)
            else:
                await send_stream(workers, stats_stream)
            if interval:
                await asyncio.sleep(interval)

    tasks = [asyncio.create_task(worker_loop()) for _ in range(max(1, int(rate) // 10))]

    # 进度报告
    while time.time() < end_time:
        await asyncio.sleep(10)
        print(f"  Requests: {stats_req.total} (ok: {stats_req.success})")
        print(f"  Streams:  {stats_stream.total} (ok: {stats_stream.success})")

    for t in tasks:
        t.cancel()


async def send_request(workers, stats):
    wtype = random.choice(["echo", "compute", "batch", "stateful"])
    if wtype not in workers:
        return

    worker = random.choice(workers[wtype])
    start = time.time()

    try:
        if wtype == "echo":
            msg = {"type": "echo", "text": f"msg_{random.randint(1, 1000)}"}
        elif wtype == "compute":
            msg = {"type": "compute", "n": random.randint(100, 5000)}
        elif wtype == "batch":
            msg = {"type": "batch", "item": random.randint(1, 100)}
        else:  # stateful
            if random.random() < 0.5:
                msg = {
                    "type": "set",
                    "key": f"k{random.randint(1, 100)}",
                    "value": random.randint(1, 1000),
                }
            else:
                msg = {"type": "get", "key": f"k{random.randint(1, 100)}"}

        await worker.ask(msg)
        stats.add(True, (time.time() - start) * 1000)
    except Exception as e:
        stats.add(False, (time.time() - start) * 1000, str(e))


async def send_stream(workers, stats):
    if "stream" not in workers:
        return

    worker = random.choice(workers["stream"])
    start = time.time()

    try:
        msg = {"type": "stream", "count": random.randint(5, 15), "delay": 0.01}
        resp = await worker.ask(msg)
        async for _ in resp.stream_reader():
            pass
        stats.add(True, (time.time() - start) * 1000)
    except Exception as e:
        stats.add(False, (time.time() - start) * 1000, str(e))


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--rate", type=float, default=100)
    parser.add_argument("--num-workers", type=int, default=50)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-dir", type=str, default="benchmark_logs")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print("Pulsing Stress Test (Single Process)")
    print(
        f"Duration: {args.duration}s, Rate: {args.rate}/s, Workers: {args.num_workers}"
    )
    print(f"{'='*50}\n")

    system = await create_actor_system(SystemConfig.with_addr(f"0.0.0.0:{args.port}"))
    print(f"System started at {system.addr}")

    # 创建 workers
    workers = {}
    for name, cls in WORKERS.items():
        workers[name] = []
        for i in range(args.num_workers):
            ref = await system.spawn(f"{name}_{i}", cls())
            workers[name].append(ref)
        print(f"Created {args.num_workers} {name} workers")

    await asyncio.sleep(1)

    # 运行压测
    stats_req, stats_stream = Stats(), Stats()
    try:
        await run_benchmark(workers, stats_req, stats_stream, args.duration, args.rate)
    except KeyboardInterrupt:
        print("\nInterrupted")

    # 结果
    print(f"\n{'='*50}")
    print("Results")
    print(f"{'='*50}")
    result = {"requests": stats_req.summary(), "streams": stats_stream.summary()}
    print(json.dumps(result, indent=2))

    with open(f"{args.log_dir}/stress_test_pulsing_single.json", "w") as f:
        json.dump(result, f, indent=2)

    # 等待 3 秒让正在进行的流式任务完成
    print("Waiting 3s for streams to complete...")
    await asyncio.sleep(3)

    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
