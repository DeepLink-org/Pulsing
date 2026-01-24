#!/usr/bin/env python3
"""
Large-Scale Stress Test Script - Multi-Process Version

Usage:
    torchrun --nproc_per_node=10 benchmarks/large_scale_stress_test.py \
        --duration 300 --rate 100
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

import pulsing as pul
from pulsing.actor import Actor, StreamMessage, SystemConfig


# ============================================================================
# Statistics
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
# Workers - Using Simplified Message Format
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
                        await writer.write({"index": i, "ts": time.time()})
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
# Stress Test
# ============================================================================


async def run_benchmark(worker_refs, stats_req, stats_stream, duration, rate):
    end_time = time.time() + duration
    interval = 1.0 / rate if rate > 0 else 0

    # Separate local and remote workers
    local = {k: v for k, v in worker_refs.items() if "_remote_" not in k}
    remote = {k: v for k, v in worker_refs.items() if "_remote_" in k}

    async def worker_loop():
        while time.time() < end_time:
            # 70% remote, 30% local
            use_remote = remote and (not local or random.random() < 0.7)
            refs = remote if use_remote else local

            if not refs:
                await asyncio.sleep(0.1)
                continue

            key = random.choice(list(refs.keys()))
            base_type = key.split("_remote_")[0] if "_remote_" in key else key

            # 70% single, 30% stream
            if random.random() < 0.7 and base_type != "stream":
                await send_request(refs[key], base_type, stats_req)
            elif base_type == "stream":
                await send_stream(refs[key], stats_stream)
            else:
                await send_request(refs[key], base_type, stats_req)

            if interval:
                await asyncio.sleep(interval)

    tasks = [asyncio.create_task(worker_loop()) for _ in range(max(1, int(rate) // 10))]

    # Progress reporting
    while time.time() < end_time:
        await asyncio.sleep(10)
        print(f"  Requests: {stats_req.total} (ok: {stats_req.success})")
        print(f"  Streams:  {stats_stream.total} (ok: {stats_stream.success})")

    for t in tasks:
        t.cancel()


async def send_request(worker, wtype, stats):
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


async def send_stream(worker, stats):
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
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--rate", type=float, default=100)
    parser.add_argument("--seed-nodes", type=str, nargs="+", default=[])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--stabilize-timeout", type=float, default=10)
    parser.add_argument("--log-dir", type=str, default="benchmark_logs")
    args = parser.parse_args()

    # Get torchrun information
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = f"{args.log_dir}/stress_test_rank_{rank}.log"

    # Log output
    class Tee:
        def __init__(self, path, orig):
            self.f = open(path, "w")
            self.orig = orig

        def write(self, s):
            self.f.write(s)
            self.f.flush()
            self.orig.write(s)

        def flush(self):
            self.f.flush()
            self.orig.flush()

    orig_stdout = sys.stdout
    sys.stdout = Tee(log_file, orig_stdout)
    sys.stderr = sys.stdout

    print(f"\n{'=' * 50}")
    print(f"Stress Test - Process {rank}/{world_size}")
    print(f"Duration: {args.duration}s, Rate: {args.rate}/s")
    print(f"{'=' * 50}\n")

    # Configure system
    port = (8000 if args.port == 0 else args.port) + rank
    addr = f"0.0.0.0:{port}"

    seeds = None
    if args.seed_nodes:
        seeds = args.seed_nodes
    elif rank > 0:
        seeds = [f"127.0.0.1:{8000 + rank - 1}"]

    system = await pul.actor_system(addr=addr, seeds=seeds)
    print(f"System started at {system.addr}")

    # Wait for cluster to stabilize
    print(f"Waiting {args.stabilize_timeout}s for cluster...")
    await asyncio.sleep(args.stabilize_timeout)

    members = await system.members()
    print(f"Cluster members: {len(members)}")

    # Create local workers
    worker_refs = {}
    for name, cls in WORKERS.items():
        ref = await system.spawn(cls(), name=f"{name}_{rank}", public=True)
        worker_refs[name] = ref
        print(f"  Spawned {name}_{rank}")

    await asyncio.sleep(args.stabilize_timeout)

    # Resolve remote workers
    print("Resolving remote workers...")
    for other_rank in range(world_size):
        if other_rank == rank:
            continue
        for name in WORKERS:
            try:
                ref = await system.resolve_named(f"{name}_{other_rank}")
                worker_refs[f"{name}_remote_{other_rank}"] = ref
            except Exception:
                pass

    print(f"Total workers: {len(worker_refs)}")

    # Run stress test
    stats_req, stats_stream = Stats(), Stats()
    try:
        await run_benchmark(
            worker_refs, stats_req, stats_stream, args.duration, args.rate
        )
    except KeyboardInterrupt:
        print("\nInterrupted")

    # Results
    print(f"\n{'=' * 50}")
    print(f"Results - Process {rank}")
    print(f"{'=' * 50}")
    result = {"requests": stats_req.summary(), "streams": stats_stream.summary()}
    print(json.dumps(result, indent=2))

    with open(f"{args.log_dir}/stress_test_stats_rank_{rank}.json", "w") as f:
        json.dump(result, f, indent=2)

    # Wait 3 seconds for ongoing stream tasks to complete
    print(f"[Process {rank}] Waiting 3s for streams to complete...")
    await asyncio.sleep(3)

    await system.shutdown()
    sys.stdout = orig_stdout


if __name__ == "__main__":
    asyncio.run(main())
