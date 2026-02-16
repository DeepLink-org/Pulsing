#!/usr/bin/env python3
"""
Queue & Topic 基线吞吐 Benchmark（单节点）

在单进程内测量 Queue 与 Topic 的基线吞吐与延迟，便于回归对比。

Usage:
    python benchmarks/baseline_throughput.py
    python benchmarks/baseline_throughput.py --duration 15 --output results.json
    python benchmarks/baseline_throughput.py --queue-only
    python benchmarks/baseline_throughput.py --topic-only --topic-subscribers 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time

import pulsing as pul
from pulsing.streaming import (
    read_queue,
    write_queue,
    PublishMode,
    read_topic,
    write_topic,
)


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    idx = min(int(len(sorted_data) * p / 100), len(sorted_data) - 1)
    return sorted_data[idx]


# =============================================================================
# Queue 基线
# =============================================================================


async def run_queue_baseline(
    system,
    storage_path: str,
    duration: float,
    num_buckets: int,
    record_size: int,
) -> dict:
    """单 writer + 单 reader，固定时长，统计写/读吞吐与延迟."""
    topic = "baseline_queue"
    write_latencies_ms: list[float] = []
    read_latencies_ms: list[float] = []
    records_written = 0
    records_read = 0

    writer = await write_queue(
        system,
        topic=topic,
        bucket_column="id",
        num_buckets=num_buckets,
        storage_path=storage_path,
    )
    reader = await read_queue(
        system,
        topic=topic,
        num_buckets=num_buckets,
        storage_path=storage_path,
    )

    end_time = time.monotonic() + duration

    async def produce():
        nonlocal records_written
        i = 0
        while time.monotonic() < end_time:
            t0 = time.perf_counter()
            try:
                rec = {"id": f"r{i}", "payload": "x" * record_size}
                await writer.put(rec)
                write_latencies_ms.append((time.perf_counter() - t0) * 1000)
                records_written += 1
                i += 1
            except Exception:
                pass

    async def consume():
        nonlocal records_read
        while time.monotonic() < end_time:
            t0 = time.perf_counter()
            try:
                batch = await reader.get(limit=50, wait=True, timeout=1.0)
                if batch:
                    read_latencies_ms.append((time.perf_counter() - t0) * 1000)
                    records_read += len(batch)
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

    await asyncio.gather(produce(), consume())

    write_latencies_ms.sort()
    read_latencies_ms.sort()

    return {
        "duration_s": duration,
        "records_written": records_written,
        "records_read": records_read,
        "write_throughput_rec_s": records_written / duration if duration > 0 else 0,
        "read_throughput_rec_s": records_read / duration if duration > 0 else 0,
        "write_latency_ms": {
            "avg": sum(write_latencies_ms) / len(write_latencies_ms)
            if write_latencies_ms
            else 0,
            "p50": _percentile(write_latencies_ms, 50),
            "p95": _percentile(write_latencies_ms, 95),
            "p99": _percentile(write_latencies_ms, 99),
        },
        "read_latency_ms": {
            "avg": sum(read_latencies_ms) / len(read_latencies_ms)
            if read_latencies_ms
            else 0,
            "p50": _percentile(read_latencies_ms, 50),
            "p95": _percentile(read_latencies_ms, 95),
            "p99": _percentile(read_latencies_ms, 99),
        },
    }


# =============================================================================
# Topic 基线
# =============================================================================


async def run_topic_baseline(
    system,
    duration: float,
    num_subscribers: int,
    payload_size: int,
) -> dict:
    """单 publisher + N subscribers，fire_and_forget，统计发布与交付吞吐."""
    topic_name = "baseline_topic"
    messages_published = 0
    delivered_per_sub: list[int] = [0] * num_subscribers
    publish_latencies_ms: list[float] = []

    writer = await write_topic(system, topic_name)
    readers = []
    locks = [asyncio.Lock() for _ in range(num_subscribers)]

    for i in range(num_subscribers):
        reader = await read_topic(system, topic_name, reader_id=f"sub_{i}")

        def make_cb(idx):
            async def cb(msg):
                async with locks[idx]:
                    delivered_per_sub[idx] += 1

            return cb

        reader.add_callback(make_cb(i))
        await reader.start()
        readers.append(reader)

    end_time = time.monotonic() + duration
    seq = 0

    while time.monotonic() < end_time:
        t0 = time.perf_counter()
        try:
            await writer.publish(
                {"seq": seq, "payload": "x" * payload_size},
                mode=PublishMode.FIRE_AND_FORGET,
            )
            publish_latencies_ms.append((time.perf_counter() - t0) * 1000)
            messages_published += 1
            seq += 1
        except Exception:
            pass

    await asyncio.sleep(0.2)

    for r in readers:
        await r.stop()

    publish_latencies_ms.sort()
    total_delivered = sum(delivered_per_sub)

    return {
        "duration_s": duration,
        "num_subscribers": num_subscribers,
        "messages_published": messages_published,
        "total_delivered": total_delivered,
        "publish_throughput_msg_s": messages_published / duration
        if duration > 0
        else 0,
        "delivered_throughput_msg_s": total_delivered / duration if duration > 0 else 0,
        "publish_latency_ms": {
            "avg": sum(publish_latencies_ms) / len(publish_latencies_ms)
            if publish_latencies_ms
            else 0,
            "p50": _percentile(publish_latencies_ms, 50),
            "p95": _percentile(publish_latencies_ms, 95),
            "p99": _percentile(publish_latencies_ms, 99),
        },
    }


# =============================================================================
# Main
# =============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Queue & Topic 基线吞吐 Benchmark（单节点）"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="每类基准运行时长（秒）",
    )
    parser.add_argument(
        "--queue-only",
        action="store_true",
        help="仅跑 Queue 基线",
    )
    parser.add_argument(
        "--topic-only",
        action="store_true",
        help="仅跑 Topic 基线",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=4,
        help="Queue 桶数",
    )
    parser.add_argument(
        "--topic-subscribers",
        type=int,
        default=1,
        help="Topic 订阅者数量",
    )
    parser.add_argument(
        "--record-size",
        type=int,
        default=100,
        help="单条记录 payload 字节数",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果写入 JSON 文件路径",
    )
    args = parser.parse_args()

    system = await pul.actor_system()
    storage_path = tempfile.mkdtemp(prefix="baseline_queue_")
    results: dict = {"queue": None, "topic": None}

    try:
        if not args.topic_only:
            print("Running Queue baseline...")
            results["queue"] = await run_queue_baseline(
                system,
                storage_path=storage_path,
                duration=args.duration,
                num_buckets=args.num_buckets,
                record_size=args.record_size,
            )

        if not args.queue_only:
            print("Running Topic baseline...")
            results["topic"] = await run_topic_baseline(
                system,
                duration=args.duration,
                num_subscribers=args.topic_subscribers,
                payload_size=args.record_size,
            )
    finally:
        await system.shutdown()
        shutil.rmtree(storage_path, ignore_errors=True)

    # 打印汇总
    print()
    print("=" * 60)
    print("Baseline Throughput Results")
    print("=" * 60)

    if results["queue"]:
        q = results["queue"]
        print("\n--- Queue ---")
        print(f"  Duration:        {q['duration_s']:.1f}s")
        print(f"  Write throughput: {q['write_throughput_rec_s']:.0f} rec/s")
        print(f"  Read throughput:  {q['read_throughput_rec_s']:.0f} rec/s")
        print(
            f"  Write latency:   avg={q['write_latency_ms']['avg']:.2f}ms "
            f"p50={q['write_latency_ms']['p50']:.2f}ms p99={q['write_latency_ms']['p99']:.2f}ms"
        )
        print(
            f"  Read latency:    avg={q['read_latency_ms']['avg']:.2f}ms "
            f"p50={q['read_latency_ms']['p50']:.2f}ms p99={q['read_latency_ms']['p99']:.2f}ms"
        )

    if results["topic"]:
        t = results["topic"]
        print("\n--- Topic ---")
        print(f"  Duration:         {t['duration_s']:.1f}s")
        print(f"  Subscribers:      {t['num_subscribers']}")
        print(f"  Publish throughput: {t['publish_throughput_msg_s']:.0f} msg/s")
        print(
            f"  Delivered total:  {t['total_delivered']} ({t['delivered_throughput_msg_s']:.0f} msg/s)"
        )
        print(
            f"  Publish latency:  avg={t['publish_latency_ms']['avg']:.2f}ms "
            f"p50={t['publish_latency_ms']['p50']:.2f}ms p99={t['publish_latency_ms']['p99']:.2f}ms"
        )

    print("\n" + "=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
