#!/usr/bin/env python3
"""
多并发扫描：在不同 生产者/消费者 并发组合下测量 Queue 与 Topic 吞吐。

扫描 (P,C) 组合，例如 P,C ∈ {1,2,4,8}，得到每种并发下的写/读吞吐，便于看扩展性。

Usage:
    python benchmarks/concurrency_sweep.py
    python benchmarks/concurrency_sweep.py --producers 1 2 4 8 --consumers 1 2 4 8 --duration 8
    python benchmarks/concurrency_sweep.py --queue-only --output sweep_queue.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time

import pulsing as pul
from pulsing.queue import read_queue, write_queue
from pulsing.topic import PublishMode, read_topic, write_topic


# =============================================================================
# Queue 多并发
# =============================================================================


async def run_queue_concurrent(
    system,
    storage_path: str,
    num_producers: int,
    num_consumers: int,
    duration: float,
    num_buckets: int,
    record_size: int,
) -> dict:
    """P 个 writer + C 个 reader (rank/world_size)，固定时长，汇总写/读吞吐."""
    topic = f"sweep_q_p{num_producers}_c{num_consumers}"
    write_counts = [0] * num_producers
    read_counts = [0] * num_consumers
    write_locks = [asyncio.Lock() for _ in range(num_producers)]
    read_locks = [asyncio.Lock() for _ in range(num_consumers)]

    async def producer(pid: int):
        writer = await write_queue(
            system,
            topic=topic,
            bucket_column="id",
            num_buckets=num_buckets,
            storage_path=storage_path,
        )
        end_time = time.monotonic() + duration
        i = 0
        while time.monotonic() < end_time:
            try:
                await writer.put({"id": f"p{pid}_{i}", "payload": "x" * record_size})
                async with write_locks[pid]:
                    write_counts[pid] += 1
                i += 1
            except Exception:
                pass

    async def consumer(rank: int):
        reader = await read_queue(
            system,
            topic=topic,
            rank=rank,
            world_size=num_consumers,
            num_buckets=num_buckets,
            storage_path=storage_path,
        )
        end_time = time.monotonic() + duration
        while time.monotonic() < end_time:
            try:
                batch = await reader.get(limit=50, wait=True, timeout=1.0)
                if batch:
                    async with read_locks[rank]:
                        read_counts[rank] += len(batch)
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

    await asyncio.gather(
        *[producer(p) for p in range(num_producers)],
        *[consumer(c) for c in range(num_consumers)],
    )

    total_written = sum(write_counts)
    total_read = sum(read_counts)
    return {
        "num_producers": num_producers,
        "num_consumers": num_consumers,
        "records_written": total_written,
        "records_read": total_read,
        "write_throughput_rec_s": total_written / duration if duration > 0 else 0,
        "read_throughput_rec_s": total_read / duration if duration > 0 else 0,
        "duration_s": duration,
    }


# =============================================================================
# Topic 多并发
# =============================================================================


async def run_topic_concurrent(
    system,
    num_publishers: int,
    num_subscribers: int,
    duration: float,
    payload_size: int,
) -> dict:
    """P 个 publisher + C 个 subscriber，fire_and_forget，汇总发布与交付吞吐."""
    topic_name = f"sweep_t_p{num_publishers}_c{num_subscribers}"
    publish_counts = [0] * num_publishers
    delivered_counts = [0] * num_subscribers
    pub_locks = [asyncio.Lock() for _ in range(num_publishers)]
    del_locks = [asyncio.Lock() for _ in range(num_subscribers)]

    # 先起订阅者
    readers = []
    for c in range(num_subscribers):
        reader = await read_topic(system, topic_name, reader_id=f"sub_{c}")

        def make_cb(cid):
            async def cb(_msg):
                async with del_locks[cid]:
                    delivered_counts[cid] += 1

            return cb

        reader.add_callback(make_cb(c))
        await reader.start()
        readers.append(reader)

    async def publisher(pid: int):
        writer = await write_topic(system, topic_name, writer_id=f"pub_{pid}")
        end_time = time.monotonic() + duration
        seq = 0
        while time.monotonic() < end_time:
            try:
                await writer.publish(
                    {"seq": seq, "payload": "x" * payload_size},
                    mode=PublishMode.FIRE_AND_FORGET,
                )
                async with pub_locks[pid]:
                    publish_counts[pid] += 1
                seq += 1
            except Exception:
                pass

    await asyncio.gather(*[publisher(p) for p in range(num_publishers)])

    await asyncio.sleep(0.3)
    for r in readers:
        await r.stop()

    total_published = sum(publish_counts)
    total_delivered = sum(delivered_counts)
    return {
        "num_publishers": num_publishers,
        "num_subscribers": num_subscribers,
        "messages_published": total_published,
        "messages_delivered": total_delivered,
        "publish_throughput_msg_s": total_published / duration if duration > 0 else 0,
        "delivered_throughput_msg_s": total_delivered / duration if duration > 0 else 0,
        "duration_s": duration,
    }


# =============================================================================
# 扫描与输出
# =============================================================================


def parse_concurrency(s: str) -> list[int]:
    """解析如 '1,2,4,8' 或 '1-4' 为整数列表."""
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.replace(",", " ").split()]


async def main():
    parser = argparse.ArgumentParser(
        description="多并发扫描：不同 生产者/消费者 下的 Queue & Topic 吞吐"
    )
    parser.add_argument(
        "--producers",
        type=str,
        default="1,2,4,8",
        help="生产者并发数列表，如 1,2,4,8 或 1-4",
    )
    parser.add_argument(
        "--consumers",
        type=str,
        default="1,2,4,8",
        help="消费者并发数列表，如 1,2,4,8 或 1-4",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="每个 (P,C) 组合运行时长（秒）",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=8,
        help="Queue 桶数",
    )
    parser.add_argument(
        "--record-size",
        type=int,
        default=100,
        help="单条记录 payload 字节数",
    )
    parser.add_argument(
        "--queue-only",
        action="store_true",
        help="仅扫描 Queue",
    )
    parser.add_argument(
        "--topic-only",
        action="store_true",
        help="仅扫描 Topic",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果 JSON 文件路径",
    )
    args = parser.parse_args()

    prods = parse_concurrency(args.producers)
    cons = parse_concurrency(args.consumers)

    system = await pul.actor_system()
    storage_path = tempfile.mkdtemp(prefix="concurrency_sweep_")
    results = {"queue": [], "topic": []}

    try:
        if not args.topic_only:
            print("Queue concurrency sweep...")
            for P in prods:
                for C in cons:
                    if C > args.num_buckets:
                        continue
                    r = await run_queue_concurrent(
                        system,
                        storage_path=storage_path,
                        num_producers=P,
                        num_consumers=C,
                        duration=args.duration,
                        num_buckets=args.num_buckets,
                        record_size=args.record_size,
                    )
                    results["queue"].append(r)
                    print(
                        f"  P={P} C={C}  write={r['write_throughput_rec_s']:.0f} rec/s  read={r['read_throughput_rec_s']:.0f} rec/s"
                    )

        if not args.queue_only:
            print("Topic concurrency sweep...")
            for P in prods:
                for C in cons:
                    r = await run_topic_concurrent(
                        system,
                        num_publishers=P,
                        num_subscribers=C,
                        duration=args.duration,
                        payload_size=args.record_size,
                    )
                    results["topic"].append(r)
                    print(
                        f"  P={P} C={C}  publish={r['publish_throughput_msg_s']:.0f} msg/s  delivered={r['delivered_throughput_msg_s']:.0f} msg/s"
                    )
    finally:
        await system.shutdown()
        shutil.rmtree(storage_path, ignore_errors=True)

    # 打印汇总表
    print()
    print("=" * 72)
    print("Concurrency Sweep Summary")
    print("=" * 72)

    if results["queue"]:
        print("\n--- Queue (rec/s) ---")
        print(f"{'P':>3} {'C':>3}  {'write_rec/s':>12}  {'read_rec/s':>12}")
        print("-" * 40)
        for r in results["queue"]:
            print(
                f"{r['num_producers']:>3} {r['num_consumers']:>3}  "
                f"{r['write_throughput_rec_s']:>12.0f}  {r['read_throughput_rec_s']:>12.0f}"
            )

    if results["topic"]:
        print("\n--- Topic (msg/s) ---")
        print(f"{'P':>3} {'C':>3}  {'publish_msg/s':>14}  {'delivered_msg/s':>16}")
        print("-" * 50)
        for r in results["topic"]:
            print(
                f"{r['num_publishers']:>3} {r['num_subscribers']:>3}  "
                f"{r['publish_throughput_msg_s']:>14.0f}  {r['delivered_throughput_msg_s']:>16.0f}"
            )

    print("\n" + "=" * 72)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "config": {
                        "duration_s": args.duration,
                        "num_buckets": args.num_buckets,
                        "record_size": args.record_size,
                        "producers": prods,
                        "consumers": cons,
                    },
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
