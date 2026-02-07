#!/usr/bin/env python3
"""
多进程压测：用 multiprocessing 启动 N 个进程，每进程一个 ActorSystem 节点，组成集群，
对 Queue 与 Topic 做多进程压力测试（不依赖 torchrun）。

Usage:
    python benchmarks/stress_multiprocessing.py --nprocs 4 --duration 20
    python benchmarks/stress_multiprocessing.py --nprocs 8 --queue-only --num-buckets 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from multiprocessing import Queue

import pulsing as pul
from pulsing.queue import read_queue, write_queue
from pulsing.topic import PublishMode, read_topic, write_topic


# =============================================================================
# 单进程内异步压测逻辑（Queue + Topic）
# =============================================================================


async def _run_worker(
    rank: int,
    world_size: int,
    seed_queue: Queue,
    result_queue: Queue,
    config: dict,
) -> None:
    """单进程 worker：加入集群后跑 Queue 与 Topic 压测，结果放入 result_queue."""
    base_port = config.get("base_port", 9100)
    duration = config["duration"]
    num_buckets = config["num_buckets"]
    record_size = config["record_size"]
    storage_path = config["storage_path"]
    run_queue_bench = config.get("queue", True)
    run_topic_bench = config.get("topic", True)
    stabilize = config.get("stabilize_timeout", 4.0)

    # rank 0 先起节点并 bind，再往 queue 里放 seed，供其余 (world_size-1) 个进程各 get 一次
    system = None
    if rank == 0:
        addr = f"127.0.0.1:{base_port}"
        system = await pul.actor_system(addr=addr)
        for _ in range(world_size - 1):
            seed_queue.put(addr)
    else:
        seed_addr = seed_queue.get(timeout=60.0)
        await asyncio.sleep(0.5)
        system = await pul.actor_system(
            addr=f"127.0.0.1:{base_port + rank}",
            seeds=[seed_addr],
        )

    await asyncio.sleep(stabilize)

    result = {"rank": rank, "world_size": world_size, "queue": None, "topic": None}

    if run_queue_bench:
        topic_q = f"mp_queue_{world_size}"
        writer = await write_queue(
            system,
            topic=topic_q,
            bucket_column="id",
            num_buckets=num_buckets,
            storage_path=storage_path,
        )
        reader = await read_queue(
            system,
            topic=topic_q,
            rank=rank,
            world_size=world_size,
            num_buckets=num_buckets,
            storage_path=storage_path,
        )
        written = 0
        read_count = 0
        end_time = time.monotonic() + duration
        while time.monotonic() < end_time:
            try:
                await writer.put(
                    {"id": f"r{rank}_{written}", "payload": "x" * record_size}
                )
                written += 1
            except Exception:
                pass
        try:
            await writer.flush()
        except Exception:
            pass
        while time.monotonic() < end_time + 2:
            try:
                batch = await reader.get(limit=100, wait=True, timeout=0.5)
                if batch:
                    read_count += len(batch)
                else:
                    break
            except asyncio.TimeoutError:
                break
            except Exception:
                break
        result["queue"] = {
            "records_written": written,
            "records_read": read_count,
            "write_rec_s": written / duration if duration > 0 else 0,
            "read_rec_s": read_count / duration if duration > 0 else 0,
        }

    if run_topic_bench:
        topic_t = f"mp_topic_{world_size}"
        recv_count = [0]
        reader = await read_topic(system, topic_t, reader_id=f"mp_{rank}")

        def on_msg(_):
            recv_count[0] += 1

        reader.add_callback(on_msg)
        await reader.start()
        writer = await write_topic(system, topic_t, writer_id=f"mp_{rank}")
        published = 0
        end_time = time.monotonic() + duration
        while time.monotonic() < end_time:
            try:
                await writer.publish(
                    {"seq": published, "payload": "x" * record_size},
                    mode=PublishMode.FIRE_AND_FORGET,
                )
                published += 1
            except Exception:
                pass
        await asyncio.sleep(0.5)
        await reader.stop()
        result["topic"] = {
            "messages_published": published,
            "messages_delivered": recv_count[0],
            "publish_msg_s": published / duration if duration > 0 else 0,
            "delivered_msg_s": recv_count[0] / duration if duration > 0 else 0,
        }

    await system.shutdown()
    result_queue.put(result)


def _worker_entry(
    rank: int,
    world_size: int,
    seed_queue: Queue,
    result_queue: Queue,
    config: dict,
) -> None:
    asyncio.run(_run_worker(rank, world_size, seed_queue, result_queue, config))


# =============================================================================
# Main：spawn 多进程并汇总
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="多进程 Queue & Topic 压测（multiprocessing）"
    )
    parser.add_argument("--nprocs", type=int, default=4, help="进程数（集群节点数）")
    parser.add_argument("--duration", type=float, default=20.0, help="压测时长（秒）")
    parser.add_argument("--num-buckets", type=int, default=8, help="Queue 桶数")
    parser.add_argument(
        "--record-size", type=int, default=100, help="单条 payload 字节数"
    )
    parser.add_argument(
        "--base-port", type=int, default=9100, help="首节点端口，其余 base_port+rank"
    )
    parser.add_argument(
        "--stabilize-timeout", type=float, default=4.0, help="集群稳定等待时间"
    )
    parser.add_argument("--queue-only", action="store_true", help="仅压 Queue")
    parser.add_argument("--topic-only", action="store_true", help="仅压 Topic")
    parser.add_argument("--output", type=str, default=None, help="汇总结果 JSON 路径")
    args = parser.parse_args()

    nprocs = args.nprocs
    run_queue = not args.topic_only
    run_topic = not args.queue_only

    storage_path = os.path.abspath(tempfile.mkdtemp(prefix="stress_mp_"))
    config = {
        "duration": args.duration,
        "num_buckets": args.num_buckets,
        "record_size": args.record_size,
        "storage_path": storage_path,
        "base_port": args.base_port,
        "stabilize_timeout": args.stabilize_timeout,
        "queue": run_queue,
        "topic": run_topic,
    }

    seed_queue = mp.Queue()
    result_queue = mp.Queue()

    procs = []
    for rank in range(nprocs):
        p = mp.Process(
            target=_worker_entry,
            args=(rank, nprocs, seed_queue, result_queue, config),
        )
        procs.append(p)

    print("Starting %d processes (cluster size=%d)..." % (nprocs, nprocs))
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=args.duration + 60)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    shutil.rmtree(storage_path, ignore_errors=True)

    results = []
    while not result_queue.empty():
        try:
            results.append(result_queue.get_nowait())
        except Exception:
            break

    if len(results) != nprocs:
        print("WARNING: got %d results, expected %d" % (len(results), nprocs))

    # 汇总
    print()
    print("=" * 60)
    print("Multiprocessing Stress Summary (cluster size=%d)" % nprocs)
    print("=" * 60)

    agg = {"queue": None, "topic": None}
    if run_queue and results:
        total_write = sum(
            r["queue"]["records_written"] for r in results if r.get("queue")
        )
        total_read = sum(r["queue"]["records_read"] for r in results if r.get("queue"))
        dur = args.duration
        agg["queue"] = {
            "total_records_written": total_write,
            "total_records_read": total_read,
            "write_throughput_rec_s": total_write / dur if dur > 0 else 0,
            "read_throughput_rec_s": total_read / dur if dur > 0 else 0,
        }
        print("\n--- Queue ---")
        print(
            "  Total written: %d  (%.0f rec/s)"
            % (total_write, agg["queue"]["write_throughput_rec_s"])
        )
        print(
            "  Total read:    %d  (%.0f rec/s)"
            % (total_read, agg["queue"]["read_throughput_rec_s"])
        )

    if run_topic and results:
        total_pub = sum(
            r["topic"]["messages_published"] for r in results if r.get("topic")
        )
        total_del = sum(
            r["topic"]["messages_delivered"] for r in results if r.get("topic")
        )
        dur = args.duration
        agg["topic"] = {
            "total_published": total_pub,
            "total_delivered": total_del,
            "publish_throughput_msg_s": total_pub / dur if dur > 0 else 0,
            "delivered_throughput_msg_s": total_del / dur if dur > 0 else 0,
        }
        print("\n--- Topic ---")
        print(
            "  Total published: %d  (%.0f msg/s)"
            % (total_pub, agg["topic"]["publish_throughput_msg_s"])
        )
        print(
            "  Total delivered: %d  (%.0f msg/s)"
            % (total_del, agg["topic"]["delivered_throughput_msg_s"])
        )

    print("\n" + "=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "nprocs": nprocs,
                    "config": {
                        "duration": args.duration,
                        "num_buckets": args.num_buckets,
                        "record_size": args.record_size,
                    },
                    "aggregate": agg,
                    "per_rank": results,
                },
                f,
                indent=2,
            )
        print("Results written to %s" % args.output)


if __name__ == "__main__":
    main()
