"""
Stress tests for the Pulsing Distributed Memory Queue.

Separated from test_queue.py for faster CI — run with:
    pytest tests/python/streaming/test_queue_stress.py -v
"""

import asyncio
import hashlib
import random
import shutil
import string
import tempfile
import time

import pytest

import pulsing as pul
from pulsing.streaming import read_queue, write_queue


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def actor_system():
    system = await pul.actor_system()
    yield system
    await system.shutdown()


@pytest.fixture
def temp_storage_path():
    path = tempfile.mkdtemp(prefix="queue_stress_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ============================================================================
# Stress Tests
# ============================================================================


@pytest.mark.asyncio
async def test_high_concurrency_writes(actor_system, temp_storage_path):
    """Stress test: many concurrent writes."""
    writer = await write_queue(
        actor_system,
        topic="stress_write",
        bucket_column="id",
        num_buckets=8,
        batch_size=100,
        storage_path=temp_storage_path,
    )

    num_writers = 10
    records_per_writer = 100

    async def write_batch(writer_id: int):
        results = []
        for i in range(records_per_writer):
            result = await writer.put(
                {
                    "id": f"writer_{writer_id}_record_{i}",
                    "writer_id": writer_id,
                    "seq": i,
                }
            )
            results.append(result)
        return results

    start = time.time()
    tasks = [write_batch(i) for i in range(num_writers)]
    all_results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    total_writes = sum(len(r) for r in all_results)
    assert total_writes == num_writers * records_per_writer

    for results in all_results:
        for result in results:
            assert result["status"] == "ok"

    print(
        f"\nHigh concurrency writes: {total_writes} records in {elapsed:.2f}s "
        f"({total_writes / elapsed:.0f} records/s)"
    )


@pytest.mark.asyncio
async def test_high_concurrency_reads(actor_system, temp_storage_path):
    """Stress test: many concurrent reads."""
    writer = await write_queue(
        actor_system,
        topic="stress_read",
        bucket_column="id",
        num_buckets=4,
        batch_size=50,
        storage_path=temp_storage_path,
    )

    for i in range(500):
        await writer.put({"id": f"record_{i}", "value": i})
    await writer.flush()

    num_readers = 10

    async def read_all(reader_id: int):
        reader = await read_queue(
            actor_system,
            topic="stress_read",
            num_buckets=4,
            storage_path=temp_storage_path,
        )
        records = await reader.get(limit=500)
        return reader_id, len(records)

    start = time.time()
    tasks = [read_all(i) for i in range(num_readers)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    for reader_id, count in results:
        assert count > 0, f"Reader {reader_id} got no data"

    total_records = sum(count for _, count in results)
    print(
        f"\nHigh concurrency reads: {num_readers} readers, {total_records} total records "
        f"in {elapsed:.2f}s"
    )


@pytest.mark.asyncio
async def test_large_records(actor_system, temp_storage_path):
    """Stress test: large record payloads."""
    writer = await write_queue(
        actor_system,
        topic="large_records",
        bucket_column="id",
        num_buckets=4,
        batch_size=10,
        storage_path=temp_storage_path,
    )

    def generate_large_record(i: int) -> dict:
        return {
            "id": f"large_{i}",
            "data": "".join(random.choices(string.ascii_letters, k=1000)),
            "seq": i,
        }

    num_records = 100

    start = time.time()
    for i in range(num_records):
        await writer.put(generate_large_record(i))
    await writer.flush()
    elapsed = time.time() - start

    print(f"\nLarge records: {num_records} x 1KB records in {elapsed:.2f}s")

    reader = await read_queue(
        actor_system,
        topic="large_records",
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    records = await reader.get(limit=num_records)
    assert len(records) == num_records


@pytest.mark.asyncio
async def test_producer_consumer_stress(actor_system, temp_storage_path):
    """Stress test: concurrent producers and consumers."""
    topic = "producer_consumer_stress"
    num_buckets = 4

    writer = await write_queue(
        actor_system,
        topic=topic,
        bucket_column="id",
        num_buckets=num_buckets,
        batch_size=50,
        storage_path=temp_storage_path,
    )

    num_producers = 5
    records_per_producer = 100
    num_consumers = 3

    produced_ids = set()
    consumed_ids = set()
    produce_done = asyncio.Event()
    lock = asyncio.Lock()

    async def producer(producer_id: int):
        nonlocal produced_ids
        for i in range(records_per_producer):
            record_id = f"p{producer_id}_r{i}"
            await writer.put({"id": record_id, "producer": producer_id, "seq": i})
            async with lock:
                produced_ids.add(record_id)
            await asyncio.sleep(0.001)
        if producer_id == num_producers - 1:
            await writer.flush()
            produce_done.set()

    async def consumer(consumer_id: int):
        nonlocal consumed_ids
        reader = await read_queue(
            actor_system,
            topic=topic,
            rank=consumer_id,
            world_size=num_consumers,
            num_buckets=num_buckets,
            storage_path=temp_storage_path,
        )

        while True:
            records = await reader.get(limit=50, wait=True, timeout=0.5)
            if records:
                async with lock:
                    for r in records:
                        consumed_ids.add(r["id"])
            elif produce_done.is_set():
                records = await reader.get(limit=100)
                async with lock:
                    for r in records:
                        consumed_ids.add(r["id"])
                break

    start = time.time()

    producer_tasks = [asyncio.create_task(producer(i)) for i in range(num_producers)]
    consumer_tasks = [asyncio.create_task(consumer(i)) for i in range(num_consumers)]

    await asyncio.gather(*producer_tasks)
    await asyncio.sleep(1.0)

    for task in consumer_tasks:
        task.cancel()

    elapsed = time.time() - start

    total_produced = num_producers * records_per_producer

    print("\nProducer-Consumer stress test:")
    print(f"  Produced: {len(produced_ids)} records")
    print(f"  Consumed: {len(consumed_ids)} records")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Throughput: {len(produced_ids) / elapsed:.0f} records/s")

    assert len(produced_ids) == total_produced


@pytest.mark.asyncio
async def test_many_buckets(actor_system, temp_storage_path):
    """Stress test: many buckets."""
    num_buckets = 32

    writer = await write_queue(
        actor_system,
        topic="many_buckets",
        bucket_column="id",
        num_buckets=num_buckets,
        batch_size=20,
        storage_path=temp_storage_path,
    )

    num_records = 500
    for i in range(num_records):
        await writer.put({"id": f"record_{i}", "value": i})
    await writer.flush()

    stats = await writer.stats()
    non_empty = sum(1 for b in stats["buckets"].values() if b.get("total_count", 0) > 0)

    print(f"\nMany buckets test: {num_buckets} buckets, {non_empty} non-empty")
    assert non_empty >= num_buckets // 2


@pytest.mark.asyncio
async def test_rapid_flush_cycles(actor_system, temp_storage_path):
    """Stress test: rapid write-flush cycles."""
    writer = await write_queue(
        actor_system,
        topic="rapid_flush",
        bucket_column="id",
        num_buckets=4,
        batch_size=5,
        storage_path=temp_storage_path,
    )

    num_cycles = 50
    records_per_cycle = 10

    start = time.time()
    for cycle in range(num_cycles):
        for i in range(records_per_cycle):
            await writer.put({"id": f"c{cycle}_r{i}", "cycle": cycle, "seq": i})
        await writer.flush()
    elapsed = time.time() - start

    total_records = num_cycles * records_per_cycle

    print(
        f"\nRapid flush: {num_cycles} cycles, {total_records} records in {elapsed:.2f}s"
    )

    reader = await read_queue(
        actor_system,
        topic="rapid_flush",
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    all_records = []
    while True:
        records = await reader.get(limit=100)
        if not records:
            break
        all_records.extend(records)

    assert len(all_records) == total_records


@pytest.mark.asyncio
async def test_data_integrity_under_stress(actor_system, temp_storage_path):
    """Stress test: verify data integrity under concurrent load."""
    writer = await write_queue(
        actor_system,
        topic="integrity_test",
        bucket_column="id",
        num_buckets=4,
        batch_size=20,
        storage_path=temp_storage_path,
    )

    num_records = 200
    expected_data = {}

    for i in range(num_records):
        record_id = f"integrity_{i}"
        value = random.randint(0, 1000000)
        checksum = hashlib.md5(f"{record_id}:{value}".encode()).hexdigest()

        await writer.put(
            {
                "id": record_id,
                "value": value,
                "checksum": checksum,
            }
        )
        expected_data[record_id] = (value, checksum)

    await writer.flush()

    reader = await read_queue(
        actor_system,
        topic="integrity_test",
        num_buckets=4,
        storage_path=temp_storage_path,
    )

    all_records = []
    while True:
        records = await reader.get(limit=100)
        if not records:
            break
        all_records.extend(records)

    assert len(all_records) == num_records

    for record in all_records:
        record_id = record["id"]
        expected_value, expected_checksum = expected_data[record_id]

        actual_checksum = hashlib.md5(
            f"{record_id}:{record['value']}".encode()
        ).hexdigest()
        assert record["checksum"] == expected_checksum, (
            f"Checksum mismatch for {record_id}"
        )
        assert actual_checksum == expected_checksum, f"Value corruption for {record_id}"
