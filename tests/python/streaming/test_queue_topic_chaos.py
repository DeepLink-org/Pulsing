"""
Queue & Topic Chaos Testing

Validates under chaotic scenarios with random delays, high concurrency, dynamic join/leave, random parameters:
- Queue: no data loss, no duplication (bucketed by rank/world_size), no deadlock
- Topic: no crash during publish when subscribers dynamically change, distinguishable delivery semantics, slow/failed subscribers kicked or timeout
- No blocking or race conditions when sharing resources with StorageManager

Run: pytest tests/python/test_queue_topic_chaos.py -v -s
"""

from __future__ import annotations

import asyncio
import random
import shutil
import tempfile
import time

import pytest

import pulsing as pul
from pulsing.streaming import (
    read_queue,
    write_queue,
    PublishMode,
    read_topic,
    write_topic,
)


# =============================================================================
# Fixtures & Random Load Utilities
# =============================================================================


@pytest.fixture
async def actor_system():
    system = await pul.actor_system()
    yield system
    await system.shutdown()


@pytest.fixture
def temp_storage_path():
    path = tempfile.mkdtemp(prefix="chaos_queue_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _random_sleep(max_ms: int = 20):
    """Short random delay to simulate chaos."""
    return asyncio.sleep(random.uniform(0, max_ms) / 1000.0)


def _chaos_sleep(
    min_ms: int = 0, max_ms: int = 50, occasional_long_ms: int | None = 120
):
    """Random delay: normally min~max_ms, small chance of long delay (simulating jitter)."""
    if occasional_long_ms and random.random() < 0.08:
        return asyncio.sleep(random.uniform(max_ms, occasional_long_ms) / 1000.0)
    return asyncio.sleep(random.uniform(min_ms, max_ms) / 1000.0)


# =============================================================================
# Queue Chaos
# =============================================================================


@pytest.mark.asyncio
async def test_queue_chaos_concurrent_producer_consumer(
    actor_system, temp_storage_path
):
    """Chaos: multiple producers + multiple consumers (rank/world_size), random put/get/delay, verify no loss or duplication."""
    random.seed(42)
    topic = "chaos_q_concurrent"
    num_buckets = random.choice([3, 4, 5, 6])
    world_size = random.choice([2, 3])
    num_producers = random.randint(2, 5)
    messages_per_producer = random.randint(30, 70)
    total_expected = num_producers * messages_per_producer

    produced_ids: set[str] = set()
    produced_lock = asyncio.Lock()

    async def producer(pid: int):
        writer = await write_queue(
            actor_system,
            topic=topic,
            bucket_column="id",
            num_buckets=num_buckets,
            storage_path=temp_storage_path,
        )
        for i in range(messages_per_producer):
            rid = f"p{pid}_r{i}"
            await writer.put({"id": rid, "producer": pid, "seq": i})
            async with produced_lock:
                produced_ids.add(rid)
            await _chaos_sleep(0, 15, 40)
            if random.random() < 0.1:
                await writer.flush()
        await writer.flush()

    async def consumer(rank: int):
        reader = await read_queue(
            actor_system,
            topic=topic,
            rank=rank,
            world_size=world_size,
            num_buckets=num_buckets,
            storage_path=temp_storage_path,
        )
        seen: set[str] = set()
        deadline = time.monotonic() + 20.0
        get_limit = random.randint(10, 40)
        while time.monotonic() < deadline:
            records = await reader.get(limit=get_limit, wait=True, timeout=1.2)
            for r in records:
                rid = r.get("id")
                if rid:
                    assert rid not in seen, f"duplicate consumption: {rid}"
                    seen.add(rid)
            await _chaos_sleep(2, 25, 50)
            if len(seen) >= total_expected:
                break
        return rank, seen

    await asyncio.gather(*[producer(i) for i in range(num_producers)])

    consumer_tasks = [asyncio.create_task(consumer(r)) for r in range(world_size)]
    results = await asyncio.gather(*consumer_tasks)

    consumed_all: set[str] = set()
    for _, seen in results:
        consumed_all |= seen

    assert (
        len(consumed_all) == total_expected
    ), f"expected {total_expected} unique ids, got {len(consumed_all)}; produced={len(produced_ids)}"
    assert consumed_all == produced_ids, "consumed set != produced set"


@pytest.mark.asyncio
async def test_queue_chaos_many_buckets_parallel_handles(
    actor_system, temp_storage_path
):
    """Chaos: many buckets, multiple writers in parallel, multiple readers in parallel; use single reader to collect all and verify total (multiple readers would split data)."""
    random.seed(43)
    topic = "chaos_q_many_buckets"
    num_buckets = random.randint(4, 12)
    num_writers = random.randint(3, 6)
    puts_per_writer = random.randint(20, 50)
    expected_count = num_writers * puts_per_writer

    async def write_batch(wid: int):
        w = await write_queue(
            actor_system,
            topic=topic,
            bucket_column="id",
            num_buckets=num_buckets,
            storage_path=temp_storage_path,
        )
        i = 0
        while i < puts_per_writer:
            if random.random() < 0.2 and i + 1 < puts_per_writer:
                await w.put(
                    [
                        {"id": f"w{wid}_{i}", "v": i},
                        {"id": f"w{wid}_{i+1}", "v": i + 1},
                    ]
                )
                i += 2
            else:
                await w.put({"id": f"w{wid}_{i}", "v": i})
                i += 1
            await _chaos_sleep(0, 15, 45)
        await w.flush()
        return wid

    await asyncio.gather(*[write_batch(w) for w in range(num_writers)])

    # Single reader reads full data, avoiding multiple readers splitting data resulting in insufficient union
    r = await read_queue(
        actor_system,
        topic=topic,
        num_buckets=num_buckets,
        storage_path=temp_storage_path,
    )
    collected = []
    for _ in range(120):
        limit = random.randint(5, 60)
        batch = await r.get(limit=limit)
        collected.extend(batch)
        if len(collected) >= expected_count:
            break
        await _chaos_sleep(1, 20, None)

    all_ids = {rec.get("id") for rec in collected}
    assert (
        len(all_ids) == expected_count
    ), f"expected {expected_count} unique ids, got {len(all_ids)}"


@pytest.mark.asyncio
async def test_queue_chaos_reader_reset_and_reread(actor_system, temp_storage_path):
    """Chaos: same reader multiple reset + get, interleaved with intermittent writes, random limit/delay."""
    random.seed(44)
    topic = "chaos_q_reset"
    num_buckets = random.choice([2, 3, 4])
    writer = await write_queue(
        actor_system,
        topic=topic,
        bucket_column="id",
        num_buckets=num_buckets,
        storage_path=temp_storage_path,
    )
    n_init = random.randint(15, 30)
    for i in range(n_init):
        await writer.put({"id": f"x{i}", "i": i})
        await _chaos_sleep(0, 5, None)
    await writer.flush()

    reader = await read_queue(
        actor_system,
        topic=topic,
        num_buckets=num_buckets,
        storage_path=temp_storage_path,
    )

    limit_a = random.randint(5, 12)
    first = await reader.get(limit=limit_a)
    first_ids = {r["id"] for r in first}
    reader.reset()
    again = await reader.get(limit=limit_a)
    again_ids = {r["id"] for r in again}
    assert first_ids == again_ids, "reset then get should see same first batch"

    await writer.put({"id": "new1", "i": 100})
    await _chaos_sleep(2, 15, 30)
    more = await reader.get(limit=random.randint(5, 15))
    ids_more = {r["id"] for r in more}
    assert "new1" in ids_more or any(r.get("id") == "new1" for r in more)


# =============================================================================
# Topic Chaos
# =============================================================================


@pytest.mark.asyncio
async def test_topic_chaos_subscribers_join_leave_during_publish(actor_system):
    """Chaos: subscribers dynamically join/leave during publishing, random phases/messages per phase/mode/delay."""
    random.seed(45)
    topic_name = "chaos_t_join_leave"
    writer = await write_topic(actor_system, topic_name)
    num_phases = random.randint(4, 8)
    messages_per_phase = random.randint(6, 18)
    modes = [PublishMode.FIRE_AND_FORGET, PublishMode.BEST_EFFORT]

    received_by_id: dict[str, list] = {}
    rec_lock = asyncio.Lock()

    async def make_reader(reader_id: str):
        reader = await read_topic(actor_system, topic_name, reader_id=reader_id)
        rec_list = []

        async def on_msg(msg):
            async with rec_lock:
                rec_list.append(msg)
            received_by_id[reader_id] = rec_list

        reader.add_callback(on_msg)
        await reader.start()
        return reader

    readers_alive: list[tuple[str, object]] = []

    for phase in range(num_phases):
        if random.random() < 0.35 and phase > 0 and readers_alive:
            _, r = readers_alive.pop()
            await r.stop()
        rid = f"r{phase}"
        reader = await make_reader(rid)
        readers_alive.append((rid, reader))

        for i in range(messages_per_phase):
            mode = random.choice(modes)
            await writer.publish(
                {"phase": phase, "seq": i},
                mode=mode,
            )
            await _chaos_sleep(1, 12, 35)

    await asyncio.sleep(0.4)

    for _, reader in readers_alive:
        await reader.stop()

    for rid, rec_list in received_by_id.items():
        assert (
            len(rec_list) > 0
        ), f"reader {rid} should have received at least one message"


@pytest.mark.asyncio
async def test_topic_chaos_many_publishers_many_subscribers(actor_system):
    """Chaos: multiple publishers + multiple subscribers, random publish mode/count/delay, verify each receives expected count."""
    random.seed(46)
    topic_name = "chaos_t_many"
    num_publishers = random.randint(3, 6)
    num_subscribers = random.randint(2, 5)
    messages_per_pub = random.randint(15, 40)
    total_messages = num_publishers * messages_per_pub

    received: list[list] = [[] for _ in range(num_subscribers)]
    locks = [asyncio.Lock() for _ in range(num_subscribers)]
    readers = []

    for i in range(num_subscribers):
        reader = await read_topic(actor_system, topic_name, reader_id=f"sub_{i}")

        async def make_cb(idx):
            async def cb(msg):
                async with locks[idx]:
                    received[idx].append(msg)

            return cb

        reader.add_callback(await make_cb(i))
        await reader.start()
        readers.append(reader)

    async def publish_batch(pid: int):
        w = await write_topic(actor_system, topic_name, writer_id=f"pub_{pid}")
        modes = [PublishMode.FIRE_AND_FORGET, PublishMode.BEST_EFFORT]
        for j in range(messages_per_pub):
            mode = random.choice(modes)
            await w.publish({"pub": pid, "seq": j}, mode=mode)
            await _chaos_sleep(0, 12, 30)

    await asyncio.gather(*[publish_batch(p) for p in range(num_publishers)])

    await asyncio.sleep(0.6)

    for i in range(num_subscribers):
        assert (
            len(received[i]) == total_messages
        ), f"subscriber {i} expected {total_messages}, got {len(received[i])}"

    for r in readers:
        await r.stop()


@pytest.mark.asyncio
async def test_topic_chaos_slow_callback_best_effort(actor_system):
    """Chaos: some subscriber callbacks are slow, random count/delay/timeout, best_effort verify no crash."""
    random.seed(47)
    topic_name = "chaos_t_slow"
    writer = await write_topic(actor_system, topic_name)
    num_messages = random.randint(12, 25)

    fast_recv = []
    reader_fast = await read_topic(actor_system, topic_name, reader_id="fast")
    reader_fast.add_callback(lambda m: fast_recv.append(m))
    await reader_fast.start()

    slow_recv = []
    reader_slow = await read_topic(actor_system, topic_name, reader_id="slow")
    slow_delay = random.uniform(0.05, 0.15)

    async def slow_cb(m):
        await asyncio.sleep(slow_delay)
        slow_recv.append(m)

    reader_slow.add_callback(slow_cb)
    await reader_slow.start()

    for i in range(num_messages):
        await writer.publish(
            {"seq": i},
            mode=PublishMode.BEST_EFFORT,
            timeout=random.uniform(1.5, 3.0),
        )
        await _chaos_sleep(2, 20, 50)

    await asyncio.sleep(0.5)

    assert (
        len(fast_recv) == num_messages
    ), f"fast subscriber should get all {num_messages}, got {len(fast_recv)}"
    await reader_fast.stop()
    await reader_slow.stop()


# =============================================================================
# Mixed: Queue + Topic Chaos Simultaneously
# =============================================================================


@pytest.mark.asyncio
async def test_chaos_mixed_queue_and_topic_same_loop(actor_system, temp_storage_path):
    """Chaos: queue + topic concurrent in same loop, random count/buckets/delay."""
    random.seed(48)
    q_topic = "chaos_mixed_q"
    t_topic = "chaos_mixed_t"
    num_buckets = random.randint(2, 6)
    q_count = random.randint(25, 55)
    t_count = random.randint(20, 45)

    q_done = asyncio.Event()
    t_done = asyncio.Event()

    async def queue_chaos():
        w = await write_queue(
            actor_system,
            topic=q_topic,
            bucket_column="id",
            num_buckets=num_buckets,
            storage_path=temp_storage_path,
        )
        for i in range(q_count):
            await w.put({"id": f"q{i}", "i": i})
            await _chaos_sleep(0, 10, 25)
        await w.flush()
        r = await read_queue(
            actor_system,
            topic=q_topic,
            num_buckets=num_buckets,
            storage_path=temp_storage_path,
        )
        records = await r.get(limit=q_count + 10)
        assert len(records) == q_count
        q_done.set()

    async def topic_chaos():
        writer = await write_topic(actor_system, t_topic)
        recv = []
        reader = await read_topic(actor_system, t_topic, reader_id="mixed_r")
        reader.add_callback(lambda m: recv.append(m))
        await reader.start()
        modes = [PublishMode.FIRE_AND_FORGET, PublishMode.BEST_EFFORT]
        for i in range(t_count):
            await writer.publish({"i": i}, mode=random.choice(modes))
            await _chaos_sleep(0, 8, 20)
        await asyncio.sleep(0.25)
        assert len(recv) == t_count
        await reader.stop()
        t_done.set()

    await asyncio.gather(queue_chaos(), topic_chaos())
    assert q_done.is_set() and t_done.is_set()


@pytest.mark.asyncio
async def test_chaos_rapid_open_close_handles(actor_system, temp_storage_path):
    """Chaos: rapidly create/discard queue writer and topic reader repeatedly, random times/delay."""
    random.seed(49)
    n_writes = random.randint(6, 12)
    n_readers = random.randint(4, 10)
    num_buckets = random.choice([2, 3, 4])

    for _ in range(n_writes):
        w = await write_queue(
            actor_system,
            topic="chaos_rapid_q",
            bucket_column="id",
            num_buckets=num_buckets,
            storage_path=temp_storage_path,
        )
        await w.put({"id": f"x{random.randint(0,1000)}", "v": 1})
        await w.flush()
        del w
        await _chaos_sleep(1, 15, 40)

    for i in range(n_readers):
        reader = await read_topic(actor_system, "chaos_rapid_t", reader_id=f"rapid_{i}")
        reader.add_callback(lambda m: None)
        await reader.start()
        await _chaos_sleep(2, 12, None)
        await reader.stop()
        del reader

    writer = await write_topic(actor_system, "chaos_rapid_t")
    result = await writer.publish({"test": True}, mode=PublishMode.FIRE_AND_FORGET)
    assert result.subscriber_count >= 0

    # -------------------------------------------------------------------------
    # Added: High Complexity / Random Load Storm
    # -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_chaos_storm_random_params(actor_system, temp_storage_path):
    """Chaos storm: fully random parameters (buckets/consumers/producers/count/get limit/delay), verify no loss or duplication."""
    random.seed(100)
    topic = "chaos_q_storm"
    num_buckets = random.randint(2, 8)
    world_size = random.randint(2, 4)
    num_producers = random.randint(2, 6)
    messages_per_producer = random.randint(20, 55)
    total_expected = num_producers * messages_per_producer

    produced_ids: set[str] = set()
    plock = asyncio.Lock()

    async def producer(pid: int):
        w = await write_queue(
            actor_system,
            topic=topic,
            bucket_column="id",
            num_buckets=num_buckets,
            storage_path=temp_storage_path,
        )
        i = 0
        while i < messages_per_producer:
            if random.random() < 0.15 and i + 1 < messages_per_producer:
                batch = [
                    {"id": f"storm_p{pid}_{i}", "p": pid, "i": i},
                    {"id": f"storm_p{pid}_{i+1}", "p": pid, "i": i + 1},
                ]
                await w.put(batch)
                async with plock:
                    produced_ids.add(batch[0]["id"])
                    produced_ids.add(batch[1]["id"])
                i += 2
            else:
                rid = f"storm_p{pid}_{i}"
                await w.put({"id": rid, "p": pid, "i": i})
                async with plock:
                    produced_ids.add(rid)
                i += 1
            await _chaos_sleep(0, 20, 60)
        await w.flush()

    async def consumer(rank: int):
        r = await read_queue(
            actor_system,
            topic=topic,
            rank=rank,
            world_size=world_size,
            num_buckets=num_buckets,
            storage_path=temp_storage_path,
        )
        seen: set[str] = set()
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            limit = random.randint(8, 50)
            records = await r.get(limit=limit, wait=True, timeout=1.5)
            for rec in records:
                rid = rec.get("id")
                if rid:
                    assert rid not in seen, f"duplicate: {rid}"
                    seen.add(rid)
            await _chaos_sleep(1, 30, 80)
            if len(seen) >= total_expected:
                break
        return seen

    await asyncio.gather(*[producer(i) for i in range(num_producers)])

    results = await asyncio.gather(*[consumer(r) for r in range(world_size)])
    consumed_all: set[str] = set()
    for s in results:
        consumed_all |= s

    assert (
        consumed_all == produced_ids
    ), f"storm: produced {len(produced_ids)} vs consumed {len(consumed_all)}"


@pytest.mark.asyncio
async def test_topic_chaos_storm_random_params(actor_system):
    """Chaos storm: fully random topic parameters (publishers/subscribers count, messages, mode, delay), verify delivery."""
    random.seed(101)
    topic_name = "chaos_t_storm"
    num_publishers = random.randint(2, 5)
    num_subscribers = random.randint(2, 5)
    messages_per_pub = random.randint(18, 45)
    total_messages = num_publishers * messages_per_pub

    received: list[list] = [[] for _ in range(num_subscribers)]
    locks = [asyncio.Lock() for _ in range(num_subscribers)]
    readers = []

    for i in range(num_subscribers):
        reader = await read_topic(actor_system, topic_name, reader_id=f"storm_sub_{i}")

        async def make_cb(idx):
            async def cb(msg):
                async with locks[idx]:
                    received[idx].append(msg)

            return cb

        reader.add_callback(await make_cb(i))
        await reader.start()
        readers.append(reader)

    async def pub(pid: int):
        w = await write_topic(actor_system, topic_name, writer_id=f"storm_pub_{pid}")
        modes = [PublishMode.FIRE_AND_FORGET, PublishMode.BEST_EFFORT]
        for j in range(messages_per_pub):
            await w.publish({"pub": pid, "seq": j}, mode=random.choice(modes))
            await _chaos_sleep(0, 15, 40)

    await asyncio.gather(*[pub(p) for p in range(num_publishers)])
    await asyncio.sleep(0.7)

    for i in range(num_subscribers):
        assert (
            len(received[i]) == total_messages
        ), f"storm sub {i}: expected {total_messages}, got {len(received[i])}"
    for r in readers:
        await r.stop()


@pytest.mark.asyncio
async def test_chaos_storm_multi_queue_multi_topic(actor_system, temp_storage_path):
    """Chaos storm: multiple queues + multiple topics running simultaneously, each with random load, verify no deadlock and data consistency."""
    random.seed(102)
    q_topics = ["chaos_storm_q1", "chaos_storm_q2"]
    t_topics = ["chaos_storm_t1", "chaos_storm_t2"]

    async def run_queue(qtopic: str):
        nb = random.randint(2, 5)
        n_msg = random.randint(20, 45)
        w = await write_queue(
            actor_system,
            topic=qtopic,
            bucket_column="id",
            num_buckets=nb,
            storage_path=temp_storage_path,
        )
        for i in range(n_msg):
            await w.put({"id": f"{qtopic}_{i}", "i": i})
            await _chaos_sleep(0, 12, 35)
        await w.flush()
        r = await read_queue(
            actor_system,
            topic=qtopic,
            num_buckets=nb,
            storage_path=temp_storage_path,
        )
        recs = await r.get(limit=n_msg + 20)
        assert len(recs) == n_msg
        return len(recs)

    async def run_topic(ttopic: str):
        n_msg = random.randint(15, 35)
        recv = []
        writer = await write_topic(actor_system, ttopic)
        reader = await read_topic(actor_system, ttopic, reader_id=f"storm_{ttopic}")
        reader.add_callback(lambda m: recv.append(m))
        await reader.start()
        for i in range(n_msg):
            await writer.publish(
                {"i": i},
                mode=random.choice(
                    [PublishMode.FIRE_AND_FORGET, PublishMode.BEST_EFFORT]
                ),
            )
            await _chaos_sleep(0, 10, 30)
        await asyncio.sleep(0.2)
        assert len(recv) == n_msg
        await reader.stop()
        return len(recv)

    results = await asyncio.gather(
        run_queue(q_topics[0]),
        run_queue(q_topics[1]),
        run_topic(t_topics[0]),
        run_topic(t_topics[1]),
    )
    assert results[0] == results[0]  # sanity
    assert len(results) == 4
