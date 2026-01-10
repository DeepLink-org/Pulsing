"""
Tests for the Pulsing Topic Pub/Sub Module.

Covers:
- Basic topic operations (publish, subscribe)
- TopicWriter/TopicReader API
- Different publish modes (fire_and_forget, wait_all_acks, etc.)
- Multiple subscribers
- Concurrent publishers and subscribers
- StorageManager integration (topic broker routing)
"""

import asyncio
import time

import pytest

from pulsing.actor import SystemConfig, create_actor_system
from pulsing.topic import (
    PublishMode,
    PublishResult,
    TopicReader,
    TopicWriter,
    read_topic,
    write_topic,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def actor_system():
    """Create a standalone actor system for testing."""
    config = SystemConfig.standalone()
    system = await create_actor_system(config)
    yield system
    await system.shutdown()


# ============================================================================
# Basic Topic Tests
# ============================================================================


@pytest.mark.asyncio
async def test_write_topic_creation(actor_system):
    """Test creating a topic writer."""
    writer = await write_topic(actor_system, "test_topic")

    assert isinstance(writer, TopicWriter)
    assert writer.topic == "test_topic"
    assert writer.writer_id.startswith("writer_")


@pytest.mark.asyncio
async def test_write_topic_custom_id(actor_system):
    """Test creating a topic writer with custom ID."""
    writer = await write_topic(actor_system, "test_topic", writer_id="my_writer")

    assert writer.writer_id == "my_writer"


@pytest.mark.asyncio
async def test_read_topic_creation(actor_system):
    """Test creating a topic reader."""
    reader = await read_topic(actor_system, "test_topic")

    assert isinstance(reader, TopicReader)
    assert reader.topic == "test_topic"
    assert reader.reader_id.startswith("reader_")
    assert not reader.is_started


@pytest.mark.asyncio
async def test_read_topic_custom_id(actor_system):
    """Test creating a topic reader with custom ID."""
    reader = await read_topic(actor_system, "test_topic", reader_id="my_reader")

    assert reader.reader_id == "my_reader"


# ============================================================================
# Publish Tests (No Subscribers)
# ============================================================================


@pytest.mark.asyncio
async def test_publish_no_subscribers(actor_system):
    """Test publishing to a topic with no subscribers."""
    writer = await write_topic(actor_system, "empty_topic")

    result = await writer.publish({"message": "hello"})

    assert isinstance(result, PublishResult)
    assert result.success is True
    assert result.delivered == 0
    assert result.failed == 0
    assert result.subscriber_count == 0


@pytest.mark.asyncio
async def test_publish_various_data_types(actor_system):
    """Test publishing different data types."""
    writer = await write_topic(actor_system, "data_types_topic")

    # Dict
    result = await writer.publish({"key": "value"})
    assert result.success is True

    # List
    result = await writer.publish([1, 2, 3])
    assert result.success is True

    # String
    result = await writer.publish("simple string")
    assert result.success is True

    # Number
    result = await writer.publish(42)
    assert result.success is True

    # Nested
    result = await writer.publish(
        {
            "nested": {"data": [1, 2, {"deep": True}]},
            "list": [{"a": 1}, {"b": 2}],
        }
    )
    assert result.success is True


# ============================================================================
# Subscribe and Receive Tests
# ============================================================================


@pytest.mark.asyncio
async def test_subscribe_and_receive(actor_system):
    """Test basic subscribe and receive."""
    writer = await write_topic(actor_system, "sub_test")
    reader = await read_topic(actor_system, "sub_test")

    received = []

    @reader.on_message
    async def handler(msg):
        received.append(msg)

    await reader.start()
    assert reader.is_started

    # Publish message
    result = await writer.publish({"type": "test", "value": 123})

    # Wait for delivery
    await asyncio.sleep(0.1)

    assert result.success is True
    assert result.delivered == 1
    assert len(received) == 1
    assert received[0]["type"] == "test"
    assert received[0]["value"] == 123

    await reader.stop()
    assert not reader.is_started


@pytest.mark.asyncio
async def test_multiple_messages(actor_system):
    """Test receiving multiple messages."""
    writer = await write_topic(actor_system, "multi_msg_topic")
    reader = await read_topic(actor_system, "multi_msg_topic")

    received = []

    @reader.on_message
    async def handler(msg):
        received.append(msg)

    await reader.start()

    # Publish multiple messages
    for i in range(10):
        await writer.publish({"seq": i})

    # Wait for delivery
    await asyncio.sleep(0.2)

    assert len(received) == 10
    for i, msg in enumerate(received):
        assert msg["seq"] == i

    await reader.stop()


@pytest.mark.asyncio
async def test_sync_callback(actor_system):
    """Test synchronous callback function."""
    writer = await write_topic(actor_system, "sync_cb_topic")
    reader = await read_topic(actor_system, "sync_cb_topic")

    received = []

    # Sync callback (not async)
    def handler(msg):
        received.append(msg)

    reader.add_callback(handler)
    await reader.start()

    await writer.publish({"data": "sync test"})
    await asyncio.sleep(0.1)

    assert len(received) == 1
    assert received[0]["data"] == "sync test"

    await reader.stop()


@pytest.mark.asyncio
async def test_multiple_callbacks(actor_system):
    """Test multiple callbacks on one reader."""
    writer = await write_topic(actor_system, "multi_cb_topic")
    reader = await read_topic(actor_system, "multi_cb_topic")

    results1 = []
    results2 = []

    @reader.on_message
    async def handler1(msg):
        results1.append(msg)

    @reader.on_message
    async def handler2(msg):
        results2.append(msg)

    await reader.start()

    await writer.publish({"value": 42})
    await asyncio.sleep(0.1)

    assert len(results1) == 1
    assert len(results2) == 1
    assert results1[0]["value"] == 42
    assert results2[0]["value"] == 42

    await reader.stop()


@pytest.mark.asyncio
async def test_remove_callback(actor_system):
    """Test removing a callback."""
    reader = await read_topic(actor_system, "remove_cb_topic")

    results = []

    def handler(msg):
        results.append(msg)

    reader.add_callback(handler)
    assert reader.remove_callback(handler) is True
    assert reader.remove_callback(handler) is False  # Already removed


# ============================================================================
# Multiple Subscribers Tests
# ============================================================================


@pytest.mark.asyncio
async def test_multiple_subscribers(actor_system):
    """Test publishing to multiple subscribers."""
    writer = await write_topic(actor_system, "multi_sub_topic")

    reader1 = await read_topic(actor_system, "multi_sub_topic", reader_id="r1")
    reader2 = await read_topic(actor_system, "multi_sub_topic", reader_id="r2")
    reader3 = await read_topic(actor_system, "multi_sub_topic", reader_id="r3")

    received1, received2, received3 = [], [], []

    reader1.add_callback(lambda m: received1.append(m))
    reader2.add_callback(lambda m: received2.append(m))
    reader3.add_callback(lambda m: received3.append(m))

    await reader1.start()
    await reader2.start()
    await reader3.start()

    result = await writer.publish({"broadcast": True})

    await asyncio.sleep(0.2)

    assert result.subscriber_count == 3
    assert result.delivered == 3

    assert len(received1) == 1
    assert len(received2) == 1
    assert len(received3) == 1

    await reader1.stop()
    await reader2.stop()
    await reader3.stop()


@pytest.mark.asyncio
async def test_subscriber_join_leave(actor_system):
    """Test subscribers joining and leaving."""
    writer = await write_topic(actor_system, "join_leave_topic")

    reader1 = await read_topic(actor_system, "join_leave_topic", reader_id="r1")
    received1 = []
    reader1.add_callback(lambda m: received1.append(m))
    await reader1.start()

    # Publish with 1 subscriber
    result1 = await writer.publish({"phase": 1})
    await asyncio.sleep(0.1)
    assert result1.subscriber_count == 1
    assert len(received1) == 1

    # Add second subscriber
    reader2 = await read_topic(actor_system, "join_leave_topic", reader_id="r2")
    received2 = []
    reader2.add_callback(lambda m: received2.append(m))
    await reader2.start()

    # Publish with 2 subscribers
    result2 = await writer.publish({"phase": 2})
    await asyncio.sleep(0.1)
    assert result2.subscriber_count == 2
    assert len(received1) == 2
    assert len(received2) == 1

    # First subscriber leaves
    await reader1.stop()

    # Publish with 1 subscriber
    result3 = await writer.publish({"phase": 3})
    await asyncio.sleep(0.1)
    assert result3.subscriber_count == 1
    assert len(received1) == 2  # No new message
    assert len(received2) == 2  # Got new message

    await reader2.stop()


# ============================================================================
# Publish Mode Tests
# ============================================================================


@pytest.mark.asyncio
async def test_publish_fire_and_forget(actor_system):
    """Test fire-and-forget publish mode."""
    writer = await write_topic(actor_system, "ff_topic")
    reader = await read_topic(actor_system, "ff_topic")

    received = []
    reader.add_callback(lambda m: received.append(m))
    await reader.start()

    result = await writer.publish(
        {"mode": "fire_and_forget"},
        mode=PublishMode.FIRE_AND_FORGET,
    )

    await asyncio.sleep(0.1)

    assert result.success is True
    assert len(received) == 1

    await reader.stop()


@pytest.mark.asyncio
async def test_publish_wait_all_acks(actor_system):
    """Test wait-all-acks publish mode."""
    writer = await write_topic(actor_system, "wait_all_topic")

    reader1 = await read_topic(actor_system, "wait_all_topic", reader_id="r1")
    reader2 = await read_topic(actor_system, "wait_all_topic", reader_id="r2")

    received1, received2 = [], []

    async def slow_handler(msg, results):
        await asyncio.sleep(0.05)
        results.append(msg)

    reader1.add_callback(lambda m: asyncio.create_task(slow_handler(m, received1)))
    reader2.add_callback(lambda m: asyncio.create_task(slow_handler(m, received2)))

    await reader1.start()
    await reader2.start()

    start = time.time()
    result = await writer.publish(
        {"mode": "wait_all"},
        mode=PublishMode.WAIT_ALL_ACKS,
    )
    _elapsed = time.time() - start

    # Should wait for responses
    assert result.success is True
    assert result.delivered == 2

    await asyncio.sleep(0.1)
    assert len(received1) == 1
    assert len(received2) == 1

    await reader1.stop()
    await reader2.stop()


@pytest.mark.asyncio
async def test_publish_best_effort(actor_system):
    """Test best-effort publish mode."""
    writer = await write_topic(actor_system, "best_effort_topic")
    reader = await read_topic(actor_system, "best_effort_topic")

    received = []
    reader.add_callback(lambda m: received.append(m))
    await reader.start()

    result = await writer.publish(
        {"mode": "best_effort"},
        mode=PublishMode.BEST_EFFORT,
    )

    assert result.success is True
    assert result.delivered == 1

    await asyncio.sleep(0.1)
    assert len(received) == 1

    await reader.stop()


# ============================================================================
# Stats Tests
# ============================================================================


@pytest.mark.asyncio
async def test_topic_stats(actor_system):
    """Test topic statistics."""
    writer = await write_topic(actor_system, "stats_topic")

    reader1 = await read_topic(actor_system, "stats_topic", reader_id="r1")
    reader2 = await read_topic(actor_system, "stats_topic", reader_id="r2")

    reader1.add_callback(lambda m: None)
    reader2.add_callback(lambda m: None)

    await reader1.start()
    await reader2.start()

    # Publish some messages
    for i in range(5):
        await writer.publish({"seq": i})

    await asyncio.sleep(0.1)

    stats = await writer.stats()

    assert stats["topic"] == "stats_topic"
    assert stats["subscriber_count"] == 2
    assert stats["total_published"] == 5
    assert stats["total_delivered"] >= 10  # 5 messages * 2 subscribers

    await reader1.stop()
    await reader2.stop()


# ============================================================================
# Concurrent Tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_publishers(actor_system):
    """Test multiple concurrent publishers."""
    topic = "concurrent_pub_topic"

    reader = await read_topic(actor_system, topic)
    received = []
    lock = asyncio.Lock()

    async def handler(msg):
        async with lock:
            received.append(msg)

    reader.add_callback(handler)
    await reader.start()

    num_publishers = 5
    messages_per_publisher = 20

    async def publish_batch(pub_id: int):
        writer = await write_topic(actor_system, topic, writer_id=f"pub_{pub_id}")
        for i in range(messages_per_publisher):
            await writer.publish({"pub_id": pub_id, "seq": i})

    # Run publishers concurrently
    tasks = [publish_batch(i) for i in range(num_publishers)]
    await asyncio.gather(*tasks)

    # Wait for delivery
    await asyncio.sleep(0.5)

    expected = num_publishers * messages_per_publisher
    assert len(received) == expected, f"Expected {expected}, got {len(received)}"

    await reader.stop()


@pytest.mark.asyncio
async def test_concurrent_subscribers(actor_system):
    """Test multiple concurrent subscribers."""
    topic = "concurrent_sub_topic"
    writer = await write_topic(actor_system, topic)

    num_subscribers = 10
    readers = []
    results = {i: [] for i in range(num_subscribers)}

    for i in range(num_subscribers):
        reader = await read_topic(actor_system, topic, reader_id=f"sub_{i}")
        reader.add_callback(lambda m, idx=i: results[idx].append(m))
        await reader.start()
        readers.append(reader)

    # Publish messages
    num_messages = 20
    for seq in range(num_messages):
        await writer.publish({"seq": seq})

    # Wait for delivery
    await asyncio.sleep(0.5)

    # All subscribers should receive all messages
    for i in range(num_subscribers):
        assert (
            len(results[i]) == num_messages
        ), f"Subscriber {i} got {len(results[i])} messages, expected {num_messages}"

    for reader in readers:
        await reader.stop()


@pytest.mark.asyncio
async def test_high_throughput(actor_system):
    """Stress test: high message throughput."""
    topic = "throughput_topic"
    writer = await write_topic(actor_system, topic)
    reader = await read_topic(actor_system, topic)

    received = []
    lock = asyncio.Lock()

    async def handler(msg):
        async with lock:
            received.append(msg)

    reader.add_callback(handler)
    await reader.start()

    num_messages = 1000

    start = time.time()
    for i in range(num_messages):
        await writer.publish({"seq": i})
    publish_elapsed = time.time() - start

    # Wait for all messages to be delivered
    max_wait = 5.0
    wait_start = time.time()
    while len(received) < num_messages and time.time() - wait_start < max_wait:
        await asyncio.sleep(0.1)

    total_elapsed = time.time() - start

    print("\nHigh throughput test:")
    print(f"  Published: {num_messages} messages in {publish_elapsed:.2f}s")
    print(f"  Throughput: {num_messages / publish_elapsed:.0f} msg/s")
    print(f"  Received: {len(received)} messages in {total_elapsed:.2f}s")

    assert len(received) == num_messages

    await reader.stop()


@pytest.mark.asyncio
async def test_producer_consumer_stress(actor_system):
    """Stress test: concurrent producers and consumers."""
    topic = "stress_topic"

    num_producers = 3
    num_consumers = 3
    messages_per_producer = 50

    all_received = {i: [] for i in range(num_consumers)}
    locks = {i: asyncio.Lock() for i in range(num_consumers)}
    produce_done = asyncio.Event()

    # Start consumers
    readers = []
    for i in range(num_consumers):
        reader = await read_topic(actor_system, topic, reader_id=f"consumer_{i}")

        async def make_handler(idx):
            async def handler(msg):
                async with locks[idx]:
                    all_received[idx].append(msg)

            return handler

        reader.add_callback(await make_handler(i))
        await reader.start()
        readers.append(reader)

    # Producer task
    async def producer(prod_id: int):
        writer = await write_topic(actor_system, topic, writer_id=f"producer_{prod_id}")
        for seq in range(messages_per_producer):
            await writer.publish({"producer": prod_id, "seq": seq})
            await asyncio.sleep(0.001)

    # Run producers
    producer_tasks = [producer(i) for i in range(num_producers)]
    await asyncio.gather(*producer_tasks)
    produce_done.set()

    # Wait for consumers
    await asyncio.sleep(1.0)

    expected_per_consumer = num_producers * messages_per_producer

    print("\nStress test results:")
    for i in range(num_consumers):
        print(f"  Consumer {i}: {len(all_received[i])} messages")
        assert len(all_received[i]) == expected_per_consumer

    for reader in readers:
        await reader.stop()


# ============================================================================
# Auto-start Tests
# ============================================================================


@pytest.mark.asyncio
async def test_read_topic_auto_start(actor_system):
    """Test auto_start parameter."""
    _writer = await write_topic(actor_system, "auto_start_topic")

    received = []

    # No callbacks before start - should warn but work
    reader = await read_topic(actor_system, "auto_start_topic", auto_start=True)
    reader.add_callback(lambda m: received.append(m))

    # Since auto_start=True but no callbacks at creation time,
    # the reader started without callbacks
    assert reader.is_started

    await reader.stop()


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_callback_error_handling(actor_system):
    """Test that callback errors don't break the system."""
    writer = await write_topic(actor_system, "error_topic")
    reader = await read_topic(actor_system, "error_topic")

    good_results = []

    def bad_callback(msg):
        raise ValueError("Intentional error")

    def good_callback(msg):
        good_results.append(msg)

    reader.add_callback(bad_callback)
    reader.add_callback(good_callback)

    await reader.start()

    # Should not raise
    await writer.publish({"test": "error handling"})
    await asyncio.sleep(0.1)

    # Good callback should still receive
    assert len(good_results) == 1

    await reader.stop()


@pytest.mark.asyncio
async def test_double_start_stop(actor_system):
    """Test double start/stop is safe."""
    reader = await read_topic(actor_system, "double_topic")
    reader.add_callback(lambda m: None)

    # Double start
    await reader.start()
    await reader.start()  # Should be no-op
    assert reader.is_started

    # Double stop
    await reader.stop()
    await reader.stop()  # Should be no-op
    assert not reader.is_started


# ============================================================================
# StorageManager Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_topic_broker_via_storage_manager(actor_system):
    """Test that topic broker is created via StorageManager."""
    from pulsing.queue.manager import get_storage_manager
    from pulsing.actor import Message

    # Ensure StorageManager exists
    manager = await get_storage_manager(actor_system)

    # Create topic via write_topic
    writer = await write_topic(actor_system, "sm_integration_topic")
    await writer.publish({"test": True})

    # Check stats include topics
    response = await manager.ask(Message.from_json("GetStats", {}))
    stats = response.to_json()

    assert "topic_count" in stats
    assert stats["topic_count"] >= 1
    assert "sm_integration_topic" in stats["topics"]


@pytest.mark.asyncio
async def test_list_topics(actor_system):
    """Test listing topics via StorageManager."""
    from pulsing.queue.manager import get_storage_manager
    from pulsing.actor import Message

    # Create some topics
    await write_topic(actor_system, "list_topic_1")
    await write_topic(actor_system, "list_topic_2")

    # Publish to ensure brokers are created
    w1 = await write_topic(actor_system, "list_topic_1")
    w2 = await write_topic(actor_system, "list_topic_2")
    await w1.publish({"test": 1})
    await w2.publish({"test": 2})

    manager = await get_storage_manager(actor_system)
    response = await manager.ask(Message.from_json("ListTopics", {}))
    data = response.to_json()

    assert "topics" in data
    assert "list_topic_1" in data["topics"]
    assert "list_topic_2" in data["topics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
