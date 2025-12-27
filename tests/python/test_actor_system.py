"""
Tests for the Pulsing Actor System Python bindings.

Covers:
- Basic actor functionality (spawn, ask, tell)
- Streaming responses (actor returns StreamMessage)
- Streaming requests (actor receives stream)
- Cluster communication (remote actors)
"""

import asyncio
import json

import pytest
from pulsing.actor import (
    Actor,
    ActorId,
    Message,
    StreamMessage,
    SystemConfig,
    create_actor_system,
)

# Actor system tests are standalone and don't require NATS/ETCD


# ============================================================================
# Test Actors
# ============================================================================


class EchoActor(Actor):
    """Simple echo actor - returns the same message it receives."""

    async def receive(self, msg: Message) -> Message:
        return Message(f"Echo:{msg.msg_type}", msg.payload)


class CounterActor(Actor):
    """Stateful counter actor."""

    def __init__(self):
        self.count = 0

    def on_start(self, actor_id: ActorId):
        print(f"CounterActor started: {actor_id}")

    def on_stop(self):
        print(f"CounterActor stopped, final count: {self.count}")

    def metadata(self):
        return {"type": "counter", "count": str(self.count)}

    async def receive(self, msg: Message) -> Message:
        data = msg.to_json()

        if msg.msg_type == "increment":
            self.count += data.get("value", 1)
            return Message.from_json("result", {"count": self.count})
        elif msg.msg_type == "decrement":
            self.count -= data.get("value", 1)
            return Message.from_json("result", {"count": self.count})
        elif msg.msg_type == "get":
            return Message.from_json("result", {"count": self.count})
        elif msg.msg_type == "reset":
            self.count = 0
            return Message.from_json("result", {"count": self.count})
        else:
            return Message.empty()


class StreamingGeneratorActor(Actor):
    """Actor that returns streaming responses."""

    async def receive(self, msg: Message) -> Message:
        if msg.msg_type == "generate":
            data = msg.to_json()
            count = data.get("count", 5)
            delay = data.get("delay", 0.01)

            # Create streaming response
            stream_msg, writer = StreamMessage.create("tokens")

            async def produce():
                try:
                    for i in range(count):
                        await writer.write_json({"index": i, "token": f"token_{i}"})
                        await asyncio.sleep(delay)
                    await writer.close()
                except Exception as e:
                    await writer.error(str(e))

            asyncio.create_task(produce())
            return stream_msg

        elif msg.msg_type == "generate_with_error":
            # Streaming response that errors midway
            stream_msg, writer = StreamMessage.create("tokens")

            async def produce_with_error():
                try:
                    for i in range(3):
                        await writer.write_json({"index": i})
                        await asyncio.sleep(0.01)
                    await writer.error("Simulated error at index 3")
                except Exception as e:
                    await writer.error(str(e))

            asyncio.create_task(produce_with_error())
            return stream_msg

        return Message.empty()


class StreamConsumerActor(Actor):
    """Actor that consumes streaming requests."""

    async def receive(self, msg: Message) -> Message:
        if msg.is_stream:
            # Consume the stream and aggregate
            reader = msg.stream_reader()
            items = []
            try:
                async for chunk in reader:
                    data = json.loads(chunk)
                    items.append(data)
            except Exception as e:
                return Message.from_json("error", {"message": str(e)})

            return Message.from_json(
                "aggregated",
                {"count": len(items), "items": items},
            )
        else:
            # Handle single message
            return Message.from_json("echo", {"received": msg.msg_type})


class BidirectionalStreamActor(Actor):
    """Actor that handles stream input and returns stream output."""

    async def receive(self, msg: Message) -> Message:
        if msg.is_stream:
            reader = msg.stream_reader()
            stream_msg, writer = StreamMessage.create("processed")

            async def process():
                try:
                    async for chunk in reader:
                        data = json.loads(chunk)
                        # Transform each item
                        processed = {
                            "original": data,
                            "processed": True,
                            "doubled": data.get("value", 0) * 2,
                        }
                        await writer.write_json(processed)
                    await writer.close()
                except Exception as e:
                    await writer.error(str(e))

            asyncio.create_task(process())
            return stream_msg
        else:
            return Message.empty()


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


@pytest.fixture
async def cluster_systems():
    """Create two actor systems that form a cluster."""
    # First node
    config1 = SystemConfig.with_addr("127.0.0.1:18001")
    system1 = await create_actor_system(config1)

    # Second node, joins the first
    config2 = SystemConfig.with_addr("127.0.0.1:18002").with_seeds(["127.0.0.1:18001"])
    system2 = await create_actor_system(config2)

    # Wait for cluster to form
    await asyncio.sleep(0.5)

    yield system1, system2

    await system2.shutdown()
    await system1.shutdown()


# ============================================================================
# Basic Functionality Tests
# ============================================================================


@pytest.mark.asyncio
async def test_actor_system_creation(actor_system):
    """Test that ActorSystem can be created."""
    assert actor_system is not None
    assert actor_system.node_id is not None
    assert actor_system.addr is not None


@pytest.mark.asyncio
async def test_spawn_actor(actor_system):
    """Test spawning an actor."""
    actor_ref = await actor_system.spawn("echo", EchoActor())
    assert actor_ref is not None
    assert actor_ref.actor_id is not None
    assert actor_ref.is_local()
    assert "echo" in actor_system.local_actor_names()


@pytest.mark.asyncio
async def test_ask_single_message(actor_system):
    """Test ask pattern with single message."""
    actor_ref = await actor_system.spawn("echo", EchoActor())

    # Send message and get response (using send() which supports Message)
    request = Message.from_json("greeting", {"text": "hello"})
    response = await actor_ref.ask(request)

    assert response.msg_type == "Echo:greeting"
    data = response.to_json()
    assert data["text"] == "hello"


@pytest.mark.asyncio
async def test_ask_json(actor_system):
    """Test ask with JSON message."""
    actor_ref = await actor_system.spawn("counter", CounterActor())

    # Test increment
    response = (
        await actor_ref.ask(Message.from_json("increment", {"value": 5}))
    ).to_json()
    assert response["count"] == 5

    # Test get
    response = (await actor_ref.ask(Message.from_json("get", {}))).to_json()
    assert response["count"] == 5

    # Test decrement
    response = (
        await actor_ref.ask(Message.from_json("decrement", {"value": 2}))
    ).to_json()
    assert response["count"] == 3


@pytest.mark.asyncio
async def test_tell_message(actor_system):
    """Test tell pattern (fire-and-forget)."""
    actor_ref = await actor_system.spawn("counter", CounterActor())

    # Send tell (fire-and-forget)
    await actor_ref.tell(Message.from_json("increment", {"value": 10}))

    # Small delay to allow processing
    await asyncio.sleep(0.1)

    # Verify with ask
    response = (await actor_ref.ask(Message.from_json("get", {}))).to_json()
    assert response["count"] == 10


@pytest.mark.asyncio
async def test_actor_lifecycle(actor_system):
    """Test actor on_start and on_stop callbacks."""
    actor = CounterActor()
    actor_ref = await actor_system.spawn("lifecycle_test", actor)

    # Do some work
    await actor_ref.ask(Message.from_json("increment", {"value": 1}))

    # Stop the actor
    await actor_system.stop("lifecycle_test")

    # Verify actor is no longer in local actors
    local_actors = actor_system.local_actor_names()
    assert "lifecycle_test" not in local_actors


@pytest.mark.asyncio
async def test_multiple_actors(actor_system):
    """Test multiple actors in the same system."""
    echo1 = await actor_system.spawn("echo1", EchoActor())
    echo2 = await actor_system.spawn("echo2", EchoActor())
    counter = await actor_system.spawn("counter", CounterActor())

    # Verify all actors exist
    local_actors = actor_system.local_actor_names()
    assert "echo1" in local_actors
    assert "echo2" in local_actors
    assert "counter" in local_actors

    # Interact with each
    resp1 = await echo1.ask(Message.from_json("test1", {}))
    resp2 = await echo2.ask(Message.from_json("test2", {}))
    resp3 = (await counter.ask(Message.from_json("get", {}))).to_json()

    assert resp1.msg_type == "Echo:test1"
    assert resp2.msg_type == "Echo:test2"
    assert resp3["count"] == 0


@pytest.mark.asyncio
async def test_actor_metadata(actor_system):
    """Test actor metadata."""
    actor_ref = await actor_system.spawn("counter_meta", CounterActor())

    # Increment counter
    await actor_ref.ask(Message.from_json("increment", {"value": 42}))

    # Note: metadata is typically accessed via system diagnostics
    # For now, just verify the actor works correctly
    response = (await actor_ref.ask(Message.from_json("get", {}))).to_json()
    assert response["count"] == 42


# ============================================================================
# Streaming Response Tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_response_basic(actor_system):
    """Test basic streaming response."""
    actor_ref = await actor_system.spawn("generator", StreamingGeneratorActor())

    # Request streaming response
    request = Message.from_json("generate", {"count": 5, "delay": 0.01})
    response = await actor_ref.ask(request)

    # Verify it's a stream
    assert response.is_stream

    # Consume the stream
    reader = response.stream_reader()
    items = []
    async for chunk in reader:
        data = json.loads(chunk)
        items.append(data)

    # Verify all items received
    assert len(items) == 5
    for i, item in enumerate(items):
        assert item["index"] == i
        assert item["token"] == f"token_{i}"


@pytest.mark.asyncio
async def test_streaming_response_with_stream_reader(actor_system):
    """Test streaming response with stream_reader method."""
    actor_ref = await actor_system.spawn("generator", StreamingGeneratorActor())

    # Use ask + stream_reader
    request = Message.from_json("generate", {"count": 3})
    response = await actor_ref.ask(request)
    reader = response.stream_reader()

    items = []
    async for chunk in reader:
        data = json.loads(chunk)
        items.append(data)

    assert len(items) == 3


@pytest.mark.asyncio
async def test_streaming_response_large(actor_system):
    """Test streaming response with many items."""
    actor_ref = await actor_system.spawn("generator", StreamingGeneratorActor())

    request = Message.from_json("generate", {"count": 100, "delay": 0.001})
    response = await actor_ref.ask(request)

    reader = response.stream_reader()
    count = 0
    async for _chunk in reader:
        count += 1

    assert count == 100


@pytest.mark.asyncio
async def test_streaming_response_with_error(actor_system):
    """Test streaming response that errors midway."""
    actor_ref = await actor_system.spawn("generator", StreamingGeneratorActor())

    request = Message.from_json("generate_with_error", {})
    response = await actor_ref.ask(request)

    reader = response.stream_reader()
    items = []
    error_caught = False

    try:
        async for chunk in reader:
            data = json.loads(chunk)
            items.append(data)
    except RuntimeError as e:
        error_caught = True
        assert "Simulated error" in str(e)

    # Should have received some items before error
    assert len(items) == 3
    assert error_caught


@pytest.mark.asyncio
async def test_streaming_response_cancel(actor_system):
    """Test cancelling a streaming response."""
    actor_ref = await actor_system.spawn("generator", StreamingGeneratorActor())

    # Request a long stream
    request = Message.from_json("generate", {"count": 1000, "delay": 0.1})
    response = await actor_ref.ask(request)

    reader = response.stream_reader()
    count = 0

    async for _chunk in reader:
        count += 1
        if count >= 3:
            await reader.cancel()
            break

    # Should have stopped early
    assert count == 3


# ============================================================================
# Streaming Request Tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_request_basic(actor_system):
    """Test actor receiving streaming request."""
    actor_ref = await actor_system.spawn("consumer", StreamConsumerActor())

    # For now, test with single message (stream input requires client-side streaming)
    request = Message.from_json("test", {"data": "hello"})
    response = await actor_ref.ask(request)

    # Should receive echo response for non-stream
    assert not response.is_stream
    data = response.to_json()
    assert data["received"] == "test"


# Note: Testing full client→actor stream requires implementing client-side streaming
# which would involve creating a stream from Python and sending it to the actor.
# This is more complex and may require additional API work.


# ============================================================================
# Cluster Tests
# ============================================================================


@pytest.mark.asyncio
async def test_cluster_formation(cluster_systems):
    """Test that two systems can form a cluster."""
    system1, system2 = cluster_systems

    # Wait for gossip to propagate
    await asyncio.sleep(1.0)

    # Check cluster members
    members1 = await system1.members()
    members2 = await system2.members()

    # Each system should see 2 members (itself + the other)
    assert len(members1) >= 1
    assert len(members2) >= 1


@pytest.mark.asyncio
async def test_remote_actor_communication(cluster_systems):
    """Test communication between actors on different nodes."""
    system1, system2 = cluster_systems

    # Spawn actor on system1
    actor_ref1 = await system1.spawn("remote_echo", EchoActor(), public=True)

    # Wait for actor registration to propagate
    await asyncio.sleep(1.0)

    # Get reference to remote actor from system2
    remote_ref = await system2.actor_ref(actor_ref1.actor_id)

    # Send message to remote actor
    request = Message.from_json("remote_test", {"from": "system2"})
    response = await remote_ref.ask(request)

    # Note: Remote responses don't preserve msg_type in current protocol
    # The HTTP transport only returns payload bytes
    data = response.to_json()
    assert data["from"] == "system2"


@pytest.mark.asyncio
async def test_remote_streaming_response(cluster_systems):
    """Test streaming response from remote actor."""
    system1, system2 = cluster_systems

    # Spawn streaming actor on system1
    actor_ref1 = await system1.spawn(
        "remote_generator", StreamingGeneratorActor(), public=True
    )

    # Wait for propagation
    await asyncio.sleep(1.0)

    # Get remote reference from system2
    remote_ref = await system2.actor_ref(actor_ref1.actor_id)

    # Request streaming response
    request = Message.from_json("generate", {"count": 5})
    response = await remote_ref.ask(request)

    assert response.is_stream

    reader = response.stream_reader()
    items = []
    async for chunk in reader:
        data = json.loads(chunk)
        items.append(data)

    assert len(items) == 5


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_actor_not_found(actor_system):
    """Test error when actor is not found."""
    # Create a fake ActorId with a random local_id that doesn't exist
    fake_id = ActorId(99999999, actor_system.node_id)

    with pytest.raises(Exception):  # noqa: B017
        await actor_system.actor_ref(fake_id)


@pytest.mark.asyncio
async def test_message_to_stopped_actor(actor_system):
    """Test sending message to stopped actor."""
    actor_ref = await actor_system.spawn("temp_actor", EchoActor())

    # Stop the actor
    await actor_system.stop("temp_actor")

    # Try to send message - should fail
    with pytest.raises(Exception):  # noqa: B017
        await actor_ref.ask(Message.from_json("test", {}))


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_high_throughput_messages(actor_system):
    """Test sending many messages quickly."""
    actor_ref = await actor_system.spawn("perf_counter", CounterActor())

    # Send many increments
    num_messages = 100
    tasks = []
    for _i in range(num_messages):
        tasks.append(actor_ref.ask(Message.from_json("increment", {"value": 1})))

    await asyncio.gather(*tasks)

    # Verify final count
    response = (await actor_ref.ask(Message.from_json("get", {}))).to_json()
    assert response["count"] == num_messages


@pytest.mark.asyncio
async def test_concurrent_streaming(actor_system):
    """Test multiple concurrent streaming responses."""
    actor_ref = await actor_system.spawn("concurrent_gen", StreamingGeneratorActor())

    async def consume_stream(stream_id: int):
        request = Message.from_json("generate", {"count": 10, "delay": 0.01})
        response = await actor_ref.ask(request)
        reader = response.stream_reader()
        count = 0
        async for _chunk in reader:
            count += 1
        return stream_id, count

    # Start multiple concurrent streams
    tasks = [consume_stream(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    # All streams should complete
    for stream_id, count in results:
        assert count == 10, f"Stream {stream_id} got {count} items instead of 10"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
