"""
Tests for SealedPyMessage - Python object serialization for Python-to-Python actor communication.

Covers:
- SealedPyMessage seal/unseal functionality
- ask/tell with arbitrary Python objects
- receive returning arbitrary Python objects
- Python-to-Python actor communication with isinstance-based dispatch
- Backward compatibility with Message.from_json
"""

import asyncio
from dataclasses import dataclass

import pytest
from pulsing.actor import (
    Actor,
    Message,
    SealedPyMessage,
)
import pulsing as pul


# ============================================================================
# Test Data Classes
# ============================================================================


@dataclass
class IncrementCommand:
    """Command to increment counter."""

    n: int = 1


@dataclass
class GetValueCommand:
    """Command to get current value."""

    pass


@dataclass
class ValueResponse:
    """Response containing value."""

    value: int


@dataclass
class ErrorResponse:
    """Error response."""

    error: str


# ============================================================================
# Test Actors
# ============================================================================


class SealedCounterActor(Actor):
    """Counter actor using Python objects instead of Message.from_json."""

    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    async def receive(self, msg):
        # Use isinstance to dispatch on message type
        if isinstance(msg, IncrementCommand):
            self.value += msg.n
            return ValueResponse(value=self.value)

        if isinstance(msg, GetValueCommand):
            return ValueResponse(value=self.value)

        # Handle dict messages
        if isinstance(msg, dict):
            action = msg.get("action")
            if action == "increment":
                self.value += msg.get("n", 1)
                return {"value": self.value}
            elif action == "get":
                return {"value": self.value}
            elif action == "reset":
                self.value = 0
                return {"value": self.value}

        # Fallback for Message (Rust actor compatibility)
        if isinstance(msg, Message):
            data = msg.to_json()
            if msg.msg_type == "increment":
                self.value += data.get("n", 1)
                return Message.from_json("result", {"value": self.value})
            elif msg.msg_type == "get":
                return Message.from_json("result", {"value": self.value})

        return ErrorResponse(error=f"Unknown message type: {type(msg)}")


class EchoAnyActor(Actor):
    """Actor that echoes back any Python object."""

    async def receive(self, msg):
        # Echo back the message wrapped in a response dict
        return {"echoed": msg, "type": type(msg).__name__}


class ListProcessorActor(Actor):
    """Actor that processes lists of items."""

    async def receive(self, msg):
        if isinstance(msg, list):
            # Process each item and return results
            return [
                item * 2 if isinstance(item, (int, float)) else str(item)
                for item in msg
            ]

        if isinstance(msg, dict) and msg.get("action") == "sum":
            items = msg.get("items", [])
            return {"sum": sum(items)}

        return None


class ComplexObjectActor(Actor):
    """Actor that handles complex nested objects."""

    async def receive(self, msg):
        if isinstance(msg, dict) and "nested" in msg:
            # Process nested structure
            nested = msg["nested"]
            result = {
                "processed": True,
                "original_keys": list(msg.keys()),
                "nested_type": type(nested).__name__,
            }
            if isinstance(nested, dict):
                result["nested_keys"] = list(nested.keys())
            return result

        return {"received": msg}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def actor_system():
    """Create a standalone actor system for testing."""
    system = await pul.actor_system()
    yield system
    await system.shutdown()


# ============================================================================
# SealedPyMessage Unit Tests
# ============================================================================


def test_sealed_message_seal_unseal_dict():
    """Test sealing and unsealing a dict."""
    original = {"key": "value", "number": 42, "nested": {"a": 1}}
    sealed = SealedPyMessage.seal(original)

    assert sealed is not None
    assert len(sealed.data) > 0

    unsealed = sealed.unseal()
    assert unsealed == original


def test_sealed_message_seal_unseal_dataclass():
    """Test sealing and unsealing a dataclass."""
    original = IncrementCommand(n=10)
    sealed = SealedPyMessage.seal(original)

    unsealed = sealed.unseal()
    assert unsealed == original
    assert isinstance(unsealed, IncrementCommand)
    assert unsealed.n == 10


def test_sealed_message_seal_unseal_list():
    """Test sealing and unsealing a list."""
    original = [1, 2, 3, "hello", {"key": "value"}]
    sealed = SealedPyMessage.seal(original)

    unsealed = sealed.unseal()
    assert unsealed == original


def test_sealed_message_seal_unseal_tuple():
    """Test sealing and unsealing a tuple."""
    original = (1, "two", 3.0)
    sealed = SealedPyMessage.seal(original)

    unsealed = sealed.unseal()
    assert unsealed == original


def test_sealed_message_seal_unseal_set():
    """Test sealing and unsealing a set."""
    original = {1, 2, 3, 4, 5}
    sealed = SealedPyMessage.seal(original)

    unsealed = sealed.unseal()
    assert unsealed == original


def test_sealed_message_seal_unseal_none():
    """Test sealing and unsealing None."""
    sealed = SealedPyMessage.seal(None)
    unsealed = sealed.unseal()
    assert unsealed is None


def test_sealed_message_repr():
    """Test SealedPyMessage repr."""
    sealed = SealedPyMessage.seal({"test": "data"})
    repr_str = repr(sealed)
    assert "SealedPyMessage" in repr_str
    assert "data_len=" in repr_str


# ============================================================================
# Actor Communication Tests - Dataclass Messages
# ============================================================================


@pytest.mark.asyncio
async def test_ask_with_dataclass(actor_system):
    """Test ask with dataclass message."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=0), name="counter"
    )

    # Send IncrementCommand dataclass
    response = await actor_ref.ask(IncrementCommand(n=5))

    assert isinstance(response, ValueResponse)
    assert response.value == 5


@pytest.mark.asyncio
async def test_ask_multiple_dataclass_messages(actor_system):
    """Test multiple ask calls with dataclass messages."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=10), name="counter"
    )

    # Multiple increments
    r1 = await actor_ref.ask(IncrementCommand(n=5))
    assert r1.value == 15

    r2 = await actor_ref.ask(IncrementCommand(n=3))
    assert r2.value == 18

    # Get value
    r3 = await actor_ref.ask(GetValueCommand())
    assert r3.value == 18


# ============================================================================
# Actor Communication Tests - Dict Messages
# ============================================================================


@pytest.mark.asyncio
async def test_ask_with_dict(actor_system):
    """Test ask with dict message."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=0), name="counter"
    )

    # Send dict message
    response = await actor_ref.ask({"action": "increment", "n": 7})

    assert isinstance(response, dict)
    assert response["value"] == 7


@pytest.mark.asyncio
async def test_ask_dict_multiple_operations(actor_system):
    """Test multiple dict-based operations."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=100), name="counter"
    )

    # Increment
    r1 = await actor_ref.ask({"action": "increment", "n": 50})
    assert r1["value"] == 150

    # Get
    r2 = await actor_ref.ask({"action": "get"})
    assert r2["value"] == 150

    # Reset
    r3 = await actor_ref.ask({"action": "reset"})
    assert r3["value"] == 0


# ============================================================================
# Actor Communication Tests - List Messages
# ============================================================================


@pytest.mark.asyncio
async def test_ask_with_list(actor_system):
    """Test ask with list message."""
    actor_ref = await actor_system.spawn(ListProcessorActor(), name="processor")

    # Send list of numbers
    response = await actor_ref.ask([1, 2, 3, 4, 5])

    assert response == [2, 4, 6, 8, 10]


@pytest.mark.asyncio
async def test_ask_with_mixed_list(actor_system):
    """Test ask with list containing mixed types."""
    actor_ref = await actor_system.spawn(ListProcessorActor(), name="processor")

    response = await actor_ref.ask([1, "hello", 3.5, "world"])

    assert response == [2, "hello", 7.0, "world"]


# ============================================================================
# Actor Communication Tests - Complex Objects
# ============================================================================


@pytest.mark.asyncio
async def test_ask_with_nested_dict(actor_system):
    """Test ask with nested dict structure."""
    actor_ref = await actor_system.spawn(ComplexObjectActor(), name="complex")

    msg = {
        "nested": {"level2": {"level3": "deep_value"}},
        "other": "data",
    }
    response = await actor_ref.ask(msg)

    assert response["processed"] is True
    assert "nested" in response["original_keys"]
    assert response["nested_type"] == "dict"


@pytest.mark.asyncio
async def test_ask_echo_any_object(actor_system):
    """Test echoing various Python objects."""
    actor_ref = await actor_system.spawn(EchoAnyActor(), name="echo")

    # Test with different types
    test_cases = [
        42,
        3.14,
        "hello",
        [1, 2, 3],
        {"key": "value"},
        (1, 2, 3),
        IncrementCommand(n=5),
    ]

    for obj in test_cases:
        response = await actor_ref.ask(obj)
        assert response["echoed"] == obj
        assert response["type"] == type(obj).__name__


# ============================================================================
# tell Tests
# ============================================================================


@pytest.mark.asyncio
async def test_tell_with_dataclass(actor_system):
    """Test tell with dataclass message."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=0), name="counter"
    )

    # Send tell (fire-and-forget)
    await actor_ref.tell(IncrementCommand(n=10))

    # Wait for processing
    await asyncio.sleep(0.1)

    # Verify with ask
    response = await actor_ref.ask(GetValueCommand())
    assert response.value == 10


@pytest.mark.asyncio
async def test_tell_with_dict(actor_system):
    """Test tell with dict message."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=0), name="counter"
    )

    # Send multiple tells
    await actor_ref.tell({"action": "increment", "n": 5})
    await actor_ref.tell({"action": "increment", "n": 3})

    await asyncio.sleep(0.1)

    response = await actor_ref.ask({"action": "get"})
    assert response["value"] == 8


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


@pytest.mark.asyncio
async def test_message_from_json_still_works(actor_system):
    """Test that Message.from_json still works for backward compatibility."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=0), name="counter"
    )

    # Use old Message.from_json style
    response = await actor_ref.ask(Message.from_json("increment", {"n": 5}))

    # Response should be a Message (for backward compatibility path)
    assert isinstance(response, Message)
    data = response.to_json()
    assert data["value"] == 5


@pytest.mark.asyncio
async def test_mixed_message_styles(actor_system):
    """Test mixing old Message style with new Python object style."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=0), name="counter"
    )

    # New style
    r1 = await actor_ref.ask(IncrementCommand(n=10))
    assert r1.value == 10

    # Old style
    r2 = await actor_ref.ask(Message.from_json("increment", {"n": 5}))
    assert r2.to_json()["value"] == 15

    # New style again
    r3 = await actor_ref.ask({"action": "get"})
    assert r3["value"] == 15


# ============================================================================
# Concurrent Access Tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_sealed_messages(actor_system):
    """Test concurrent access with sealed messages."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=0), name="counter"
    )

    # Send many concurrent increments
    tasks = [actor_ref.ask(IncrementCommand(n=1)) for _ in range(50)]
    results = await asyncio.gather(*tasks)

    # All should return ValueResponse
    for r in results:
        assert isinstance(r, ValueResponse)

    # Final value should be 50
    final = await actor_ref.ask(GetValueCommand())
    assert final.value == 50


@pytest.mark.asyncio
async def test_concurrent_dict_messages(actor_system):
    """Test concurrent access with dict messages."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=0), name="counter"
    )

    tasks = [actor_ref.ask({"action": "increment", "n": 1}) for _ in range(30)]
    await asyncio.gather(*tasks)

    response = await actor_ref.ask({"action": "get"})
    assert response["value"] == 30


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_unknown_message_type_returns_error(actor_system):
    """Test that unknown message types return error response."""
    actor_ref = await actor_system.spawn(
        SealedCounterActor(initial_value=0), name="counter"
    )

    # Send an unknown type
    response = await actor_ref.ask("just a string")

    assert isinstance(response, ErrorResponse)
    assert "Unknown message type" in response.error


# ============================================================================
# Custom Class Tests
# ============================================================================


@dataclass
class CustomRequest:
    """Custom request with multiple fields."""

    operation: str
    values: list
    metadata: dict = None


@dataclass
class CustomResponse:
    """Custom response."""

    success: bool
    result: any = None
    error: str = None


class CustomHandlerActor(Actor):
    """Actor that handles custom request/response types."""

    async def receive(self, msg):
        if isinstance(msg, CustomRequest):
            if msg.operation == "sum":
                total = sum(msg.values)
                return CustomResponse(success=True, result=total)
            elif msg.operation == "multiply":
                result = 1
                for v in msg.values:
                    result *= v
                return CustomResponse(success=True, result=result)
            else:
                return CustomResponse(
                    success=False, error=f"Unknown operation: {msg.operation}"
                )

        return CustomResponse(success=False, error="Invalid message type")


@pytest.mark.asyncio
async def test_custom_request_response_types(actor_system):
    """Test custom request/response dataclass types."""
    actor_ref = await actor_system.spawn(CustomHandlerActor(), name="custom")

    # Sum operation
    r1 = await actor_ref.ask(CustomRequest(operation="sum", values=[1, 2, 3, 4, 5]))
    assert r1.success is True
    assert r1.result == 15

    # Multiply operation
    r2 = await actor_ref.ask(
        CustomRequest(
            operation="multiply", values=[2, 3, 4], metadata={"source": "test"}
        )
    )
    assert r2.success is True
    assert r2.result == 24

    # Unknown operation
    r3 = await actor_ref.ask(CustomRequest(operation="unknown", values=[]))
    assert r3.success is False
    assert "Unknown operation" in r3.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
