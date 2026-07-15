import asyncio
import ctypes
import gc

import pytest
import pulsing as pul
from pulsing.core import Actor, TensorMessage
from pulsing.core.remote import _WrappedActor


def test_tensor_message_public_api_and_exports():
    first = bytearray(b"abc")
    second = memoryview(bytearray(b"defg"))
    message = TensorMessage(b"opaque-metadata", [first, second])

    assert pul.TensorMessage is TensorMessage
    assert message.metadata == b"opaque-metadata"
    assert message.version == 1
    assert isinstance(message.buffers, list)
    assert [bytes(buffer) for buffer in message.buffers] == [b"abc", b"defg"]
    assert message.total_bytes == 7


def test_tensor_message_allows_control_only_message():
    message = TensorMessage(b"control", [], version=3)

    assert message.metadata == b"control"
    assert message.buffers == []
    assert message.version == 3
    assert message.total_bytes == 0


def test_tensor_message_rejects_non_contiguous_buffer_without_copying():
    non_contiguous = memoryview(bytearray(b"abcdef"))[::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        TensorMessage(b"meta", [non_contiguous])


class _TensorEcho(Actor):
    async def receive(self, message):
        assert isinstance(message, TensorMessage)
        return message


@pytest.mark.asyncio
async def test_local_actor_tensor_roundtrip_shares_payload_and_keeps_owner_alive():
    system = await pul.actor_system()
    try:
        ref = await system.spawn(_TensorEcho(), name="tensor-local-echo")
        source = bytearray(b"payload")
        request = TensorMessage(b"schema", [source])

        response = await ref.ask(request)
        assert isinstance(response, TensorMessage)
        assert response.metadata == b"schema"
        assert bytes(response.buffers[0]) == b"payload"

        # The local mailbox path retains the original buffer export instead of
        # materializing a payload copy.
        source[0] = ord("P")
        assert bytes(response.buffers[0]) == b"Payload"

        torch = pytest.importorskip("torch")
        source_ptr = ctypes.addressof(ctypes.c_ubyte.from_buffer(source))
        tensor = torch.frombuffer(response.buffers[0], dtype=torch.uint8)
        assert tensor.data_ptr() == source_ptr
        assert bytes(tensor.tolist()) == b"Payload"

        del request
        del source
        del response
        gc.collect()
        # torch.frombuffer retains the read-only exporter, which retains Rust
        # Bytes and ultimately the original Python buffer lease.
        assert bytes(tensor.tolist()) == b"Payload"
    finally:
        await system.shutdown()


@pytest.mark.parametrize("transport", ["raw", "http2"])
@pytest.mark.asyncio
async def test_remote_actor_tensor_roundtrip_uses_selected_transport(
    transport, monkeypatch
):
    monkeypatch.setenv("PULSING_TENSOR_TRANSPORT", transport)
    before = pul.tensor_transport_stats()
    server = await pul.actor_system(addr="127.0.0.1:0")
    client = None
    try:
        actor_name = f"tensor-remote-echo-{transport}"
        await server.spawn(_TensorEcho(), name=actor_name, public=True)
        client = await pul.actor_system(addr="127.0.0.1:0", seeds=[server.addr])
        remote_ref = await client.resolve_named(
            actor_name, node_id=server.node_id.id, timeout=5.0
        )
        assert not remote_ref.is_local()

        source = bytearray(b"remote-payload")
        response = await remote_ref.ask(TensorMessage(b"remote-schema", [source]))

        assert isinstance(response, TensorMessage)
        assert response.metadata == b"remote-schema"
        assert bytes(response.buffers[0]) == b"remote-payload"
        assert not response.buffers[0].readonly

        # Crossing TCP materializes a receive allocation, so it must no longer
        # alias the sender's Python bytearray.
        source[0] = ord("R")
        assert bytes(response.buffers[0]) == b"remote-payload"

        torch = pytest.importorskip("torch")
        tensor = torch.frombuffer(response.buffers[0], dtype=torch.uint8)
        tensor[0] = ord("R")
        del response
        gc.collect()
        assert bytes(tensor.tolist()) == b"Remote-payload"

        after = pul.tensor_transport_stats()
        if transport == "raw":
            assert after["raw_frames_sent"] >= before["raw_frames_sent"] + 2
            assert after["active_copy_model"] == "direct_tcp"
        else:
            assert (
                after["http2_fallback_frames"]
                >= before["http2_fallback_frames"] + 1
            )
            assert after["active_copy_model"] == "packed_http2_compatibility"
    finally:
        if client is not None:
            await client.shutdown()
        await server.shutdown()


class _TensorPutAck(Actor):
    async def receive(self, message):
        assert isinstance(message, TensorMessage)
        return {"ok": True, "received_bytes": message.total_bytes}


@pytest.mark.parametrize("transport", ["raw", "http2"])
@pytest.mark.asyncio
async def test_tensor_request_supports_pickled_single_ack(transport, monkeypatch):
    monkeypatch.setenv("PULSING_TENSOR_TRANSPORT", transport)
    server = await pul.actor_system(addr="127.0.0.1:0")
    client = None
    try:
        actor_name = f"tensor-put-ack-{transport}"
        await server.spawn(_TensorPutAck(), name=actor_name, public=True)
        client = await pul.actor_system(addr="127.0.0.1:0", seeds=[server.addr])
        remote_ref = await client.resolve_named(
            actor_name, node_id=server.node_id.id, timeout=5.0
        )

        response = await remote_ref.ask(TensorMessage(b"put", [b"123", b"4567"]))

        assert response == {"ok": True, "received_bytes": 7}
        expected_model = (
            "direct_tcp" if transport == "raw" else "packed_http2_compatibility"
        )
        assert pul.tensor_transport_stats()["active_copy_model"] == expected_model
    finally:
        if client is not None:
            await client.shutdown()
        await server.shutdown()


@pytest.mark.asyncio
async def test_raw_tensor_tell_receives_protocol_ack(monkeypatch):
    monkeypatch.setenv("PULSING_TENSOR_TRANSPORT", "raw")
    before = pul.tensor_transport_stats()
    server = await pul.actor_system(addr="127.0.0.1:0")
    client = None
    try:
        await server.spawn(_TensorEcho(), name="tensor-tell-ack", public=True)
        client = await pul.actor_system(addr="127.0.0.1:0", seeds=[server.addr])
        remote_ref = await client.resolve_named(
            "tensor-tell-ack", node_id=server.node_id.id, timeout=5.0
        )

        # tell() completes only after the raw peer returns its protocol ACK.
        await remote_ref.tell(TensorMessage(b"tell", [b"payload"]))

        after = pul.tensor_transport_stats()
        assert after["raw_frames_sent"] >= before["raw_frames_sent"] + 2
        assert after["active_copy_model"] == "direct_tcp"
    finally:
        if client is not None:
            await client.shutdown()
        await server.shutdown()


@pytest.mark.asyncio
async def test_raw_tensor_pool_reconnects_only_before_next_request(monkeypatch):
    monkeypatch.setenv("PULSING_TENSOR_TRANSPORT", "raw")
    monkeypatch.setenv("PULSING_TENSOR_MAX_REQUESTS_PER_CONNECTION", "1")
    before = pul.tensor_transport_stats()
    server = await pul.actor_system(addr="127.0.0.1:0")
    client = None
    try:
        await server.spawn(_TensorEcho(), name="tensor-reconnect", public=True)
        client = await pul.actor_system(addr="127.0.0.1:0", seeds=[server.addr])
        remote_ref = await client.resolve_named(
            "tensor-reconnect", node_id=server.node_id.id, timeout=5.0
        )

        first = await remote_ref.ask(TensorMessage(b"one", [b"first"]))
        assert bytes(first.buffers[0]) == b"first"
        # Let the peer FIN become visible before checking out the pooled socket.
        await asyncio.sleep(0.05)
        second = await remote_ref.ask(TensorMessage(b"two", [b"second"]))
        assert bytes(second.buffers[0]) == b"second"

        after = pul.tensor_transport_stats()
        assert (
            after["raw_connections_accepted"]
            >= before["raw_connections_accepted"] + 2
        )
    finally:
        if client is not None:
            await client.shutdown()
        await server.shutdown()


class _TensorService:
    def __init__(self):
        self.calls = 0

    async def receive_tensor(self, message):
        self.calls += 1
        return TensorMessage(message.metadata + b"-reply", message.buffers, message.version)


@pytest.mark.asyncio
async def test_wrapped_actor_dispatches_receive_tensor_and_returns_direct_message():
    service = _TensorService()
    wrapped = _WrappedActor(service)
    request = TensorMessage(b"request", [b"data"])

    response = await wrapped.receive(request)

    assert isinstance(response, TensorMessage)
    assert response.metadata == b"request-reply"
    assert bytes(response.buffers[0]) == b"data"
    assert service.calls == 1
