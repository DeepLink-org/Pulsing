import pytest
import pulsing as pul
from pulsing.core import Actor, ZeroCopyDescriptor


class _ZeroCopyPayload:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __zerocopy__(self, _ctx):
        return ZeroCopyDescriptor(
            buffers=[memoryview(self.payload)],
            dtype="u8",
            shape=[len(self.payload)],
            strides=[1],
            transport="inline",
            checksum=None,
            version=1,
        )


class _Inspector(Actor):
    async def receive(self, msg):
        if isinstance(msg, ZeroCopyDescriptor):
            buffers = msg.buffers
            return {
                "kind": "descriptor",
                "version": msg.version,
                "buffer_count": len(buffers),
                "first_size": len(buffers[0]),
            }
        return {"kind": "normal", "type": type(msg).__name__}


@pytest.fixture
async def actor_system():
    system = await pul.actor_system()
    yield system
    await system.shutdown()


@pytest.mark.asyncio
async def test_zerocopy_auto_uses_descriptor(actor_system, monkeypatch):
    monkeypatch.setenv("PULSING_ZEROCOPY", "auto")
    ref = await actor_system.spawn(_Inspector(), name="zc-auto")
    resp = await ref.ask(_ZeroCopyPayload(b"hello"))
    assert resp["kind"] == "descriptor"
    assert resp["version"] == 1
    assert resp["buffer_count"] == 1
    assert resp["first_size"] == 5


@pytest.mark.asyncio
async def test_zerocopy_off_falls_back_pickle(actor_system, monkeypatch):
    monkeypatch.setenv("PULSING_ZEROCOPY", "off")
    ref = await actor_system.spawn(_Inspector(), name="zc-off")
    resp = await ref.ask(_ZeroCopyPayload(b"hello"))
    assert resp["kind"] == "normal"
    assert resp["type"] == "_ZeroCopyPayload"


@pytest.mark.asyncio
async def test_zerocopy_force_rejects_non_descriptor(actor_system, monkeypatch):
    monkeypatch.setenv("PULSING_ZEROCOPY", "force")
    ref = await actor_system.spawn(_Inspector(), name="zc-force")
    with pytest.raises(Exception):
        await ref.ask({"x": 1})


@pytest.mark.asyncio
async def test_zerocopy_small_payload_single_path(actor_system, monkeypatch):
    """Small payload below stream threshold stays on single-message path."""
    monkeypatch.setenv("PULSING_ZEROCOPY", "auto")
    monkeypatch.setenv("PULSING_ZEROCOPY_STREAM_THRESHOLD", "1048576")
    ref = await actor_system.spawn(_Inspector(), name="zc-small")
    resp = await ref.ask(_ZeroCopyPayload(b"tiny"))
    assert resp["kind"] == "descriptor"
    assert resp["first_size"] == 4


@pytest.mark.asyncio
async def test_zerocopy_large_payload_stream_path(actor_system, monkeypatch):
    """Large payload above stream threshold goes through descriptor-first stream."""
    monkeypatch.setenv("PULSING_ZEROCOPY", "auto")
    monkeypatch.setenv("PULSING_ZEROCOPY_STREAM_THRESHOLD", "4096")
    monkeypatch.setenv("PULSING_ZEROCOPY_CHUNK_BYTES", "4096")
    ref = await actor_system.spawn(_Inspector(), name="zc-stream")
    big = bytes(range(256)) * 64  # 16384 bytes, 4 chunks
    resp = await ref.ask(_ZeroCopyPayload(big))
    assert resp["kind"] == "descriptor"
    assert resp["first_size"] == len(big)
    assert resp["buffer_count"] == 1


@pytest.mark.asyncio
async def test_zerocopy_stream_data_integrity(actor_system, monkeypatch):
    """Data transmitted via stream path arrives intact."""
    monkeypatch.setenv("PULSING_ZEROCOPY", "auto")
    monkeypatch.setenv("PULSING_ZEROCOPY_STREAM_THRESHOLD", "4096")
    monkeypatch.setenv("PULSING_ZEROCOPY_CHUNK_BYTES", "4096")

    class _DataVerifier(Actor):
        async def receive(self, msg):
            if isinstance(msg, ZeroCopyDescriptor):
                data = bytes(msg.buffers[0])
                return {
                    "size": len(data),
                    "checksum": sum(data) % 65536,
                }
            return {}

    ref = await actor_system.spawn(_DataVerifier(), name="zc-verify")
    payload = bytes(range(256)) * 80  # 20480 bytes
    expected_checksum = sum(payload) % 65536
    resp = await ref.ask(_ZeroCopyPayload(payload))
    assert resp["size"] == len(payload)
    assert resp["checksum"] == expected_checksum
