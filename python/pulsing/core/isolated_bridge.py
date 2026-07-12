"""Parent-side bridge actor: cluster traffic is handled here and forwarded to a child process."""

from __future__ import annotations

import asyncio
import logging
import pickle
from dataclasses import dataclass
from typing import Any

from pulsing.core.remote import Actor

logger = logging.getLogger(__name__)


@dataclass
class IsolatedSpawnHandle:
    """Result of ``spawn(actor, new_process=True, ...)`` with a real ``actor``.

    ``ref`` is the cluster-visible actor on the parent node; ``process`` is the
    isolated worker (terminate it to tear down the child).
    """

    ref: Any  # ActorRef — avoid circular import typing
    process: asyncio.subprocess.Process


async def _write_frame(writer: asyncio.StreamWriter, obj: Any) -> None:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if len(raw) > 0xFFFFFF:
        raise ValueError("isolated IPC payload too large")
    writer.write(len(raw).to_bytes(4, "big") + raw)
    await writer.drain()


async def _read_frame(reader: asyncio.StreamReader) -> Any:
    hdr = await reader.readexactly(4)
    n = int.from_bytes(hdr, "big")
    if n > 0xFFFFFF:
        raise ValueError("isolated IPC frame too large")
    body = await reader.readexactly(n)
    return pickle.loads(body)


async def wait_child_ready(reader: asyncio.StreamReader) -> None:
    line = await reader.readline()
    if line != b"READY\n":
        raise RuntimeError(
            f"isolated worker protocol error, expected READY, got {line!r}"
        )


class IsolatedBridgeActor(Actor):
    """Forwards each ``receive`` to the child over IPC (pickle-framed payloads)."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        pickle_path: str,
    ):
        self._reader = reader
        self._writer = writer
        self._pickle_path = pickle_path
        self._lock = asyncio.Lock()

    async def receive(self, msg: Any) -> Any:
        async with self._lock:
            try:
                await _write_frame(self._writer, {"kind": "call", "msg": msg})
                resp = await _read_frame(self._reader)
            except (
                BrokenPipeError,
                ConnectionResetError,
                asyncio.IncompleteReadError,
            ) as e:
                logger.warning("isolated child IPC lost: %s", e)
                return {"__error__": "isolated worker disconnected"}
            if not isinstance(resp, dict):
                return {"__error__": "invalid isolated worker response"}
            kind = resp.get("kind")
            if kind == "error":
                return {"__error__": str(resp.get("message", "isolated worker error"))}
            if kind == "stream_unsupported":
                return {
                    "__error__": "streaming responses are not supported for isolated actors (MVP)"
                }
            if kind == "result":
                return resp.get("value")
            return {"__error__": f"invalid isolated worker response kind: {kind!r}"}
