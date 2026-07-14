"""Child process: Connect (out-cluster) to parent gateway + run one actor behind IPC.

Started by the parent with ``python -m pulsing.isolated_worker``. Environment:

- ``PULSING_GATEWAY_ADDR`` — gateway for :class:`pulsing.connect.Connect`
- ``PULSING_IPC_HOST`` / ``PULSING_IPC_PORT`` — TCP to parent bridge
- ``PULSING_ISOLATED_PICKLE_PATH`` — pickled :class:`pulsing.core.remote._WrappedActor`

**Security:** unpickling executes code; only use with trusted actors / trusted parents.
IPC uses pickle for request/response bodies (same trust boundary as the initial worker pickle).
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import sys
from typing import Any

logger = logging.getLogger(__name__)


async def _read_frame(reader: asyncio.StreamReader) -> Any:
    hdr = await reader.readexactly(4)
    n = int.from_bytes(hdr, "big")
    body = await reader.readexactly(n)
    return pickle.loads(body)


async def _write_frame(writer: asyncio.StreamWriter, obj: object) -> None:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    writer.write(len(raw).to_bytes(4, "big") + raw)
    await writer.drain()


async def _async_main() -> None:
    from pulsing.connect import Connect

    path = os.environ.get("PULSING_ISOLATED_PICKLE_PATH")
    gw = os.environ.get("PULSING_GATEWAY_ADDR")
    host = os.environ.get("PULSING_IPC_HOST", "127.0.0.1")
    port_s = os.environ.get("PULSING_IPC_PORT")
    if not path or not gw or not port_s:
        print(
            "isolated_worker: missing PULSING_ISOLATED_PICKLE_PATH / "
            "PULSING_GATEWAY_ADDR / PULSING_IPC_PORT",
            file=sys.stderr,
        )
        raise SystemExit(2)

    with open(path, "rb") as f:
        wrapped = pickle.load(f)

    try:
        os.unlink(path)
    except OSError:
        pass

    connect = await Connect.to(gw)
    _keep = connect  # noqa: F841 — hold gateway connection open (out-cluster attach)

    reader, writer = await asyncio.open_connection(host, int(port_s))
    writer.write(b"READY\n")
    await writer.drain()

    while True:
        try:
            envelope = await _read_frame(reader)
        except asyncio.IncompleteReadError:
            break
        if not isinstance(envelope, dict) or envelope.get("kind") != "call":
            await _write_frame(
                writer,
                {"kind": "error", "message": "invalid envelope (expected kind=call)"},
            )
            continue
        inner = envelope.get("msg")
        try:
            out = await wrapped.receive(inner)
            if hasattr(out, "is_stream") and getattr(out, "is_stream", False):
                await _write_frame(writer, {"kind": "stream_unsupported"})
            else:
                try:
                    await _write_frame(writer, {"kind": "result", "value": out})
                except (pickle.PicklingError, TypeError) as ser_e:
                    await _write_frame(
                        writer,
                        {
                            "kind": "error",
                            "message": f"unpicklable isolated result: {ser_e}",
                        },
                    )
        except Exception as e:
            logger.exception("isolated worker actor error")
            await _write_frame(writer, {"kind": "error", "message": str(e)})

    await connect.close()
    writer.close()
    await writer.wait_closed()


def main() -> None:
    logging.basicConfig(level=os.environ.get("PULSING_LOG_LEVEL", "WARNING"))
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
