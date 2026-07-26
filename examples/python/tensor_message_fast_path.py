#!/usr/bin/env python3
"""Send contiguous CPU tensor buffers over Pulsing's TensorMessage fast path.

The example uses two ActorSystems so the request always crosses TCP. It keeps
the schema in opaque JSON metadata and uses only the standard library; a real
TensorDict integration can replace ``array`` with contiguous CPU tensor views.

Usage:
    python examples/python/tensor_message_fast_path.py
    python examples/python/tensor_message_fast_path.py --transport http2
"""

import argparse
import asyncio
import json
import os
import sys
from array import array

import pulsing as pul


@pul.remote
class TensorSink:
    """Consume a TensorMessage and return a small application-level ACK."""

    async def receive_tensor(self, message: pul.TensorMessage) -> dict:
        metadata = json.loads(message.metadata)
        if metadata["dtype"] != "float32-native":
            raise ValueError(f"unsupported dtype: {metadata['dtype']}")

        values = message.buffers[0].cast("f")
        expected_values = metadata["shape"][0] * metadata["shape"][1]
        if len(values) != expected_values:
            raise ValueError(
                f"shape expects {expected_values} values, received {len(values)}"
            )

        # Returning only after processing gives ask() application-level
        # completion semantics. A storage actor would write its bucket here.
        return {
            "ok": True,
            "received_bytes": message.total_bytes,
            "sum": sum(values),
            "version": message.version,
        }


async def run(transport: str) -> None:
    os.environ["PULSING_TENSOR_TRANSPORT"] = transport

    server = await pul.actor_system(addr="127.0.0.1:0")
    client = None
    try:
        actor_name = "tensor-message-fast-path"
        await TensorSink.spawn(system=server, name=actor_name, public=True)

        client = await pul.actor_system(
            addr="127.0.0.1:0",
            seeds=[server.addr],
        )
        sink = await TensorSink.resolve(
            name=actor_name,
            system=client,
            node_id=server.node_id.id,
            timeout=5.0,
        )
        if sink.ref.is_local():
            raise RuntimeError("example must resolve a remote actor")

        source = array("f", range(12))
        metadata = json.dumps(
            {
                "dtype": "float32-native",
                "shape": [3, 4],
                "byte_order": sys.byteorder,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        message = pul.TensorMessage(
            metadata=metadata,
            buffers=[memoryview(source).cast("B")],
            version=1,
        )

        # Use .ref.ask() for TensorMessage. Calling a generated proxy method
        # would put the value inside the ordinary pickle method envelope.
        ack = await sink.ref.ask(message)
        assert ack == {
            "ok": True,
            "received_bytes": 48,
            "sum": 66.0,
            "version": 1,
        }

        stats = pul.tensor_transport_stats()
        print(f"transport={transport}")
        print(f"ack={ack}")
        print(f"tensor_transport_stats={stats}")
    finally:
        if client is not None:
            await client.shutdown()
        await server.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("raw", "http2"),
        default="raw",
        help="raw uses the direct TCP path; http2 exercises compatibility mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.transport))
