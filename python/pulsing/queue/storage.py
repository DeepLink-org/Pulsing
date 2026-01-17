"""Bucket Storage Actor - Using Pluggable Backend"""

import asyncio
import logging
from typing import Any

from pulsing.actor import Actor, ActorId, Message, StreamMessage

from .backend import StorageBackend, get_backend_class

logger = logging.getLogger(__name__)


class BucketStorage(Actor):
    """Storage Actor for a Single Bucket

    Uses pluggable StorageBackend for data storage.

    Args:
        bucket_id: Bucket ID
        storage_path: Storage path
        batch_size: Batch size
        backend: Backend name or backend class
            - "memory": Pure in-memory backend
            - "lance": Lance persistent backend (default)
            - Custom class: Class implementing StorageBackend protocol
        backend_options: Additional parameters passed to backend
    """

    def __init__(
        self,
        bucket_id: int,
        storage_path: str,
        batch_size: int = 100,
        backend: str | type = "lance",
        backend_options: dict[str, Any] | None = None,
    ):
        self.bucket_id = bucket_id
        self.storage_path = storage_path
        self.batch_size = batch_size
        self._backend_type = backend
        self._backend_options = backend_options or {}

        # Backend instance (initialized in on_start)
        self._backend: StorageBackend | None = None

    def on_start(self, actor_id: ActorId) -> None:
        # Create backend instance
        backend_class = get_backend_class(self._backend_type)
        self._backend = backend_class(
            bucket_id=self.bucket_id,
            storage_path=self.storage_path,
            batch_size=self.batch_size,
            **self._backend_options,
        )
        backend_name = getattr(backend_class, "__name__", str(self._backend_type))
        logger.info(
            f"BucketStorage[{self.bucket_id}] started with {backend_name} at {self.storage_path}"
        )

    def on_stop(self) -> None:
        logger.info(f"BucketStorage[{self.bucket_id}] stopping")

    async def receive(self, msg: Message) -> Message | StreamMessage | None:
        msg_type = msg.msg_type
        data = msg.to_json()

        if msg_type == "Put":
            record = data.get("record")
            if not record:
                return Message.from_json("Error", {"error": "Missing 'record'"})

            await self._backend.put(record)
            return Message.from_json("PutResponse", {"status": "ok"})

        elif msg_type == "PutBatch":
            records = data.get("records")
            if not records:
                return Message.from_json("Error", {"error": "Missing 'records'"})

            await self._backend.put_batch(records)
            return Message.from_json(
                "PutBatchResponse", {"status": "ok", "count": len(records)}
            )

        elif msg_type == "Get":
            limit = data.get("limit", 100)
            offset = data.get("offset", 0)
            records = await self._backend.get(limit, offset)
            return Message.from_json("GetResponse", {"records": records})

        elif msg_type == "GetStream":
            limit = data.get("limit", 100)
            offset = data.get("offset", 0)
            wait: bool = data.get("wait", False)
            timeout: float | None = data.get("timeout", None)

            stream_msg, writer = StreamMessage.create("GetStream")

            async def produce():
                try:
                    async for records in self._backend.get_stream(
                        limit, offset, wait, timeout
                    ):
                        await writer.write({"records": records})
                    writer.close()
                except Exception as e:
                    logger.error(f"BucketStorage[{self.bucket_id}] stream error: {e}")
                    await writer.error(str(e))
                    writer.close()

            asyncio.create_task(produce())
            return stream_msg

        elif msg_type == "Flush":
            await self._backend.flush()
            return Message.from_json("FlushResponse", {"status": "ok"})

        elif msg_type == "Stats":
            stats = await self._backend.stats()
            return Message.from_json("StatsResponse", stats)

        else:
            return Message.from_json("Error", {"error": f"Unknown: {msg_type}"})
