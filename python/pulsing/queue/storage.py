"""Bucket 存储 Actor - 使用可插拔后端"""

import asyncio
import logging
from typing import Any

from pulsing.actor import Actor, ActorId, Message, StreamMessage

from .backend import StorageBackend, get_backend_class

logger = logging.getLogger(__name__)


class BucketStorage(Actor):
    """单个 Bucket 的存储 Actor

    使用可插拔的 StorageBackend 实现数据存储。

    Args:
        bucket_id: 桶 ID
        storage_path: 存储路径
        batch_size: 批处理大小
        backend: 后端名称或后端类
            - "memory": 纯内存后端
            - "lance": Lance 持久化后端（默认）
            - 自定义类: 实现 StorageBackend 协议的类
        backend_options: 传递给后端的额外参数
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

        # 后端实例（在 on_start 中初始化）
        self._backend: StorageBackend | None = None

    def on_start(self, actor_id: ActorId) -> None:
        # 创建后端实例
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
