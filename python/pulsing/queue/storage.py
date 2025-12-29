"""Bucket 存储 Actor - 每个 bucket 一个 Actor，负责数据缓冲和持久化"""

import asyncio
import logging
from pathlib import Path
from typing import Any

try:
    import lance
    import pyarrow as pa

    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

from pulsing.actor import Actor, ActorId, Message, StreamMessage

logger = logging.getLogger(__name__)


class BucketStorage(Actor):
    """单个 Bucket 的存储 Actor

    每个 bucket 对应一个 Lance 文件和一个 Storage Actor。
    数据分为内存缓冲和已持久化两部分，两部分数据同时对消费者可见。
    """

    def __init__(
        self,
        bucket_id: int,
        storage_path: str,
        batch_size: int = 100,
    ):
        self.bucket_id = bucket_id
        self.storage_path = Path(storage_path)
        self.batch_size = batch_size

        # 内存缓冲区
        self.buffer: list[dict[str, Any]] = []

        # 已持久化的记录数
        self.persisted_count: int = 0

        # 锁和条件变量
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Lance 数据集路径
        self._dataset_path = self.storage_path / "data.lance"

    def on_start(self, actor_id: ActorId) -> None:
        if not LANCE_AVAILABLE:
            logger.error("Lance not available. pip install lance pyarrow")
        # 恢复已持久化的记录数
        if self._dataset_path.exists() and LANCE_AVAILABLE:
            try:
                self.persisted_count = lance.dataset(self._dataset_path).count_rows()
            except Exception:
                pass
        logger.info(f"BucketStorage[{self.bucket_id}] started: {self.storage_path}")

    def on_stop(self) -> None:
        logger.info(f"BucketStorage[{self.bucket_id}] stopping")

    def _total_count(self) -> int:
        """总记录数 = 已持久化 + 内存缓冲"""
        return self.persisted_count + len(self.buffer)

    async def _flush(self) -> None:
        """将内存缓冲持久化到 Lance"""
        async with self._lock:
            if not self.buffer:
                return
            records = self.buffer[:]
            self.buffer = []

        if not LANCE_AVAILABLE:
            async with self._lock:
                self.buffer = records + self.buffer
            return

        try:
            # 构建 PyArrow Table
            field_names = set()
            for record in records:
                field_names.update(record.keys())

            arrays = {}
            for field_name in field_names:
                values = [record.get(field_name) for record in records]
                if all(isinstance(v, int) for v in values if v is not None):
                    arrays[field_name] = pa.array(values, type=pa.int64())
                elif all(isinstance(v, float) for v in values if v is not None):
                    arrays[field_name] = pa.array(values, type=pa.float64())
                elif all(isinstance(v, bool) for v in values if v is not None):
                    arrays[field_name] = pa.array(values, type=pa.bool_())
                else:
                    arrays[field_name] = pa.array(
                        [str(v) if v is not None else None for v in values],
                        type=pa.string(),
                    )

            table = pa.table(arrays)

            if self._dataset_path.exists():
                existing = lance.dataset(self._dataset_path).to_table()
                combined = pa.concat_tables([existing, table])
                lance.write_dataset(combined, self._dataset_path, mode="overwrite")
            else:
                lance.write_dataset(table, self._dataset_path, mode="create")

            self.persisted_count += len(records)
            logger.debug(
                f"BucketStorage[{self.bucket_id}] flushed {len(records)} records"
            )

        except Exception as e:
            logger.error(f"BucketStorage[{self.bucket_id}] flush error: {e}")
            async with self._lock:
                self.buffer = records + self.buffer

    def _read_records(self, offset: int, limit: int) -> list[dict[str, Any]]:
        """读取数据：先读持久化数据，再读内存缓冲，合并返回"""
        records = []

        # 1. 从持久化数据读取
        if (
            offset < self.persisted_count
            and self._dataset_path.exists()
            and LANCE_AVAILABLE
        ):
            try:
                dataset = lance.dataset(self._dataset_path)
                persisted_limit = min(limit, self.persisted_count - offset)
                table = dataset.to_table(offset=offset, limit=persisted_limit)
                if table.num_rows > 0:
                    columns = table.column_names
                    for i in range(table.num_rows):
                        records.append({col: table[col][i].as_py() for col in columns})
            except Exception as e:
                logger.error(f"BucketStorage[{self.bucket_id}] read error: {e}")

        # 2. 从内存缓冲读取
        if len(records) < limit:
            buffer_offset = max(0, offset - self.persisted_count)
            buffer_limit = limit - len(records)
            buffer_records = self.buffer[buffer_offset : buffer_offset + buffer_limit]
            records.extend(buffer_records)

        return records

    async def receive(self, msg: Message) -> Message | StreamMessage | None:
        msg_type = msg.msg_type
        data = msg.to_json()

        if msg_type == "Put":
            record = data.get("record")
            if not record:
                return Message.from_json("Error", {"error": "Missing 'record'"})

            async with self._condition:
                self.buffer.append(record)
                should_flush = len(self.buffer) >= self.batch_size
                self._condition.notify_all()

            if should_flush:
                await self._flush()
                async with self._condition:
                    self._condition.notify_all()

            return Message.from_json("PutResponse", {"status": "ok"})

        elif msg_type == "Get":
            limit = data.get("limit", 100)
            offset = data.get("offset", 0)

            async with self._lock:
                records = self._read_records(offset, limit)

            return Message.from_json("GetResponse", {"records": records})

        elif msg_type == "GetStream":
            limit = data.get("limit", 100)
            offset = data.get("offset", 0)
            wait: bool = data.get("wait", False)
            timeout: float | None = data.get("timeout", None)

            stream_msg, writer = StreamMessage.create("GetStream")

            async def produce():
                try:
                    current_offset = offset
                    remaining = limit

                    while remaining > 0:
                        async with self._condition:
                            total = self._total_count()

                            # 没有数据时等待
                            if current_offset >= total:
                                if wait:
                                    try:
                                        if timeout:
                                            await asyncio.wait_for(
                                                self._condition.wait(), timeout=timeout
                                            )
                                        else:
                                            await self._condition.wait()
                                        continue
                                    except asyncio.TimeoutError:
                                        writer.close()
                                        return
                                else:
                                    writer.close()
                                    return

                            # 读取数据
                            records = self._read_records(
                                current_offset, min(remaining, 100)
                            )

                        if records:
                            await writer.write_json(
                                {
                                    "records": records,
                                    "offset": current_offset,
                                }
                            )
                            current_offset += len(records)
                            remaining -= len(records)
                        elif not wait:
                            break

                    writer.close()

                except Exception as e:
                    logger.error(f"BucketStorage[{self.bucket_id}] stream error: {e}")
                    await writer.error(str(e))
                    writer.close()

            asyncio.create_task(produce())
            return stream_msg

        elif msg_type == "Flush":
            await self._flush()
            return Message.from_json("FlushResponse", {"status": "ok"})

        elif msg_type == "Stats":
            async with self._lock:
                return Message.from_json(
                    "StatsResponse",
                    {
                        "bucket_id": self.bucket_id,
                        "buffer_size": len(self.buffer),
                        "persisted_count": self.persisted_count,
                        "total_count": self._total_count(),
                    },
                )

        else:
            return Message.from_json("Error", {"error": f"Unknown: {msg_type}"})
