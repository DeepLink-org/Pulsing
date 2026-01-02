"""存储后端协议 - 可插拔的存储实现

定义 StorageBackend 协议，允许不同的存储实现：
- MemoryBackend: 纯内存（内置默认）
- 第三方实现: 如 persisting 提供的 LanceBackend、PersistingBackend

使用方式：
    # 使用内置后端
    writer = await write_queue(system, "topic", backend="memory")
    
    # 使用 persisting 提供的持久化后端
    from persisting.queue import LanceBackend
    from pulsing.queue import register_backend
    register_backend("lance", LanceBackend)
    writer = await write_queue(system, "topic", backend="lance")
    
    # 或者直接传类
    writer = await write_queue(system, "topic", backend=LanceBackend)
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from typing import Any, AsyncIterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class StorageBackend(Protocol):
    """存储后端协议
    
    所有存储后端必须实现此协议。可以通过继承或鸭子类型实现。
    """

    @abstractmethod
    async def put(self, record: dict[str, Any]) -> None:
        """写入单条记录"""
        ...

    @abstractmethod
    async def put_batch(self, records: list[dict[str, Any]]) -> None:
        """批量写入记录"""
        ...

    @abstractmethod
    async def get(self, limit: int, offset: int) -> list[dict[str, Any]]:
        """读取记录"""
        ...

    @abstractmethod
    async def get_stream(
        self,
        limit: int,
        offset: int,
        wait: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """流式读取记录"""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """刷新缓冲区到持久化存储"""
        ...

    @abstractmethod
    async def stats(self) -> dict[str, Any]:
        """获取统计信息"""
        ...

    @abstractmethod
    def total_count(self) -> int:
        """总记录数"""
        ...


class MemoryBackend:
    """纯内存后端 - 内置默认实现
    
    特点：
    - 无持久化，数据仅存在于内存
    - 支持阻塞等待新数据
    - 轻量级，适合测试和临时数据
    
    如需持久化能力，请使用 persisting 包提供的后端：
    - persisting.queue.LanceBackend: Lance 持久化
    - persisting.queue.PersistingBackend: 增强版（WAL、监控等）
    """

    def __init__(self, bucket_id: int, **kwargs):
        self.bucket_id = bucket_id
        self.buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    async def put(self, record: dict[str, Any]) -> None:
        async with self._condition:
            self.buffer.append(record)
            self._condition.notify_all()

    async def put_batch(self, records: list[dict[str, Any]]) -> None:
        async with self._condition:
            self.buffer.extend(records)
            self._condition.notify_all()

    async def get(self, limit: int, offset: int) -> list[dict[str, Any]]:
        async with self._lock:
            return self.buffer[offset : offset + limit]

    async def get_stream(
        self,
        limit: int,
        offset: int,
        wait: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        current_offset = offset
        remaining = limit

        while remaining > 0:
            async with self._condition:
                total = len(self.buffer)

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
                            return
                    else:
                        return

                records = self.buffer[current_offset : current_offset + min(remaining, 100)]

            if records:
                yield records
                current_offset += len(records)
                remaining -= len(records)
            elif not wait:
                break

    async def flush(self) -> None:
        pass  # 纯内存，无需 flush

    async def stats(self) -> dict[str, Any]:
        return {
            "bucket_id": self.bucket_id,
            "buffer_size": len(self.buffer),
            "persisted_count": 0,
            "total_count": len(self.buffer),
            "backend": "memory",
        }

    def total_count(self) -> int:
        return len(self.buffer)


# ============================================================
# 后端注册表
# ============================================================

# 内置后端映射（仅 memory）
_BUILTIN_BACKENDS: dict[str, type] = {
    "memory": MemoryBackend,
}

# 第三方后端注册（如 persisting 提供的 lance）
_REGISTERED_BACKENDS: dict[str, type] = {}


def register_backend(name: str, backend_class: type) -> None:
    """注册自定义后端
    
    Example:
        from persisting.queue import LanceBackend
        register_backend("lance", LanceBackend)
        
        writer = await write_queue(system, "topic", backend="lance")
    """
    if not isinstance(backend_class, type):
        raise TypeError(f"backend_class must be a class, got {type(backend_class)}")
    _REGISTERED_BACKENDS[name] = backend_class
    logger.info(f"Registered storage backend: {name}")


def get_backend_class(backend: str | type) -> type:
    """获取后端类
    
    Args:
        backend: 后端名称（str）或后端类（type）
        
    Returns:
        后端类
    """
    if isinstance(backend, type):
        return backend

    if backend in _REGISTERED_BACKENDS:
        return _REGISTERED_BACKENDS[backend]

    if backend in _BUILTIN_BACKENDS:
        return _BUILTIN_BACKENDS[backend]

    available = list(_BUILTIN_BACKENDS.keys()) + list(_REGISTERED_BACKENDS.keys())
    raise ValueError(
        f"Unknown backend: {backend}. Available: {available}. "
        f"Use register_backend() to add custom backends, or install 'persisting' for Lance support."
    )


def list_backends() -> list[str]:
    """列出所有可用的后端"""
    return list(_BUILTIN_BACKENDS.keys()) + list(_REGISTERED_BACKENDS.keys())
