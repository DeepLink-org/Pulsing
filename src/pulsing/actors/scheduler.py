"""Worker 调度器 - 负载均衡策略"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class Scheduler(ABC):
    """调度器基类"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        self._system = actor_system
        self._worker_name = worker_name
        self._lock = asyncio.Lock()

    async def get_available_workers(self):
        try:
            return await self._system.get_named_instances(self._worker_name)
        except Exception:
            return []

    async def get_worker_count(self) -> int:
        return len(await self.get_available_workers())

    async def get_healthy_worker_count(self) -> int:
        workers = await self.get_available_workers()
        return sum(1 for w in workers if w.get("status") == "Alive")

    async def _resolve_worker(self, node_id: Optional[str] = None):
        try:
            # 如果指定了 node_id，则请求解析该特定节点的 Actor
            return await self._system.resolve_named(self._worker_name, node_id=node_id)
        except Exception:
            return None

    @abstractmethod
    async def select_worker(self):
        """选择一个 worker，返回 ActorRef 或 None"""
        pass


class RoundRobinScheduler(Scheduler):
    """轮询调度器"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._index = 0

    async def select_worker(self):
        workers = await self.get_available_workers()
        if not workers:
            return None

        async with self._lock:
            self._index = (self._index + 1) % len(workers)
            selected_worker = workers[self._index]

        return await self._resolve_worker(node_id=selected_worker.get("node_id"))


class RandomScheduler(Scheduler):
    """随机调度器"""

    async def select_worker(self):
        import random

        workers = await self.get_available_workers()
        if not workers:
            return None

        selected_worker = random.choice(workers)
        return await self._resolve_worker(node_id=selected_worker.get("node_id"))


class LeastConnectionScheduler(Scheduler):
    """最少连接调度器"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._request_counts = {}

    async def select_worker(self):
        workers = await self.get_available_workers()
        if not workers:
            return None

        async with self._lock:
            selected_worker = min(
                workers, key=lambda w: self._request_counts.get(w.get("node_id"), 0)
            )
            node_id = selected_worker.get("node_id")
            self._request_counts[node_id] = self._request_counts.get(node_id, 0) + 1

        return await self._resolve_worker(node_id=node_id)
