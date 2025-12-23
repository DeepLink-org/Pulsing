"""Worker 调度器 - 负载均衡策略"""

from abc import ABC, abstractmethod
from typing import Optional
import asyncio


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
    
    async def _resolve_worker(self):
        try:
            return await self._system.resolve_named(self._worker_name)
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
        
        return await self._resolve_worker()


class RandomScheduler(Scheduler):
    """随机调度器"""
    
    async def select_worker(self):
        workers = await self.get_available_workers()
        return await self._resolve_worker() if workers else None


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
            selected = min(workers, key=lambda w: self._request_counts.get(w.get("node_id"), 0))
            self._request_counts[selected["node_id"]] = self._request_counts.get(selected["node_id"], 0) + 1
        
        return await self._resolve_worker()

