"""
调度器 - 负责从 Worker 中选择一个

职责分离：
- Rust Gossip 层：服务发现（找到所有 worker 节点，resolve_named 返回 ActorRef）
- Python 调度器：选择策略（RoundRobin、Random、LeastConnection）

设计理念：
- 按需查询：每次 select_worker() 时从 Gossip 获取最新列表
- 无缓存：不缓存 actor_ref（resolve_named 本身很快，Rust 层有缓存）
- 轻量级：只维护调度器自己的状态（如 RoundRobin 的 index）
"""

from abc import ABC, abstractmethod
from typing import Optional
import asyncio


class Scheduler(ABC):
    """调度器基类 - 按需选择 Worker"""
    
    def __init__(self, actor_system, worker_name: str = "worker"):
        self._system = actor_system
        self._worker_name = worker_name
        self._lock = asyncio.Lock()
    
    async def get_available_workers(self):
        """从 Gossip 获取当前可用的 worker 列表"""
        try:
            return await self._system.get_named_instances(self._worker_name)
        except Exception:
            return []
    
    async def get_worker_count(self) -> int:
        """获取 worker 总数"""
        return len(await self.get_available_workers())
    
    async def get_healthy_worker_count(self) -> int:
        """获取健康 worker 数量（Alive 状态）"""
        workers = await self.get_available_workers()
        return sum(1 for w in workers if w.get("status") == "Alive")
    
    async def _resolve_worker(self):
        """解析 worker ActorRef（子类共用）"""
        try:
            return await self._system.resolve_named(self._worker_name)
        except Exception:
            return None
    
    @abstractmethod
    async def select_worker(self):
        """选择一个 worker 并返回其 ActorRef
        
        Returns:
            ActorRef 或 None
        """
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
    """最少连接调度器 - 基于请求计数"""
    
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

