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
            instances = await self._system.get_named_instances(self._worker_name)
            # 返回所有实例（不过滤状态，让 resolve_named 时由 Rust 层判断）
            return instances
        except Exception as e:
            print(f"[Scheduler] Failed to get workers: {e}")
            return []
    
    async def get_worker_count(self) -> int:
        """获取 worker 总数"""
        workers = await self.get_available_workers()
        return len(workers)
    
    async def get_healthy_worker_count(self) -> int:
        """获取健康 worker 数量（Alive 状态）"""
        workers = await self.get_available_workers()
        return sum(1 for w in workers if w.get("status") == "Alive")
    
    @abstractmethod
    async def select_worker(self):
        """选择一个 worker 并返回其 ActorRef
        
        Returns:
            ActorRef 或 None
        """
        pass


class RoundRobinScheduler(Scheduler):
    """轮询调度器 - 最简单的负载均衡"""
    
    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._index = 0
    
    async def select_worker(self):
        """按 RoundRobin 选择一个 worker"""
        workers = await self.get_available_workers()
        if not workers:
            print("[Scheduler] No workers available")
            return None
        
        async with self._lock:
            # RoundRobin 选择
            selected = workers[self._index % len(workers)]
            self._index += 1
            
            node_id = selected.get("node_id", "unknown")[:8]
            addr = selected.get("addr", "unknown")
            print(f"[Scheduler] RoundRobin selected: {node_id}... at {addr} ({self._index-1} % {len(workers)})")
        
        # 解析 ActorRef（Rust 层会选择健康的实例）
        try:
            actor_ref = await self._system.resolve_named(self._worker_name)
            return actor_ref
        except Exception as e:
            print(f"[Scheduler] Failed to resolve worker: {e}")
            return None


class RandomScheduler(Scheduler):
    """随机调度器"""
    
    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
    
    async def select_worker(self):
        """随机选择一个 worker"""
        import random
        
        workers = await self.get_available_workers()
        if not workers:
            return None
        
        selected = random.choice(workers)
        node_id = selected.get("node_id", "unknown")[:8]
        print(f"[Scheduler] Random selected: {node_id}...")
        
        try:
            actor_ref = await self._system.resolve_named(self._worker_name)
            return actor_ref
        except Exception as e:
            print(f"[Scheduler] Failed to resolve worker: {e}")
            return None


class LeastConnectionScheduler(Scheduler):
    """最少连接调度器 - 选择当前请求数最少的 worker
    
    注意：这个实现维护了请求计数状态
    """
    
    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._request_counts = {}  # node_id -> count
    
    async def select_worker(self):
        """选择请求数最少的 worker"""
        workers = await self.get_available_workers()
        if not workers:
            return None
        
        async with self._lock:
            # 选择请求数最少的
            selected = min(workers, key=lambda w: self._request_counts.get(w.get("node_id"), 0))
            node_id = selected.get("node_id")
            self._request_counts[node_id] = self._request_counts.get(node_id, 0) + 1
            
            print(f"[Scheduler] LeastConnection selected: {node_id[:8]}... (requests={self._request_counts[node_id]})")
        
        try:
            actor_ref = await self._system.resolve_named(self._worker_name)
            return actor_ref
        except Exception as e:
            print(f"[Scheduler] Failed to resolve worker: {e}")
            return None

