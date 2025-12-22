"""
Worker Discovery - 基于 Actor System 的服务发现

通过 Named Actor 机制自动发现集群中的 Worker。
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Awaitable

from pulsing.actor import ActorSystem, ActorRef


@dataclass
class DiscoveredWorker:
    """发现的 Worker 信息"""
    node_id: str
    addr: str
    actor_ref: Optional[ActorRef] = None
    discovered_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    is_healthy: bool = True
    request_count: int = 0

    @property
    def worker_id(self) -> str:
        return self.node_id


class WorkerDiscovery:
    """
    Worker 发现服务
    
    通过 Named Actor 机制发现集群中的 Worker：
    1. Worker 以 public=True 注册，名称为 "worker"
    2. Router 定期调用 get_named_instances("worker") 获取所有实例
    """
    
    def __init__(
        self,
        system: ActorSystem,
        on_worker_added: Optional[Callable[[DiscoveredWorker], Awaitable[None]]] = None,
        on_worker_removed: Optional[Callable[[str], Awaitable[None]]] = None,
        scan_interval: float = 5.0,
        worker_name: str = "worker",
    ):
        self.system = system
        self.on_worker_added = on_worker_added
        self.on_worker_removed = on_worker_removed
        self.scan_interval = scan_interval
        self.worker_name = worker_name
        
        self._workers: Dict[str, DiscoveredWorker] = {}
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动发现服务"""
        if self._running:
            return
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        print(f"[WorkerDiscovery] Started")
    
    async def stop(self):
        """停止发现服务"""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        print("[WorkerDiscovery] Stopped")
    
    async def _scan_loop(self):
        while self._running:
            try:
                await self._discover_workers()
            except Exception as e:
                print(f"[WorkerDiscovery] Error: {e}")
            await asyncio.sleep(self.scan_interval)
    
    async def _discover_workers(self):
        """发现 Worker 实例"""
        try:
            instances = await self.system.get_named_instances(self.worker_name)
        except Exception:
            instances = []
        
        current_nodes = set()
        
        for instance in instances:
            node_id = instance.get("node_id", "")
            addr = instance.get("addr", "")
            status = instance.get("status", "")
            
            # 排除 Dead 状态
            if status == "Dead":
                continue
            
            current_nodes.add(node_id)
            is_healthy = (status == "Alive")
            
            if node_id not in self._workers:
                await self._add_worker(node_id, addr, is_healthy)
            else:
                self._workers[node_id].last_seen = time.time()
                self._workers[node_id].is_healthy = is_healthy
        
        # 移除消失的 Worker
        for node_id in set(self._workers.keys()) - current_nodes:
            await self._remove_worker(node_id)
    
    async def _add_worker(self, node_id: str, addr: str, is_healthy: bool = True):
        actor_ref = None
        try:
            actor_ref = await self.system.resolve_named(self.worker_name)
        except Exception:
            pass
        
        worker = DiscoveredWorker(
            node_id=node_id,
            addr=addr,
            actor_ref=actor_ref,
            is_healthy=is_healthy,
        )
        self._workers[node_id] = worker
        print(f"[WorkerDiscovery] + {node_id[:8]}... at {addr}")
        
        if self.on_worker_added:
            try:
                await self.on_worker_added(worker)
            except Exception as e:
                print(f"[WorkerDiscovery] Callback error: {e}")
    
    async def _remove_worker(self, node_id: str):
        if node_id in self._workers:
            worker = self._workers.pop(node_id)
            print(f"[WorkerDiscovery] - {node_id[:8]}... at {worker.addr}")
            
            if self.on_worker_removed:
                try:
                    await self.on_worker_removed(node_id)
                except Exception as e:
                    print(f"[WorkerDiscovery] Callback error: {e}")
    
    def get_workers(self) -> List[DiscoveredWorker]:
        return list(self._workers.values())
    
    def get_worker_count(self) -> int:
        return len(self._workers)
