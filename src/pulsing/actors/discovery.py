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
    
    async def stop(self):
        """停止发现服务"""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
    
    async def _scan_loop(self):
        while self._running:
            try:
                await self._discover_workers()
            except Exception:
                pass  # 静默处理扫描错误
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
            
            # 修正地址：0.0.0.0 替换为 127.0.0.1
            if addr.startswith("0.0.0.0:"):
                addr = addr.replace("0.0.0.0:", "127.0.0.1:")
            
            # 只处理 Alive 状态的 Worker，其他状态（Suspect/Dead）视为不可用
            if status != "Alive":
                # 如果之前存在，标记为需要移除
                if node_id in self._workers:
                    current_nodes.add(node_id)  # 保留在当前节点集合中，避免立即删除
                    worker = self._workers[node_id]
                    if worker.is_healthy:
                        # 状态变化：从健康变为不健康
                        worker.is_healthy = False
                        worker.actor_ref = None
                        if self.on_worker_removed:
                            try:
                                await self.on_worker_removed(node_id)
                            except Exception:
                                pass
                    worker.last_seen = time.time()
                continue
            
            # 只有 Alive 状态才处理
            current_nodes.add(node_id)
            is_healthy = True
            
            if node_id not in self._workers:
                await self._add_worker(node_id, addr, is_healthy)
            else:
                worker = self._workers[node_id]
                worker.last_seen = time.time()
                # 如果之前不健康，现在恢复健康，重新添加
                if not worker.is_healthy:
                    worker.is_healthy = True
                    # 重新解析 actor_ref
                    try:
                        worker.actor_ref = await self.system.resolve_named(self.worker_name)
                        await asyncio.wait_for(
                            worker.actor_ref.ask_json("HealthCheck", {}),
                            timeout=2.0
                        )
                        if self.on_worker_added:
                            try:
                                await self.on_worker_added(worker)
                            except Exception:
                                pass
                    except Exception:
                        worker.is_healthy = False
                        worker.actor_ref = None
        
        # 移除消失的 Worker
        for node_id in set(self._workers.keys()) - current_nodes:
            await self._remove_worker(node_id)
        
        # 清理长时间不健康的 Worker（超过 30 秒）
        now = time.time()
        to_remove = []
        for node_id, worker in self._workers.items():
            if not worker.is_healthy and (now - worker.last_seen) > 30:
                to_remove.append(node_id)
        for node_id in to_remove:
            await self._remove_worker(node_id)
    
    async def _add_worker(self, node_id: str, addr: str, is_healthy: bool = True):
        actor_ref = None
        # 只有状态为 Alive 的 Worker 才解析 actor_ref
        if is_healthy:
            try:
                actor_ref = await self.system.resolve_named(self.worker_name)
                # 测试连接：发送一个 HealthCheck（设置短超时）
                await asyncio.wait_for(
                    actor_ref.ask_json("HealthCheck", {}),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                # 连接超时，标记为不健康
                is_healthy = False
                actor_ref = None
            except Exception:
                # 其他错误，也标记为不健康
                is_healthy = False
                actor_ref = None
        
        worker = DiscoveredWorker(
            node_id=node_id,
            addr=addr,
            actor_ref=actor_ref,
            is_healthy=is_healthy,
        )
        self._workers[node_id] = worker
        
        if self.on_worker_added and is_healthy:
            try:
                await self.on_worker_added(worker)
            except Exception:
                pass  # 静默处理回调错误
    
    async def _remove_worker(self, node_id: str):
        if node_id in self._workers:
            self._workers.pop(node_id)
            
            if self.on_worker_removed:
                try:
                    await self.on_worker_removed(node_id)
                except Exception:
                    pass  # 静默处理回调错误
    
    def get_workers(self) -> List[DiscoveredWorker]:
        return list(self._workers.values())
    
    def get_worker_count(self) -> int:
        return len(self._workers)
