"""
Router Actor - 基于 RoundRobin 调度的路由器

提供请求路由功能，支持：
- Worker 自动发现
- RoundRobin 负载均衡
- OpenAI 兼容 HTTP API
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import deque

from aiohttp import web

from pulsing.actor import Actor, ActorRef, Message, ActorId

from .base import BaseServiceActor


@dataclass
class WorkerInfo:
    """Worker 信息"""
    worker_id: str
    actor_ref: Optional[ActorRef]
    endpoint: str
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    request_count: int = 0
    is_healthy: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)


class RoundRobinScheduler:
    """RoundRobin 调度器"""
    
    def __init__(self):
        self._workers: Dict[str, WorkerInfo] = {}
        self._worker_queue: deque = deque()
        self._lock = asyncio.Lock()
    
    async def add_worker(self, worker_info: WorkerInfo) -> bool:
        async with self._lock:
            if worker_info.worker_id in self._workers:
                return False
            self._workers[worker_info.worker_id] = worker_info
            self._worker_queue.append(worker_info.worker_id)
            return True
    
    async def remove_worker(self, worker_id: str) -> bool:
        async with self._lock:
            if worker_id not in self._workers:
                return False
            del self._workers[worker_id]
            self._worker_queue = deque(w for w in self._worker_queue if w != worker_id)
            return True
    
    async def get_next_worker(self) -> Optional[WorkerInfo]:
        """获取下一个可用的 Worker (RoundRobin)"""
        async with self._lock:
            if not self._worker_queue:
                return None
            
            for _ in range(len(self._worker_queue)):
                worker_id = self._worker_queue[0]
                self._worker_queue.rotate(-1)
                
                worker = self._workers.get(worker_id)
                if worker and worker.is_healthy:
                    worker.request_count += 1
                    return worker
            return None
    
    async def get_workers(self) -> List[WorkerInfo]:
        async with self._lock:
            return list(self._workers.values())
    
    async def get_worker_count(self) -> int:
        async with self._lock:
            return len(self._workers)
    
    async def get_healthy_worker_count(self) -> int:
        async with self._lock:
            return sum(1 for w in self._workers.values() if w.is_healthy)


class RouterActorHandler(Actor):
    """Router Actor 消息处理器（保留用于 Actor 间通信）"""
    
    def __init__(self):
        self.scheduler = RoundRobinScheduler()
        self._actor_id: Optional[ActorId] = None
    
    def on_start(self, actor_id: ActorId) -> None:
        self._actor_id = actor_id
    
    def on_stop(self) -> None:
        pass
    
    def metadata(self) -> Dict[str, str]:
        return {"type": "router", "scheduler": "round_robin"}
    
    async def receive(self, msg: Message) -> Message:
        # 简单返回状态
        return Message.from_json("Ok", {"status": "running"})


class RouterActor(BaseServiceActor):
    """
    Router Actor 服务
    
    基于 RoundRobin 调度的请求路由器，支持：
    - 自动 Worker 发现
    - OpenAI 兼容 HTTP API
    
    API 端点:
        GET  /v1/models           - 模型列表
        POST /v1/chat/completions - 聊天补全
        POST /v1/completions      - 文本补全
        GET  /v1/workers          - Worker 列表
        GET  /health              - 健康检查
    
    使用示例:
        pulsing actor --type router --http_port 8080
    """
    
    def __init__(
        self,
        namespace: str = "dynamo",
        addr: Optional[str] = None,
        seeds: Optional[List[str]] = None,
        http_host: str = "0.0.0.0",
        http_port: int = 8080,
        model_name: str = "pulsing-model",
        discovery_interval: float = 5.0,
    ):
        super().__init__(namespace=namespace, addr=addr, seeds=seeds, public=True)
        self._handler: Optional[RouterActorHandler] = None
        self._http_host = http_host
        self._http_port = http_port
        self._http_runner: Optional[web.AppRunner] = None
        self._model_name = model_name
        self._discovery_interval = discovery_interval
        self._discovery = None
    
    @property
    def service_name(self) -> str:
        return "router"
    
    def _create_actor(self) -> Actor:
        self._handler = RouterActorHandler()
        return self._handler
    
    @property
    def scheduler(self) -> Optional[RoundRobinScheduler]:
        return self._handler.scheduler if self._handler else None
    
    def _create_http_app(self) -> web.Application:
        from .openai_server import OpenAIServer
        server = OpenAIServer(
            router_scheduler=self._handler.scheduler,
            model_name=self._model_name,
            actor_system=self._system,  # 传递 ActorSystem 用于 resolve_named
        )
        return server.create_app()
    
    async def _on_worker_discovered(self, discovered_worker):
        """Worker 发现回调"""
        worker_info = WorkerInfo(
            worker_id=discovered_worker.worker_id,
            actor_ref=discovered_worker.actor_ref,
            endpoint=discovered_worker.addr,
        )
        if await self._handler.scheduler.add_worker(worker_info):
            print(f"[Router] + Worker {discovered_worker.worker_id[:8]}...")
    
    async def _on_worker_removed(self, worker_id: str):
        """Worker 移除回调"""
        if await self._handler.scheduler.remove_worker(worker_id):
            print(f"[Router] - Worker {worker_id[:8]}...")
    
    async def start(self) -> ActorRef:
        actor_ref = await super().start()
        
        # 启动 Worker 发现
        from .discovery import WorkerDiscovery
        self._discovery = WorkerDiscovery(
            system=self._system,
            on_worker_added=self._on_worker_discovered,
            on_worker_removed=self._on_worker_removed,
            scan_interval=self._discovery_interval,
        )
        await self._discovery.start()
        
        # 启动 HTTP 服务
        app = self._create_http_app()
        self._http_runner = web.AppRunner(app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, self._http_host, self._http_port)
        await site.start()
        
        print(f"[Router] HTTP: http://{self._http_host}:{self._http_port}")
        return actor_ref
    
    async def stop(self):
        if self._discovery:
            await self._discovery.stop()
        if self._http_runner:
            await self._http_runner.cleanup()
        await super().stop()
