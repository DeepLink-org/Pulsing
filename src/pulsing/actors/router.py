"""
Router Actor - 基于 Actor System 的请求路由器

提供请求路由功能，支持：
- 自动 Worker 发现（通过 Named Actor）
- OpenAI 兼容 HTTP API
"""

import asyncio
from typing import Optional, List

from aiohttp import web

from pulsing.actor import Actor, ActorRef, Message, ActorId

from .base import BaseServiceActor


class SimpleActor(Actor):
    """简单的 Actor 实现"""
    
    def on_start(self, actor_id: ActorId) -> None:
        pass
    
    def on_stop(self) -> None:
        pass
    
    def metadata(self) -> dict:
        return {}
    
    async def receive(self, msg: Message) -> Message:
        return Message.from_json("Ok", {"status": "running"})


class RouterActor(BaseServiceActor):
    """
    Router Actor 服务
    
    基于 Actor System 的请求路由器，支持：
    - 自动 Worker 发现（通过 Named Actor 机制）
    - OpenAI 兼容 HTTP API
    
    API 端点:
        GET  /v1/models           - 模型列表
        POST /v1/chat/completions - 聊天补全
        POST /v1/completions      - 文本补全
        GET  /health              - 健康检查
    
    使用示例:
        pulsing actor --type router --http-port 8080
    """
    
    def __init__(
        self,
        namespace: str = "dynamo",
        addr: Optional[str] = None,
        seeds: Optional[List[str]] = None,
        http_host: str = "0.0.0.0",
        http_port: int = 8080,
        model_name: str = "pulsing-model",
        scheduler_class=None,
    ):
        super().__init__(namespace=namespace, addr=addr, seeds=seeds, public=True)
        self._http_host = http_host
        self._http_port = http_port
        self._model_name = model_name
        self._scheduler_class = scheduler_class
        self._scheduler = None
        self._http_runner: Optional[web.AppRunner] = None
    
    @property
    def service_name(self) -> str:
        return "router"
    
    def _create_actor(self) -> Actor:
        return SimpleActor()
    
    async def start(self) -> ActorRef:
        actor_ref = await super().start()
        
        # 创建调度器
        from .scheduler import RoundRobinScheduler
        scheduler_class = self._scheduler_class or RoundRobinScheduler
        self._scheduler = scheduler_class(self._system)
        
        # 创建 HTTP 服务器
        from .openai_server import OpenAIServer
        server = OpenAIServer(self._system, self._model_name, self._scheduler)
        app = server.create_app()
        
        self._http_runner = web.AppRunner(app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, self._http_host, self._http_port)
        await site.start()
        return actor_ref
    
    async def stop(self):
        # 停止 HTTP 服务器
        if self._http_runner:
            await self._http_runner.cleanup()
        
        await super().stop()
