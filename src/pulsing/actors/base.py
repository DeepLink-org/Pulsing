"""
Base Service Actor - 服务 Actor 基类
"""

import asyncio
import signal
from abc import ABC, abstractmethod
from typing import Optional, List

from pulsing.actor import ActorSystem, SystemConfig, Actor, ActorRef


class BaseServiceActor(ABC):
    """
    服务 Actor 基类
    
    子类需要实现：
    - service_name: 服务名称
    - _create_actor(): 创建 Actor 实例
    """
    
    def __init__(
        self,
        namespace: str = "dynamo",
        addr: Optional[str] = None,
        seeds: Optional[List[str]] = None,
        public: bool = True,
    ):
        self.namespace = namespace
        self.addr = addr
        self.seeds = seeds or []
        self.public = public
        
        self._system: Optional[ActorSystem] = None
        self._actor_ref: Optional[ActorRef] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    @property
    @abstractmethod
    def service_name(self) -> str:
        pass
    
    @abstractmethod
    def _create_actor(self) -> Actor:
        pass
    
    def _build_config(self) -> SystemConfig:
        config = SystemConfig.with_addr(self.addr) if self.addr else SystemConfig.standalone()
        if self.seeds:
            config = config.with_seeds(self.seeds)
        return config
    
    async def start(self) -> ActorRef:
        if self._running:
            raise RuntimeError("Already running")
        
        config = self._build_config()
        self._system = await ActorSystem.create(config)
        
        actor = self._create_actor()
        self._actor_ref = await self._system.spawn(
            self.service_name, actor, public=self.public
        )
        
        self._running = True
        print(f"[{self.service_name}] Started at {self._system.addr}")
        return self._actor_ref
    
    async def stop(self):
        if not self._running:
            return
        
        self._running = False
        
        if self._system:
            try:
                await self._system.stop(self.service_name)
            except Exception:
                pass
            try:
                await self._system.shutdown()
            except Exception:
                pass
        
        self._shutdown_event.set()
        print(f"[{self.service_name}] Stopped")
    
    async def run(self):
        """运行服务直到收到关闭信号"""
        await self.start()
        
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        
        await self._shutdown_event.wait()
    
    @property
    def actor_ref(self) -> Optional[ActorRef]:
        return self._actor_ref
    
    @property
    def system(self) -> Optional[ActorSystem]:
        return self._system
    
    @property
    def is_running(self) -> bool:
        return self._running
