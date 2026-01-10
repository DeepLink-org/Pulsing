"""
PulsingRuntime - 基于 Pulsing 的 AutoGen 兼容运行时

特点:
- 单机/分布式统一 API
- 无需中央协调器，Gossip 自动发现
- 内置负载均衡和故障转移
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import signal
import uuid
from asyncio import Future, Task
from collections import defaultdict
from typing import (
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    Dict,
    Mapping,
    Sequence,
    Set,
    Type,
    TypeVar,
    cast,
)

from pulsing.actor import (
    Actor,
    ActorRef,
    ActorSystem,
    Message,
    SystemConfig,
    create_actor_system,
)

logger = logging.getLogger("pulsing.autogen")
T = TypeVar("T")


class PulsingRuntime:
    """基于 Pulsing 的 AutoGen 兼容运行时

    特点:
    - 单机/分布式统一 API
    - 无需中央协调器
    - Gossip 自动发现
    - 内置负载均衡和故障转移

    Usage:
        # 单机模式
        runtime = PulsingRuntime()

        # 分布式模式 - 第一个节点
        runtime = PulsingRuntime(addr="0.0.0.0:8000")

        # 分布式模式 - 加入集群
        runtime = PulsingRuntime(
            addr="0.0.0.0:8000",
            seeds=["first-node:8000"]
        )
    """

    def __init__(
        self,
        *,
        addr: str | None = None,
        seeds: list[str] | None = None,
    ) -> None:
        """
        Args:
            addr: 本节点监听地址，None 表示单机模式
            seeds: 种子节点地址列表，用于加入集群
        """
        self._addr = addr
        self._seeds = seeds or []

        # Pulsing ActorSystem
        self._system: ActorSystem | None = None

        # Agent 管理
        self._agent_factories: Dict[str, Callable[[], Any | Awaitable[Any]]] = {}
        self._instantiated_agents: Dict[str, Any] = {}  # agent_key -> Agent instance
        self._agent_refs: Dict[str, ActorRef] = {}  # agent_key -> ActorRef

        # 订阅管理: topic_key -> set of agent_types
        self._subscriptions: DefaultDict[str, Set[str]] = defaultdict(set)

        # 运行状态
        self._running = False
        self._run_task: Task | None = None
        self._pending_requests: Dict[str, Future[Any]] = {}
        self._background_tasks: Set[Task[Any]] = set()

    @property
    def is_distributed(self) -> bool:
        """是否为分布式模式"""
        return self._addr is not None

    async def start(self) -> None:
        """启动运行时"""
        if self._running:
            raise RuntimeError("Runtime is already running")

        # 创建 Pulsing ActorSystem
        if self._addr:
            # 分布式模式
            config = SystemConfig.with_addr(self._addr)
            if self._seeds:
                config = config.with_seeds(self._seeds)
        else:
            # 单机模式
            config = SystemConfig.standalone()

        self._system = await create_actor_system(config)
        self._running = True

        mode = "distributed" if self.is_distributed else "standalone"
        logger.info(f"PulsingRuntime started in {mode} mode")
        if self._addr:
            logger.info(f"Listening on {self._addr}, seeds: {self._seeds}")

    async def stop(self) -> None:
        """停止运行时"""
        if not self._running:
            raise RuntimeError("Runtime is not running")

        self._running = False

        # 等待所有后台任务完成
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("PulsingRuntime stopped")

    async def stop_when_idle(self) -> None:
        """空闲时停止运行时"""
        # 等待所有待处理请求完成
        while self._pending_requests:
            await asyncio.sleep(0.1)
        await self.stop()

    async def stop_when_signal(
        self, signals: Sequence[signal.Signals] = (signal.SIGTERM, signal.SIGINT)
    ) -> None:
        """收到信号时停止"""
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def signal_handler():
            logger.info("Received exit signal, shutting down...")
            shutdown_event.set()

        for sig in signals:
            loop.add_signal_handler(sig, signal_handler)

        await shutdown_event.wait()
        await self.stop()

    # ========================================================================
    # AutoGen AgentRuntime 协议实现
    # ========================================================================

    async def send_message(
        self,
        message: Any,
        recipient: Any,  # AgentId
        *,
        sender: Any | None = None,
        cancellation_token: Any | None = None,
        message_id: str | None = None,
    ) -> Any:
        """发送消息到指定 Agent (RPC 模式)"""
        if not self._running:
            raise RuntimeError("Runtime is not running")

        # 解析 recipient
        agent_type = recipient.type if hasattr(recipient, "type") else str(recipient)
        agent_key = recipient.key if hasattr(recipient, "key") else "default"
        full_key = f"{agent_type}/{agent_key}"

        # 确保 Agent 已创建
        await self._ensure_agent(agent_type, agent_key)

        # 获取 ActorRef
        actor_ref = self._agent_refs.get(full_key)
        if actor_ref is None:
            raise LookupError(f"Agent '{full_key}' not found")

        # 构造消息
        msg_id = message_id or str(uuid.uuid4())
        envelope = {
            "__autogen_msg__": True,
            "payload": message,
            "sender": {
                "type": sender.type if sender else None,
                "key": sender.key if sender else None,
            }
            if sender
            else None,
            "topic_id": None,
            "is_rpc": True,
            "message_id": msg_id,
        }

        # 发送并等待响应
        response = await actor_ref.ask(envelope)

        # 处理响应
        response = self._deserialize_response(response)

        # 如果是我们的 AutoGen 响应格式
        if isinstance(response, dict) and "__autogen_response__" in response:
            if "__error__" in response:
                raise RuntimeError(response["__error__"])
            return response.get("result")

        # 直接返回
        return response

    def _deserialize_response(self, response: Any) -> Any:
        """反序列化响应"""
        import pickle

        # 1. 如果是 Pulsing Message 对象
        if hasattr(response, "msg_type") and hasattr(response, "payload"):
            msg_type = response.msg_type
            payload = response.payload

            # 如果 msg_type 为空，说明是 pickled Python 对象
            if not msg_type and isinstance(payload, bytes):
                try:
                    return pickle.loads(payload)
                except Exception:
                    pass

            # 如果有 msg_type，尝试 to_json
            if hasattr(response, "to_json"):
                try:
                    return response.to_json()
                except Exception:
                    pass

        # 2. 如果是 bytes，尝试 unpickle
        if isinstance(response, bytes):
            try:
                return pickle.loads(response)
            except Exception:
                pass

        # 3. 直接返回
        return response

    async def publish_message(
        self,
        message: Any,
        topic_id: Any,  # TopicId
        *,
        sender: Any | None = None,
        cancellation_token: Any | None = None,
        message_id: str | None = None,
    ) -> None:
        """发布消息到 Topic (Pub/Sub 模式)"""
        if not self._running:
            raise RuntimeError("Runtime is not running")

        # 解析 topic
        topic_type = topic_id.type if hasattr(topic_id, "type") else str(topic_id)
        topic_source = topic_id.source if hasattr(topic_id, "source") else "default"
        topic_key = f"{topic_type}/{topic_source}"

        # 获取订阅者
        subscriber_types = self._subscriptions.get(topic_key, set())

        # 构造消息
        msg_id = message_id or str(uuid.uuid4())
        envelope = {
            "__autogen_msg__": True,
            "payload": message,
            "sender": {
                "type": sender.type if sender else None,
                "key": sender.key if sender else None,
            }
            if sender
            else None,
            "topic_id": {
                "type": topic_type,
                "source": topic_source,
            },
            "is_rpc": False,
            "message_id": msg_id,
        }

        # 发送给所有订阅者
        tasks = []
        sender_key = f"{sender.type}/{sender.key}" if sender else None

        for agent_type in subscriber_types:
            agent_key = "default"  # 默认 key
            full_key = f"{agent_type}/{agent_key}"

            # 不发给自己
            if full_key == sender_key:
                continue

            # 确保 Agent 已创建
            await self._ensure_agent(agent_type, agent_key)

            actor_ref = self._agent_refs.get(full_key)
            if actor_ref:
                # 使用 tell (不等待响应)
                task = asyncio.create_task(actor_ref.ask(envelope))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def register_factory(
        self,
        type: str | Any,  # str | AgentType
        agent_factory: Callable[[], T | Awaitable[T]],
        *,
        expected_class: type[T] | None = None,
        eager: bool = True,  # 立即创建 Agent 实例
    ) -> Any:  # AgentType
        """注册 Agent 工厂

        Args:
            type: Agent 类型名称
            agent_factory: 创建 Agent 的工厂函数
            expected_class: 预期的 Agent 类型 (用于验证)
            eager: 是否立即创建 Agent 实例 (分布式模式需要 True)
        """
        agent_type = type.type if hasattr(type, "type") else str(type)

        if agent_type in self._agent_factories:
            raise ValueError(f"Agent type '{agent_type}' already registered")

        self._agent_factories[agent_type] = agent_factory
        logger.debug(f"Registered agent factory: {agent_type}")

        # 立即创建实例（分布式模式下必须，否则其他节点无法发现）
        if eager and self._running:
            await self._ensure_agent(agent_type, "default")

        # 返回 AgentType (兼容 AutoGen)
        try:
            from autogen_core import AgentType as AutoGenAgentType

            return AutoGenAgentType(agent_type)
        except ImportError:
            return agent_type

    async def register_agent_instance(
        self,
        agent_instance: Any,  # Agent
        agent_id: Any,  # AgentId
    ) -> Any:  # AgentId
        """注册 Agent 实例"""
        agent_type = agent_id.type if hasattr(agent_id, "type") else str(agent_id)
        agent_key = agent_id.key if hasattr(agent_id, "key") else "default"
        full_key = f"{agent_type}/{agent_key}"

        if full_key in self._instantiated_agents:
            raise ValueError(f"Agent '{full_key}' already exists")

        # 绑定运行时
        if hasattr(agent_instance, "bind_id_and_runtime"):
            await agent_instance.bind_id_and_runtime(id=agent_id, runtime=self)

        # 创建包装器并 spawn
        wrapper = AutoGenAgentWrapper(agent_instance, self)

        actor_ref = await self._system.spawn(full_key, wrapper, public=True)

        self._instantiated_agents[full_key] = agent_instance
        self._agent_refs[full_key] = actor_ref

        logger.debug(f"Registered agent instance: {full_key}")
        return agent_id

    async def add_subscription(self, subscription: Any) -> None:
        """添加订阅"""
        # 解析订阅
        if hasattr(subscription, "topic_type"):
            topic_type = subscription.topic_type
        elif hasattr(subscription, "id"):
            topic_type = subscription.id
        else:
            topic_type = "default"

        if hasattr(subscription, "agent_type"):
            agent_type = subscription.agent_type
        else:
            agent_type = str(subscription)

        # 使用 "default" 作为默认 source
        topic_key = f"{topic_type}:default"
        self._subscriptions[topic_key].add(agent_type)

        logger.debug(f"Added subscription: {agent_type} -> {topic_key}")

    async def remove_subscription(self, id: str) -> None:
        """移除订阅"""
        # 简化实现：遍历查找
        for topic_key, agent_types in self._subscriptions.items():
            agent_types.discard(id)

    async def get(
        self,
        id_or_type: Any,  # AgentId | AgentType | str
        /,
        key: str = "default",
        *,
        lazy: bool = True,
    ) -> Any:  # AgentId
        """获取 Agent ID"""
        if hasattr(id_or_type, "type") and hasattr(id_or_type, "key"):
            # 已经是 AgentId
            return id_or_type

        agent_type = id_or_type.type if hasattr(id_or_type, "type") else str(id_or_type)

        if not lazy:
            await self._ensure_agent(agent_type, key)

        try:
            from autogen_core import AgentId as AutoGenAgentId

            return AutoGenAgentId(agent_type, key)
        except ImportError:
            return f"{agent_type}:{key}"

    async def try_get_underlying_agent_instance(
        self,
        id: Any,  # AgentId
        type: Type[T] = object,  # type: ignore
    ) -> T:
        """获取底层 Agent 实例"""
        agent_type = id.type if hasattr(id, "type") else str(id)
        agent_key = id.key if hasattr(id, "key") else "default"
        full_key = f"{agent_type}/{agent_key}"

        if full_key not in self._instantiated_agents:
            await self._ensure_agent(agent_type, agent_key)

        instance = self._instantiated_agents.get(full_key)
        if instance is None:
            raise LookupError(f"Agent '{full_key}' not found")

        if not isinstance(instance, type):
            raise TypeError(f"Agent is not of type {type.__name__}")

        return cast(T, instance)

    async def agent_metadata(self, agent: Any) -> Any:
        """获取 Agent 元数据"""
        instance = await self.try_get_underlying_agent_instance(agent)
        if hasattr(instance, "metadata"):
            return instance.metadata
        return {}

    async def save_state(self) -> Mapping[str, Any]:
        """保存运行时状态"""
        state = {}
        for key, agent in self._instantiated_agents.items():
            if hasattr(agent, "save_state"):
                state[key] = dict(await agent.save_state())
        return state

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """加载运行时状态"""
        for key, agent_state in state.items():
            if key in self._instantiated_agents:
                agent = self._instantiated_agents[key]
                if hasattr(agent, "load_state"):
                    await agent.load_state(agent_state)

    async def agent_save_state(self, agent: Any) -> Mapping[str, Any]:
        """保存单个 Agent 状态"""
        instance = await self.try_get_underlying_agent_instance(agent)
        if hasattr(instance, "save_state"):
            return await instance.save_state()
        return {}

    async def agent_load_state(self, agent: Any, state: Mapping[str, Any]) -> None:
        """加载单个 Agent 状态"""
        instance = await self.try_get_underlying_agent_instance(agent)
        if hasattr(instance, "load_state"):
            await instance.load_state(state)

    def add_message_serializer(self, serializer: Any) -> None:
        """添加消息序列化器 (Pulsing 使用 pickle，此方法为兼容性保留)"""
        pass

    # ========================================================================
    # 内部方法
    # ========================================================================

    async def _ensure_agent(self, agent_type: str, agent_key: str = "default") -> None:
        """确保 Agent 可用 (本地或远程)"""
        full_key = f"{agent_type}/{agent_key}"

        # 1. 检查是否已有本地引用
        if full_key in self._agent_refs:
            return

        # 2. 尝试在集群中查找远程 Agent
        if self.is_distributed:
            try:
                actor_ref = await self._system.resolve_named(full_key)
                if actor_ref:
                    self._agent_refs[full_key] = actor_ref
                    logger.debug(f"Found remote agent: {full_key}")
                    return
            except Exception as e:
                logger.debug(f"Agent '{full_key}' not found in cluster: {e}")

        # 3. 如果本地有工厂，则创建本地实例
        if agent_type not in self._agent_factories:
            raise LookupError(f"Agent type '{agent_type}' not registered")

        # 调用工厂创建 Agent
        factory = self._agent_factories[agent_type]
        agent = factory()
        if inspect.isawaitable(agent):
            agent = await agent

        # 创建 AgentId
        try:
            from autogen_core import AgentId as AutoGenAgentId

            agent_id = AutoGenAgentId(agent_type, agent_key)
        except ImportError:
            agent_id = full_key

        # 绑定运行时
        if hasattr(agent, "bind_id_and_runtime"):
            await agent.bind_id_and_runtime(id=agent_id, runtime=self)

        # 创建包装器并 spawn
        wrapper = AutoGenAgentWrapper(agent, self)

        actor_ref = await self._system.spawn(full_key, wrapper, public=True)

        self._instantiated_agents[full_key] = agent
        self._agent_refs[full_key] = actor_ref

        logger.debug(f"Created local agent: {full_key}")


# 导入包装器
from .agent_wrapper import AutoGenAgentWrapper
