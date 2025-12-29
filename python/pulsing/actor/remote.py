"""
@as_actor 装饰器 - 类似 Ray 的分布式对象封装

用法：
    from pulsing.actor import as_actor, create_actor_system, SystemConfig

    @as_actor
    class Counter:
        def __init__(self, init_value=0):
            self.value = init_value

        def increment(self, n=1):
            self.value += n
            return self.value

    system = await create_actor_system(config)

    # 本地创建
    counter = await Counter.local(system, init_value=10)

    # 远程创建（随机选择一个远程节点）
    counter = await Counter.remote(system, init_value=10)

    # 调用方法（自动转为 Actor 消息）
    result = await counter.increment(5)  # 返回 15
"""

import asyncio
import inspect
import logging
import random
import uuid
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pulsing._core import ActorRef, ActorSystem, Message, StreamMessage

logger = logging.getLogger(__name__)


class _ActorBase(ABC):
    """Actor 基类（避免循环导入）"""

    def on_start(self, actor_id) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def metadata(self) -> dict[str, str]:
        return {}

    @abstractmethod
    async def receive(self, msg: Message) -> Message | StreamMessage | None:
        pass


T = TypeVar("T")

# 全局类注册表
_actor_class_registry: dict[str, type] = {}

# SystemActor 名称
SYSTEM_ACTOR_NAME = "_pulsing_system_actor"


class ActorProxy:
    """Actor 代理：将方法调用自动转为 ask 消息"""

    def __init__(self, actor_ref: ActorRef, method_names: list[str]):
        self._ref = actor_ref
        self._method_names = set(method_names)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"Cannot access private attribute: {name}")
        if name not in self._method_names:
            raise AttributeError(f"No method '{name}'")
        return _MethodCaller(self._ref, name)

    @property
    def ref(self) -> ActorRef:
        """获取底层 ActorRef"""
        return self._ref


class _MethodCaller:
    """方法调用器：执行远程方法调用"""

    def __init__(self, actor_ref: ActorRef, method_name: str):
        self._ref = actor_ref
        self._method = method_name

    async def __call__(self, *args, **kwargs) -> Any:
        msg = Message.from_json(
            "Call",
            {"method": self._method, "args": list(args), "kwargs": kwargs},
        )
        resp = await self._ref.ask(msg)
        data = resp.to_json()

        if resp.msg_type == "Error":
            raise RuntimeError(data.get("error", "Remote call failed"))
        return data.get("result")


class _WrappedActor(_ActorBase):
    """将用户类包装为 Actor"""

    def __init__(self, instance: Any):
        self._instance = instance

    def on_start(self, actor_id) -> None:
        if hasattr(self._instance, "on_start"):
            self._instance.on_start(actor_id)

    def on_stop(self) -> None:
        if hasattr(self._instance, "on_stop"):
            self._instance.on_stop()

    async def receive(self, msg: Message) -> Message | StreamMessage | None:
        if msg.msg_type != "Call":
            return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})

        data = msg.to_json()
        method = data.get("method")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})

        if not method or method.startswith("_"):
            return Message.from_json("Error", {"error": f"Invalid method: {method}"})

        func = getattr(self._instance, method, None)
        if func is None or not callable(func):
            return Message.from_json("Error", {"error": f"Not found: {method}"})

        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return Message.from_json("Result", {"result": result})
        except Exception as e:
            return Message.from_json("Error", {"error": str(e)})


class SystemActor(_ActorBase):
    """系统 Actor - 每个节点一个，处理远程 Actor 创建请求"""

    def __init__(self, system: ActorSystem):
        self.system = system

    async def receive(self, msg: Message) -> Message | None:
        data = msg.to_json()

        if msg.msg_type == "CreateActor":
            return await self._create_actor(data)
        return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})

    async def _create_actor(self, data: dict) -> Message:
        class_name = data.get("class_name")
        actor_name = data.get("actor_name")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})
        public = data.get("public", True)

        cls = _actor_class_registry.get(class_name)
        if cls is None:
            return Message.from_json(
                "Error", {"error": f"Class '{class_name}' not found"}
            )

        try:
            instance = cls(*args, **kwargs)
            actor = _WrappedActor(instance)
            actor_ref = await self.system.spawn(actor_name, actor, public=public)

            method_names = [
                n
                for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
                if not n.startswith("_")
            ]

            return Message.from_json(
                "Created",
                {
                    "actor_id": actor_ref.actor_id.local_id,
                    "node_id": self.system.node_id.id,
                    "methods": method_names,
                },
            )
        except Exception as e:
            logger.exception(f"Create actor failed: {e}")
            return Message.from_json("Error", {"error": str(e)})


async def _ensure_system_actor(system: ActorSystem) -> ActorRef:
    """确保本节点有 SystemActor"""
    try:
        return await system.resolve_named(SYSTEM_ACTOR_NAME)
    except Exception:
        pass

    try:
        actor = SystemActor(system)
        return await system.spawn(SYSTEM_ACTOR_NAME, actor, public=True)
    except Exception as e:
        if "already exists" in str(e).lower():
            return await system.resolve_named(SYSTEM_ACTOR_NAME)
        raise


class ActorClass:
    """Actor 类包装器"""

    def __init__(self, cls: type):
        self._cls = cls
        self._class_name = f"{cls.__module__}.{cls.__name__}"
        self._methods = [
            n
            for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not n.startswith("_")
        ]
        # 注册类
        _actor_class_registry[self._class_name] = cls

    async def local(
        self,
        system: ActorSystem,
        *args,
        name: str | None = None,
        **kwargs,
    ) -> ActorProxy:
        """本地创建 Actor"""
        await _ensure_system_actor(system)

        instance = self._cls(*args, **kwargs)
        actor = _WrappedActor(instance)
        actor_name = name or f"{self._cls.__name__}_{uuid.uuid4().hex[:8]}"
        actor_ref = await system.spawn(actor_name, actor, public=True)

        return ActorProxy(actor_ref, self._methods)

    async def remote(
        self,
        system: ActorSystem,
        *args,
        name: str | None = None,
        **kwargs,
    ) -> ActorProxy:
        """远程创建 Actor（随机选择一个远程节点）"""
        await _ensure_system_actor(system)

        members = await system.members()
        local_id = system.node_id.id

        # 过滤出远程节点
        remote_nodes = [m for m in members if int(m["node_id"]) != local_id]

        if not remote_nodes:
            # 没有远程节点，回退到本地创建
            logger.warning("No remote nodes, fallback to local")
            return await self.local(system, *args, name=name, **kwargs)

        # 随机选一个
        target = random.choice(remote_nodes)
        target_id = int(target["node_id"])

        # 获取目标节点的 SystemActor
        system_actor = await system.resolve_named(SYSTEM_ACTOR_NAME, node_id=target_id)

        actor_name = name or f"{self._cls.__name__}_{uuid.uuid4().hex[:8]}"

        # 发送创建请求
        resp = await system_actor.ask(
            Message.from_json(
                "CreateActor",
                {
                    "class_name": self._class_name,
                    "actor_name": actor_name,
                    "args": list(args),
                    "kwargs": kwargs,
                    "public": True,
                },
            )
        )

        data = resp.to_json()
        if resp.msg_type == "Error":
            raise RuntimeError(f"Remote create failed: {data.get('error')}")

        # 构建远程 ActorRef
        from pulsing._core import ActorId, NodeId

        remote_id = ActorId(data["actor_id"], NodeId(data["node_id"]))
        actor_ref = await system.actor_ref(remote_id)

        return ActorProxy(actor_ref, data.get("methods", self._methods))

    def __call__(self, *args, **kwargs):
        """直接调用返回本地实例（非 Actor）"""
        return self._cls(*args, **kwargs)


def as_actor(cls: type[T]) -> ActorClass:
    """@as_actor 装饰器

    将普通类转换为可分布式部署的 Actor。

    Example:
        @as_actor
        class Counter:
            def __init__(self, init_value=0):
                self.value = init_value

            def increment(self, n=1):
                self.value += n
                return self.value

        # 本地创建
        counter = await Counter.local(system, init_value=10)

        # 远程创建
        counter = await Counter.remote(system, init_value=10)

        # 调用
        result = await counter.increment(5)
    """
    return ActorClass(cls)


# 保留 remote 作为别名（向后兼容）
remote = as_actor
RemoteClass = ActorClass
