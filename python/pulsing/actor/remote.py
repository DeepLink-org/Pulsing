"""
@remote decorator - Ray-like distributed object wrapper

Usage:
    from pulsing.actor import init, shutdown, remote

    @remote
    class Counter:
        def __init__(self, init_value=0):
            self.value = init_value

        def increment(self, n=1):
            self.value += n
            return self.value

    async def main():
        await init()

        # Create actor
        counter = await Counter.spawn(init_value=10)

        # Call methods (automatically converted to actor messages)
        result = await counter.increment(5)  # Returns 15

        await shutdown()
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
    """Actor base class (avoids circular imports)"""

    def on_start(self, actor_id) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def metadata(self) -> dict[str, str]:
        return {}

    @abstractmethod
    async def receive(self, msg) -> Any:
        """Handle incoming message. Can receive and return any Python object."""
        pass


T = TypeVar("T")

# Global class registry
_actor_class_registry: dict[str, type] = {}

# Actor instance metadata registry (actor_name -> metadata)
_actor_metadata_registry: dict[str, dict[str, str]] = {}


def _register_actor_metadata(name: str, cls: type):
    """Register actor metadata for later retrieval"""
    import inspect

    metadata = {
        "python_class": f"{cls.__module__}.{cls.__name__}",
        "python_module": cls.__module__,
    }

    # Try to get source file
    try:
        source_file = inspect.getfile(cls)
        metadata["python_file"] = source_file
    except (TypeError, OSError):
        pass

    _actor_metadata_registry[name] = metadata


def get_actor_metadata(name: str) -> dict[str, str] | None:
    """Get metadata for an actor by name"""
    return _actor_metadata_registry.get(name)


# Python actor service name (different from Rust SystemActor "system/core")
PYTHON_ACTOR_SERVICE_NAME = "_python_actor_service"


class ActorProxy:
    """Actor proxy: automatically converts method calls to ask messages

    支持两种方法类型：
    - 普通方法: 同步 ask/response
    - async 方法: 流式响应，不阻塞 actor
    """

    def __init__(
        self,
        actor_ref: ActorRef,
        method_names: list[str] | None = None,
        async_methods: set[str] | None = None,
    ):
        self._ref = actor_ref
        self._method_names = set(method_names) if method_names else None
        self._async_methods = async_methods or set()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"Cannot access private attribute: {name}")
        # 如果没有方法列表，允许任意方法调用（动态模式）
        if self._method_names is not None and name not in self._method_names:
            raise AttributeError(f"No method '{name}'")
        is_async = name in self._async_methods
        return _MethodCaller(self._ref, name, is_async=is_async)

    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef"""
        return self._ref

    @classmethod
    def from_ref(
        cls,
        actor_ref: ActorRef,
        methods: list[str] | None = None,
        async_methods: set[str] | None = None,
    ) -> "ActorProxy":
        """从 ActorRef 创建 ActorProxy

        Args:
            actor_ref: 底层 actor 引用
            methods: 可选的方法名列表。如果不提供，允许调用任意方法（动态模式）
            async_methods: async 方法名集合，这些方法使用流式响应

        Example:
            # 动态模式 - 允许任意方法调用
            ref = await system.resolve_named("my_counter")
            proxy = ActorProxy.from_ref(ref)
            await proxy.increment(5)  # 可调用任意方法

            # 静态模式 - 只允许指定的方法
            proxy = ActorProxy.from_ref(ref, methods=["increment", "get_value"])

            # 带 async 方法标记
            proxy = ActorProxy.from_ref(
                ref,
                methods=["get", "generate"],
                async_methods={"generate"}
            )
        """
        return cls(actor_ref, methods, async_methods)


class _MethodCaller:
    """Method caller: executes remote method calls"""

    def __init__(self, actor_ref: ActorRef, method_name: str, is_async: bool = False):
        self._ref = actor_ref
        self._method = method_name
        self._is_async = is_async

    def __call__(self, *args, **kwargs):
        if self._is_async:
            # 返回一个可以被 await 或 async for 的对象
            return _AsyncMethodCall(self._ref, self._method, args, kwargs)
        else:
            # 返回一个 coroutine 用于同步方法
            return self._sync_call(*args, **kwargs)

    async def _sync_call(self, *args, **kwargs) -> Any:
        """同步方法调用"""
        call_msg = {
            "__call__": self._method,
            "args": args,
            "kwargs": kwargs,
            "__async__": False,
        }
        resp = await self._ref.ask(call_msg)

        # Handle normal response
        if isinstance(resp, dict):
            if "__error__" in resp:
                raise RuntimeError(resp["__error__"])
            return resp.get("__result__")
        elif isinstance(resp, Message):
            # Fallback for Rust actor communication
            data = resp.to_json()
            if resp.msg_type == "Error":
                raise RuntimeError(data.get("error", "Remote call failed"))
            return data.get("result")
        return resp


class _AsyncMethodCall:
    """异步方法调用 - 支持 await 和 async for

    用法:
        # 直接 await 获取最终结果
        result = await service.generate("hello")

        # 流式获取中间结果
        async for chunk in service.generate("hello"):
            print(chunk)
    """

    def __init__(
        self, actor_ref: ActorRef, method_name: str, args: tuple, kwargs: dict
    ):
        self._ref = actor_ref
        self._method = method_name
        self._args = args
        self._kwargs = kwargs
        self._stream_reader = None
        self._final_result = None
        self._got_result = False

    async def _get_stream(self):
        """获取流（惰性初始化）"""
        if self._stream_reader is None:
            call_msg = {
                "__call__": self._method,
                "args": self._args,
                "kwargs": self._kwargs,
                "__async__": True,
            }
            resp = await self._ref.ask(call_msg)

            # 响应可能是 PyMessage（流式）或直接的 Python 对象
            if isinstance(resp, Message):
                # 检查是否是流式消息
                if resp.is_stream:
                    self._stream_reader = resp.stream_reader()
                else:
                    # 不是流式，可能是错误
                    data = resp.to_json()
                    if resp.msg_type == "Error":
                        raise RuntimeError(data.get("error", "Remote call failed"))
                    # 包装为单次迭代器
                    self._stream_reader = _SingleValueIterator(data)
            else:
                # 普通 Python 对象（可能是 dict）
                self._stream_reader = _SingleValueIterator(resp)

        return self._stream_reader

    def __aiter__(self):
        """支持异步迭代，获取中间结果"""
        return self

    async def __anext__(self):
        """获取下一个流式数据"""
        reader = await self._get_stream()
        try:
            item = await reader.__anext__()
            # 检查是否是最终结果
            if isinstance(item, dict):
                if "__final__" in item:
                    self._final_result = item.get("__result__")
                    self._got_result = True
                    raise StopAsyncIteration
                if "__error__" in item:
                    raise RuntimeError(item["__error__"])
                if "__yield__" in item:
                    return item["__yield__"]
            return item
        except StopAsyncIteration:
            raise

    def __await__(self):
        """支持 await，获取最终结果"""
        return self._await_result().__await__()

    async def _await_result(self):
        """消费整个流，返回最终结果"""
        async for _ in self:
            pass  # 消费所有 yield 的中间值
        if self._got_result:
            return self._final_result
        return None


class _SingleValueIterator:
    """单值异步迭代器 - 将单个值包装为异步迭代器"""

    def __init__(self, value):
        self._value = value
        self._consumed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._consumed:
            raise StopAsyncIteration
        self._consumed = True
        return self._value


class _WrappedActor(_ActorBase):
    """Wraps user class as an Actor"""

    def __init__(self, instance: Any):
        self._instance = instance
        # Store original class info for metadata extraction
        self._original_class = instance.__class__

    @property
    def __original_module__(self):
        """Return original class module for Rust metadata extraction"""
        return self._original_class.__module__

    @property
    def __original_qualname__(self):
        """Return original class qualified name for Rust metadata extraction"""
        return self._original_class.__qualname__

    @property
    def __original_file__(self):
        """Return original class file path for Rust metadata extraction"""
        try:
            return inspect.getfile(self._original_class)
        except (TypeError, OSError):
            return None

    def on_start(self, actor_id) -> None:
        if hasattr(self._instance, "on_start"):
            self._instance.on_start(actor_id)

    def on_stop(self) -> None:
        if hasattr(self._instance, "on_stop"):
            self._instance.on_stop()

    async def receive(self, msg) -> Any:
        # Handle new dict-based call format (Python-to-Python)
        if isinstance(msg, dict) and "__call__" in msg:
            method = msg["__call__"]
            args = msg.get("args", ())
            kwargs = msg.get("kwargs", {})
            is_async_call = msg.get("__async__", False)

            if not method or method.startswith("_"):
                return {"__error__": f"Invalid method: {method}"}

            func = getattr(self._instance, method, None)
            if func is None or not callable(func):
                return {"__error__": f"Not found: {method}"}

            # 检测是否是 async 方法（包括异步生成器）
            is_async_method = (
                inspect.iscoroutinefunction(func)
                or inspect.isasyncgenfunction(func)
                or (
                    hasattr(func, "__func__")
                    and (
                        inspect.iscoroutinefunction(func.__func__)
                        or inspect.isasyncgenfunction(func.__func__)
                    )
                )
            )

            # 对于 async 方法，使用流式响应
            if is_async_method and is_async_call:
                return self._handle_async_method(func, args, kwargs)

            # 普通方法或未标记为 async 调用的情况
            try:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                return {"__result__": result}
            except Exception as e:
                return {"__error__": str(e)}

        # Handle legacy Message-based call format (for Rust actor compatibility)
        if isinstance(msg, Message):
            if msg.msg_type != "Call":
                return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})

            data = msg.to_json()
            method = data.get("method")
            args = data.get("args", [])
            kwargs = data.get("kwargs", {})

            if not method or method.startswith("_"):
                return Message.from_json(
                    "Error", {"error": f"Invalid method: {method}"}
                )

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

        return {"__error__": f"Unknown message type: {type(msg)}"}

    def _handle_async_method(self, func, args, kwargs) -> StreamMessage:
        """处理 async 方法，返回流式响应"""
        stream_msg, writer = StreamMessage.create("AsyncMethodStream")

        async def execute():
            try:
                result = func(*args, **kwargs)

                # 检查结果类型
                if inspect.isasyncgen(result):
                    # 异步生成器
                    async for item in result:
                        await writer.write({"__yield__": item})
                    await writer.write({"__final__": True, "__result__": None})
                elif asyncio.iscoroutine(result):
                    # 普通 async 函数
                    final_result = await result
                    await writer.write({"__final__": True, "__result__": final_result})
                elif inspect.isgenerator(result):
                    # 同步生成器
                    for item in result:
                        await writer.write({"__yield__": item})
                    await writer.write({"__final__": True, "__result__": None})
                else:
                    # 普通返回值
                    await writer.write({"__final__": True, "__result__": result})
            except Exception as e:
                await writer.write({"__error__": str(e)})
            finally:
                await writer.close()

        # 在后台任务中执行，不阻塞 actor
        asyncio.create_task(execute())
        return stream_msg


class PythonActorService(_ActorBase):
    """Python Actor creation service - one per node, handles Python actor creation requests.

    Note: Rust SystemActor (path "system/core") handles system-level operations,
    this service specifically handles Python actor creation.
    """

    def __init__(self, system: ActorSystem):
        self.system = system

    async def receive(self, msg: Message) -> Message | None:
        data = msg.to_json()

        if msg.msg_type == "CreateActor":
            return await self._create_actor(data)
        elif msg.msg_type == "ListRegistry":
            # List registered actor classes
            return Message.from_json(
                "Registry",
                {"classes": list(_actor_class_registry.keys())},
            )
        return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})

    async def _create_actor(self, data: dict) -> Message:
        class_name = data.get("class_name")
        actor_name = data.get("actor_name")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})
        public = data.get("public", True)

        # Supervision config
        restart_policy = data.get("restart_policy", "never")
        max_restarts = data.get("max_restarts", 3)
        min_backoff = data.get("min_backoff", 0.1)
        max_backoff = data.get("max_backoff", 30.0)

        cls = _actor_class_registry.get(class_name)
        if cls is None:
            return Message.from_json(
                "Error", {"error": f"Class '{class_name}' not found"}
            )

        try:
            if restart_policy != "never":
                # For supervision, we must provide a factory
                def factory():
                    instance = cls(*args, **kwargs)
                    return _WrappedActor(instance)

                actor_ref = await self.system.spawn(
                    actor_name,
                    factory,
                    public=public,
                    restart_policy=restart_policy,
                    max_restarts=max_restarts,
                    min_backoff=min_backoff,
                    max_backoff=max_backoff,
                )
            else:
                # Standard spawn
                instance = cls(*args, **kwargs)
                actor = _WrappedActor(instance)
                actor_ref = await self.system.spawn(actor_name, actor, public=public)

            # Register actor metadata
            _register_actor_metadata(actor_name, cls)

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


class ActorClass:
    """Actor class wrapper

    Provides two ways to create actors:

    1. Simple API (uses global system):
        await init()
        counter = await Counter.spawn(init=10)

    2. Explicit system:
        system = await create_actor_system(config)
        counter = await Counter.local(system, init=10)
    """

    def __init__(
        self,
        cls: type,
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0,
    ):
        self._cls = cls
        self._class_name = f"{cls.__module__}.{cls.__name__}"
        self._restart_policy = restart_policy
        self._max_restarts = max_restarts
        self._min_backoff = min_backoff
        self._max_backoff = max_backoff

        # 收集所有公开方法
        self._methods = []
        self._async_methods = set()

        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            self._methods.append(name)
            # 检测是否是 async 方法（包括 async 函数和异步生成器）
            if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(
                method
            ):
                self._async_methods.add(name)

        # Register class
        _actor_class_registry[self._class_name] = cls

    async def spawn(
        self,
        *args,
        name: str | None = None,
        **kwargs,
    ) -> ActorProxy:
        """Create actor using global system (simple API)

        Must call `await init()` before using this method.

        Example:
            from pulsing.actor import init, remote

            await init()

            @remote
            class Counter:
                def __init__(self, init=0): self.value = init
                def incr(self): self.value += 1; return self.value

            counter = await Counter.spawn(init=10)
            result = await counter.incr()
        """
        # Import here to avoid circular import
        from . import _global_system

        if _global_system is None:
            raise RuntimeError(
                "Actor system not initialized. Call 'await init()' first."
            )

        return await self.local(_global_system, *args, name=name, **kwargs)

    async def local(
        self,
        system: ActorSystem,
        *args,
        name: str | None = None,
        **kwargs,
    ) -> ActorProxy:
        """Create actor locally with explicit system.

        Note: Use create_actor_system() to create ActorSystem,
        which automatically registers PythonActorService.
        """
        actor_name = name or f"{self._cls.__name__}_{uuid.uuid4().hex[:8]}"

        if self._restart_policy != "never":

            def factory():
                instance = self._cls(*args, **kwargs)
                return _WrappedActor(instance)

            actor_ref = await system.spawn(
                actor_name,
                factory,
                public=True,
                restart_policy=self._restart_policy,
                max_restarts=self._max_restarts,
                min_backoff=self._min_backoff,
                max_backoff=self._max_backoff,
            )
        else:
            instance = self._cls(*args, **kwargs)
            actor = _WrappedActor(instance)
            actor_ref = await system.spawn(actor_name, actor, public=True)

        # Register actor metadata
        _register_actor_metadata(actor_name, self._cls)

        return ActorProxy(actor_ref, self._methods, self._async_methods)

    async def remote(
        self,
        system: ActorSystem,
        *args,
        name: str | None = None,
        **kwargs,
    ) -> ActorProxy:
        """Create actor remotely (randomly selects a remote node).

        Note: Use create_actor_system() to create ActorSystem,
        which automatically registers PythonActorService.
        """

        members = await system.members()
        local_id = system.node_id.id

        # Filter out remote nodes
        remote_nodes = [m for m in members if int(m["node_id"]) != local_id]

        if not remote_nodes:
            # No remote nodes, fallback to local creation
            logger.warning("No remote nodes, fallback to local")
            return await self.local(system, *args, name=name, **kwargs)

        # Randomly select one
        target = random.choice(remote_nodes)
        target_id = int(target["node_id"])

        # Get target node's Python actor creation service
        service_ref = await system.resolve_named(
            PYTHON_ACTOR_SERVICE_NAME, node_id=target_id
        )

        actor_name = name or f"{self._cls.__name__}_{uuid.uuid4().hex[:8]}"

        # Send creation request
        resp = await service_ref.ask(
            Message.from_json(
                "CreateActor",
                {
                    "class_name": self._class_name,
                    "actor_name": actor_name,
                    "args": list(args),
                    "kwargs": kwargs,
                    "public": True,
                    # Supervision config
                    "restart_policy": self._restart_policy,
                    "max_restarts": self._max_restarts,
                    "min_backoff": self._min_backoff,
                    "max_backoff": self._max_backoff,
                },
            )
        )

        data = resp.to_json()
        if resp.msg_type == "Error":
            raise RuntimeError(f"Remote create failed: {data.get('error')}")

        # Build remote ActorRef
        from pulsing._core import ActorId, NodeId

        remote_id = ActorId(data["actor_id"], NodeId(data["node_id"]))
        actor_ref = await system.actor_ref(remote_id)

        return ActorProxy(
            actor_ref, data.get("methods", self._methods), self._async_methods
        )

    def __call__(self, *args, **kwargs):
        """Direct call returns local instance (not an Actor)"""
        return self._cls(*args, **kwargs)

    def proxy(self, actor_ref: ActorRef) -> ActorProxy:
        """将 ActorRef 包装成带类型的 ActorProxy

        Args:
            actor_ref: 底层 actor 引用

        Returns:
            ActorProxy: 带方法类型信息的代理

        Example:
            ref = await system.resolve_named("my_counter")
            counter = Counter.proxy(ref)
            await counter.increment()
        """
        return ActorProxy(actor_ref, self._methods, self._async_methods)

    async def resolve(
        self,
        name: str,
        *,
        system: ActorSystem | None = None,
        node_id: int | None = None,
    ) -> ActorProxy:
        """通过名字解析 actor，返回带类型的 ActorProxy

        Args:
            name: actor 名字
            system: ActorSystem 实例，如果不提供则使用全局 system
            node_id: 目标节点 ID，如果不提供则在集群中查找

        Returns:
            ActorProxy: 带方法类型信息的代理

        Example:
            @remote
            class Counter:
                def __init__(self, init=0): self.value = init
                async def generate(self, prompt): ...  # async 方法，流式响应

            # 节点 A 创建 actor
            counter = await Counter.spawn(name="my_counter")

            # 节点 B 解析并调用
            counter = await Counter.resolve("my_counter")

            # 调用 async 方法，可以流式获取结果
            result = counter.generate("hello")
            async for chunk in result:
                print(chunk)
            # 或者直接 await 获取最终结果
            final = await counter.generate("hello")
        """
        from . import _global_system

        if system is None:
            if _global_system is None:
                raise RuntimeError(
                    "Actor system not initialized. Call 'await init()' first."
                )
            system = _global_system

        actor_ref = await system.resolve_named(name, node_id=node_id)
        return ActorProxy(actor_ref, self._methods, self._async_methods)


def remote(
    cls: type[T] | None = None,
    *,
    restart_policy: str = "never",
    max_restarts: int = 3,
    min_backoff: float = 0.1,
    max_backoff: float = 30.0,
) -> ActorClass:
    """@remote decorator

    Converts a regular class into a distributed deployable Actor.

    Supports supervision configuration:
    - restart_policy: "never" (default), "always", "on-failure"
    - max_restarts: maximum number of restarts (default: 3)
    - min_backoff: minimum backoff in seconds (default: 0.1)
    - max_backoff: maximum backoff in seconds (default: 30.0)

    Example:
        @remote(restart_policy="on-failure", max_restarts=5)
        class Counter:
            ...
    """

    def wrapper(cls):
        return ActorClass(
            cls,
            restart_policy=restart_policy,
            max_restarts=max_restarts,
            min_backoff=min_backoff,
            max_backoff=max_backoff,
        )

    if cls is None:
        return wrapper

    return wrapper(cls)


# ============================================================================
# System operation helper functions (calls Rust SystemActor)
# ============================================================================


async def list_actors(system: ActorSystem) -> list[dict]:
    """List all actors on the current node."""
    sys_actor = await system.system()
    # SystemMessage uses serde tag format
    resp = await sys_actor.ask(
        Message.from_json("SystemMessage", {"type": "ListActors"})
    )
    data = resp.to_json()
    if data.get("type") == "Error":
        raise RuntimeError(data.get("message"))
    return data.get("actors", [])


async def get_metrics(system: ActorSystem) -> dict:
    """Get system metrics."""
    sys_actor = await system.system()
    resp = await sys_actor.ask(
        Message.from_json("SystemMessage", {"type": "GetMetrics"})
    )
    return resp.to_json()


async def get_node_info(system: ActorSystem) -> dict:
    """Get node info."""
    sys_actor = await system.system()
    resp = await sys_actor.ask(
        Message.from_json("SystemMessage", {"type": "GetNodeInfo"})
    )
    return resp.to_json()


async def health_check(system: ActorSystem) -> dict:
    """Health check."""
    sys_actor = await system.system()
    resp = await sys_actor.ask(
        Message.from_json("SystemMessage", {"type": "HealthCheck"})
    )
    return resp.to_json()


async def ping(system: ActorSystem, node_id: int | None = None) -> dict:
    """Ping node.

    Args:
        system: ActorSystem instance
        node_id: Target node ID (None means local node)
    """
    if node_id is None:
        sys_actor = await system.system()
    else:
        sys_actor = await system.remote_system(node_id)
    resp = await sys_actor.ask(Message.from_json("SystemMessage", {"type": "Ping"}))
    return resp.to_json()


async def resolve(
    name: str,
    *,
    system: ActorSystem | None = None,
    node_id: int | None = None,
    methods: list[str] | None = None,
) -> ActorProxy:
    """通过名字解析一个 named actor，返回可调用方法的 ActorProxy

    Args:
        name: actor 名字
        system: ActorSystem 实例，如果不提供则使用全局 system
        node_id: 目标节点 ID，如果不提供则在集群中查找
        methods: 可选的方法名列表。如果不提供，允许调用任意方法（动态模式）

    Returns:
        ActorProxy: 可以直接调用方法的代理对象

    Example:
        from pulsing.actor import init, remote, resolve

        await init()

        @remote
        class Counter:
            def __init__(self, init=0): self.value = init
            def increment(self): self.value += 1; return self.value

        # 节点 A 创建 actor
        counter = await Counter.spawn(name="my_counter")

        # 节点 B 解析并调用
        proxy = await resolve("my_counter")
        result = await proxy.increment()  # 远程调用
    """
    from . import _global_system

    if system is None:
        if _global_system is None:
            raise RuntimeError(
                "Actor system not initialized. Call 'await init()' first."
            )
        system = _global_system

    actor_ref = await system.resolve_named(name, node_id=node_id)
    return ActorProxy(actor_ref, methods)


RemoteClass = ActorClass
# Keep old name as alias (backward compatibility)
SystemActor = PythonActorService
