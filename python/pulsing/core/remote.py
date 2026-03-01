"""@remote decorator, ActorClass, and actor lifecycle management."""

import asyncio
import inspect
import logging
import random
import uuid
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pulsing._core import ActorRef, ActorSystem, Message, StreamMessage
from pulsing.exceptions import PulsingActorError, PulsingRuntimeError

from .protocol import (
    _consume_task_exception,
    _normalize_actor_name,
    _unwrap_call,
    _wrap_call,
    _wrap_response,
)
from .proxy import ActorProxy, _DelayedCallProxy

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ============================================================================
# Actor base class
# ============================================================================


class Actor(ABC):
    """Base class for Python actors. Implement `receive` to handle messages."""

    def on_start(self, actor_id) -> None:  # noqa: B027
        pass

    def on_stop(self) -> None:  # noqa: B027
        pass

    def metadata(self) -> dict[str, str]:
        return {}

    @abstractmethod
    async def receive(self, msg) -> Any:
        """Handle incoming message."""
        pass


# ============================================================================
# Actor class registry
# ============================================================================

_actor_class_registry: dict[str, type] = {}

_actor_metadata_registry: dict[str, dict[str, str]] = {}


def _register_actor_metadata(name: str, cls: type):
    """Register actor metadata for later retrieval."""
    metadata = {
        "python_class": f"{cls.__module__}.{cls.__name__}",
        "python_module": cls.__module__,
    }

    try:
        source_file = inspect.getfile(cls)
        metadata["python_file"] = source_file
    except (TypeError, OSError):
        pass

    _actor_metadata_registry[name] = metadata


def get_actor_metadata(name: str) -> dict[str, str] | None:
    """Get metadata for an actor by name."""
    return _actor_metadata_registry.get(name)


def _unwrap_class(cls) -> type:
    """Unwrap ActorClass / Ray ActorClass to get the original user class."""
    if isinstance(cls, ActorClass):
        return cls._cls
    try:
        from ray.actor import ActorClass as RayActorClass

        if isinstance(cls, RayActorClass):
            if hasattr(cls, "__ray_metadata__"):
                meta = cls.__ray_metadata__
                if hasattr(meta, "modified_class"):
                    return meta.modified_class
    except ImportError:
        pass
    return cls


def _extract_methods(cls: type) -> tuple[list[str], set[str]]:
    """Extract public method names and async method set from a class.

    Handles @pul.remote ActorClass and Ray-wrapped classes by unwrapping first.
    """
    cls = _unwrap_class(cls)
    methods = []
    async_methods = set()
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        methods.append(name)
        if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
            async_methods.add(name)
    return methods, async_methods


# ============================================================================
# _WrappedActor — wraps a user class instance as an Actor
# ============================================================================


class _WrappedActor(Actor):
    """Wraps user class as an Actor"""

    def __init__(self, instance: Any):
        self._instance = instance
        self._original_class = instance.__class__

    @property
    def __original_module__(self):
        return self._original_class.__module__

    @property
    def __original_qualname__(self):
        return self._original_class.__qualname__

    @property
    def __original_file__(self):
        try:
            return inspect.getfile(self._original_class)
        except (TypeError, OSError):
            return None

    def _inject_delayed(self, actor_ref: ActorRef) -> None:
        """Inject ``self.delayed(sec)`` on the user instance after spawn."""
        self._instance.delayed = lambda delay_sec: _DelayedCallProxy(
            actor_ref, delay_sec
        )

    def on_start(self, actor_id):
        if hasattr(self._instance, "on_start"):
            r = self._instance.on_start(actor_id)
            if asyncio.iscoroutine(r):
                return r
        return None

    def on_stop(self):
        if hasattr(self._instance, "on_stop"):
            r = self._instance.on_stop()
            if asyncio.iscoroutine(r):
                return r
        return None

    def metadata(self) -> dict[str, str]:
        if hasattr(self._instance, "metadata") and callable(self._instance.metadata):
            return self._instance.metadata()
        return {}

    async def receive(self, msg) -> Any:
        if isinstance(msg, dict):
            method, args, kwargs, is_async_call = _unwrap_call(msg)

            if not method or method.startswith("_"):
                return _wrap_response(error=f"Invalid method: {method}")

            _MISSING = object()
            attr = getattr(self._instance, method, _MISSING)
            if attr is _MISSING:
                return _wrap_response(error=f"Not found: {method}")

            if not callable(attr):
                return _wrap_response(result=attr)

            func = attr

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

            if is_async_method and is_async_call:
                return self._stream_result(func(*args, **kwargs))

            try:
                result = func(*args, **kwargs)
                if inspect.isgenerator(result) or inspect.isasyncgen(result):
                    return self._stream_result(result)
                if asyncio.iscoroutine(result):
                    result = await result
                return _wrap_response(result=result)
            except Exception as e:
                return _wrap_response(error=str(e))

        return _wrap_response(error=f"Unknown message type: {type(msg)}")

    @staticmethod
    async def _safe_stream_write(writer, obj: dict) -> bool:
        try:
            await writer.write(obj)
            return True
        except (RuntimeError, OSError, ConnectionError) as e:
            if "closed" in str(e).lower() or "stream" in str(e).lower():
                return False
            raise

    @staticmethod
    async def _safe_stream_close(writer) -> None:
        try:
            await writer.close()
        except (RuntimeError, OSError, ConnectionError):
            pass

    def _stream_result(self, result_or_gen) -> StreamMessage:
        """Stream a generator, coroutine, or plain value back to the caller."""
        stream_msg, writer = StreamMessage.create("Stream")

        async def _iter_to_stream(gen):
            if inspect.isasyncgen(gen):
                async for item in gen:
                    if not await self._safe_stream_write(writer, {"__yield__": item}):
                        return
            else:
                for item in gen:
                    if not await self._safe_stream_write(writer, {"__yield__": item}):
                        return

        async def execute():
            try:
                r = result_or_gen
                if inspect.isasyncgen(r) or inspect.isgenerator(r):
                    await _iter_to_stream(r)
                    final = None
                elif asyncio.iscoroutine(r):
                    final = await r
                else:
                    final = r
                await self._safe_stream_write(
                    writer, {"__final__": True, "__result__": final}
                )
            except Exception as e:
                await self._safe_stream_write(writer, {"__error__": str(e)})
            finally:
                await self._safe_stream_close(writer)

        task = asyncio.create_task(execute())
        task.add_done_callback(_consume_task_exception)
        return stream_msg


# ============================================================================
# ActorClass & @remote decorator
# ============================================================================

from .service import PYTHON_ACTOR_SERVICE_NAME


class ActorClass:
    """Actor class wrapper.

    Usage::

        await init()
        counter = await Counter.spawn(init=10)             # local, global system
        counter = await Counter.spawn(system=s, init=10)   # local, explicit system
        counter = await Counter.spawn(placement="remote")  # random remote node
        counter = await Counter.spawn(placement=node_id)   # specific node
    """

    def __init__(
        self,
        cls: type,
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0,
    ):
        unwrapped = _unwrap_class(cls)
        self._ray_cls = cls if unwrapped is not cls else None
        cls = unwrapped
        self._cls = cls
        self._class_name = f"{cls.__module__}.{cls.__name__}"
        self._restart_policy = restart_policy
        self._max_restarts = max_restarts
        self._min_backoff = min_backoff
        self._max_backoff = max_backoff

        self._methods, self._async_methods = _extract_methods(cls)

        _actor_class_registry[self._class_name] = cls

        if self._ray_cls is not None:
            self.remote = self._ray_cls.remote

    async def spawn(
        self,
        *args,
        system: ActorSystem | None = None,
        name: str | None = None,
        public: bool | None = None,
        placement: "str | int" = "local",
        **kwargs,
    ) -> ActorProxy:
        """Create an actor and return its proxy."""
        if system is None:
            from . import get_system

            system = get_system()

        if public is None:
            public = name is not None

        if placement == "local":
            return await self._spawn_local(
                system, *args, name=name, public=public, **kwargs
            )
        elif placement == "remote":
            return await self._spawn_remote(
                system, None, *args, name=name, public=public, **kwargs
            )
        elif isinstance(placement, int):
            return await self._spawn_remote(
                system, placement, *args, name=name, public=public, **kwargs
            )
        else:
            raise ValueError(
                f"Invalid placement {placement!r}. Use 'local', 'remote', or an int node_id."
            )

    async def _spawn_local(
        self,
        system: ActorSystem,
        *args,
        name: str | None = None,
        public: bool = False,
        **kwargs,
    ) -> ActorProxy:
        actor_name = _normalize_actor_name(self._cls.__name__, name)

        if self._restart_policy != "never":
            _wrapped_holder: list[_WrappedActor] = []

            def factory():
                instance = self._cls(*args, **kwargs)
                wrapped = _WrappedActor(instance)
                _wrapped_holder.append(wrapped)
                return wrapped

            actor_ref = await system.spawn(
                factory,
                name=actor_name,
                public=public,
                restart_policy=self._restart_policy,
                max_restarts=self._max_restarts,
                min_backoff=self._min_backoff,
                max_backoff=self._max_backoff,
            )
            if _wrapped_holder:
                _wrapped_holder[-1]._inject_delayed(actor_ref)
        else:
            instance = self._cls(*args, **kwargs)
            actor = _WrappedActor(instance)
            actor_ref = await system.spawn(actor, name=actor_name, public=public)
            actor._inject_delayed(actor_ref)

        _register_actor_metadata(actor_name, self._cls)
        return ActorProxy(actor_ref, self._methods, self._async_methods)

    async def _spawn_remote(
        self,
        system: ActorSystem,
        node_id: int | None,
        *args,
        name: str | None = None,
        public: bool = False,
        **kwargs,
    ) -> ActorProxy:
        """Spawn on a specific remote node (node_id=None means random)."""
        if node_id is None:
            members = await system.members()
            local_id = str(system.node_id.id)
            remote_nodes = [m for m in members if m["node_id"] != local_id]
            if not remote_nodes:
                logger.warning("No remote nodes available, falling back to local spawn")
                return await self._spawn_local(
                    system, *args, name=name, public=public, **kwargs
                )
            node_id = int(random.choice(remote_nodes)["node_id"])

        service_ref = await system.resolve_named(
            PYTHON_ACTOR_SERVICE_NAME, node_id=node_id
        )

        actor_name = _normalize_actor_name(self._cls.__name__, name)

        resp = await service_ref.ask(
            Message.from_json(
                "CreateActor",
                {
                    "class_name": self._class_name,
                    "actor_name": actor_name,
                    "args": list(args),
                    "kwargs": kwargs,
                    "public": public,
                    "restart_policy": self._restart_policy,
                    "max_restarts": self._max_restarts,
                    "min_backoff": self._min_backoff,
                    "max_backoff": self._max_backoff,
                },
            ),
        )

        data = resp.to_json()
        if resp.msg_type == "Error":
            raise PulsingRuntimeError(f"Remote create failed: {data.get('error')}")

        from pulsing._core import ActorId

        actor_id = data["actor_id"]
        if isinstance(actor_id, str):
            actor_id = int(actor_id)
        actor_ref = await system.actor_ref(ActorId(actor_id))
        return ActorProxy(
            actor_ref, data.get("methods", self._methods), self._async_methods
        )

    def __call__(self, *args, **kwargs):
        """Direct call returns local instance (not an Actor)"""
        return self._cls(*args, **kwargs)

    async def resolve(
        self,
        name: str,
        *,
        system: ActorSystem | None = None,
        node_id: int | None = None,
        timeout: float | None = None,
    ) -> ActorProxy:
        """Resolve actor by name, return typed ActorProxy."""
        if system is None:
            from . import get_system

            system = get_system()

        actor_ref = await system.resolve_named(name, node_id=node_id, timeout=timeout)
        return ActorProxy(actor_ref, self._methods, self._async_methods)


def remote(
    cls: type[T] | None = None,
    *,
    restart_policy: str = "never",
    max_restarts: int = 3,
    min_backoff: float = 0.1,
    max_backoff: float = 30.0,
) -> ActorClass:
    """@remote decorator — converts a regular class into a distributed Actor."""

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
# resolve() — top-level name resolution
# ============================================================================


async def resolve(
    name: str,
    *,
    cls: type | None = None,
    node_id: int | None = None,
    timeout: float | None = None,
) -> ActorProxy:
    """Resolve a named actor and return a ready-to-use proxy.

    Args:
        name: Actor name to resolve.
        cls: Optional class for typed proxy (validates method names).
             If omitted, returns an untyped proxy that accepts any method call.
        node_id: Target node ID (None = any node via load balancing).
        timeout: Retry timeout in seconds for waiting on gossip propagation.

    Examples::

        proxy = await pul.resolve("counter", cls=Counter, timeout=30)
        result = await proxy.incr()

        proxy = await pul.resolve("service")
        result = await proxy.some_method()
    """
    from . import get_system

    ref = await get_system().resolve(name, node_id=node_id, timeout=timeout)
    if cls is not None:
        methods, async_methods = _extract_methods(cls)
        return ActorProxy(ref, methods, async_methods)
    return ActorProxy(ref)


# ============================================================================
# Backward-compatible re-exports
# ============================================================================
# Many modules do `from pulsing.core.remote import X`; keep them working.

from .protocol import (  # noqa: E402, F401
    _check_response,
    _unwrap_response,
)
from .proxy import _AsyncMethodCall, _MethodCaller  # noqa: E402, F401
from .service import (  # noqa: E402, F401
    PythonActorService,
    PythonActorServiceProxy,
    SystemActorProxy,
    get_python_actor_service,
    get_system_actor,
)
