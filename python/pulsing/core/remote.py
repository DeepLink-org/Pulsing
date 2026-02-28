"""Ray-like distributed object wrapper."""

import asyncio
import inspect
import logging
import random
import uuid
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pulsing._core import ActorRef, ActorSystem, Message, StreamMessage
from pulsing.exceptions import PulsingActorError, PulsingRuntimeError


def _consume_task_exception(task: asyncio.Task) -> None:
    """Consume exception from background task to avoid 'Task exception was never retrieved'."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except (RuntimeError, OSError, ConnectionError) as e:
        if "closed" in str(e).lower() or "stream" in str(e).lower():
            logging.getLogger(__name__).debug("Stream closed before response: %s", e)
        else:
            logging.getLogger(__name__).exception("Stream task failed: %s", e)
    except Exception:
        logging.getLogger(__name__).exception("Stream task failed")


# Wire format version (single protocol)
_PULSING_WIRE_VERSION = "1"


def _wrap_call(method: str, args: tuple, kwargs: dict, is_async: bool) -> dict:
    """Wrap method call for wire format (namespace isolation).

    Format:
        {
            "__pulsing_proto__": version,
            "__pulsing__": { "call": method_name, "async": is_async },
            "user_data": { "args": args, "kwargs": kwargs }
        }
    """
    return {
        "__pulsing_proto__": _PULSING_WIRE_VERSION,
        "__pulsing__": {
            "call": method,
            "async": is_async,
        },
        "user_data": {
            "args": args,
            "kwargs": kwargs,
        },
    }


def _unwrap_call(msg: dict) -> tuple[str, tuple, dict, bool]:
    """Unwrap call message. Returns (method_name, args, kwargs, is_async)."""
    pulsing = msg.get("__pulsing__", {})
    user_data = msg.get("user_data", {})
    return (
        pulsing.get("call", ""),
        tuple(user_data.get("args", ())),
        dict(user_data.get("kwargs", {})),
        pulsing.get("async", False),
    )


def _wrap_response(result: Any = None, error: str | None = None) -> dict:
    """Wrap response for wire format."""
    if error:
        return {
            "__pulsing_proto__": _PULSING_WIRE_VERSION,
            "__pulsing__": {"error": error},
            "user_data": {},
        }
    return {
        "__pulsing_proto__": _PULSING_WIRE_VERSION,
        "__pulsing__": {"result": result},
        "user_data": {},
    }


def _unwrap_response(resp: dict) -> tuple[Any, str | None]:
    """Unwrap response. Returns (result, error) - one of them will be None.

    Accepts:
    - Wire format: {"__pulsing__": {"result": ..., "error": ...}}
    - Message JSON: {"result": ..., "error": ...}  (Rust actor responses)
    """
    pulsing = resp.get("__pulsing__", {})
    if isinstance(pulsing, dict):
        if "error" in pulsing:
            return (None, pulsing["error"])
        if "result" in pulsing:
            return (pulsing["result"], None)
    if "error" in resp:
        return (None, resp["error"])
    if "result" in resp:
        return (resp["result"], None)
    return (None, None)


async def _ask_convert_errors(ref, msg) -> Any:
    """Call ref.ask(msg); Rust raises typed Pulsing exceptions directly."""
    return await ref.ask(msg)


logger = logging.getLogger(__name__)


class _ActorBase(ABC):
    """Actor base class."""

    def on_start(self, actor_id) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def metadata(self) -> dict[str, str]:
        return {}

    @abstractmethod
    async def receive(self, msg) -> Any:
        """Handle incoming message."""
        pass


T = TypeVar("T")

_actor_class_registry: dict[str, type] = {}

_actor_metadata_registry: dict[str, dict[str, str]] = {}


def _register_actor_metadata(name: str, cls: type):
    """Register actor metadata for later retrieval."""
    import inspect

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


def _extract_methods(cls: type) -> tuple[list[str], set[str]]:
    """Extract public method names and async method set from a class.

    Handles @pul.remote ActorClass and Ray-wrapped classes by unwrapping first.
    """
    # If it's an ActorClass (@pul.remote decorated), extract the original class
    if isinstance(cls, ActorClass):
        cls = cls._cls

    # If it's a Ray ActorClass, extract the original class
    try:
        from ray.actor import ActorClass as RayActorClass

        if isinstance(cls, RayActorClass):
            if hasattr(cls, "__ray_metadata__"):
                meta = cls.__ray_metadata__
                if hasattr(meta, "modified_class"):
                    cls = meta.modified_class
    except ImportError:
        pass

    methods = []
    async_methods = set()
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        methods.append(name)
        if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
            async_methods.add(name)
    return methods, async_methods


PYTHON_ACTOR_SERVICE_NAME = "system/python_actor_service"


class ActorProxy:
    """Actor proxy."""

    def __init__(
        self,
        actor_ref: ActorRef,
        method_names: list[str] | None = None,
        async_methods: set[str] | None = None,
    ):
        self._ref = actor_ref
        self._method_names = set(method_names) if method_names else None
        # None means "any proxy": allow any method, treat all as async (streaming support)
        self._async_methods = async_methods

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"Cannot access private attribute: {name}")
        if self._method_names is not None and name not in self._method_names:
            raise AttributeError(f"No method '{name}'")
        # When _async_methods is None (any proxy), treat all methods as async
        is_async = self._async_methods is None or name in self._async_methods
        return _MethodCaller(self._ref, name, is_async=is_async)

    def as_any(self) -> "ActorProxy":
        """Return an untyped proxy that forwards any method call to the remote actor."""
        return ActorProxy(self._ref, method_names=None, async_methods=None)

    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef."""
        return self._ref

    @classmethod
    def from_ref(
        cls,
        actor_ref: ActorRef,
        methods: list[str] | None = None,
        async_methods: set[str] | None = None,
    ) -> "ActorProxy":
        """Create ActorProxy from ActorRef."""
        return cls(actor_ref, methods, async_methods)


class _MethodCaller:
    """Method caller. Supports two usage patterns:
    - await proxy.method(args)  — method call
    - await proxy.attr          — attribute access (no args)
    """

    def __init__(self, actor_ref: ActorRef, method_name: str, is_async: bool = False):
        self._ref = actor_ref
        self._method = method_name
        self._is_async = is_async

    def __call__(self, *args, **kwargs):
        if self._is_async:
            return _AsyncMethodCall(self._ref, self._method, args, kwargs)
        else:
            return self._sync_call(*args, **kwargs)

    def __await__(self):
        """Support await proxy.attr for direct attribute access"""
        return self().__await__()

    async def _sync_call(self, *args, **kwargs) -> Any:
        """Synchronous method call."""
        call_msg = _wrap_call(self._method, args, kwargs, False)
        resp = await _ask_convert_errors(self._ref, call_msg)

        if isinstance(resp, dict):
            result, error = _unwrap_response(resp)
            if error:
                raise PulsingActorError(error, actor_name=str(self._ref.actor_id.id))
            return result
        elif isinstance(resp, Message):
            if resp.is_stream:
                # Sync generator: return an awaitable/iterable stream reader
                return _AsyncMethodCall.from_message(self._ref, resp)
            data = resp.to_json()
            if not isinstance(data, dict):
                return resp
            if resp.msg_type == "Error":
                raise PulsingActorError(
                    data.get("error", "Remote call failed"),
                    actor_name=str(self._ref.actor_id.id),
                )
            result, error = _unwrap_response(data)
            if error:
                raise PulsingActorError(error, actor_name=str(self._ref.actor_id.id))
            if result is not None:
                return result
            return data.get("result")
        return resp


class _AsyncMethodCall:
    """Async method call — supports await (final result) and async for (stream).

    Usage:
        result = await actor.generate("hello")        # get final result
        async for chunk in actor.generate("hello"):   # stream chunks
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

    @classmethod
    def from_message(cls, ref: ActorRef, message: Message) -> "_AsyncMethodCall":
        """Build from a pre-acquired streaming Message (sync generator return path)."""
        obj = cls.__new__(cls)
        obj._ref = ref
        obj._method = ""
        obj._args = ()
        obj._kwargs = {}
        obj._stream_reader = message.stream_reader()
        obj._final_result = None
        obj._got_result = False
        return obj

    async def _ensure_stream(self) -> None:
        """Send RPC and resolve the response.

        For streaming responses, initialises _stream_reader.
        For direct responses (non-streaming), resolves _final_result immediately
        so __anext__ can stop without an extra iterator allocation.
        """
        if self._stream_reader is not None or self._got_result:
            return

        call_msg = _wrap_call(self._method, self._args, self._kwargs, True)
        resp = await _ask_convert_errors(self._ref, call_msg)

        if isinstance(resp, Message):
            if resp.is_stream:
                self._stream_reader = resp.stream_reader()
            else:
                data = resp.to_json()
                if resp.msg_type == "Error":
                    raise PulsingActorError(
                        data.get("error", "Remote call failed"),
                        actor_name=str(self._ref.actor_id.id),
                    )
                result, error = _unwrap_response(data)
                if error:
                    raise PulsingActorError(
                        error, actor_name=str(self._ref.actor_id.id)
                    )
                self._final_result = result
                self._got_result = True
        else:
            # Direct dict from Python actor called with is_async=True
            if isinstance(resp, dict):
                pulsing = resp.get("__pulsing__", {})
                if isinstance(pulsing, dict):
                    if "error" in pulsing:
                        raise PulsingActorError(
                            pulsing["error"], actor_name=str(self._ref.actor_id.id)
                        )
                    self._final_result = pulsing.get("result")
                    self._got_result = True
                    return
            self._final_result = resp
            self._got_result = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._ensure_stream()
        if self._got_result:
            raise StopAsyncIteration
        try:
            item = await self._stream_reader.__anext__()
            if isinstance(item, dict):
                pulsing = item.get("__pulsing__", {})
                if isinstance(pulsing, dict):
                    if "error" in pulsing:
                        raise PulsingActorError(
                            pulsing["error"], actor_name=str(self._ref.actor_id.id)
                        )
                    if pulsing.get("final"):
                        self._final_result = pulsing.get("result")
                        self._got_result = True
                        raise StopAsyncIteration
                if "__yield__" in item:
                    return item["__yield__"]
            return item
        except StopAsyncIteration:
            raise

    def __await__(self):
        return self._await_result().__await__()

    async def _await_result(self):
        async for _ in self:
            pass
        if self._got_result:
            return self._final_result
        return None


class _DelayedCallProxy:
    """Proxy returned by ``self.delayed(sec)`` — any method call becomes a delayed message to self.

    Usage inside a @remote class::

        task = self.delayed(5.0).some_method(arg1, arg2)
        task.cancel()  # cancel if needed

    Returns an ``asyncio.Task`` that fires after the delay.
    """

    __slots__ = ("_ref", "_delay_sec")

    def __init__(self, ref: ActorRef, delay_sec: float):
        self._ref = ref
        self._delay_sec = delay_sec

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def caller(*args, **kwargs):
            msg = _wrap_call(name, args, kwargs, is_async=True)
            delay = max(0.0, self._delay_sec)

            async def _send():
                await asyncio.sleep(delay)
                await self._ref.tell(msg)

            return asyncio.create_task(_send())

        return caller


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

    def _inject_delayed(self, actor_ref: ActorRef) -> None:
        """Inject ``self.delayed(sec)`` on the user instance after spawn."""
        self._instance.delayed = lambda delay_sec: _DelayedCallProxy(
            actor_ref, delay_sec
        )

    def on_start(self, actor_id):
        """调用用户 on_start；若为 async 则返回 coroutine 供 Rust 端 run_coroutine_threadsafe 执行。"""
        if hasattr(self._instance, "on_start"):
            r = self._instance.on_start(actor_id)
            if asyncio.iscoroutine(r):
                return r
        return None

    def on_stop(self):
        """调用用户 on_stop；若为 async 则返回 coroutine 供 Rust 端执行。"""
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
        # Handle dict-based call format
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

            # Detect if it's an async method (including async generators)
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

            # For async methods, use streaming response
            if is_async_method and is_async_call:
                return self._handle_async_method(func, args, kwargs)

            # Regular method or not marked as async call
            try:
                result = func(*args, **kwargs)
                # Check if result is a generator (sync or async) FIRST
                # This must come before the coroutine check to avoid awaiting generators
                if inspect.isgenerator(result) or inspect.isasyncgen(result):
                    return self._handle_generator_result(result)
                if asyncio.iscoroutine(result):
                    result = await result
                return _wrap_response(result=result)
            except Exception as e:
                return _wrap_response(error=str(e))

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

        return _wrap_response(error=f"Unknown message type: {type(msg)}")

    @staticmethod
    async def _safe_stream_write(writer, obj: dict) -> bool:
        """Write to stream; return False if stream already closed (e.g. caller cancelled)."""
        try:
            await writer.write(obj)
            return True
        except (RuntimeError, OSError, ConnectionError) as e:
            if "closed" in str(e).lower() or "stream" in str(e).lower():
                return False
            raise

    @staticmethod
    async def _safe_stream_close(writer) -> None:
        """Close stream; ignore if already closed."""
        try:
            await writer.close()
        except (RuntimeError, OSError, ConnectionError):
            pass

    def _handle_generator_result(self, gen) -> StreamMessage:
        """Handle generator result, return streaming response"""
        stream_msg, writer = StreamMessage.create("GeneratorStream")

        async def execute():
            try:
                if inspect.isasyncgen(gen):
                    async for item in gen:
                        if not await self._safe_stream_write(
                            writer, {"__yield__": item}
                        ):
                            return
                else:
                    for item in gen:
                        if not await self._safe_stream_write(
                            writer, {"__yield__": item}
                        ):
                            return
                await self._safe_stream_write(
                    writer, {"__pulsing__": {"final": True, "result": None}}
                )
            except Exception as e:
                await self._safe_stream_write(
                    writer, {"__pulsing__": {"error": str(e)}}
                )
            finally:
                await self._safe_stream_close(writer)

        task = asyncio.create_task(execute())
        task.add_done_callback(_consume_task_exception)
        return stream_msg

    def _handle_async_method(self, func, args, kwargs) -> StreamMessage:
        """Handle async method, return streaming response"""
        stream_msg, writer = StreamMessage.create("AsyncMethodStream")

        async def execute():
            try:
                result = func(*args, **kwargs)

                # Check result type
                if inspect.isasyncgen(result):
                    async for item in result:
                        if not await self._safe_stream_write(
                            writer, {"__yield__": item}
                        ):
                            return
                    await self._safe_stream_write(
                        writer, {"__pulsing__": {"final": True, "result": None}}
                    )
                elif asyncio.iscoroutine(result):
                    final_result = await result
                    await self._safe_stream_write(
                        writer, {"__pulsing__": {"final": True, "result": final_result}}
                    )
                elif inspect.isgenerator(result):
                    for item in result:
                        if not await self._safe_stream_write(
                            writer, {"__yield__": item}
                        ):
                            return
                    await self._safe_stream_write(
                        writer, {"__pulsing__": {"final": True, "result": None}}
                    )
                else:
                    await self._safe_stream_write(
                        writer, {"__pulsing__": {"final": True, "result": result}}
                    )
            except Exception as e:
                await self._safe_stream_write(
                    writer, {"__pulsing__": {"error": str(e)}}
                )
            finally:
                await self._safe_stream_close(writer)

        task = asyncio.create_task(execute())
        task.add_done_callback(_consume_task_exception)
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
                    factory,
                    name=actor_name,
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
                actor_ref = await self.system.spawn(
                    actor, name=actor_name, public=public
                )

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
                    # actor_id is now a UUID (u128), transmit as string for JSON
                    "actor_id": str(actor_ref.actor_id.id),
                    "node_id": str(self.system.node_id.id),
                    "methods": method_names,
                },
            )
        except Exception as e:
            logger.exception(f"Create actor failed: {e}")
            return Message.from_json("Error", {"error": str(e)})


class ActorClass:
    """Actor class wrapper.

    Usage::

        await init()
        counter = await Counter.spawn(init=10)             # local, global system
        counter = await Counter.spawn(system=s, init=10)   # local, explicit system
        counter = await Counter.spawn(placement="remote")  # random remote node
        counter = await Counter.spawn(placement=node_id)   # specific node
    """

    @staticmethod
    def _unwrap_ray_class(cls):
        """Extract original user class if cls is a Ray ActorClass."""
        try:
            from ray.actor import ActorClass as RayActorClass
        except ImportError:
            return cls
        if isinstance(cls, RayActorClass):
            for base in type(cls).__bases__:
                if base is not RayActorClass and base.__name__ != "Generic":
                    return base
        return cls

    def __init__(
        self,
        cls: type,
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0,
    ):
        unwrapped = self._unwrap_ray_class(cls)
        # Keep Ray handle so .remote() remains available for Ray-wrapped classes
        self._ray_cls = cls if unwrapped is not cls else None
        cls = unwrapped
        self._cls = cls
        self._class_name = f"{cls.__module__}.{cls.__name__}"
        self._restart_policy = restart_policy
        self._max_restarts = max_restarts
        self._min_backoff = min_backoff
        self._max_backoff = max_backoff

        self._methods = []
        self._async_methods = set()

        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            self._methods.append(name)
            if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(
                method
            ):
                self._async_methods.add(name)

        _actor_class_registry[self._class_name] = cls

        # If original class was decorated with @ray.remote, expose Ray's .remote()
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
        """Create an actor and return its proxy.

        Args:
            *args: Positional arguments forwarded to the class constructor.
            system: ActorSystem to use. Defaults to the global system
                (requires ``await init()`` to have been called first).
            name: Optional actor name. When given, ``public`` defaults to True.
            public: Whether the actor is cluster-discoverable.
                Defaults to True when *name* is set, False otherwise.
            placement: Where to place the actor.
                - ``"local"`` *(default)*: spawn on the current node.
                - ``"remote"``: spawn on a randomly-chosen remote node;
                  falls back to local if no remote nodes are available.
                - ``int``: spawn on the node with that specific node_id.
            **kwargs: Keyword arguments forwarded to the class constructor.

        Example::

            await init()

            @remote
            class Counter:
                def __init__(self, init=0): self.value = init
                def incr(self): self.value += 1; return self.value

            counter = await Counter.spawn(init=10)
            result = await counter.incr()
        """
        from . import _global_system

        if system is None:
            system = _global_system
        if system is None:
            raise PulsingRuntimeError(
                "Actor system not initialized. Call 'await init()' first."
            )

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
        actor_name = (
            name
            if (name and "/" in name)
            else (
                f"actors/{name}"
                if name
                else f"actors/{self._cls.__name__}_{uuid.uuid4().hex[:8]}"
            )
        )

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

        actor_name = (
            name
            if (name and "/" in name)
            else (
                f"actors/{name}"
                if name
                else f"actors/{self._cls.__name__}_{uuid.uuid4().hex[:8]}"
            )
        )

        resp = await _ask_convert_errors(
            service_ref,
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
        """Resolve actor by name, return typed ActorProxy

        Args:
            name: Actor name
            system: ActorSystem instance, uses global system if not provided
            node_id: Target node ID, searches in cluster if not provided
            timeout: Seconds to wait for the name to appear (gossip convergence).
                     None means no wait (error immediately if not found).

        Returns:
            ActorProxy: Proxy with method type information

        Example:
            @remote
            class Counter:
                def __init__(self, init=0): self.value = init
                async def generate(self, prompt): ...  # async method, streaming response

            # Node A creates actor
            counter = await Counter.spawn(name="my_counter")

            # Node B resolves and calls
            counter = await Counter.resolve("my_counter")

            # Call async method, can stream results
            result = counter.generate("hello")
            async for chunk in result:
                print(chunk)
            # Or directly await to get final result
            final = await counter.generate("hello")
        """
        from . import _global_system

        if system is None:
            if _global_system is None:
                raise RuntimeError(
                    "Actor system not initialized. Call 'await init()' first."
                )
            system = _global_system

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


class SystemActorProxy:
    """Proxy for SystemActor with direct method calls.

    Example:
        system_proxy = await get_system_actor(system)
        actors = await system_proxy.list_actors()
        metrics = await system_proxy.get_metrics()
        await system_proxy.ping()
    """

    def __init__(self, actor_ref: ActorRef):
        self._ref = actor_ref

    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef."""
        return self._ref

    async def _ask(self, msg_type: str) -> dict:
        """Send SystemMessage and return response."""
        resp = await _ask_convert_errors(
            self._ref,
            Message.from_json("SystemMessage", {"type": msg_type}),
        )
        return resp.to_json()

    async def list_actors(self) -> list[dict]:
        """List all actors on this node."""
        data = await self._ask("ListActors")
        if data.get("type") == "Error":
            # System error: system message failed
            raise PulsingRuntimeError(data.get("message"))
        return data.get("actors", [])

    async def get_metrics(self) -> dict:
        """Get system metrics."""
        return await self._ask("GetMetrics")

    async def get_node_info(self) -> dict:
        """Get node info."""
        return await self._ask("GetNodeInfo")

    async def health_check(self) -> dict:
        """Health check."""
        return await self._ask("HealthCheck")

    async def ping(self) -> dict:
        """Ping this node."""
        return await self._ask("Ping")


async def get_system_actor(
    system: ActorSystem, node_id: int | None = None
) -> SystemActorProxy:
    """Get SystemActorProxy for direct method calls.

    Args:
        system: ActorSystem instance
        node_id: Target node ID (None means local node)

    Returns:
        SystemActorProxy with methods: list_actors(), get_metrics(), etc.

    Example:
        sys = await get_system_actor(system)
        actors = await sys.list_actors()
        await sys.ping()
    """
    if node_id is None:
        actor_ref = await system.system()
    else:
        actor_ref = await system.remote_system(node_id)
    return SystemActorProxy(actor_ref)


class PythonActorServiceProxy:
    """Proxy for PythonActorService with direct method calls.

    Example:
        service = await get_python_actor_service(system)
        classes = await service.list_registry()
        actor_ref = await service.create_actor("MyClass", name="my_actor")
    """

    def __init__(self, actor_ref: ActorRef):
        self._ref = actor_ref

    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef."""
        return self._ref

    async def list_registry(self) -> list[str]:
        """List registered actor classes.

        Returns:
            List of registered class names
        """
        resp = await _ask_convert_errors(
            self._ref, Message.from_json("ListRegistry", {})
        )
        data = resp.to_json()
        return data.get("classes", [])

    async def create_actor(
        self,
        class_name: str,
        *args,
        name: str | None = None,
        public: bool = True,
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0,
        **kwargs,
    ) -> dict:
        """Create a Python actor.

        Args:
            class_name: Name of the registered actor class
            *args: Positional arguments for the class constructor
            name: Optional actor name
            public: Whether the actor should be publicly resolvable
            restart_policy: "never", "always", or "on_failure"
            max_restarts: Maximum restart attempts
            min_backoff: Minimum backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            **kwargs: Keyword arguments for the class constructor

        Returns:
            {"actor_id": "...", "node_id": "...", "actor_name": "..."}

        Raises:
            RuntimeError: If creation fails
        """
        resp = await _ask_convert_errors(
            self._ref,
            Message.from_json(
                "CreateActor",
                {
                    "class_name": class_name,
                    "actor_name": name,
                    "args": args,
                    "kwargs": kwargs,
                    "public": public,
                    "restart_policy": restart_policy,
                    "max_restarts": max_restarts,
                    "min_backoff": min_backoff,
                    "max_backoff": max_backoff,
                },
            ),
        )
        data = resp.to_json()
        if resp.msg_type == "Error" or data.get("error"):
            # System error: actor creation failed
            raise PulsingRuntimeError(data.get("error", "Unknown error"))
        return data


async def get_python_actor_service(
    system: ActorSystem, node_id: int | None = None
) -> PythonActorServiceProxy:
    """Get PythonActorServiceProxy for direct method calls.

    Args:
        system: ActorSystem instance
        node_id: Target node ID (None means local node)

    Returns:
        PythonActorServiceProxy with methods: list_registry(), create_actor()

    Example:
        service = await get_python_actor_service(system)
        classes = await service.list_registry()
    """
    service_ref = await system.resolve_named(PYTHON_ACTOR_SERVICE_NAME, node_id=node_id)
    return PythonActorServiceProxy(service_ref)


# Legacy helper functions (for backwards compatibility)
async def list_actors(system: ActorSystem) -> list[dict]:
    """List all actors on the current node."""
    proxy = await get_system_actor(system)
    return await proxy.list_actors()


async def get_metrics(system: ActorSystem) -> dict:
    """Get system metrics."""
    proxy = await get_system_actor(system)
    return await proxy.get_metrics()


async def get_node_info(system: ActorSystem) -> dict:
    """Get node info."""
    proxy = await get_system_actor(system)
    return await proxy.get_node_info()


async def health_check(system: ActorSystem) -> dict:
    """Health check."""
    proxy = await get_system_actor(system)
    return await proxy.health_check()


async def ping(system: ActorSystem, node_id: int | None = None) -> dict:
    """Ping node.

    Args:
        system: ActorSystem instance
        node_id: Target node ID (None means local node)
    """
    proxy = await get_system_actor(system, node_id)
    return await proxy.ping()


async def resolve(
    name: str,
    *,
    node_id: int | None = None,
    timeout: float | None = None,
):
    """Resolve a named actor by name.

    Returns an ActorRef that supports .ask(), .tell(), .as_any(), and .as_type().
    Use .as_any() to get an untyped proxy that forwards any method call.
    Use .as_type(Counter) to get a typed proxy with method validation.

    For typed ActorProxy with method calls, use Counter.resolve(name) instead.

    Args:
        name: Actor name
        node_id: Target node ID, searches in cluster if not provided
        timeout: Seconds to wait for the name to appear (gossip convergence).
                 None means no wait (error immediately if not found).

    Returns:
        ActorRef: Actor reference with .as_any() / .as_type() for proxy generation.

    Example:
        from pulsing.core import init, remote, resolve

        await init()

        # By name only (no type needed)
        ref = await resolve("channel.discord")
        proxy = ref.as_any()
        await proxy.send_text(chat_id, content)

        # Wait for name to appear (gossip convergence)
        ref = await resolve("peer_node", timeout=30)

        # Low-level ask
        ref = await resolve("my_counter")
        result = await ref.ask({"__call__": "increment", "args": [], "kwargs": {}})
    """
    from . import _global_system

    if _global_system is None:
        raise RuntimeError("Actor system not initialized. Call 'await init()' first.")

    return await _global_system.resolve(name, node_id=node_id, timeout=timeout)


def as_any(ref: ActorRef) -> ActorProxy:
    """Return an untyped proxy that forwards any method call to the remote actor.

    Use when you have an ActorRef and want to call methods by name
    without the typed class.

    Args:
        ref: ActorRef from resolve(name).

    Example:
        ref = await resolve("channel.discord")
        proxy = as_any(ref)  # or proxy = ref.as_any()
        await proxy.send_text(chat_id, content)
    """
    return ref.as_any()


RemoteClass = ActorClass
# Keep old name as alias (backward compatibility)
SystemActor = PythonActorService
