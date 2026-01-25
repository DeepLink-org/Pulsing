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


def _convert_rust_error(err: RuntimeError) -> Exception:
    """Convert Rust-raised RuntimeError to appropriate Pulsing exception.

    Rust layer prefixes error messages with markers:
    - "ACTOR_ERROR:" -> PulsingActorError (or specific subclasses)
    - "RUNTIME_ERROR:" -> PulsingRuntimeError

    The error message format for ActorError:
    - "ACTOR_ERROR:Business error [code]: message" -> PulsingBusinessError
    - "ACTOR_ERROR:System error: message" -> PulsingSystemError
    - "ACTOR_ERROR:Timeout: operation 'op' timed out..." -> PulsingTimeoutError
    - "ACTOR_ERROR:Unsupported operation: op" -> PulsingUnsupportedError
    """
    from pulsing.exceptions import (
        PulsingBusinessError,
        PulsingSystemError,
        PulsingTimeoutError,
        PulsingUnsupportedError,
    )

    err_msg = str(err)

    if err_msg.startswith("ACTOR_ERROR:"):
        msg = err_msg.replace("ACTOR_ERROR:", "")

        # Try to identify specific ActorError type from message
        if msg.startswith("Business error ["):
            # Extract code, message, and details from "Business error [code]: message"
            import re

            match = re.match(r"Business error \[(\d+)\]: (.+)", msg)
            if match:
                code = int(match.group(1))
                message = match.group(2)
                return PulsingBusinessError(code, message)

        if msg.startswith("System error: "):
            # Extract error message from "System error: message"
            error_msg = msg.replace("System error: ", "")
            # Default to recoverable=True (we don't have recoverable flag in message)
            return PulsingSystemError(error_msg, recoverable=True)

        if msg.startswith("Timeout: operation '"):
            # Extract operation and duration from "Timeout: operation 'op' timed out after Xms"
            import re

            match = re.match(
                r"Timeout: operation '([^']+)' timed out after (\d+)ms", msg
            )
            if match:
                operation = match.group(1)
                duration_ms = int(match.group(2))
                return PulsingTimeoutError(operation, duration_ms)

        if msg.startswith("Unsupported operation: "):
            # Extract operation from "Unsupported operation: op"
            operation = msg.replace("Unsupported operation: ", "")
            return PulsingUnsupportedError(operation)

        # Fallback: generic PulsingActorError
        return PulsingActorError(msg)
    elif err_msg.startswith("RUNTIME_ERROR:"):
        msg = err_msg.replace("RUNTIME_ERROR:", "")
        return PulsingRuntimeError(msg)
    else:
        # Unknown format, wrap as RuntimeError
        return PulsingRuntimeError(err_msg)


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
        self._async_methods = async_methods or set()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"Cannot access private attribute: {name}")
        if self._method_names is not None and name not in self._method_names:
            raise AttributeError(f"No method '{name}'")
        is_async = name in self._async_methods
        return _MethodCaller(self._ref, name, is_async=is_async)

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
    """Method caller."""

    def __init__(self, actor_ref: ActorRef, method_name: str, is_async: bool = False):
        self._ref = actor_ref
        self._method = method_name
        self._is_async = is_async

    def __call__(self, *args, **kwargs):
        if self._is_async:
            return _AsyncMethodCall(self._ref, self._method, args, kwargs)
        else:
            return self._sync_call(*args, **kwargs)

    async def _sync_call(self, *args, **kwargs) -> Any:
        """Synchronous method call."""
        call_msg = {
            "__call__": self._method,
            "args": args,
            "kwargs": kwargs,
            "__async__": False,
        }
        resp = await self._ref.ask(call_msg)

        if isinstance(resp, dict):
            if "__error__" in resp:
                # Actor execution error
                try:
                    raise PulsingActorError(
                        resp["__error__"], actor_name=str(self._ref.actor_id.id)
                    )
                except RuntimeError as e:
                    # If it's a Rust error, convert it
                    raise _convert_rust_error(e) from e
            return resp.get("__result__")
        elif isinstance(resp, Message):
            if resp.is_stream:
                return _SyncGeneratorStreamReader(resp)
            data = resp.to_json()
            if resp.msg_type == "Error":
                # Actor execution error
                raise PulsingActorError(
                    data.get("error", "Remote call failed"),
                    actor_name=str(self._ref.actor_id.id),
                )
            return data.get("result")
        return resp


class _AsyncMethodCall:
    """Async method call - supports await and async for

    Usage:
        # Directly await to get final result
        result = await service.generate("hello")

        # Stream intermediate results
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
        """Get stream (lazy initialization)"""
        if self._stream_reader is None:
            call_msg = {
                "__call__": self._method,
                "args": self._args,
                "kwargs": self._kwargs,
                "__async__": True,
            }
            resp = await self._ref.ask(call_msg)

            # Response may be PyMessage (streaming) or direct Python object
            if isinstance(resp, Message):
                # Check if it's a streaming message
                if resp.is_stream:
                    self._stream_reader = resp.stream_reader()
                else:
                    # Not streaming, might be an error
                    data = resp.to_json()
                    if resp.msg_type == "Error":
                        # Actor execution error
                        raise PulsingActorError(
                            data.get("error", "Remote call failed"),
                            actor_name=str(self._ref.actor_id.id),
                        )
                    # Wrap as single-value iterator
                    self._stream_reader = _SingleValueIterator(data)
            else:
                # Regular Python object (might be dict)
                self._stream_reader = _SingleValueIterator(resp)

        return self._stream_reader

    def __aiter__(self):
        """Support async iteration, get intermediate results"""
        return self

    async def __anext__(self):
        """Get next streaming data"""
        reader = await self._get_stream()
        try:
            item = await reader.__anext__()
            # Check if it's the final result
            if isinstance(item, dict):
                if "__final__" in item:
                    self._final_result = item.get("__result__")
                    self._got_result = True
                    raise StopAsyncIteration
                if "__error__" in item:
                    # Actor execution error
                    raise PulsingActorError(
                        item["__error__"], actor_name=str(self._ref.actor_id.id)
                    )
                if "__yield__" in item:
                    return item["__yield__"]
            return item
        except StopAsyncIteration:
            raise

    def __await__(self):
        """Support await, get final result"""
        return self._await_result().__await__()

    async def _await_result(self):
        """Consume entire stream, return final result"""
        async for _ in self:
            pass  # Consume all yielded intermediate values
        if self._got_result:
            return self._final_result
        return None


class _SingleValueIterator:
    """Single-value async iterator - wraps a single value as async iterator"""

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


class _SyncGeneratorStreamReader:
    """Stream reader for sync generator returned from non-async method"""

    def __init__(self, message: Message):
        self._reader = message.stream_reader()
        self._final_result = None
        self._got_result = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = await self._reader.__anext__()
            if isinstance(item, dict):
                if "__final__" in item:
                    self._final_result = item.get("__result__")
                    self._got_result = True
                    raise StopAsyncIteration
                if "__error__" in item:
                    # Actor execution error
                    raise PulsingActorError(
                        item["__error__"], actor_name=str(self._ref.actor_id.id)
                    )
                if "__yield__" in item:
                    return item["__yield__"]
            return item
        except StopAsyncIteration:
            raise


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

    def _handle_generator_result(self, gen) -> StreamMessage:
        """Handle generator result, return streaming response"""
        stream_msg, writer = StreamMessage.create("GeneratorStream")

        async def execute():
            try:
                if inspect.isasyncgen(gen):
                    async for item in gen:
                        await writer.write({"__yield__": item})
                else:
                    for item in gen:
                        await writer.write({"__yield__": item})
                await writer.write({"__final__": True, "__result__": None})
            except Exception as e:
                await writer.write({"__error__": str(e)})
            finally:
                await writer.close()

        asyncio.create_task(execute())
        return stream_msg

    def _handle_async_method(self, func, args, kwargs) -> StreamMessage:
        """Handle async method, return streaming response"""
        stream_msg, writer = StreamMessage.create("AsyncMethodStream")

        async def execute():
            try:
                result = func(*args, **kwargs)

                # Check result type
                if inspect.isasyncgen(result):
                    # Async generator
                    async for item in result:
                        await writer.write({"__yield__": item})
                    await writer.write({"__final__": True, "__result__": None})
                elif asyncio.iscoroutine(result):
                    # Regular async function
                    final_result = await result
                    await writer.write({"__final__": True, "__result__": final_result})
                elif inspect.isgenerator(result):
                    # Synchronous generator
                    for item in result:
                        await writer.write({"__yield__": item})
                    await writer.write({"__final__": True, "__result__": None})
                else:
                    # Regular return value
                    await writer.write({"__final__": True, "__result__": result})
            except Exception as e:
                await writer.write({"__error__": str(e)})
            finally:
                await writer.close()

        # Execute in background task, non-blocking actor
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
    """Actor class wrapper

    Provides two ways to create actors:

    1. Simple API (uses global system):
        await init()
        counter = await Counter.spawn(init=10)

    2. Explicit system:
        system = await pul.actor_system()
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

        # Collect all public methods
        self._methods = []
        self._async_methods = set()

        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            self._methods.append(name)
            # Detect if it's an async method (including async functions and async generators)
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
        public: bool | None = None,
        **kwargs,
    ) -> ActorProxy:
        """Create actor using global system (simple API)

        Must call `await init()` before using this method.

        Args:
            *args: Positional arguments for the class constructor
            name: Optional actor name (if provided, defaults to public=True)
            public: Whether the actor should be publicly resolvable (default: True if name provided)
            **kwargs: Keyword arguments for the class constructor

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
            raise PulsingRuntimeError(
                "Actor system not initialized. Call 'await init()' first."
            )

        # Default public=True if name is provided
        if public is None:
            public = name is not None

        return await self.local(
            _global_system, *args, name=name, public=public, **kwargs
        )

    async def local(
        self,
        system: ActorSystem,
        *args,
        name: str | None = None,
        public: bool | None = None,
        **kwargs,
    ) -> ActorProxy:
        """Create actor locally with explicit system.

        Args:
            system: The ActorSystem to spawn the actor in
            *args: Positional arguments for the class constructor
            name: Optional actor name (if provided, defaults to public=True)
            public: Whether the actor should be publicly resolvable (default: True if name provided)
            **kwargs: Keyword arguments for the class constructor

        Note: Use pul.actor_system() to create ActorSystem,
        which automatically registers PythonActorService.
        """
        # Default public=True if name is provided
        if public is None:
            public = name is not None

        # Actor name must follow namespace/name format
        if name:
            # Ensure user-provided name has namespace
            actor_name = name if "/" in name else f"actors/{name}"
        else:
            actor_name = f"actors/{self._cls.__name__}_{uuid.uuid4().hex[:8]}"

        if self._restart_policy != "never":

            def factory():
                instance = self._cls(*args, **kwargs)
                return _WrappedActor(instance)

            actor_ref = await system.spawn(
                factory,
                name=actor_name,
                public=public,
                restart_policy=self._restart_policy,
                max_restarts=self._max_restarts,
                min_backoff=self._min_backoff,
                max_backoff=self._max_backoff,
            )
        else:
            instance = self._cls(*args, **kwargs)
            actor = _WrappedActor(instance)
            actor_ref = await system.spawn(actor, name=actor_name, public=public)

        # Register actor metadata
        _register_actor_metadata(actor_name, self._cls)

        return ActorProxy(actor_ref, self._methods, self._async_methods)

    async def remote(
        self,
        system: ActorSystem,
        *args,
        name: str | None = None,
        public: bool | None = None,
        **kwargs,
    ) -> ActorProxy:
        """Create actor remotely (randomly selects a remote node).

        Args:
            system: The ActorSystem to spawn the actor in
            *args: Positional arguments for the class constructor
            name: Optional actor name (if provided, defaults to public=True)
            public: Whether the actor should be publicly resolvable (default: True if name provided)
            **kwargs: Keyword arguments for the class constructor

        Note: Use pul.actor_system() to create ActorSystem,
        which automatically registers PythonActorService.
        """
        # Default public=True if name is provided
        if public is None:
            public = name is not None

        members = await system.members()
        # members["node_id"] is string, convert local_id to string for comparison
        local_id = str(system.node_id.id)

        # Filter out remote nodes (node_id is string)
        remote_nodes = [m for m in members if m["node_id"] != local_id]

        if not remote_nodes:
            # No remote nodes, fallback to local creation
            logger.warning("No remote nodes, fallback to local")
            return await self.local(system, *args, name=name, public=public, **kwargs)

        # Randomly select one
        target = random.choice(remote_nodes)
        # Convert back to int for resolve_named
        target_id = int(target["node_id"])

        # Get target node's Python actor creation service
        service_ref = await system.resolve_named(
            PYTHON_ACTOR_SERVICE_NAME, node_id=target_id
        )

        # Actor name must follow namespace/name format
        if name:
            actor_name = name if "/" in name else f"actors/{name}"
        else:
            actor_name = f"actors/{self._cls.__name__}_{uuid.uuid4().hex[:8]}"

        # Send creation request
        resp = await service_ref.ask(
            Message.from_json(
                "CreateActor",
                {
                    "class_name": self._class_name,
                    "actor_name": actor_name,
                    "args": list(args),
                    "kwargs": kwargs,
                    "public": public,
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
            # System error: actor creation failed
            raise PulsingRuntimeError(f"Remote create failed: {data.get('error')}")

        # Build remote ActorRef
        from pulsing._core import ActorId

        # actor_id is now a UUID (u128), may be transmitted as string
        actor_id = data["actor_id"]
        if isinstance(actor_id, str):
            actor_id = int(actor_id)
        remote_id = ActorId(actor_id)
        actor_ref = await system.actor_ref(remote_id)

        return ActorProxy(
            actor_ref, data.get("methods", self._methods), self._async_methods
        )

    def __call__(self, *args, **kwargs):
        """Direct call returns local instance (not an Actor)"""
        return self._cls(*args, **kwargs)

    def proxy(self, actor_ref: ActorRef) -> ActorProxy:
        """Wrap ActorRef into typed ActorProxy

        Args:
            actor_ref: Underlying actor reference

        Returns:
            ActorProxy: Proxy with method type information

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
        """Resolve actor by name, return typed ActorProxy

        Args:
            name: Actor name
            system: ActorSystem instance, uses global system if not provided
            node_id: Target node ID, searches in cluster if not provided

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
        resp = await self._ref.ask(
            Message.from_json("SystemMessage", {"type": msg_type})
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
        resp = await self._ref.ask(Message.from_json("ListRegistry", {}))
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
        resp = await self._ref.ask(
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
            )
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
) -> ActorRef:
    """Resolve a named actor by name, return ActorRef

    For typed ActorProxy with method calls, use Counter.resolve(name) instead.

    Args:
        name: Actor name
        node_id: Target node ID, searches in cluster if not provided

    Returns:
        ActorRef: Low-level actor reference for ask/tell operations.

    Example:
        from pulsing.actor import init, remote, resolve

        await init()

        @remote
        class Counter:
            def __init__(self, init=0): self.value = init
            def increment(self): self.value += 1; return self.value

        # Create actor
        counter = await Counter.spawn(name="my_counter")

        # Method 1: Use typed resolve (recommended)
        proxy = await Counter.resolve("my_counter")
        result = await proxy.increment()

        # Method 2: Use low-level resolve + ask
        ref = await resolve("my_counter")
        result = await ref.ask({"method": "increment", "args": [], "kwargs": {}})
    """
    from . import _global_system

    if _global_system is None:
        raise RuntimeError("Actor system not initialized. Call 'await init()' first.")

    return await _global_system.resolve(name, node_id=node_id)


RemoteClass = ActorClass
# Keep old name as alias (backward compatibility)
SystemActor = PythonActorService
