"""Ray-like distributed object wrapper."""

import asyncio
import inspect
import logging
import os
import random
import uuid
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pulsing._core import ActorRef, ActorSystem, Message, StreamMessage
from pulsing.exceptions import PulsingActorError, PulsingRuntimeError

# Protocol version configuration
# Default to v1 for backward compatibility
_DEFAULT_PROTOCOL_VERSION = int(os.getenv("PULSING_PROTOCOL_VERSION", "1"))


def _get_protocol_version() -> int:
    """Get protocol version from environment or default to v1."""
    return _DEFAULT_PROTOCOL_VERSION


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


def _detect_protocol_version(msg: dict) -> int:
    """Auto-detect protocol version from message.

    Returns:
        1 for v1 protocol, 2 for v2 protocol
    """
    if "__pulsing_proto__" in msg:
        version_str = msg["__pulsing_proto__"]
        if isinstance(version_str, str) and version_str.startswith("v"):
            return int(version_str[1:])
        return int(version_str)
    # v1 compatibility: check for __call__ field
    if "__call__" in msg:
        return 1
    return 1  # default to v1


def _wrap_call_v1(method: str, args: tuple, kwargs: dict, is_async: bool) -> dict:
    """v1 protocol: legacy format (backward compatible).

    Format:
        {
            "__call__": method_name,
            "args": args,
            "kwargs": kwargs,
            "__async__": is_async
        }
    """
    return {
        "__call__": method,
        "args": args,
        "kwargs": kwargs,
        "__async__": is_async,
    }


def _wrap_call_v2(method: str, args: tuple, kwargs: dict, is_async: bool) -> dict:
    """v2 protocol: namespace isolation.

    Format:
        {
            "__pulsing_proto__": "v2",
            "__pulsing__": {
                "call": method_name,
                "async": is_async
            },
            "user_data": {
                "args": args,
                "kwargs": kwargs
            }
        }
    """
    return {
        "__pulsing_proto__": "v2",
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
    """Unwrap call message, supporting both v1 and v2 protocols.

    Returns:
        (method_name, args, kwargs, is_async)
    """
    version = _detect_protocol_version(msg)

    if version == 2:
        pulsing = msg.get("__pulsing__", {})
        user_data = msg.get("user_data", {})
        return (
            pulsing.get("call", ""),
            tuple(user_data.get("args", ())),
            dict(user_data.get("kwargs", {})),
            pulsing.get("async", False),
        )
    else:  # v1
        return (
            msg.get("__call__", ""),
            tuple(msg.get("args", ())),
            dict(msg.get("kwargs", {})),
            msg.get("__async__", False),
        )


def _wrap_response_v1(result: Any = None, error: str | None = None) -> dict:
    """v1 protocol response format."""
    if error:
        return {"__error__": error}
    return {"__result__": result}


def _wrap_response_v2(result: Any = None, error: str | None = None) -> dict:
    """v2 protocol response format."""
    if error:
        return {
            "__pulsing_proto__": "v2",
            "__pulsing__": {"error": error},
            "user_data": {},
        }
    return {
        "__pulsing_proto__": "v2",
        "__pulsing__": {"result": result},
        "user_data": {},
    }


def _unwrap_response(resp: dict) -> tuple[Any, str | None]:
    """Unwrap response, supporting both v1 and v2 protocols.

    Returns:
        (result, error) - one of them will be None
    """
    version = _detect_protocol_version(resp)

    if version == 2:
        pulsing = resp.get("__pulsing__", {})
        if "error" in pulsing:
            return (None, pulsing["error"])
        return (pulsing.get("result"), None)
    else:  # v1
        if "__error__" in resp:
            return (None, resp["__error__"])
        return (resp.get("__result__"), None)


_PULSING_ERROR_PREFIX = "__PULSING_ERROR__:"


def _convert_rust_error(err: RuntimeError) -> Exception:
    """Convert Rust-raised RuntimeError to appropriate Pulsing exception.

    Rust layer encodes errors as JSON envelopes with prefix "__PULSING_ERROR__:".
    The JSON format:
      Actor errors:  {"category": "actor", "error": {"type": "business", "code": 400, ...}}
      Runtime errors: {"category": "runtime", "kind": "actor_not_found", "message": "...", ...}

    This replaces the previous regex-based string prefix parsing with
    reliable JSON deserialization.
    """
    import json

    from pulsing.exceptions import (
        PulsingBusinessError,
        PulsingSystemError,
        PulsingTimeoutError,
        PulsingUnsupportedError,
    )

    err_msg = str(err)

    if not err_msg.startswith(_PULSING_ERROR_PREFIX):
        # Not a structured Pulsing error, wrap as generic RuntimeError
        return PulsingRuntimeError(err_msg)

    json_str = err_msg[len(_PULSING_ERROR_PREFIX) :]
    try:
        envelope = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        # JSON parse failed, fall back to generic error
        return PulsingRuntimeError(err_msg)

    category = envelope.get("category")

    if category == "actor":
        actor_err = envelope.get("error", {})
        err_type = actor_err.get("type")

        if err_type == "business":
            code = actor_err.get("code", 0)
            message = actor_err.get("message", "Unknown error")
            details = actor_err.get("details")
            return PulsingBusinessError(code, message, details=details)

        if err_type == "system":
            error = actor_err.get("error", "Unknown error")
            recoverable = actor_err.get("recoverable", True)
            return PulsingSystemError(error, recoverable=recoverable)

        if err_type == "timeout":
            operation = actor_err.get("operation", "unknown")
            duration_ms = actor_err.get("duration_ms", 0)
            return PulsingTimeoutError(operation, duration_ms)

        if err_type == "unsupported":
            operation = actor_err.get("operation", "unknown")
            return PulsingUnsupportedError(operation)

        # Unknown actor error type, generic fallback
        return PulsingActorError(str(actor_err))

    if category == "runtime":
        message = envelope.get("message", "Unknown runtime error")
        return PulsingRuntimeError(message)

    # Unknown category
    return PulsingRuntimeError(err_msg)


async def _ask_convert_errors(ref, msg) -> Any:
    """Call ref.ask(msg) and convert Rust RuntimeError to Pulsing exceptions."""
    try:
        return await ref.ask(msg)
    except RuntimeError as e:
        raise _convert_rust_error(e) from e


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


class ActorRefView:
    """Wrapper around ActorRef that adds .as_any() for an untyped proxy.

    Returned by resolve(name). Delegates .ask(), .tell(), and other
    ActorRef attributes to the underlying ref. Use .as_any() to get
    a proxy that forwards any method call to the remote actor.
    """

    __slots__ = ("_ref",)

    def __init__(self, ref: ActorRef):
        self._ref = ref

    def as_any(self) -> "ActorProxy":
        """Return an untyped proxy that forwards any method call to the remote actor."""
        return ActorProxy(self._ref, method_names=None, async_methods=None)

    def __getattr__(self, name: str):
        return getattr(self._ref, name)


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
        # Use configured protocol version (default v1)
        protocol_version = _get_protocol_version()
        if protocol_version == 2:
            call_msg = _wrap_call_v2(self._method, args, kwargs, False)
        else:
            call_msg = _wrap_call_v1(self._method, args, kwargs, False)

        resp = await _ask_convert_errors(self._ref, call_msg)

        if isinstance(resp, dict):
            result, error = _unwrap_response(resp)
            if error:
                # Actor execution error
                try:
                    raise PulsingActorError(
                        error, actor_name=str(self._ref.actor_id.id)
                    )
                except RuntimeError as e:
                    # If it's a Rust error, convert it
                    raise _convert_rust_error(e) from e
            return result
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
            # Use configured protocol version (default v1)
            protocol_version = _get_protocol_version()
            if protocol_version == 2:
                call_msg = _wrap_call_v2(self._method, self._args, self._kwargs, True)
            else:
                call_msg = _wrap_call_v1(self._method, self._args, self._kwargs, True)
            resp = await _ask_convert_errors(self._ref, call_msg)

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
                # Single-value response (non-streaming): {"__result__": value}
                if "__result__" in item:
                    self._final_result = item.get("__result__")
                    self._got_result = True
                    raise StopAsyncIteration
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
                    raise PulsingActorError(item["__error__"])
                if "__yield__" in item:
                    return item["__yield__"]
            return item
        except StopAsyncIteration:
            raise


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
            msg = _wrap_call_v1(name, args, kwargs, is_async=True)
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

    def on_start(self, actor_id) -> None:
        if hasattr(self._instance, "on_start"):
            self._instance.on_start(actor_id)

    def on_stop(self) -> None:
        if hasattr(self._instance, "on_stop"):
            self._instance.on_stop()

    async def receive(self, msg) -> Any:
        # Handle dict-based call format (supporting both v1 and v2)
        if isinstance(msg, dict):
            # Detect protocol version
            version = _detect_protocol_version(msg)
            method, args, kwargs, is_async_call = _unwrap_call(msg)

            if not method or method.startswith("_"):
                error_msg = f"Invalid method: {method}"
                if version == 2:
                    return _wrap_response_v2(error=error_msg)
                return _wrap_response_v1(error=error_msg)

            func = getattr(self._instance, method, None)
            if func is None or not callable(func):
                error_msg = f"Not found: {method}"
                if version == 2:
                    return _wrap_response_v2(error=error_msg)
                return _wrap_response_v1(error=error_msg)

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
                # Use same protocol version as request
                if version == 2:
                    return _wrap_response_v2(result=result)
                return _wrap_response_v1(result=result)
            except Exception as e:
                error_msg = str(e)
                if version == 2:
                    return _wrap_response_v2(error=error_msg)
                return _wrap_response_v1(error=error_msg)

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
                    writer, {"__final__": True, "__result__": None}
                )
            except Exception as e:
                await self._safe_stream_write(writer, {"__error__": str(e)})
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
                        writer, {"__final__": True, "__result__": None}
                    )
                elif asyncio.iscoroutine(result):
                    final_result = await result
                    await self._safe_stream_write(
                        writer, {"__final__": True, "__result__": final_result}
                    )
                elif inspect.isgenerator(result):
                    for item in result:
                        if not await self._safe_stream_write(
                            writer, {"__yield__": item}
                        ):
                            return
                    await self._safe_stream_write(
                        writer, {"__final__": True, "__result__": None}
                    )
                else:
                    await self._safe_stream_write(
                        writer, {"__final__": True, "__result__": result}
                    )
            except Exception as e:
                await self._safe_stream_write(writer, {"__error__": str(e)})
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
                    # Supervision config
                    "restart_policy": self._restart_policy,
                    "max_restarts": self._max_restarts,
                    "min_backoff": self._min_backoff,
                    "max_backoff": self._max_backoff,
                },
            ),
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
):
    """Resolve a named actor by name.

    Returns an object that supports .ask(), .tell(), and .as_any().
    Use .as_any() to get an untyped proxy that forwards any method call.

    For typed ActorProxy with method calls, use Counter.resolve(name) instead.

    Args:
        name: Actor name
        node_id: Target node ID, searches in cluster if not provided

    Returns:
        ActorRefView: Ref-like object with .as_any() for untyped proxy.

    Example:
        from pulsing.actor import init, remote, resolve

        await init()

        # By name only (no type needed)
        ref = await resolve("channel.discord")
        proxy = ref.as_any()
        await proxy.send_text(chat_id, content)

        # Low-level ask
        ref = await resolve("my_counter")
        result = await ref.ask({"__call__": "increment", "args": [], "kwargs": {}})
    """
    from . import _global_system

    if _global_system is None:
        raise RuntimeError("Actor system not initialized. Call 'await init()' first.")

    try:
        ref = await _global_system.resolve(name, node_id=node_id)
        return ActorRefView(ref)
    except RuntimeError as e:
        raise _convert_rust_error(e) from e


def as_any(ref: ActorRef | ActorRefView) -> ActorProxy:
    """Return an untyped proxy that forwards any method call to the remote actor.

    Use when you have an ActorRef (or ref from resolve()) and want to call
    methods by name without the typed class.

    Args:
        ref: ActorRef from resolve(name), or raw ActorRef from system.resolve_named().

    Example:
        ref = await resolve("channel.discord")
        proxy = as_any(ref)  # or proxy = ref.as_any()
        await proxy.send_text(chat_id, content)
    """
    if isinstance(ref, ActorRefView):
        return ref.as_any()
    return ActorProxy(ref, method_names=None, async_methods=None)


RemoteClass = ActorClass
# Keep old name as alias (backward compatibility)
SystemActor = PythonActorService
