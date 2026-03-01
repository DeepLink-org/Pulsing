"""Caller-side proxy classes for actor method invocation."""

import asyncio
from typing import Any

from pulsing._core import ActorRef, Message
from pulsing.exceptions import PulsingActorError

from .protocol import _check_response, _wrap_call


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
        self._async_methods = async_methods

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"Cannot access private attribute: {name}")
        if self._method_names is not None and name not in self._method_names:
            raise AttributeError(f"No method '{name}'")
        is_async = self._async_methods is None or name in self._async_methods
        return _MethodCaller(self._ref, name, is_async=is_async)

    def as_any(self) -> "ActorProxy":
        """Return an untyped proxy that forwards any method call to the remote actor."""
        return ActorProxy(self._ref, method_names=None, async_methods=None)

    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef."""
        return self._ref


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
        resp = await self._ref.ask(call_msg)
        result = _check_response(resp, self._ref)
        if isinstance(result, Message) and result.is_stream:
            return _AsyncMethodCall.from_message(self._ref, result)
        return result


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
        resp = await self._ref.ask(call_msg)
        result = _check_response(resp, self._ref)

        if isinstance(result, Message) and result.is_stream:
            self._stream_reader = result.stream_reader()
        else:
            self._final_result = result
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
                if "__error__" in item:
                    raise PulsingActorError(
                        item["__error__"], actor_name=str(self._ref.actor_id.id)
                    )
                if item.get("__final__"):
                    self._final_result = item.get("__result__")
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
