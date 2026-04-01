"""High-level Python connector for out-cluster actor access."""

import asyncio
import inspect
import logging
from typing import Any

from pulsing._core import ConnectActorRef, PulsingConnect
from pulsing.core.protocol import _check_response, _wrap_call

logger = logging.getLogger(__name__)


class ConnectProxy:
    """Proxy for a remote actor accessed through the out-cluster connector.

    Supports the same method-call syntax as in-cluster ActorProxy::

        counter = await conn.resolve("services/counter")
        result = await counter.increment()

        async for token in await llm.generate(prompt="Hello"):
            print(token, end="")
    """

    def __init__(
        self,
        actor_ref: ConnectActorRef,
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
        return _ConnectMethodCaller(self._ref, name, is_async=is_async)

    @property
    def ref(self) -> ConnectActorRef:
        return self._ref

    @property
    def path(self) -> str:
        return self._ref.path

    @property
    def gateway(self) -> str:
        return self._ref.gateway


class _ConnectMethodCaller:
    """Dispatches a method call to the remote actor via the gateway."""

    def __init__(
        self, actor_ref: ConnectActorRef, method_name: str, is_async: bool = False
    ):
        self._ref = actor_ref
        self._method = method_name
        self._is_async = is_async

    def __call__(self, *args, **kwargs):
        if self._is_async:
            return _ConnectAsyncMethodCall(self._ref, self._method, args, kwargs)
        return self._sync_call(*args, **kwargs)

    def __await__(self):
        return self().__await__()

    async def _sync_call(self, *args, **kwargs) -> Any:
        call_msg = _wrap_call(self._method, args, kwargs, False)
        resp = await self._ref.ask(call_msg)
        return _connect_check_response(resp, self._ref)


class _ConnectAsyncMethodCall:
    """Async method call supporting both await (final result) and async-for (streaming)."""

    def __init__(
        self, actor_ref: ConnectActorRef, method_name: str, args: tuple, kwargs: dict
    ):
        self._ref = actor_ref
        self._method = method_name
        self._args = args
        self._kwargs = kwargs
        self._stream_reader = None
        self._final_result = None
        self._got_result = False

    async def _ensure_stream(self):
        if self._stream_reader is not None or self._got_result:
            return

        from pulsing._core import Message

        call_msg = _wrap_call(self._method, self._args, self._kwargs, True)
        resp = await self._ref.ask(call_msg)
        result = _connect_check_response(resp, self._ref)

        if isinstance(result, Message) and result.is_stream:
            self._stream_reader = result.stream_reader()
        else:
            self._final_result = result
            self._got_result = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        from pulsing.exceptions import PulsingActorError

        await self._ensure_stream()
        if self._got_result:
            raise StopAsyncIteration
        try:
            item = await self._stream_reader.__anext__()
            if isinstance(item, dict):
                if "__error__" in item:
                    raise PulsingActorError(
                        item["__error__"], actor_name=self._ref.path
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


def _connect_check_response(resp, ref) -> Any:
    """Unwrap response, raise on errors."""
    from pulsing._core import Message
    from pulsing.exceptions import PulsingActorError

    if isinstance(resp, Message):
        if resp.is_stream:
            return resp
        try:
            resp = resp.to_json()
        except ValueError:
            import pickle

            resp = pickle.loads(resp.payload)
    if isinstance(resp, dict):
        if "__error__" in resp:
            raise PulsingActorError(resp["__error__"], actor_name=ref.path)
        if "__result__" in resp:
            return resp["__result__"]
        if "error" in resp:
            raise PulsingActorError(resp["error"], actor_name=ref.path)
        if "result" in resp:
            return resp["result"]
    return resp


class Connect:
    """High-level out-cluster connector.

    Usage::

        # Single gateway
        conn = await Connect.to("10.0.1.1:8080")

        # Multiple gateways (failover)
        conn = await Connect.to(["10.0.1.1:8080", "10.0.1.2:8080"])

        # Resolve and call actors
        counter = await conn.resolve("services/counter")
        result = await counter.increment()

        # Streaming
        llm = await conn.resolve("services/llm")
        async for token in await llm.generate(prompt="Hello"):
            print(token, end="")

        await conn.close()
    """

    def __init__(self, inner: PulsingConnect):
        self._inner = inner

    @classmethod
    async def to(cls, addrs: "str | list[str]") -> "Connect":
        """Connect to a Pulsing cluster gateway.

        Args:
            addrs: Single gateway address or list of addresses for failover.
        """
        if isinstance(addrs, str):
            inner = await PulsingConnect.connect(addrs)
        else:
            inner = await PulsingConnect.connect_multi(addrs)
        return cls(inner)

    async def resolve(
        self,
        name: str,
        *,
        cls: type | None = None,
    ) -> ConnectProxy:
        """Resolve a named actor and return a proxy.

        Args:
            name: Actor name/path (e.g., "services/counter").
            cls: Optional class for typed proxy (validates method names).
        """
        if "/" not in name:
            name = f"actors/{name}"

        actor_ref = await self._inner.resolve(name)

        if cls is not None:
            methods, async_methods = _extract_methods(cls)
            return ConnectProxy(actor_ref, methods, async_methods)
        return ConnectProxy(actor_ref)

    async def spawn(
        self,
        actor_cls,
        *args,
        name: str | None = None,
        public: bool = True,
        **kwargs,
    ) -> ConnectProxy:
        """Spawn a @remote actor on the cluster via the gateway.

        The actor class must already be registered (via @remote) on the
        target cluster node. The actor is created on the gateway node
        and can be accessed through the returned proxy.

        Args:
            actor_cls: A @remote-decorated class (ActorClass) or plain class.
            name: Optional actor name/path. Auto-generated if omitted.
            public: Whether the actor is discoverable via resolve (default True).
            *args, **kwargs: Constructor arguments for the actor.

        Returns:
            ConnectProxy for the newly created actor.
        """
        from pulsing._core import Message
        from pulsing.core.remote import ActorClass
        from pulsing.core.protocol import _normalize_actor_name

        if isinstance(actor_cls, ActorClass):
            class_name = actor_cls._class_name
            original_cls = actor_cls._cls
            methods = list(actor_cls._methods)
            async_methods = actor_cls._async_methods
        else:
            original_cls = actor_cls
            class_name = f"{actor_cls.__module__}.{actor_cls.__name__}"
            methods, async_methods = _extract_methods(actor_cls)

        actor_name = _normalize_actor_name(original_cls.__name__, name)

        service_ref = await self._inner.actor_ref("system/python_actor_service")

        msg = Message.from_json(
            "CreateActor",
            {
                "class_name": class_name,
                "actor_name": actor_name,
                "args": list(args),
                "kwargs": kwargs,
                "public": public,
            },
        )

        resp = await service_ref.ask(msg)

        if isinstance(resp, Message):
            if resp.msg_type == "Error":
                data = resp.to_json()
                raise RuntimeError(f"Remote spawn failed: {data.get('error')}")
        elif isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Remote spawn failed: {resp['error']}")

        actor_ref = await self._inner.actor_ref(actor_name)
        return ConnectProxy(actor_ref, methods, async_methods)

    async def refresh_gateways(self):
        """Refresh the gateway list from the cluster."""
        await self._inner.refresh_gateways()

    async def close(self):
        """Close the connector."""
        self._inner.close()


def _extract_methods(cls: type) -> tuple[list[str], set[str]]:
    """Extract public method names and async method set from a class."""
    methods = []
    async_methods = set()
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        methods.append(name)
        if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
            async_methods.add(name)
    return methods, async_methods
