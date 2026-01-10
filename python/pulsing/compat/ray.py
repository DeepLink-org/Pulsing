"""
Ray-compatible API for Pulsing

This module provides a Ray-like synchronous API for easy migration.
For new projects, we recommend using the native async API in pulsing.actor.

Migration from Ray:
    # Before (Ray)
    import ray
    ray.init()

    @ray.remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = Counter.remote(init=10)
    result = ray.get(counter.incr.remote())
    ray.shutdown()

    # After (Pulsing compat)
    from pulsing.compat import ray  # Only change this line!

    ray.init()

    @ray.remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = Counter.remote(init=10)
    result = ray.get(counter.incr.remote())
    ray.shutdown()

Note: This is a synchronous wrapper around async Pulsing.
For better performance in async environments, use pulsing.actor directly.
"""

import asyncio
import inspect
from typing import Any, TypeVar

T = TypeVar("T")

# Global state
_system = None
_loop = None


class ObjectRef:
    """Ray-compatible ObjectRef (wraps async coroutine)"""

    def __init__(self, coro_or_result: Any, is_ready: bool = False):
        self._coro = coro_or_result
        self._result = coro_or_result if is_ready else None
        self._is_ready = is_ready

    def _get_sync(self, timeout: float = None) -> Any:
        """Get result synchronously"""
        if self._is_ready:
            return self._result

        if _loop is None:
            raise RuntimeError("Not initialized. Call ray.init() first.")

        async def _get():
            return await self._coro

        if timeout:
            coro = asyncio.wait_for(_get(), timeout)
        else:
            coro = _get()

        self._result = _loop.run_until_complete(coro)
        self._is_ready = True
        return self._result


class _MethodCaller:
    """Method caller that returns ObjectRef"""

    def __init__(self, proxy, method_name: str):
        self._proxy = proxy
        self._method = method_name

    def remote(self, *args, **kwargs) -> ObjectRef:
        """Call method remotely (Ray-style)"""
        method = getattr(self._proxy, self._method)
        coro = method(*args, **kwargs)
        return ObjectRef(coro)


class _ActorHandle:
    """Ray-compatible actor handle"""

    def __init__(self, proxy, methods: list[str]):
        self._proxy = proxy
        self._methods = set(methods)

    def __getattr__(self, name: str) -> _MethodCaller:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._methods:
            raise AttributeError(f"No method '{name}'")
        return _MethodCaller(self._proxy, name)


class _ActorClass:
    """Ray-compatible actor class wrapper"""

    def __init__(self, cls: type):
        self._cls = cls
        self._pulsing_class = None
        self._methods = [
            n for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not n.startswith("_")
        ]

    def _ensure_wrapped(self):
        if self._pulsing_class is None:
            from pulsing.actor import remote
            self._pulsing_class = remote(self._cls)

    def remote(self, *args, **kwargs) -> _ActorHandle:
        """Create actor (Ray-style, synchronous)"""
        if _system is None:
            raise RuntimeError("Not initialized. Call ray.init() first.")

        self._ensure_wrapped()

        async def create():
            proxy = await self._pulsing_class.local(_system, *args, **kwargs)
            return _ActorHandle(proxy, self._methods)

        return _loop.run_until_complete(create())

    def options(self, **kwargs) -> "_ActorClass":
        """Set actor options (Ray compatibility, limited support)"""
        # TODO: Support num_cpus, num_gpus, etc.
        return self

    def __call__(self, *args, **kwargs):
        """Direct instantiation (not as actor)"""
        return self._cls(*args, **kwargs)


def init(
    address: str = None,
    *,
    ignore_reinit_error: bool = False,
    **kwargs,
) -> None:
    """Initialize Pulsing (Ray-compatible)

    Args:
        address: Ignored (use SystemConfig for Pulsing configuration)
        ignore_reinit_error: If True, ignore if already initialized

    Example:
        from pulsing.compat import ray
        ray.init()
    """
    global _system, _loop

    if _system is not None:
        if ignore_reinit_error:
            return
        raise RuntimeError("Already initialized. Call ray.shutdown() first.")

    from pulsing.actor import SystemConfig, create_actor_system

    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)

    config = SystemConfig.standalone()
    _system = _loop.run_until_complete(create_actor_system(config))


def shutdown() -> None:
    """Shutdown Pulsing (Ray-compatible)"""
    global _system, _loop

    if _system is not None:
        try:
            _loop.run_until_complete(_system.shutdown())
        except Exception:
            pass
        _system = None
        _loop = None


def is_initialized() -> bool:
    """Check if initialized"""
    return _system is not None


def remote(cls: type[T]) -> _ActorClass:
    """@ray.remote decorator (Ray-compatible)

    Example:
        @ray.remote
        class Counter:
            def __init__(self, init=0): self.value = init
            def incr(self): self.value += 1; return self.value

        counter = Counter.remote(init=10)
    """
    return _ActorClass(cls)


def get(refs: Any, *, timeout: float = None) -> Any:
    """Get results from ObjectRefs (Ray-compatible)

    Args:
        refs: Single ObjectRef or list of ObjectRefs
        timeout: Timeout in seconds

    Example:
        result = ray.get(counter.incr.remote())
        results = ray.get([ref1, ref2, ref3])
    """
    if _system is None:
        raise RuntimeError("Not initialized. Call ray.init() first.")

    if isinstance(refs, list):
        return [r._get_sync(timeout) for r in refs]
    return refs._get_sync(timeout)


def put(value: Any) -> ObjectRef:
    """Put value (Ray-compatible)

    Note: Pulsing doesn't have distributed object store.
    This just wraps the value for API compatibility.
    """
    return ObjectRef(value, is_ready=True)


def wait(
    refs: list,
    *,
    num_returns: int = 1,
    timeout: float = None,
) -> tuple[list, list]:
    """Wait for ObjectRefs (Ray-compatible)

    Returns:
        (ready, remaining) tuple
    """
    ready, remaining = [], list(refs)
    for ref in refs[:num_returns]:
        try:
            get(ref, timeout=timeout)
            ready.append(ref)
            remaining.remove(ref)
        except Exception:
            break
    return ready, remaining


__all__ = [
    "init",
    "shutdown",
    "is_initialized",
    "remote",
    "get",
    "put",
    "wait",
    "ObjectRef",
]
