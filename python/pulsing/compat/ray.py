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
import concurrent.futures
import inspect
import threading
from typing import Any, TypeVar

T = TypeVar("T")

# Global state
_system = None
_loop = None
_thread = None
_loop_ready = None


def _ensure_not_initialized(ignore_reinit_error: bool) -> None:
    global _system
    if _system is not None:
        if ignore_reinit_error:
            return
        raise RuntimeError("Already initialized. Call ray.shutdown() first.")


def _start_background_loop() -> None:
    """Start a dedicated event loop in a background thread.

    This is required when the caller is already inside a running event loop.
    """
    global _thread, _loop, _loop_ready
    if _thread is not None:
        return

    ready = threading.Event()
    _loop_ready = ready

    def _thread_main():
        global _loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _loop = loop
        ready.set()
        loop.run_forever()
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        finally:
            loop.close()

    t = threading.Thread(
        target=_thread_main, name="pulsing-compat-ray-loop", daemon=True
    )
    _thread = t
    t.start()
    ready.wait()


def _run_coro_sync(coro: Any, timeout=None) -> Any:
    """Run a coroutine to completion and return its result.

    - If we have a background loop thread: schedule via run_coroutine_threadsafe().
    - Otherwise: run on the local event loop with run_until_complete().
    """
    if _loop is None:
        raise RuntimeError("Not initialized. Call ray.init() first.")

    if _thread is not None:
        fut = asyncio.run_coroutine_threadsafe(coro, _loop)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError("Timed out waiting for result") from e
    else:
        # Local (non-running) loop only
        return _loop.run_until_complete(coro)


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

        async def _get():
            return await self._coro

        if timeout is not None:
            coro = asyncio.wait_for(_get(), timeout)
        else:
            coro = _get()

        self._result = _run_coro_sync(coro, timeout=timeout)
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
            n
            for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
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

        return _run_coro_sync(create())

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

    _ensure_not_initialized(ignore_reinit_error)

    from pulsing.actor import ActorSystem, SystemConfig
    from pulsing.actor.remote import PYTHON_ACTOR_SERVICE_NAME, PythonActorService

    # If we're already inside a running event loop (e.g., Jupyter/pytest-asyncio),
    # we must not call run_until_complete() on it. Use a dedicated background loop.
    in_running_loop = True
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        in_running_loop = False

    if in_running_loop:
        _start_background_loop()
    else:
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)

    async def _create_system():
        system = await ActorSystem.create(SystemConfig.standalone(), _loop)
        service = PythonActorService(system)
        await system.spawn(service, name=PYTHON_ACTOR_SERVICE_NAME, public=True)
        return system

    _system = _run_coro_sync(_create_system())


def shutdown() -> None:
    """Shutdown Pulsing (Ray-compatible)"""
    global _system, _loop, _thread, _loop_ready

    if _system is not None:
        try:
            _run_coro_sync(_system.shutdown())
        except Exception:
            pass
        _system = None
        if _thread is not None and _loop is not None:
            try:
                _loop.call_soon_threadsafe(_loop.stop)
            except Exception:
                pass
            try:
                _thread.join(timeout=2.0)
            except Exception:
                pass
            _thread = None
            _loop_ready = None
            _loop = None
        else:
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
