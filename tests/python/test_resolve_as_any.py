"""
Tests for resolve() proxy behavior.

Covers:
- resolve(name) returns untyped ActorProxy
- resolve(name, cls=X) returns typed ActorProxy
- typed_proxy.as_any() returns an any proxy with the same underlying ref
- proxy.ref gives underlying ActorRef for low-level .ask()/.tell()
"""

import asyncio

import pytest

from pulsing.exceptions import PulsingRuntimeError

import pulsing as pul
from pulsing.core import Actor, ActorRef, ActorProxy, remote


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def initialized_pul():
    """Initialize global pulsing system for testing."""
    await pul.init()
    yield
    await pul.shutdown()


# ============================================================================
# Test: resolve() returns ActorProxy
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_returns_untyped_proxy(initialized_pul):
    """resolve(name) returns an untyped ActorProxy."""
    await pul.spawn(
        _EchoActor(),
        name="resolve_proxy_echo",
        public=True,
    )

    proxy = await pul.resolve("resolve_proxy_echo")
    assert isinstance(proxy, ActorProxy)
    assert hasattr(proxy, "ref")


@pytest.mark.asyncio
async def test_resolve_with_cls_returns_typed_proxy(initialized_pul):
    """resolve(name, cls=X) returns a typed ActorProxy."""
    await _ServiceWithMethods.spawn(name="resolve_typed_svc", public=True)

    proxy = await pul.resolve("resolve_typed_svc", cls=_ServiceWithMethods)
    assert isinstance(proxy, ActorProxy)

    result = await proxy.get_value()
    assert result == 0


# ============================================================================
# Test: untyped proxy forwards any method call
# ============================================================================


class _EchoActor(Actor):
    """Simple actor that echoes and has a named method for proxy calls."""

    async def receive(self, msg):
        if isinstance(msg, dict) and "echo" in msg:
            return msg["echo"]
        return msg


@pul.remote
class _ServiceWithMethods:
    """Remote service with sync and async methods."""

    def __init__(self):
        self.value = 0

    def get_value(self):
        return self.value

    def set_value(self, n: int):
        self.value = n
        return self.value

    async def async_incr(self):
        self.value += 1
        return self.value

    def echo(self, text: str):
        return text


@pytest.mark.asyncio
async def test_untyped_proxy_calls_sync_method(initialized_pul):
    """Untyped proxy from resolve() forwards sync method calls."""
    await _ServiceWithMethods.spawn(name="untyped_svc", public=True)

    proxy = await pul.resolve("untyped_svc")

    result = await proxy.get_value()
    assert result == 0

    result = await proxy.set_value(42)
    assert result == 42

    result = await proxy.get_value()
    assert result == 42


@pytest.mark.asyncio
async def test_untyped_proxy_calls_async_method(initialized_pul):
    """Untyped proxy from resolve() forwards async method calls."""
    await _ServiceWithMethods.spawn(name="untyped_async_svc", public=True)

    proxy = await pul.resolve("untyped_async_svc")

    result = await proxy.async_incr()
    assert result == 1
    result = await proxy.async_incr()
    assert result == 2


@pytest.mark.asyncio
async def test_untyped_proxy_method_with_args(initialized_pul):
    """Untyped proxy forwards args and kwargs correctly."""
    await _ServiceWithMethods.spawn(name="untyped_echo_svc", public=True)

    proxy = await pul.resolve("untyped_echo_svc")

    result = await proxy.echo("hello")
    assert result == "hello"


# ============================================================================
# Test: typed proxy validates methods
# ============================================================================


@pytest.mark.asyncio
async def test_typed_proxy_calls_method(initialized_pul):
    """resolve(name, cls=X) proxy calls methods correctly."""
    await _ServiceWithMethods.spawn(name="typed_svc", public=True)

    proxy = await pul.resolve("typed_svc", cls=_ServiceWithMethods)

    result = await proxy.get_value()
    assert result == 0

    result = await proxy.set_value(99)
    assert result == 99

    result = await proxy.get_value()
    assert result == 99


@pytest.mark.asyncio
async def test_typed_proxy_rejects_invalid_method(initialized_pul):
    """Typed proxy rejects methods not on the class."""
    await _ServiceWithMethods.spawn(name="typed_reject_svc", public=True)

    proxy = await pul.resolve("typed_reject_svc", cls=_ServiceWithMethods)

    with pytest.raises(AttributeError, match="No method"):
        proxy.nonexistent_method


@pytest.mark.asyncio
async def test_typed_proxy_async_method(initialized_pul):
    """Typed proxy correctly handles async methods."""
    await _ServiceWithMethods.spawn(name="typed_async_svc", public=True)

    proxy = await pul.resolve("typed_async_svc", cls=_ServiceWithMethods)

    result = await proxy.async_incr()
    assert result == 1
    result = await proxy.async_incr()
    assert result == 2


# ============================================================================
# Test: typed proxy.as_any()
# ============================================================================


@pytest.mark.asyncio
async def test_typed_proxy_as_any(initialized_pul):
    """typed_proxy.as_any() returns an untyped proxy with same underlying ref."""
    await _ServiceWithMethods.spawn(name="typed_any_svc", public=True)

    typed = await pul.resolve("typed_any_svc", cls=_ServiceWithMethods)
    result_typed = await typed.get_value()
    assert result_typed == 0

    any_proxy = typed.as_any()
    result_any = await any_proxy.get_value()
    assert result_any == 0

    await any_proxy.set_value(100)
    assert await typed.get_value() == 100


# ============================================================================
# Test: proxy.ref for low-level access
# ============================================================================


@pytest.mark.asyncio
async def test_proxy_ref_ask_still_works(initialized_pul):
    """proxy.ref.ask(msg) still works for low-level messaging."""
    await pul.spawn(_EchoActor(), name="compat_ask_echo", public=True)

    proxy = await pul.resolve("compat_ask_echo")
    ref = proxy.ref
    assert isinstance(ref, ActorRef)
    result = await ref.ask({"echo": "hello"})
    assert result == "hello"


@pytest.mark.asyncio
async def test_proxy_ref_tell_still_works(initialized_pul):
    """proxy.ref.tell(msg) still works for fire-and-forget."""

    class _CountTell(Actor):
        def __init__(self):
            self.n = 0

        async def receive(self, msg):
            self.n += 1
            if msg == "get":
                return self.n
            return None

    await pul.spawn(_CountTell(), name="compat_tell_count", public=True)

    proxy = await pul.resolve("compat_tell_count")
    ref = proxy.ref
    await ref.tell(None)
    await ref.tell(None)
    await asyncio.sleep(0.05)
    result = await ref.ask("get")
    assert result == 3


# ============================================================================
# Test: ActorRef.as_any() / .as_type() still work (low-level API)
# ============================================================================


@pytest.mark.asyncio
async def test_raw_ref_as_any(initialized_pul):
    """ActorRef.as_any() still works via system.resolve()."""
    from pulsing.core import get_system

    await _ServiceWithMethods.spawn(name="raw_ref_svc", public=True)

    system = get_system()
    raw_ref = await system.resolve("raw_ref_svc")
    proxy = raw_ref.as_any()

    result = await proxy.get_value()
    assert result == 0


@pytest.mark.asyncio
async def test_raw_ref_as_type(initialized_pul):
    """ActorRef.as_type(cls) still works via system.resolve()."""
    from pulsing.core import get_system

    await _ServiceWithMethods.spawn(name="raw_ref_type_svc", public=True)

    system = get_system()
    raw_ref = await system.resolve("raw_ref_type_svc")
    proxy = raw_ref.as_type(_ServiceWithMethods)

    result = await proxy.get_value()
    assert result == 0


# ============================================================================
# Test: Counter.resolve(name, timeout=...)
# ============================================================================


@pytest.mark.asyncio
async def test_counter_resolve_with_timeout(initialized_pul):
    """Counter.resolve(name, timeout=...) passes timeout to underlying resolve."""
    await _ServiceWithMethods.spawn(name="timeout_svc", public=True)

    proxy = await _ServiceWithMethods.resolve("timeout_svc", timeout=5)
    result = await proxy.get_value()
    assert result == 0


@pytest.mark.asyncio
async def test_counter_resolve_timeout_not_found(initialized_pul):
    """Counter.resolve(name, timeout=...) raises PulsingRuntimeError if not found."""
    with pytest.raises(PulsingRuntimeError):
        await _ServiceWithMethods.resolve("nonexistent_actor", timeout=0.3)
