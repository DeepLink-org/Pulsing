"""
Tests for resolve().as_any() and as_any(ref): untyped proxy that forwards any method call.

Covers:
- resolve(name) returns an object with .as_any()
- ref.as_any() returns a proxy; await proxy.method(...) works without knowing the actor type
- as_any(ref) function works with ref from resolve() or raw ActorRef
- typed_proxy.as_any() returns an any proxy with the same underlying ref
- ref.ask() / ref.tell() still work (backward compatibility)
"""

import asyncio

import pytest

import pulsing as pul
from pulsing.actor import Actor, ActorRefView, as_any, remote


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
# Test: resolve() returns object with .as_any()
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_returns_ref_view_with_as_any(initialized_pul):
    """resolve(name) returns an object that has .as_any() method."""
    await pul.spawn(
        _EchoActor(),
        name="as_any_echo",
        public=True,
    )

    ref = await pul.resolve("as_any_echo")
    assert ref is not None
    assert hasattr(ref, "as_any")
    assert callable(getattr(ref, "as_any"))

    proxy = ref.as_any()
    assert proxy is not None
    assert hasattr(proxy, "ref")


@pytest.mark.asyncio
async def test_resolve_returns_actor_ref_view(initialized_pul):
    """resolve(name) returns ActorRefView (or equivalent with .as_any())."""
    await pul.spawn(_EchoActor(), name="ref_view_echo", public=True)

    ref = await pul.resolve("ref_view_echo")
    assert isinstance(ref, ActorRefView)


# ============================================================================
# Test: ref.as_any() proxy forwards any method call
# ============================================================================


class _EchoActor(Actor):
    """Simple actor that echoes and has a named method for proxy calls."""

    async def receive(self, msg):
        if isinstance(msg, dict) and "echo" in msg:
            return msg["echo"]
        return msg


@pul.remote
class _ServiceWithMethods:
    """Remote service with sync and async methods for as_any tests."""

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
async def test_as_any_proxy_calls_sync_method(initialized_pul):
    """ref.as_any() returns a proxy; await proxy.sync_method() works."""
    await _ServiceWithMethods.spawn(name="as_any_svc", public=True)

    ref = await pul.resolve("as_any_svc")
    proxy = ref.as_any()

    result = await proxy.get_value()
    assert result == 0

    result = await proxy.set_value(42)
    assert result == 42

    result = await proxy.get_value()
    assert result == 42


@pytest.mark.asyncio
async def test_as_any_proxy_calls_async_method(initialized_pul):
    """await proxy.async_method() works through as_any() proxy."""
    await _ServiceWithMethods.spawn(name="as_any_async_svc", public=True)

    ref = await pul.resolve("as_any_async_svc")
    proxy = ref.as_any()

    result = await proxy.async_incr()
    assert result == 1
    result = await proxy.async_incr()
    assert result == 2


@pytest.mark.asyncio
async def test_as_any_proxy_method_with_args(initialized_pul):
    """proxy.method(args, kwargs) forwards correctly."""
    await _ServiceWithMethods.spawn(name="as_any_echo_svc", public=True)

    ref = await pul.resolve("as_any_echo_svc")
    proxy = ref.as_any()

    result = await proxy.echo("hello")
    assert result == "hello"


# ============================================================================
# Test: as_any(ref) function
# ============================================================================


@pytest.mark.asyncio
async def test_as_any_function_with_ref_from_resolve(initialized_pul):
    """as_any(ref) works when ref is from pul.resolve()."""
    await _ServiceWithMethods.spawn(name="as_any_fn_svc", public=True)

    ref = await pul.resolve("as_any_fn_svc")
    proxy = as_any(ref)

    result = await proxy.get_value()
    assert result == 0


@pytest.mark.asyncio
async def test_as_any_function_with_raw_ref(initialized_pul):
    """as_any(ref) works when ref is raw ActorRef from system.resolve()."""
    from pulsing.actor import get_system

    await _ServiceWithMethods.spawn(name="as_any_raw_svc", public=True)

    system = get_system()
    raw_ref = await system.resolve("as_any_raw_svc")
    proxy = as_any(raw_ref)

    result = await proxy.get_value()
    assert result == 0


# ============================================================================
# Test: typed proxy.as_any()
# ============================================================================


@pytest.mark.asyncio
async def test_typed_proxy_as_any(initialized_pul):
    """typed_proxy.as_any() returns a proxy that can call the same methods."""
    await _ServiceWithMethods.spawn(name="typed_any_svc", public=True)

    typed = await _ServiceWithMethods.resolve("typed_any_svc")
    result_typed = await typed.get_value()
    assert result_typed == 0

    any_proxy = typed.as_any()
    result_any = await any_proxy.get_value()
    assert result_any == 0

    await any_proxy.set_value(100)
    assert await typed.get_value() == 100


# ============================================================================
# Test: backward compatibility — ref.ask() / ref.tell() still work
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_ref_ask_still_works(initialized_pul):
    """After resolve(), ref.ask(msg) still works (ActorRefView delegates to _ref)."""
    await pul.spawn(_EchoActor(), name="compat_ask_echo", public=True)

    ref = await pul.resolve("compat_ask_echo")
    result = await ref.ask({"echo": "hello"})
    assert result == "hello"


@pytest.mark.asyncio
async def test_resolve_ref_tell_still_works(initialized_pul):
    """After resolve(), ref.tell(msg) still works."""

    class _CountTell(Actor):
        def __init__(self):
            self.n = 0

        async def receive(self, msg):
            self.n += 1
            if msg == "get":
                return self.n
            return None

    await pul.spawn(_CountTell(), name="compat_tell_count", public=True)

    ref = await pul.resolve("compat_tell_count")
    await ref.tell(None)
    await ref.tell(None)
    await asyncio.sleep(0.05)
    result = await ref.ask("get")
    assert result == 3
