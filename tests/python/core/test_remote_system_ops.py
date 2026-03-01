"""Tests for remote.py system operation helpers.

Covers: SystemActorProxy, PythonActorServiceProxy, get_system_actor,
get_python_actor_service, resolve.
"""

import asyncio

import pytest

from pulsing.core import init, shutdown, get_system


# ============================================================================
# Shared fixture — replaces per-test init()/shutdown() boilerplate
# ============================================================================


@pytest.fixture
async def system():
    sys = await init()
    yield sys
    await shutdown()


# ============================================================================
# SystemActorProxy operations
# ============================================================================


@pytest.mark.asyncio
async def test_system_actor_proxy_legacy_ops(system):
    """Test system operations via SystemActorProxy (replaces legacy helper functions)."""
    from pulsing.core.remote import get_system_actor

    proxy = await get_system_actor(system)

    actors = await proxy.list_actors()
    assert isinstance(actors, list)

    metrics = await proxy.get_metrics()
    assert isinstance(metrics, dict)

    info = await proxy.get_node_info()
    assert isinstance(info, dict)

    result = await proxy.health_check()
    assert isinstance(result, dict)

    pong = await proxy.ping()
    assert isinstance(pong, dict)


@pytest.mark.asyncio
async def test_system_actor_proxy_all_methods(system):
    from pulsing.core.remote import get_system_actor

    proxy = await get_system_actor(system)
    assert proxy.ref is not None

    actors = await proxy.list_actors()
    assert isinstance(actors, list)

    metrics = await proxy.get_metrics()
    assert isinstance(metrics, dict)

    node_info = await proxy.get_node_info()
    assert isinstance(node_info, dict)

    health = await proxy.health_check()
    assert isinstance(health, dict)

    pong = await proxy.ping()
    assert isinstance(pong, dict)


# ============================================================================
# PythonActorServiceProxy
# ============================================================================


@pytest.mark.asyncio
async def test_python_actor_service_proxy_list_registry(system):
    from pulsing.core.remote import get_python_actor_service, remote

    @remote
    class RegisteredActor:
        def hello(self):
            return "hi"

    service = await get_python_actor_service(system)
    assert service.ref is not None

    classes = await service.list_registry()
    assert isinstance(classes, list)
    assert any("RegisteredActor" in c for c in classes)


@pytest.mark.asyncio
async def test_python_actor_service_proxy_create_actor(system):
    from pulsing.core.remote import get_python_actor_service, remote

    @remote
    class CreatableActor:
        def __init__(self, val=0):
            self.val = val

        def get_val(self):
            return self.val

    service = await get_python_actor_service(system)
    class_name = f"{CreatableActor._cls.__module__}.{CreatableActor._cls.__name__}"
    result = await service.create_actor(class_name, name="created_test", val=42)
    assert "actor_id" in result
    assert "node_id" in result


@pytest.mark.asyncio
async def test_python_actor_service_proxy_create_unknown_class(system):
    from pulsing.core.remote import get_python_actor_service
    from pulsing.exceptions import PulsingRuntimeError

    service = await get_python_actor_service(system)
    with pytest.raises(PulsingRuntimeError):
        await service.create_actor("nonexistent.module.FakeClass", name="should_fail")


# ============================================================================
# resolve() function
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_function(system):
    from pulsing.core import remote
    from pulsing.core.remote import resolve

    @remote
    class ResolveTarget:
        def echo(self, msg):
            return msg

    await ResolveTarget.spawn(name="resolve_target_test", public=True)
    ref = await resolve("resolve_target_test")
    assert ref is not None


@pytest.mark.asyncio
async def test_resolve_without_init():
    from pulsing.core.remote import resolve
    from pulsing.exceptions import PulsingRuntimeError

    with pytest.raises(PulsingRuntimeError, match="not initialized"):
        await resolve("anything")


# ============================================================================
# _WrappedActor async on_start / on_stop
# ============================================================================


@pytest.mark.asyncio
async def test_async_on_start(system):
    from pulsing.core import remote

    on_start_called = []

    @remote
    class AsyncOnStartActor:
        async def on_start(self, actor_id):
            on_start_called.append(str(actor_id))

        def ping(self):
            return "pong"

    actor = await AsyncOnStartActor.spawn()
    assert await actor.ping() == "pong"
    await asyncio.sleep(0.05)
    assert len(on_start_called) >= 1


@pytest.mark.asyncio
async def test_async_on_stop(system):
    from pulsing.core import remote

    on_stop_called = []

    @remote
    class AsyncOnStopActor:
        async def on_stop(self):
            on_stop_called.append("stopped")

        def ping(self):
            return "pong"

    actor = await AsyncOnStopActor.spawn(name="async_stop_test")
    assert await actor.ping() == "pong"
    await get_system().stop("async_stop_test")
    await asyncio.sleep(0.1)
    assert "stopped" in on_stop_called


# ============================================================================
# _WrappedActor receive with invalid/private method via raw ask
# ============================================================================


@pytest.mark.asyncio
async def test_receive_empty_method_name(system):
    from pulsing.core import remote
    from pulsing.core.remote import _wrap_call

    @remote
    class RawActor:
        def ping(self):
            return "pong"

    actor = await RawActor.spawn()
    msg = _wrap_call("", (), {}, False)
    resp = await actor.ref.ask(msg)
    assert isinstance(resp, dict)


@pytest.mark.asyncio
async def test_receive_private_method_via_raw(system):
    from pulsing.core import remote
    from pulsing.core.remote import _wrap_call

    @remote
    class RawActor2:
        def ping(self):
            return "pong"

    actor = await RawActor2.spawn()
    msg = _wrap_call("_secret", (), {}, False)
    resp = await actor.ref.ask(msg)
    assert isinstance(resp, dict)


@pytest.mark.asyncio
async def test_receive_nonexistent_method_via_raw(system):
    from pulsing.core import remote
    from pulsing.core.remote import _wrap_call

    @remote
    class RawActor3:
        def ping(self):
            return "pong"

    actor = await RawActor3.spawn()
    msg = _wrap_call("does_not_exist", (), {}, False)
    resp = await actor.ref.ask(msg)
    assert isinstance(resp, dict)
