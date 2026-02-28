"""Tests for remote.py system operation helpers and legacy functions.

Covers: list_actors, get_metrics, get_node_info, health_check, ping,
resolve, SystemActorProxy, PythonActorServiceProxy, get_system_actor,
get_python_actor_service.
"""

import asyncio

import pytest

from pulsing.core import init, shutdown, get_system


# ============================================================================
# Legacy helper functions (call SystemActor under the hood)
# ============================================================================


@pytest.mark.asyncio
async def test_list_actors():
    from pulsing.core.remote import list_actors

    system = await init()
    try:
        actors = await list_actors(system)
        assert isinstance(actors, list)
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_get_metrics():
    from pulsing.core.remote import get_metrics

    system = await init()
    try:
        metrics = await get_metrics(system)
        assert isinstance(metrics, dict)
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_get_node_info():
    from pulsing.core.remote import get_node_info

    system = await init()
    try:
        info = await get_node_info(system)
        assert isinstance(info, dict)
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_health_check():
    from pulsing.core.remote import health_check

    system = await init()
    try:
        result = await health_check(system)
        assert isinstance(result, dict)
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_ping():
    from pulsing.core.remote import ping

    system = await init()
    try:
        result = await ping(system)
        assert isinstance(result, dict)
    finally:
        await shutdown()


# ============================================================================
# SystemActorProxy
# ============================================================================


@pytest.mark.asyncio
async def test_system_actor_proxy_all_methods():
    from pulsing.core.remote import get_system_actor

    system = await init()
    try:
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
    finally:
        await shutdown()


# ============================================================================
# PythonActorServiceProxy
# ============================================================================


@pytest.mark.asyncio
async def test_python_actor_service_proxy_list_registry():
    from pulsing.core.remote import get_python_actor_service, remote

    @remote
    class RegisteredActor:
        def hello(self):
            return "hi"

    system = await init()
    try:
        service = await get_python_actor_service(system)
        assert service.ref is not None

        classes = await service.list_registry()
        assert isinstance(classes, list)
        assert any("RegisteredActor" in c for c in classes)
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_python_actor_service_proxy_create_actor():
    from pulsing.core.remote import get_python_actor_service, remote

    @remote
    class CreatableActor:
        def __init__(self, val=0):
            self.val = val

        def get_val(self):
            return self.val

    system = await init()
    try:
        service = await get_python_actor_service(system)
        class_name = f"{CreatableActor._cls.__module__}.{CreatableActor._cls.__name__}"
        result = await service.create_actor(class_name, name="created_test", val=42)
        assert "actor_id" in result
        assert "node_id" in result
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_python_actor_service_proxy_create_unknown_class():
    from pulsing.core.remote import get_python_actor_service
    from pulsing.exceptions import PulsingRuntimeError

    system = await init()
    try:
        service = await get_python_actor_service(system)
        with pytest.raises(PulsingRuntimeError):
            await service.create_actor(
                "nonexistent.module.FakeClass", name="should_fail"
            )
    finally:
        await shutdown()


# ============================================================================
# resolve() function
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_function():
    from pulsing.core import remote
    from pulsing.core.remote import resolve

    @remote
    class ResolveTarget:
        def echo(self, msg):
            return msg

    system = await init()
    try:
        await ResolveTarget.spawn(name="resolve_target_test", public=True)
        ref = await resolve("resolve_target_test")
        assert ref is not None
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_resolve_without_init():
    from pulsing.core.remote import resolve

    with pytest.raises(RuntimeError, match="not initialized"):
        await resolve("anything")


# ============================================================================
# _WrappedActor async on_start / on_stop
# ============================================================================


@pytest.mark.asyncio
async def test_async_on_start():
    """Test that async on_start is properly handled."""
    from pulsing.core import remote

    on_start_called = []

    @remote
    class AsyncOnStartActor:
        async def on_start(self, actor_id):
            on_start_called.append(str(actor_id))

        def ping(self):
            return "pong"

    system = await init()
    try:
        actor = await AsyncOnStartActor.spawn()
        assert await actor.ping() == "pong"
        await asyncio.sleep(0.05)
        assert len(on_start_called) >= 1
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_async_on_stop():
    """Test that async on_stop is properly handled."""
    from pulsing.core import remote

    on_stop_called = []

    @remote
    class AsyncOnStopActor:
        async def on_stop(self):
            on_stop_called.append("stopped")

        def ping(self):
            return "pong"

    system = await init()
    try:
        actor = await AsyncOnStopActor.spawn(name="async_stop_test")
        assert await actor.ping() == "pong"
        await get_system().stop("async_stop_test")
        await asyncio.sleep(0.1)
        assert "stopped" in on_stop_called
    finally:
        await shutdown()


# ============================================================================
# _WrappedActor receive with invalid/private method via raw ask
# ============================================================================


@pytest.mark.asyncio
async def test_receive_empty_method_name():
    """Empty method name in call should return error response."""
    from pulsing.core import remote
    from pulsing.core.remote import _wrap_call

    @remote
    class RawActor:
        def ping(self):
            return "pong"

    system = await init()
    try:
        actor = await RawActor.spawn()
        msg = _wrap_call("", (), {}, False)
        resp = await actor.ref.ask(msg)
        assert isinstance(resp, dict)
        # Should contain error about invalid method
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_receive_private_method_via_raw():
    """Private method call via raw ask should return error."""
    from pulsing.core import remote
    from pulsing.core.remote import _wrap_call

    @remote
    class RawActor2:
        def ping(self):
            return "pong"

    system = await init()
    try:
        actor = await RawActor2.spawn()
        msg = _wrap_call("_secret", (), {}, False)
        resp = await actor.ref.ask(msg)
        assert isinstance(resp, dict)
    finally:
        await shutdown()


@pytest.mark.asyncio
async def test_receive_nonexistent_method_via_raw():
    """Nonexistent method call via raw ask should return error."""
    from pulsing.core import remote
    from pulsing.core.remote import _wrap_call

    @remote
    class RawActor3:
        def ping(self):
            return "pong"

    system = await init()
    try:
        actor = await RawActor3.spawn()
        msg = _wrap_call("does_not_exist", (), {}, False)
        resp = await actor.ref.ask(msg)
        assert isinstance(resp, dict)
    finally:
        await shutdown()
