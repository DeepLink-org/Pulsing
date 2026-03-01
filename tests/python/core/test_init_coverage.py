"""Tests for core/__init__.py — cover uncovered branches.

Targets:
- get_system() when not initialized
- is_initialized()
- init() with head_node + head_addr conflict
- init() idempotency (double init returns same system)
- shutdown() when no system
- Actor base class
"""

import asyncio

import pytest

from pulsing.core import (
    Actor,
    ActorId,
    ActorRef,
    ActorSystem,
    Message,
    StreamMessage,
    SystemConfig,
    get_system,
    init,
    is_initialized,
    shutdown,
)
from pulsing.exceptions import PulsingRuntimeError


# ============================================================================
# Global system lifecycle
# ============================================================================


class TestGetSystem:
    @pytest.mark.asyncio
    async def test_get_system_before_init_raises(self):
        assert not is_initialized()
        with pytest.raises(PulsingRuntimeError, match="not initialized"):
            get_system()

    @pytest.mark.asyncio
    async def test_is_initialized_false_by_default(self):
        assert is_initialized() is False


class TestInit:
    @pytest.mark.asyncio
    async def test_head_node_and_head_addr_conflict(self):
        with pytest.raises(ValueError, match="Cannot set both"):
            await init(addr="0.0.0.0:9999", is_head_node=True, head_addr="1.2.3.4:8000")

    @pytest.mark.asyncio
    async def test_init_and_shutdown(self):
        system = await init()
        assert is_initialized() is True
        assert get_system() is system
        await shutdown()
        assert is_initialized() is False

    @pytest.mark.asyncio
    async def test_double_init_returns_same(self):
        system1 = await init()
        system2 = await init()
        assert system1 is system2
        await shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_when_not_initialized(self):
        assert not is_initialized()
        await shutdown()  # should not raise


# ============================================================================
# Actor base class
# ============================================================================


class TestActorBaseClass:
    def test_on_start_default_noop(self):
        class MyActor(Actor):
            async def receive(self, msg):
                return msg

        actor = MyActor()
        result = actor.on_start(ActorId(1))
        assert result is None

    def test_on_stop_default_noop(self):
        class MyActor(Actor):
            async def receive(self, msg):
                return msg

        actor = MyActor()
        result = actor.on_stop()
        assert result is None

    def test_metadata_default_empty(self):
        class MyActor(Actor):
            async def receive(self, msg):
                return msg

        actor = MyActor()
        assert actor.metadata() == {}

    def test_cannot_instantiate_without_receive(self):
        with pytest.raises(TypeError):

            class BadActor(Actor):
                pass

            BadActor()
