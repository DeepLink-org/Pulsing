"""
Tests for receive error behavior (business errors don't kill actor, panic stops without recovery).

Covers:
1. When receive returns/raises error: error returned to caller, actor doesn't exit, can process next message
2. Multiple receive errors: each error only returned to caller, actor stays alive
"""

import pytest

import pulsing as pul
from pulsing.core import Actor


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def system():
    """Create a standalone ActorSystem for testing."""
    sys = await pul.actor_system()
    yield sys
    await sys.shutdown()


# ============================================================================
# Actor: returns error for specific message, processes other messages normally
# ============================================================================


class ErrorOnBadMessageActor(Actor):
    """Raises when receiving 'bad', echoes other messages."""

    async def receive(self, msg):
        if msg == "bad":
            raise ValueError("intentional receive error")
        return msg


# ============================================================================
# Test: receive error only returned to caller, actor doesn't exit
# ============================================================================


@pytest.mark.asyncio
async def test_receive_error_returned_to_caller_actor_stays_alive(system):
    """When receive returns/raises error: caller receives error, actor doesn't exit, next message processed normally."""
    ref = await system.spawn(ErrorOnBadMessageActor(), name="error_on_bad")

    # 1st message: trigger error, should receive exception
    with pytest.raises(Exception):
        await ref.ask("bad")

    # 2nd message: actor still alive, should return normally
    result = await ref.ask("ok")
    assert result == "ok"


@pytest.mark.asyncio
async def test_receive_multiple_errors_then_success(system):
    """Multiple receive errors: each error only returned to caller, actor stays alive, final message succeeds."""
    ref = await system.spawn(ErrorOnBadMessageActor(), name="multi_error")

    for _ in range(3):
        with pytest.raises(Exception):
            await ref.ask("bad")

    result = await ref.ask("ok")
    assert result == "ok"
