"""
Tests for receive error behavior (业务错误不杀 actor、panic 停止不恢复).

Covers:
1. receive 返回/抛出错误时：错误返回给调用者，actor 不退出，可继续处理下一条消息
2. 多次 receive 错误：每次错误只回传调用方，actor 始终存活
"""

import pytest

import pulsing as pul
from pulsing.actor import Actor


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
# Actor: 对特定消息返回错误，其它消息正常处理
# ============================================================================


class ErrorOnBadMessageActor(Actor):
    """收到 'bad' 时 raise，其它消息 echo."""

    async def receive(self, msg):
        if msg == "bad":
            raise ValueError("intentional receive error")
        return msg


# ============================================================================
# Test: receive 出错只回传调用者，actor 不退出
# ============================================================================


@pytest.mark.asyncio
async def test_receive_error_returned_to_caller_actor_stays_alive(system):
    """receive 返回/抛出错误时：调用者收到错误，actor 不退出，下一条消息正常处理。"""
    ref = await system.spawn(ErrorOnBadMessageActor(), name="error_on_bad")

    # 第一条：触发错误，应收到异常
    with pytest.raises(Exception):
        await ref.ask("bad")

    # 第二条：actor 仍存活，应正常返回
    result = await ref.ask("ok")
    assert result == "ok"


@pytest.mark.asyncio
async def test_receive_multiple_errors_then_success(system):
    """多次 receive 出错：每次错误只回传调用方，actor 始终存活，最后一条正常。"""
    ref = await system.spawn(ErrorOnBadMessageActor(), name="multi_error")

    for _ in range(3):
        with pytest.raises(Exception):
            await ref.ask("bad")

    result = await ref.ask("ok")
    assert result == "ok"
