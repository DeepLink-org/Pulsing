"""Lightweight tests for pulsing.torchrun sync bridge usage."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("torch.distributed")

import pulsing.integrations.torchrun as ptorch


def test_init_in_torchrun_uses_run_sync_bridge():
    system = SimpleNamespace(addr="0.0.0.0:12345")

    def fake_run_sync(awaitable, timeout=None):
        assert timeout == 60
        awaitable.close()
        return system

    with (
        patch.object(ptorch.dist, "is_initialized", return_value=True),
        patch.object(ptorch.dist, "get_rank", return_value=0),
        patch.object(ptorch.dist, "broadcast_object_list") as mock_broadcast,
        patch("pulsing.integrations.torchrun.run_sync", side_effect=fake_run_sync),
    ):
        assert ptorch.init_in_torchrun() is system

    mock_broadcast.assert_called_once()


def test_init_in_torchrun_nonzero_rank_joins_seed():
    calls = []

    async def fake_do_init(addr, seeds=None):
        calls.append((addr, seeds))
        return SimpleNamespace(addr=f"joined:{seeds[0]}")

    def fake_broadcast(object_list, src):
        assert src == 0
        object_list[0] = "seed.host:9999"

    with (
        patch.object(ptorch.dist, "is_initialized", return_value=True),
        patch.object(ptorch.dist, "get_rank", return_value=1),
        patch.object(ptorch.dist, "broadcast_object_list", side_effect=fake_broadcast),
        patch.object(ptorch, "_do_init", fake_do_init),
        patch(
            "pulsing.integrations.torchrun.run_sync",
            side_effect=lambda awaitable, timeout=None: asyncio.run(awaitable),
        ),
    ):
        system = ptorch.init_in_torchrun()

    assert system.addr == "joined:seed.host:9999"
    assert calls == [("0.0.0.0:0", ["seed.host:9999"])]
