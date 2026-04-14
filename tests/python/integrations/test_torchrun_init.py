"""Lightweight tests for pulsing.torchrun sync bridge usage."""

from __future__ import annotations

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
