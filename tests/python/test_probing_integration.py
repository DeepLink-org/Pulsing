"""Optional integration with probing (skip if probing not installed)."""

import pytest

pytest.importorskip("probing")

import pulsing as pul
from pulsing.integrations.probing import (
    refresh_probing_tables_async,
    start_probing_integration,
    stop_probing_integration,
)


@pytest.mark.asyncio
async def test_refresh_triggers_snapshots():
    """refresh_probing_tables_async triggers Rust-side memtable writes."""
    stop_probing_integration()

    await pul.init()
    try:
        stop_probing_integration()
        assert await refresh_probing_tables_async() is True
    finally:
        await pul.shutdown()


@pytest.mark.asyncio
async def test_start_probing_integration_idempotent():
    stop_probing_integration()
    await pul.init()
    try:
        assert start_probing_integration() is True
        assert start_probing_integration() is True
    finally:
        await pul.shutdown()
