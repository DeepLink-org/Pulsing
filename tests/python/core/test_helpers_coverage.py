"""Tests for core/helpers.py — covers run_until_signal and spawn_and_run with mocking."""

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pulsing.core.helpers import run_sync, run_until_signal, spawn_and_run


# ============================================================================
# run_until_signal
# ============================================================================


class TestRunUntilSignal:
    @pytest.mark.asyncio
    async def test_signal_triggers_shutdown(self):
        """Simulate SIGTERM by directly calling the registered handler."""
        captured_handlers = {}

        def mock_add_signal_handler(sig, handler):
            captured_handlers[sig] = handler

        mock_system = MagicMock()
        mock_system.stop = AsyncMock()

        with (
            patch("pulsing.core.helpers.asyncio.get_running_loop") as mock_loop_fn,
            patch("pulsing.core.get_system", return_value=mock_system),
            patch("pulsing.core.shutdown", new_callable=AsyncMock) as mock_shutdown,
        ):
            loop = MagicMock()
            loop.add_signal_handler = mock_add_signal_handler
            mock_loop_fn.return_value = loop

            async def run_with_trigger():
                task = asyncio.create_task(run_until_signal("test_actor"))
                await asyncio.sleep(0.01)
                captured_handlers[signal.SIGTERM]()
                await task

            await run_with_trigger()

            mock_system.stop.assert_awaited_once_with("test_actor")
            mock_shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_signal_without_actor_name(self):
        captured_handlers = {}

        def mock_add_signal_handler(sig, handler):
            captured_handlers[sig] = handler

        mock_system = MagicMock()
        mock_system.stop = AsyncMock()

        with (
            patch("pulsing.core.helpers.asyncio.get_running_loop") as mock_loop_fn,
            patch("pulsing.core.get_system", return_value=mock_system),
            patch("pulsing.core.shutdown", new_callable=AsyncMock),
        ):
            loop = MagicMock()
            loop.add_signal_handler = mock_add_signal_handler
            mock_loop_fn.return_value = loop

            task = asyncio.create_task(run_until_signal(None))
            await asyncio.sleep(0.01)
            captured_handlers[signal.SIGINT]()
            await task

            mock_system.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_error_handled(self):
        captured_handlers = {}

        def mock_add_signal_handler(sig, handler):
            captured_handlers[sig] = handler

        mock_system = MagicMock()
        mock_system.stop = AsyncMock(side_effect=RuntimeError("stop failed"))

        with (
            patch("pulsing.core.helpers.asyncio.get_running_loop") as mock_loop_fn,
            patch("pulsing.core.get_system", return_value=mock_system),
            patch("pulsing.core.shutdown", new_callable=AsyncMock),
        ):
            loop = MagicMock()
            loop.add_signal_handler = mock_add_signal_handler
            mock_loop_fn.return_value = loop

            task = asyncio.create_task(run_until_signal("err_actor"))
            await asyncio.sleep(0.01)
            captured_handlers[signal.SIGTERM]()
            await task  # should not raise

    @pytest.mark.asyncio
    async def test_shutdown_error_handled(self):
        captured_handlers = {}

        def mock_add_signal_handler(sig, handler):
            captured_handlers[sig] = handler

        mock_system = MagicMock()
        mock_system.stop = AsyncMock()

        with (
            patch("pulsing.core.helpers.asyncio.get_running_loop") as mock_loop_fn,
            patch("pulsing.core.get_system", return_value=mock_system),
            patch(
                "pulsing.core.shutdown",
                new_callable=AsyncMock,
                side_effect=RuntimeError("shutdown failed"),
            ),
        ):
            loop = MagicMock()
            loop.add_signal_handler = mock_add_signal_handler
            mock_loop_fn.return_value = loop

            task = asyncio.create_task(run_until_signal("actor"))
            await asyncio.sleep(0.01)
            captured_handlers[signal.SIGTERM]()
            await task  # should not raise


# ============================================================================
# spawn_and_run
# ============================================================================


class TestSpawnAndRun:
    @pytest.mark.asyncio
    async def test_spawn_and_run_calls_init_and_spawn(self):
        mock_system = MagicMock()
        mock_system.spawn = AsyncMock()
        mock_system.addr = "127.0.0.1:8000"

        mock_actor = MagicMock()

        with (
            patch(
                "pulsing.core.init", new_callable=AsyncMock, return_value=mock_system
            ) as mock_init,
            patch(
                "pulsing.core.helpers.run_until_signal", new_callable=AsyncMock
            ) as mock_signal,
        ):
            await spawn_and_run(
                mock_actor,
                name="test",
                addr="0.0.0.0:9000",
                seeds=["seed:8000"],
                public=True,
            )

            mock_init.assert_awaited_once_with(addr="0.0.0.0:9000", seeds=["seed:8000"])
            mock_system.spawn.assert_awaited_once_with(
                mock_actor, name="test", public=True
            )
            mock_signal.assert_awaited_once_with("test")

    @pytest.mark.asyncio
    async def test_spawn_and_run_defaults(self):
        mock_system = MagicMock()
        mock_system.spawn = AsyncMock()
        mock_system.addr = "127.0.0.1:0"

        with (
            patch(
                "pulsing.core.init", new_callable=AsyncMock, return_value=mock_system
            ),
            patch("pulsing.core.helpers.run_until_signal", new_callable=AsyncMock),
        ):
            await spawn_and_run(MagicMock(), name="default_actor")
            mock_system.spawn.assert_awaited_once()


# ============================================================================
# run_sync
# ============================================================================


class TestRunSync:
    def test_delegates_to_async_bridge(self):
        coro = AsyncMock()
        shared_loop = object()

        with (
            patch("pulsing.core.helpers.get_shared_loop", return_value=shared_loop),
            patch(
                "pulsing.core.helpers._bridge_run_sync", return_value="shared"
            ) as bridge,
        ):
            assert run_sync(coro) == "shared"
            bridge.assert_called_once_with(
                coro,
                loop=shared_loop,
                timeout=30,
                same_loop="worker",
                missing_loop="run",
            )
