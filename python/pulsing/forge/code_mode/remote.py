# SPDX-License-Identifier: Apache-2.0
"""Remote Code Mode client — exec/wait via CodeCellRegistryActor."""

from __future__ import annotations

from typing import Any

from pulsing._async_bridge import run_sync
from pulsing.forge.code_mode.protocol import RuntimeResponse, WaitArgs
from pulsing.forge.code_mode.service import CodeModeService
from pulsing.forge.code_mode.tools_bridge import ToolsBridge
from pulsing.forge.result import ToolResult


class RemoteCodeModeClient:
    """Host-side facade that delegates cells to a registry actor."""

    def __init__(self, registry_name: str, host_name: str) -> None:
        self._registry_name = registry_name
        self._host_name = host_name

    def execute(self, source: str, tools: ToolsBridge) -> RuntimeResponse:
        del tools  # registry builds its own bridge from host_name
        raw = run_sync(self._ask_execute(source), timeout=120.0)
        return RuntimeResponse.from_dict(dict(raw))

    def wait(self, args: WaitArgs) -> RuntimeResponse:
        payload: dict[str, Any] = {
            "cell_id": args.cell_id,
            "yield_time_ms": args.yield_time_ms,
            "terminate": args.terminate,
        }
        if args.max_tokens is not None:
            payload["max_tokens"] = args.max_tokens
        raw = run_sync(self._ask_wait(payload), timeout=120.0)
        return RuntimeResponse.from_dict(dict(raw))

    async def _ask_execute(self, source: str) -> dict[str, Any]:
        import pulsing as pul

        proxy = await pul.resolve(self._registry_name, timeout=30.0)
        return await proxy.execute(source, host_name=self._host_name)

    async def _ask_wait(self, payload: dict[str, Any]) -> dict[str, Any]:
        import pulsing as pul

        proxy = await pul.resolve(self._registry_name, timeout=30.0)
        return await proxy.wait(payload)


class LocalCodeModeClient(CodeModeService):
    """Alias for in-process registry (tests)."""

    pass
