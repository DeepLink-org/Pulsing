# SPDX-License-Identifier: Apache-2.0
"""Rust-native Forge runtime (``pulsing._core.ForgeRuntime``) with Python fallback."""

from __future__ import annotations

from typing import Any, Callable

from pulsing.forge.events import ForgeEvent
from pulsing.forge.result import ToolResult

try:
    from pulsing._core import ForgeRuntime as _RustForgeRuntime

    RUST_FORGE_AVAILABLE = True
except ImportError:
    _RustForgeRuntime = None  # type: ignore[misc, assignment]
    RUST_FORGE_AVAILABLE = False


def rust_forge_available() -> bool:
    return RUST_FORGE_AVAILABLE


class RustForgeAdapter:
    """Thin wrapper around ``pulsing._core.ForgeRuntime``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @classmethod
    def create(
        cls,
        *,
        cwd: str,
        sandbox_policy: str = "off",
        dangerously_disable_sandbox: bool = False,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        user_input_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        exec_approval_callback: Callable[[dict[str, Any]], str] | None = None,
        request_permissions_callback: Callable[[dict[str, Any]], dict[str, Any]]
        | None = None,
        tokens_remaining_callback: Callable[[], int | None] | None = None,
        plugin_install_callback: Callable[[dict[str, Any]], bool | str] | None = None,
        auto_approve: bool = False,
        start_mcp: bool = True,
    ) -> RustForgeAdapter:
        if not RUST_FORGE_AVAILABLE:
            raise RuntimeError(
                "Rust ForgeRuntime is not available; rebuild with maturin develop"
            )
        rt = _RustForgeRuntime(
            cwd,
            sandbox_policy,
            dangerously_disable_sandbox,
            auto_approve,
            event_callback,
            user_input_callback,
            exec_approval_callback,
            request_permissions_callback,
            tokens_remaining_callback,
            plugin_install_callback,
            start_mcp,
        )
        return cls(rt)

    def tool_names(self) -> list[str]:
        return list(self._inner.tool_names())

    def refresh_mcp(self) -> None:
        self._inner.refresh_mcp()

    def mcp_tool_names(self) -> list[str]:
        if hasattr(self._inner, "mcp_tool_names"):
            return list(self._inner.mcp_tool_names())
        return []

    def mcp_tool_specs(self) -> list[dict[str, Any]]:
        if hasattr(self._inner, "mcp_tool_specs"):
            raw = self._inner.mcp_tool_specs()
            return [dict(item) for item in raw]
        return []

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        raw = self._inner.call_tool(name, dict(arguments or {}))
        return ToolResult(
            content=str(raw.get("content", "")),
            is_error=bool(raw.get("is_error")),
            structured=raw.get("structured"),
        )


def forge_event_dict(event: ForgeEvent) -> dict[str, Any]:
    return event.to_dict()
