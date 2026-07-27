# SPDX-License-Identifier: Apache-2.0
"""Legacy hybrid tool adapter.

This type no longer owns the default Agent loop or Session. Keep its use
explicit while Python-only tools migrate behind Rust-registered workers.
"""

from __future__ import annotations

from typing import Any, Callable

from pulsing.forge.codex_parity import PYTHON_ONLY_HOST
from pulsing.forge.integrated import FORGE_TOOL_NAMES
from pulsing.forge.result import ToolResult
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.rust_runtime import RUST_FORGE_AVAILABLE, RustForgeAdapter
from pulsing.forge.session import LocalToolSession, ToolSession, UpdatePlanArgs

# Session tools executed on the Rust path keep state in a separate in-process
# `LocalToolSession` unless an `event_callback` is wired. Mirror successful
# calls onto the Python `ToolSession` so host/tests see one plan snapshot.
_RUST_SESSION_SYNC = frozenset({"update_plan", "new_context"})


class HybridForgeRuntime:
    """Explicit compatibility adapter for mixed Rust/Python tool execution.

    Rust executes isolated + most Host tools; Python handles tools without Rust
    handlers (``exec``, ``wait``, Extension×8) so all 32 Forge tools are callable.
    """

    def __init__(
        self,
        *,
        python: LocalToolRuntime,
        rust: RustForgeAdapter | None,
        sync_rust_session: bool = False,
    ) -> None:
        self._python = python
        self._rust = rust
        self._rust_names = (
            frozenset(rust.tool_names()) if rust is not None else frozenset()
        )
        self._sync_rust_session = sync_rust_session

    @classmethod
    def create(
        cls,
        *,
        cwd: str,
        sandbox_policy: str = "off",
        dangerously_disable_sandbox: bool = False,
        session: ToolSession | None = None,
        auto_approve: bool = False,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        user_input_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        exec_approval_callback: Callable[[dict[str, Any]], str] | None = None,
        request_permissions_callback: (
            Callable[[dict[str, Any]], dict[str, Any]] | None
        ) = None,
        tokens_remaining_callback: Callable[[], int | None] | None = None,
        plugin_install_callback: Callable[[dict[str, Any]], bool | str] | None = None,
        start_mcp: bool = True,
    ) -> HybridForgeRuntime:
        sess = session or LocalToolSession()
        if tokens_remaining_callback is None:
            session_tokens = getattr(sess, "tokens_remaining", None)
            if callable(session_tokens):
                tokens_remaining_callback = session_tokens
        catalog = getattr(sess, "tool_catalog", None)
        python = LocalToolRuntime(
            cwd=cwd,
            sandbox_policy=sandbox_policy,
            dangerously_disable_sandbox=dangerously_disable_sandbox,
            session=sess,
            tool_catalog=catalog,
        )
        rust: RustForgeAdapter | None = None
        if RUST_FORGE_AVAILABLE:
            rust = RustForgeAdapter.create(
                cwd=cwd,
                sandbox_policy=sandbox_policy,
                dangerously_disable_sandbox=dangerously_disable_sandbox,
                auto_approve=auto_approve,
                event_callback=event_callback,
                user_input_callback=user_input_callback,
                exec_approval_callback=exec_approval_callback,
                request_permissions_callback=request_permissions_callback,
                tokens_remaining_callback=tokens_remaining_callback,
                plugin_install_callback=plugin_install_callback,
                start_mcp=start_mcp,
            )
        return cls(
            python=python,
            rust=rust,
            sync_rust_session=event_callback is None,
        )

    @property
    def python_runtime(self) -> LocalToolRuntime:
        return self._python

    @property
    def rust_runtime(self) -> RustForgeAdapter | None:
        return self._rust

    def refresh_mcp(self) -> None:
        if self._rust is not None:
            self._rust.refresh_mcp()

    def close(self) -> None:
        self._python.close()

    def tool_names(self) -> list[str]:
        return sorted(FORGE_TOOL_NAMES)

    def _mirror_rust_session(self, name: str, args: dict[str, Any]) -> None:
        sess = self._python.session
        if name == "update_plan":
            try:
                parsed = UpdatePlanArgs.from_dict(args)
            except (TypeError, ValueError):
                return
            sess.update_plan(parsed)
        elif name == "new_context":
            sess.request_new_context()

    def _use_python(self, name: str) -> bool:
        from pulsing.forge.mcp.naming import is_mcp_dynamic_tool

        if is_mcp_dynamic_tool(name):
            return False
        if name in PYTHON_ONLY_HOST:
            return True
        if self._rust is None:
            return True
        if name in FORGE_TOOL_NAMES and name not in self._rust_names:
            return True
        return False

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        from pulsing.forge.mcp.naming import is_mcp_dynamic_tool

        args = dict(arguments or {})
        if self._use_python(name):
            return self._python.call_tool(name, args)
        assert self._rust is not None
        out = self._rust.call_tool(name, args)
        if (
            out.is_error
            and out.content.startswith("Unknown tool:")
            and not is_mcp_dynamic_tool(name)
        ):
            return self._python.call_tool(name, args)
        if self._sync_rust_session and not out.is_error and name in _RUST_SESSION_SYNC:
            self._mirror_rust_session(name, args)
        return out

    async def acall_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        return self.call_tool(name, arguments)
