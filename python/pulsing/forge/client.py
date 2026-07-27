# SPDX-License-Identifier: Apache-2.0
"""Python interface to the canonical Rust Forge session control plane."""

from __future__ import annotations

from typing import Any

try:
    from pulsing._core import ForgeClient as _NativeForgeClient

    RUST_FORGE_CLIENT_AVAILABLE = True
except ImportError:
    _NativeForgeClient = None  # type: ignore[misc, assignment]
    RUST_FORGE_CLIENT_AVAILABLE = False


class ForgeClient:
    """Thin client; all Session, Turn, Agent loop, and event state lives in Rust."""

    def __init__(self) -> None:
        if not RUST_FORGE_CLIENT_AVAILABLE:
            raise RuntimeError(
                "Rust ForgeClient is required; rebuild the extension with "
                "`maturin develop`"
            )
        self._inner = _NativeForgeClient()

    def create_session(
        self,
        *,
        cwd: str = ".",
        provider: str = "demo",
        model: str = "demo",
        max_tokens: int = 8192,
        max_turns: int = 20,
        sandbox: str = "off",
        auto_approve: bool = True,
        tool_names: list[str] | tuple[str, ...] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        return str(
            self._inner.create_session(
                cwd,
                provider,
                model,
                max_tokens,
                max_turns,
                sandbox,
                list(tool_names) if tool_names is not None else None,
                system_prompt,
                auto_approve,
            )
        )

    def start_turn(self, session_id: str, input: str) -> dict[str, Any]:
        return dict(self._inner.start_turn(session_id, input))

    def wait_turn(
        self,
        session_id: str,
        turn_id: str,
        after_seq: int,
    ) -> dict[str, Any]:
        return dict(self._inner.wait_turn(session_id, turn_id, after_seq))

    def cancel_turn(self, session_id: str, turn_id: str) -> dict[str, Any]:
        return dict(self._inner.cancel_turn(session_id, turn_id))

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return dict(self._inner.snapshot(session_id))


__all__ = ["ForgeClient", "RUST_FORGE_CLIENT_AVAILABLE"]
