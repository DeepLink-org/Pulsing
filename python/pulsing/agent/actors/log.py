# SPDX-License-Identifier: Apache-2.0
"""Per-agent scroll log (in-memory ring buffer)."""

from __future__ import annotations

import time
from typing import Any

_MAX_LINES = 400


def init_log(agent: Any) -> None:
    agent._log_seq = 0
    agent._log_lines: list[tuple[int, str]] = []


def append_log(agent: Any, message: str) -> None:
    text = (message or "").strip()
    if not text:
        return
    agent._log_seq = int(getattr(agent, "_log_seq", 0)) + 1
    stamp = time.strftime("%H:%M:%S")
    agent._log_lines.append((agent._log_seq, f"[{stamp}] {text}"))
    if len(agent._log_lines) > _MAX_LINES:
        agent._log_lines = agent._log_lines[-_MAX_LINES:]


def get_logs(agent: Any, *, since: int = 0) -> dict[str, Any]:
    since = max(0, int(since))
    lines = [text for seq, text in agent._log_lines if seq > since]
    return {
        "name": getattr(agent, "_cluster_short_name", "") or "",
        "since": since,
        "next": int(getattr(agent, "_log_seq", 0)),
        "lines": lines,
    }
