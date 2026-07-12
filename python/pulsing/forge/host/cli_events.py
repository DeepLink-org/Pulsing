# SPDX-License-Identifier: Apache-2.0
"""Default CLI sink for Forge + LLM streaming events."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from pulsing.forge.events import ForgeEvent, ForgeEventKind
from pulsing.forge.result import ToolResult


def _preview_args(arguments: dict[str, Any], limit: int = 120) -> str:
    try:
        text = json.dumps(arguments, ensure_ascii=False)
    except TypeError:
        text = str(arguments)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


@dataclass
class CliEventSink:
    """Print assistant text and tool activity to a terminal (Codex-like feedback)."""

    stream_assistant: bool = True
    verbose_tools: bool = True
    out: TextIO = sys.stdout
    err: TextIO = sys.stderr

    def on_assistant_delta(self, text: str) -> None:
        if self.stream_assistant and text:
            print(text, end="", flush=True, file=self.out)

    def on_assistant_end(self) -> None:
        if self.stream_assistant:
            print(file=self.out, flush=True)

    def on_tool_begin(self, name: str, arguments: dict[str, Any] | None = None) -> None:
        args = dict(arguments or {})
        if self.verbose_tools and args:
            print(f"\n→ {name}({_preview_args(args)})", flush=True, file=self.out)
        else:
            print(f"\n→ {name}", flush=True, file=self.out)

    def on_tool_end(self, name: str, result: ToolResult) -> None:
        preview = (result.content or "").replace("\n", "\\n")[:200]
        tag = "error" if result.is_error else "ok"
        print(f"← {name} [{tag}] {preview}", flush=True, file=self.out)

    def on_plan_updated(self, plan: list[dict[str, Any]]) -> None:
        if not plan:
            return
        print("\n── plan ──", file=self.out)
        for item in plan:
            status = item.get("status", "?")
            step = item.get("step", "")
            print(f"  [{status}] {step}", file=self.out)
        print("──────────", flush=True, file=self.out)

    def on_error(self, message: str) -> None:
        print(f"\n! {message}", flush=True, file=self.err)

    def on_forge_event(self, raw: dict[str, Any]) -> None:
        """Rust Forge ``event_callback`` — tool begin/end handled by Host to avoid dupes."""
        ev = ForgeEvent.from_dict(raw)
        kind = ev.kind
        if kind in (
            ForgeEventKind.TOOL_BEGIN.value,
            ForgeEventKind.TOOL_END.value,
        ):
            return
        if kind == ForgeEventKind.EXEC_OUTPUT_DELTA.value:
            chunk = str(ev.payload.get("chunk") or "")
            if chunk:
                print(chunk, end="", flush=True, file=self.out)
            return
        if kind == ForgeEventKind.PLAN_UPDATED.value:
            plan = ev.payload.get("plan")
            if isinstance(plan, list):
                self.on_plan_updated(plan)
            return
        if kind == ForgeEventKind.USER_INPUT_REQUEST.value:
            print(
                "\n? user input requested (auto-approved in ForgeAgent)", file=self.out
            )
            return
        if kind == ForgeEventKind.EXEC_APPROVAL_REQUEST.value:
            print("\n? exec approval requested (auto-approved)", file=self.out)
            return
        if kind == ForgeEventKind.REQUEST_PERMISSIONS.value:
            print("\n? permissions requested (auto-approved)", file=self.out)
