# SPDX-License-Identifier: Apache-2.0
"""In-process code cell state (Actor-ready; no pulsing actor dependency in Forge)."""

from __future__ import annotations

import builtins
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pulsing.forge.code_mode.exceptions import CellExit, CellYield
from pulsing.forge.code_mode.protocol import (
    CellId,
    ContentItem,
    ParsedExecSource,
    RuntimeResponse,
)
from pulsing.forge.code_mode.tools_bridge import ToolsBridge

# Cells have no OS-level sandbox (see code_mode/handlers.py:_rejects_sandbox_policy),
# so this is defense-in-depth only: it blocks the obvious escapes (import os,
# open, eval/exec/compile, __import__) but is NOT a real security boundary —
# attribute-based tricks (e.g. `().__class__.__mro__`) are a language feature
# that no builtins allowlist can remove. Real isolation needs a process/OS
# boundary, tracked as a follow-up (see codex_parity.py notes for "exec").
_ALLOWED_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "callable",
        "chr",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
        "NotImplemented",
        "BaseException",
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "StopIteration",
        "RuntimeError",
        "ArithmeticError",
        "ZeroDivisionError",
        "AttributeError",
        "NotImplementedError",
        "OverflowError",
        "LookupError",
        "AssertionError",
        "NameError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "StopAsyncIteration",
        "GeneratorExit",
    }
)


def _restricted_builtins() -> dict[str, Any]:
    return {
        name: getattr(builtins, name)
        for name in _ALLOWED_BUILTINS
        if hasattr(builtins, name)
    }


class CellStatus(str, Enum):
    RUNNING = "running"
    YIELDED = "yielded"
    DONE = "done"
    TERMINATED = "terminated"
    ERROR = "error"


_TERMINAL_STATUSES = (CellStatus.DONE, CellStatus.ERROR, CellStatus.TERMINATED)


@dataclass
class CodeCell:
    cell_id: CellId
    parsed: ParsedExecSource
    stored_values: dict[str, Any] = field(default_factory=dict)
    content_items: list[ContentItem] = field(default_factory=list)
    status: CellStatus = CellStatus.RUNNING
    error_text: str | None = None
    _terminate_requested: bool = field(default=False, repr=False)
    _segment_index: int = field(default=0, repr=False)
    _tools: ToolsBridge | None = field(default=None, repr=False)

    @classmethod
    def new(cls, parsed: ParsedExecSource) -> CodeCell:
        cid = CellId(f"cell-{uuid.uuid4().hex[:12]}")
        return cls(cell_id=cid, parsed=parsed)

    def append_text(self, value: Any) -> None:
        if value is None:
            text = "null"
        elif isinstance(value, str):
            text = value
        else:
            import json

            try:
                text = json.dumps(value, ensure_ascii=False)
            except TypeError:
                text = str(value)
        self.content_items.append(ContentItem(text=text))

    def build_namespace(self, tools: ToolsBridge) -> dict[str, Any]:
        cell = self

        def text(value: Any) -> None:
            cell.append_text(value)

        def store(key: str, value: Any) -> None:
            cell.stored_values[str(key)] = value

        def load(key: str) -> Any:
            return cell.stored_values.get(str(key))

        def yield_control() -> None:
            raise CellYield()

        def exit() -> None:
            raise CellExit()

        def notify(value: Any) -> None:
            cell.append_text(value)

        return {
            "__builtins__": _restricted_builtins(),
            "tools": tools,
            "text": text,
            "store": store,
            "load": load,
            "yield_control": yield_control,
            "exit": exit,
            "notify": notify,
            "ALL_TOOLS": sorted(tools._allowed),
        }

    def _segments(self) -> list[str]:
        segs = self.parsed.segments
        return segs if segs else [self.parsed.source]

    def run(self, tools: ToolsBridge) -> None:
        if self.status in _TERMINAL_STATUSES:
            return
        if self.status == CellStatus.YIELDED:
            self.status = CellStatus.RUNNING
        self._tools = tools
        namespace = self.build_namespace(tools)
        segments = self._segments()
        while self._segment_index < len(segments):
            if self._terminate_requested:
                break
            try:
                src = segments[self._segment_index]
                exec(compile(src, "<forge-exec>", "exec"), namespace, namespace)
            except CellYield:
                self._segment_index += 1
                self.status = CellStatus.YIELDED
                return
            except CellExit:
                self.status = CellStatus.DONE
                return
            except Exception as exc:
                self.status = CellStatus.ERROR
                message = str(exc)
                self.error_text = (
                    f"{type(exc).__name__}: {message}"
                    if message
                    else type(exc).__name__
                )
                return
            self._segment_index += 1
        if not self._terminate_requested and self.status == CellStatus.RUNNING:
            self.status = CellStatus.DONE

    def mark_terminated(self) -> None:
        """Request termination; no-op if the cell already reached a terminal state.

        Codex ``wait(..., terminate=true)`` on an already-finished cell must not
        clobber its result/error with a bogus ``terminated`` status.
        """
        if self.status in _TERMINAL_STATUSES:
            return
        self.status = CellStatus.TERMINATED
        self._terminate_requested = True

    def to_response(self, *, max_tokens: int | None = None) -> RuntimeResponse:
        items = self.content_items
        if max_tokens is not None and max_tokens > 0:
            budget = max_tokens
            trimmed: list[ContentItem] = []
            for item in items:
                if budget <= 0:
                    break
                text = item.text[:budget]
                trimmed.append(ContentItem(text=text))
                budget -= len(text)
            items = trimmed

        if self.status == CellStatus.ERROR:
            return RuntimeResponse(
                kind="result",
                cell_id=self.cell_id,
                content_items=items,
                error_text=self.error_text,
            )
        if self.status == CellStatus.TERMINATED:
            return RuntimeResponse(
                kind="terminated", cell_id=self.cell_id, content_items=items
            )
        if self.status == CellStatus.YIELDED:
            return RuntimeResponse(
                kind="yielded", cell_id=self.cell_id, content_items=items
            )
        return RuntimeResponse(kind="result", cell_id=self.cell_id, content_items=items)
