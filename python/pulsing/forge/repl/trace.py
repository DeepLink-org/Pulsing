# SPDX-License-Identifier: Apache-2.0
"""JSONL trace format for Forge session replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

TraceKind = Literal["tool_call", "forge_event", "session", "meta"]


@dataclass
class TraceRecord:
    seq: int
    kind: TraceKind
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    event: dict[str, Any] | None = None
    session: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: v for k, v in out.items() if v is not None}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TraceRecord:
        return cls(
            seq=int(raw.get("seq", 0)),
            kind=str(raw.get("kind", "meta")),  # type: ignore[arg-type]
            tool=raw.get("tool"),
            arguments=dict(raw.get("arguments") or {}) or None,
            result=dict(raw.get("result") or {}) or None,
            event=dict(raw.get("event") or {}) or None,
            session=dict(raw.get("session") or {}) or None,
            meta=dict(raw.get("meta") or {}) or None,
        )


@dataclass
class TraceLog:
    records: list[TraceRecord] = field(default_factory=list)
    _next_seq: int = 1

    def append(self, record: TraceRecord) -> None:
        if record.seq <= 0:
            record.seq = self._next_seq
            self._next_seq += 1
        else:
            self._next_seq = max(self._next_seq, record.seq + 1)
        self.records.append(record)

    def tool_calls(self) -> list[TraceRecord]:
        return [r for r in self.records if r.kind == "tool_call"]


def load_trace(path: str | Path) -> TraceLog:
    log = TraceLog()
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        log.append(TraceRecord.from_dict(json.loads(line)))
    if log.records:
        log._next_seq = max(r.seq for r in log.records) + 1
    return log


def save_trace(path: str | Path, log: TraceLog) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in log.records]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
