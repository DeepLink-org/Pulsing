# SPDX-License-Identifier: Apache-2.0
"""Memories backend wire types (aligned with codex-rs/ext/memories)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

DEFAULT_LIST_MAX_RESULTS = 2000
MAX_LIST_RESULTS = 2000
DEFAULT_SEARCH_MAX_RESULTS = 200
MAX_SEARCH_RESULTS = 200
MAX_SEARCH_QUERIES = 16
MAX_SEARCH_CONTEXT_LINES = 50
MAX_SEARCH_MATCH_WINDOW_LINES = 200
DEFAULT_READ_MAX_TOKENS = 20_000

AD_HOC_NOTES_DIR = ("extensions", "ad_hoc", "notes")
AD_HOC_NOTE_FILENAME_MAX_BYTES = 128
AD_HOC_NOTE_SLUG_MAX_BYTES = 80
AD_HOC_NOTE_MAX_BYTES = 256 * 1024
TIMESTAMP_PREFIX_LEN = len("YYYY-MM-DDTHH-MM-SS-")


class MemoryEntryType(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"


class SearchMatchModeKind(str, Enum):
    ANY = "any"
    ALL_ON_SAME_LINE = "all_on_same_line"
    ALL_WITHIN_LINES = "all_within_lines"


@dataclass
class SearchMatchMode:
    kind: SearchMatchModeKind = SearchMatchModeKind.ANY
    line_count: int = 1

    @classmethod
    def from_wire(cls, raw: Any) -> SearchMatchMode:
        if raw is None:
            return cls()
        if isinstance(raw, str):
            try:
                return cls(kind=SearchMatchModeKind(raw))
            except ValueError:
                return cls()
        if isinstance(raw, dict):
            typ = raw.get("type") or raw.get("kind") or "any"
            if typ == "all_within_lines":
                raw_line_count = raw.get("line_count")
                line_count = int(raw_line_count if raw_line_count is not None else 1)
                return cls(
                    kind=SearchMatchModeKind.ALL_WITHIN_LINES,
                    line_count=line_count,
                )
            try:
                return cls(kind=SearchMatchModeKind(str(typ)))
            except ValueError:
                return cls()
        return cls()

    def to_wire(self) -> dict[str, Any]:
        if self.kind == SearchMatchModeKind.ALL_WITHIN_LINES:
            return {"type": "all_within_lines", "line_count": self.line_count}
        return {"type": self.kind.value}


@dataclass(frozen=True)
class MemoryEntry:
    path: str
    entry_type: MemoryEntryType

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "entry_type": self.entry_type.value}


@dataclass
class ListMemoriesResponse:
    path: str | None
    entries: list[MemoryEntry]
    next_cursor: str | None = None
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "entries": [e.to_dict() for e in self.entries],
            "next_cursor": self.next_cursor,
            "truncated": self.truncated,
        }


@dataclass
class ReadMemoryResponse:
    path: str
    start_line_number: int
    content: str
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line_number": self.start_line_number,
            "content": self.content,
            "truncated": self.truncated,
        }


@dataclass
class MemorySearchMatch:
    path: str
    match_line_number: int
    content_start_line_number: int
    content: str
    matched_queries: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "match_line_number": self.match_line_number,
            "content_start_line_number": self.content_start_line_number,
            "content": self.content,
            "matched_queries": self.matched_queries,
        }


@dataclass
class SearchMemoriesResponse:
    queries: list[str]
    match_mode: SearchMatchMode
    path: str | None
    matches: list[MemorySearchMatch]
    next_cursor: str | None = None
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "match_mode": self.match_mode.to_wire(),
            "path": self.path,
            "matches": [m.to_dict() for m in self.matches],
            "next_cursor": self.next_cursor,
            "truncated": self.truncated,
        }


class MemoriesBackendError(Exception):
    """Model-facing memory backend error (maps to Codex RespondToModel errors)."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal

    @classmethod
    def invalid_filename(cls, filename: str, reason: str) -> MemoriesBackendError:
        return cls(f"filename '{filename}' {reason}")

    @classmethod
    def invalid_path(cls, path: str, reason: str) -> MemoriesBackendError:
        return cls(f"path '{path}' {reason}")

    @classmethod
    def invalid_cursor(cls, cursor: str, reason: str) -> MemoriesBackendError:
        return cls(f"cursor '{cursor}' {reason}")

    @classmethod
    def not_found(cls, path: str) -> MemoriesBackendError:
        return cls(f"path '{path}' was not found")

    @classmethod
    def not_file(cls, path: str) -> MemoriesBackendError:
        return cls(f"path '{path}' is not a file")

    @classmethod
    def write_failed(cls, filename: str, exc: OSError) -> MemoriesBackendError:
        reason = exc.strerror or str(exc)
        return cls(f"failed to write ad-hoc note '{filename}': {reason}")
