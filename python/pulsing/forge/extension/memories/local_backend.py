# SPDX-License-Identifier: Apache-2.0
"""Local filesystem memories backend (codex-rs/ext/memories/local)."""

from __future__ import annotations

import os
import re
import secrets
import stat
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pulsing.forge.extension.memories.backend import (
    AD_HOC_NOTES_DIR,
    AD_HOC_NOTE_FILENAME_MAX_BYTES,
    AD_HOC_NOTE_MAX_BYTES,
    AD_HOC_NOTE_SLUG_MAX_BYTES,
    DEFAULT_LIST_MAX_RESULTS,
    DEFAULT_READ_MAX_TOKENS,
    DEFAULT_SEARCH_MAX_RESULTS,
    MAX_LIST_RESULTS,
    MAX_SEARCH_CONTEXT_LINES,
    MAX_SEARCH_MATCH_WINDOW_LINES,
    MAX_SEARCH_QUERIES,
    MAX_SEARCH_RESULTS,
    TIMESTAMP_PREFIX_LEN,
    ListMemoriesResponse,
    MemoriesBackendError,
    MemoryEntry,
    MemoryEntryType,
    MemorySearchMatch,
    ReadMemoryResponse,
    SearchMatchMode,
    SearchMatchModeKind,
    SearchMemoriesResponse,
)
from pulsing.forge.extension.memories.path_utils import (
    assert_readable_memory_file,
    display_relative_path,
    is_hidden_component,
    is_hidden_path,
    is_symlink,
    metadata_or_none,
    read_sorted_dir_paths,
    reject_symlink,
)
from pulsing.forge.extension.paths import memories_root

_AD_HOC_FILENAME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]{0,79}\.md$"
)


def clamp_max_results(requested: int | None, default: int, maximum: int) -> int:
    value = int(requested if requested is not None else default)
    return max(1, min(value, maximum))


def truncate_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Approximate Codex token truncation (~4 chars per token)."""
    if max_tokens <= 0:
        max_tokens = DEFAULT_READ_MAX_TOKENS
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def default_ad_hoc_filename(slug: str) -> str:
    """Build a timestamp+slug filename with a random suffix.

    The suffix keeps concurrent/rapid calls (second-granularity timestamp,
    possibly identical slug) from colliding on the same path.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    clean = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-") or "note"
    suffix = f"-{secrets.token_hex(3)}"
    clean = clean[: AD_HOC_NOTE_SLUG_MAX_BYTES - len(suffix)]
    return f"{ts}-{clean}{suffix}.md"


@dataclass
class LocalMemoriesStore:
    root: Path

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or memories_root()).resolve()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MemoriesBackendError(str(exc), fatal=True) from exc

    @classmethod
    def from_codex_home(cls, codex_home: Path) -> LocalMemoriesStore:
        return cls(codex_home / "memories")

    def resolve_scoped_path(self, relative_path: str | None) -> Path:
        if not relative_path:
            return self.root
        relative = Path(relative_path)
        if relative.is_absolute():
            raise MemoriesBackendError.invalid_path(
                relative_path, "must stay within the memories root"
            )
        if any(p == ".." for p in relative.parts):
            raise MemoriesBackendError.invalid_path(
                relative_path, "must stay within the memories root"
            )
        for part in relative.parts:
            if is_hidden_component(part):
                raise MemoriesBackendError.not_found(relative_path)

        scoped = self.root
        components = list(relative.parts)
        for idx, part in enumerate(components):
            scoped = scoped / part
            meta = metadata_or_none(scoped)
            if meta is None:
                if idx + 1 < len(components):
                    for remaining in components[idx + 1 :]:
                        scoped = scoped / remaining
                return scoped.resolve()
            reject_symlink(display_relative_path(self.root, scoped), scoped)
            if not scoped.resolve().is_relative_to(self.root):
                raise MemoriesBackendError.invalid_path(
                    relative_path, "must stay within the memories root"
                )
            if idx + 1 < len(components) and not os.path.isdir(scoped):
                raise MemoriesBackendError.invalid_path(
                    relative_path,
                    "traverses through a non-directory path component",
                )
        return scoped.resolve()

    def list_memories(
        self,
        *,
        path: str | None = None,
        cursor: str | None = None,
        max_results: int | None = None,
    ) -> ListMemoriesResponse:
        limit = clamp_max_results(
            max_results, DEFAULT_LIST_MAX_RESULTS, MAX_LIST_RESULTS
        )
        start_path = self.resolve_scoped_path(path)
        start_index = _parse_cursor(cursor)

        meta = metadata_or_none(start_path)
        if meta is None:
            raise MemoriesBackendError.not_found(path or "")

        reject_symlink(display_relative_path(self.root, start_path), start_path)

        if os.path.isfile(start_path):
            entries = [
                MemoryEntry(
                    path=display_relative_path(self.root, start_path),
                    entry_type=MemoryEntryType.FILE,
                )
            ]
        elif os.path.isdir(start_path):
            entries = []
            for child in read_sorted_dir_paths(start_path):
                if is_hidden_path(child):
                    continue
                child_meta = metadata_or_none(child)
                if child_meta is None or is_symlink(child):
                    continue
                if stat.S_ISDIR(child_meta.st_mode):
                    entry_type = MemoryEntryType.DIRECTORY
                elif stat.S_ISREG(child_meta.st_mode):
                    entry_type = MemoryEntryType.FILE
                else:
                    continue
                entries.append(
                    MemoryEntry(
                        path=display_relative_path(self.root, child),
                        entry_type=entry_type,
                    )
                )
        else:
            entries = []

        if start_index > len(entries):
            raise MemoriesBackendError.invalid_cursor(
                str(start_index), "exceeds result count"
            )

        end_index = min(start_index + limit, len(entries))
        next_cursor = str(end_index) if end_index < len(entries) else None
        return ListMemoriesResponse(
            path=path,
            entries=entries[start_index:end_index],
            next_cursor=next_cursor,
            truncated=next_cursor is not None,
        )

    def read_memory(
        self,
        *,
        path: str,
        line_offset: int = 1,
        max_lines: int | None = None,
        max_tokens: int = DEFAULT_READ_MAX_TOKENS,
    ) -> ReadMemoryResponse:
        if line_offset < 1:
            raise MemoriesBackendError("line_offset must be a 1-indexed line number")
        if max_lines is not None and max_lines < 1:
            raise MemoriesBackendError("max_lines must be a positive integer")

        file_path = self.resolve_scoped_path(path)
        assert_readable_memory_file(
            root=self.root, relative_path=path, file_path=file_path
        )

        try:
            original = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise MemoriesBackendError(
                f"path '{path}' is not valid UTF-8 text"
            ) from exc
        start_byte = _line_start_byte_offset(original, line_offset)
        end_byte = _line_end_byte_offset(original, start_byte, max_lines)
        slice_text = original[start_byte:end_byte]
        content, token_truncated = truncate_tokens(slice_text, max_tokens)
        truncated = end_byte < len(original) or token_truncated or content != slice_text
        return ReadMemoryResponse(
            path=path,
            start_line_number=line_offset,
            content=content,
            truncated=truncated,
        )

    def search_memories(
        self,
        *,
        queries: list[str],
        match_mode: SearchMatchMode | None = None,
        path: str | None = None,
        cursor: str | None = None,
        context_lines: int = 0,
        case_sensitive: bool = True,
        normalized: bool = False,
        max_results: int | None = None,
    ) -> SearchMemoriesResponse:
        cleaned = [q.strip() for q in queries]
        if not cleaned:
            raise MemoriesBackendError("at least one query is required")
        if any(not q for q in cleaned):
            raise MemoriesBackendError("queries must not contain empty strings")
        if len(cleaned) > MAX_SEARCH_QUERIES:
            raise MemoriesBackendError(
                f"queries must contain at most {MAX_SEARCH_QUERIES} entries"
            )
        mode = match_mode or SearchMatchMode()
        if mode.kind == SearchMatchModeKind.ALL_WITHIN_LINES:
            if mode.line_count <= 0:
                raise MemoriesBackendError(
                    "all_within_lines.line_count must be a positive integer"
                )
            if mode.line_count > MAX_SEARCH_MATCH_WINDOW_LINES:
                raise MemoriesBackendError(
                    f"all_within_lines.line_count must be at most {MAX_SEARCH_MATCH_WINDOW_LINES}"
                )

        limit = clamp_max_results(
            max_results, DEFAULT_SEARCH_MAX_RESULTS, MAX_SEARCH_RESULTS
        )
        bounded_context_lines = max(0, min(context_lines, MAX_SEARCH_CONTEXT_LINES))
        start_path = self.resolve_scoped_path(path)
        start_index = _parse_cursor(cursor)

        meta = metadata_or_none(start_path)
        if meta is None:
            raise MemoriesBackendError.not_found(path or "")

        reject_symlink(display_relative_path(self.root, start_path), start_path)

        matcher = _SearchMatcher(cleaned, mode, case_sensitive, normalized)
        matches: list[MemorySearchMatch] = []
        _search_entries(
            self.root, start_path, meta, matcher, bounded_context_lines, matches
        )

        matches.sort(key=lambda m: (m.path, m.match_line_number))
        if start_index > len(matches):
            raise MemoriesBackendError.invalid_cursor(
                str(start_index), "exceeds result count"
            )

        end_index = min(start_index + limit, len(matches))
        next_cursor = str(end_index) if end_index < len(matches) else None
        return SearchMemoriesResponse(
            queries=cleaned,
            match_mode=mode,
            path=path,
            matches=matches[start_index:end_index],
            next_cursor=next_cursor,
            truncated=next_cursor is not None,
        )

    def add_ad_hoc_note(self, *, filename: str, note: str) -> dict[str, object]:
        _validate_ad_hoc_filename(filename)
        if not note.strip():
            raise MemoriesBackendError("ad-hoc note must not be empty")
        note_bytes = note.encode("utf-8")
        if len(note_bytes) > AD_HOC_NOTE_MAX_BYTES:
            raise MemoriesBackendError(
                f"ad-hoc note exceeds {AD_HOC_NOTE_MAX_BYTES} byte limit"
                f" ({len(note_bytes)} bytes given)"
            )

        notes_dir = self._ensure_notes_dir()
        dest = notes_dir / filename
        if dest.parent != notes_dir:
            raise MemoriesBackendError.invalid_filename(
                filename, "must not contain path separators"
            )
        # O_EXCL makes create-if-absent atomic, closing the check-then-write
        # race that a plain `exists()` check followed by `write_text()` would
        # leave open between two concurrent callers targeting the same name.
        try:
            fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise MemoriesBackendError(
                f"ad-hoc note '{filename}' already exists"
            ) from exc
        except OSError as exc:
            raise MemoriesBackendError.write_failed(filename, exc) from exc

        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(note_bytes)
        except OSError as exc:
            with suppress(OSError):
                dest.unlink()
            raise MemoriesBackendError.write_failed(filename, exc) from exc
        return {}

    def _ensure_notes_dir(self) -> Path:
        path = self.root
        for component in AD_HOC_NOTES_DIR:
            path = path / component
            meta = metadata_or_none(path)
            if meta is None:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise MemoriesBackendError.write_failed(str(path), exc) from exc
                meta = metadata_or_none(path)
            if meta is None:
                raise MemoriesBackendError.not_found(str(path))
            reject_symlink(str(path), path)
            if not os.path.isdir(path):
                raise MemoriesBackendError.invalid_path(
                    str(path), "must be a directory"
                )
        return path


def _parse_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        value = int(cursor)
    except ValueError as exc:
        raise MemoriesBackendError.invalid_cursor(
            cursor, "must be a non-negative integer"
        ) from exc
    if value < 0:
        raise MemoriesBackendError.invalid_cursor(
            cursor, "must be a non-negative integer"
        )
    return value


def _line_start_byte_offset(content: str, line_offset: int) -> int:
    if line_offset == 1:
        return 0
    current = 1
    for idx, ch in enumerate(content):
        if ch == "\n":
            current += 1
            if current == line_offset:
                return idx + 1
    raise MemoriesBackendError("line_offset exceeds file length")


def _line_end_byte_offset(content: str, start_byte: int, max_lines: int | None) -> int:
    if max_lines is None:
        return len(content)
    lines_seen = 1
    for relative_idx, ch in enumerate(content[start_byte:]):
        if ch == "\n":
            if lines_seen == max_lines:
                return start_byte + relative_idx + 1
            lines_seen += 1
    return len(content)


def _validate_ad_hoc_filename(filename: str) -> None:
    if len(filename.encode("utf-8")) > AD_HOC_NOTE_FILENAME_MAX_BYTES:
        raise MemoriesBackendError.invalid_filename(
            filename, "must be at most 128 bytes"
        )
    if not filename.endswith(".md"):
        raise MemoriesBackendError.invalid_filename(filename, "must end with .md")
    if not _AD_HOC_FILENAME_RE.match(filename):
        raise MemoriesBackendError.invalid_filename(
            filename,
            "must use YYYY-MM-DDTHH-MM-SS-<slug>.md with lowercase slug",
        )
    stem = filename[:-3]
    slug = stem[TIMESTAMP_PREFIX_LEN:]
    if not slug or len(slug.encode("utf-8")) > AD_HOC_NOTE_SLUG_MAX_BYTES:
        raise MemoriesBackendError.invalid_filename(
            filename, "slug must be 1 to 80 bytes"
        )


class _SearchMatcher:
    def __init__(
        self,
        queries: list[str],
        match_mode: SearchMatchMode,
        case_sensitive: bool,
        normalized: bool,
    ) -> None:
        self.queries = queries
        self.match_mode = match_mode
        self.case_sensitive = case_sensitive
        self.normalized = normalized
        self.prepared = [_prepare_query(q, case_sensitive, normalized) for q in queries]
        empty = next(
            (q for q, p in zip(queries, self.prepared, strict=True) if not p), None
        )
        if empty is not None:
            raise MemoriesBackendError(f"query {empty!r} is empty after normalization")

    def matched_query_flags(self, line: str) -> list[bool]:
        hay = _prepare_query(line, self.case_sensitive, self.normalized)
        return [q in hay for q in self.prepared]

    def matched_queries(self, flags: list[bool]) -> list[str]:
        return [q for q, ok in zip(self.queries, flags, strict=True) if ok]


def _prepare_query(value: str, case_sensitive: bool, normalized: bool) -> str:
    out = value if case_sensitive else value.lower()
    if normalized:
        out = "".join(ch for ch in out if ch.isalnum())
    return out


def _search_entries(
    root: Path,
    current: Path,
    meta: os.stat_result,
    matcher: _SearchMatcher,
    context_lines: int,
    matches: list[MemorySearchMatch],
) -> None:
    if os.path.isfile(current):
        _search_file(root, current, matcher, context_lines, matches)
        return
    if not os.path.isdir(current):
        return

    pending = [current]
    while pending:
        dir_path = pending.pop()
        for path in read_sorted_dir_paths(dir_path):
            if is_hidden_path(path):
                continue
            child_meta = metadata_or_none(path)
            if child_meta is None or is_symlink(path):
                continue
            if os.path.isdir(path):
                pending.append(path)
            elif os.path.isfile(path):
                _search_file(root, path, matcher, context_lines, matches)


def _search_file(
    root: Path,
    path: Path,
    matcher: _SearchMatcher,
    context_lines: int,
    matches: list[MemorySearchMatch],
) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    except OSError:
        return

    lines = content.splitlines()
    line_matches = [matcher.matched_query_flags(line) for line in lines]
    mode = matcher.match_mode.kind

    if mode == SearchMatchModeKind.ANY:
        for idx, flags in enumerate(line_matches):
            if any(flags):
                matches.append(
                    _build_match(
                        root, path, lines, idx, idx, context_lines, matcher, flags
                    )
                )
    elif mode == SearchMatchModeKind.ALL_ON_SAME_LINE:
        for idx, flags in enumerate(line_matches):
            if all(flags):
                matches.append(
                    _build_match(
                        root, path, lines, idx, idx, context_lines, matcher, flags
                    )
                )
    elif mode == SearchMatchModeKind.ALL_WITHIN_LINES:
        window = matcher.match_mode.line_count
        windows: list[tuple[int, int, list[bool]]] = []
        for start in range(len(lines)):
            if not any(line_matches[start]):
                continue
            last = min(start + window - 1, len(lines) - 1)
            flags = [False] * len(matcher.queries)
            for end in range(start, last + 1):
                for i, matched in enumerate(line_matches[end]):
                    flags[i] = flags[i] or matched
                if all(flags):
                    windows.append((start, end, flags.copy()))
                    break
        # windows are already ordered by ascending start; a window is redundant
        # (dominated) iff a later, narrower window is fully contained in it, so a
        # single backward sweep with a running minimum end suffices (avoids O(n^2)
        # all-pairs containment checks on files with many matching lines).
        min_end_after = None
        kept: list[tuple[int, int, list[bool]]] = []
        for start, end, flags in reversed(windows):
            if min_end_after is not None and min_end_after <= end:
                continue
            kept.append((start, end, flags))
            min_end_after = end if min_end_after is None else min(min_end_after, end)
        for start, end, flags in reversed(kept):
            matches.append(
                _build_match(
                    root, path, lines, start, end, context_lines, matcher, flags
                )
            )


def _build_match(
    root: Path,
    path: Path,
    lines: list[str],
    match_start: int,
    match_end: int,
    context_lines: int,
    matcher: _SearchMatcher,
    flags: list[bool],
) -> MemorySearchMatch:
    content_start = max(0, match_start - context_lines)
    content_end = min(len(lines), match_end + context_lines + 1)
    return MemorySearchMatch(
        path=display_relative_path(root, path),
        match_line_number=match_start + 1,
        content_start_line_number=content_start + 1,
        content="\n".join(lines[content_start:content_end]),
        matched_queries=matcher.matched_queries(flags),
    )
