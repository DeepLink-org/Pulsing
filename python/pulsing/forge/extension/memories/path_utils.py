# SPDX-License-Identifier: Apache-2.0
"""Path helpers for local memories (codex ext/memories/local/path.rs)."""

from __future__ import annotations

import os
from pathlib import Path

from pulsing.forge.extension.memories.backend import MemoriesBackendError

MAX_READ_FILE_BYTES = 2 * 1024 * 1024


def _os_error_message(exc: OSError) -> str:
    return exc.strerror or str(exc)


def display_relative_path(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = [p for p in rel.parts if p]
    return "/".join(parts)


def is_hidden_path(path: Path) -> bool:
    name = path.name
    return bool(name.startswith("."))


def is_hidden_component(name: str) -> bool:
    return name.startswith(".")


def is_symlink(path: Path) -> bool:
    return path.is_symlink()


def reject_symlink(path: str, resolved: Path) -> None:
    if is_symlink(resolved):
        raise MemoriesBackendError.invalid_path(path, "must not be a symlink")


def validate_optional_scoped_path(path: str | None) -> str | None:
    """Normalize optional list/search scope; reject escapes before filesystem access."""
    if path is None:
        return None
    rel = str(path).strip()
    if not rel:
        return None
    if "\x00" in rel:
        raise MemoriesBackendError.invalid_path(
            rel, "must stay within the memories root"
        )
    candidate = Path(rel)
    if candidate.is_absolute():
        raise MemoriesBackendError.invalid_path(
            rel, "must stay within the memories root"
        )
    if any(part == ".." for part in candidate.parts):
        raise MemoriesBackendError.invalid_path(
            rel, "must stay within the memories root"
        )
    for part in candidate.parts:
        if is_hidden_component(part):
            raise MemoriesBackendError.not_found(rel)
    return rel


def validate_read_path(path: str) -> str:
    """Normalize and reject obvious path-escape inputs before filesystem access."""
    rel = validate_optional_scoped_path(path)
    if rel is None:
        raise MemoriesBackendError("path is required")
    return rel


def assert_readable_memory_file(
    *,
    root: Path,
    relative_path: str,
    file_path: Path,
    max_bytes: int = MAX_READ_FILE_BYTES,
) -> None:
    """Re-check symlink, root scope, and size immediately before reading."""
    meta = metadata_or_none(file_path)
    if meta is None:
        raise MemoriesBackendError.not_found(relative_path)
    reject_symlink(relative_path, file_path)
    if not file_path.resolve().is_relative_to(root.resolve()):
        raise MemoriesBackendError.invalid_path(
            relative_path, "must stay within the memories root"
        )
    if not os.path.isfile(file_path):
        raise MemoriesBackendError.not_file(relative_path)
    if meta.st_size > max_bytes:
        raise MemoriesBackendError(
            f"path '{relative_path}' exceeds {max_bytes} byte read limit ({meta.st_size} bytes)"
        )


def read_sorted_dir_paths(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        return []
    try:
        paths = sorted(dir_path.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        raise MemoriesBackendError(_os_error_message(exc), fatal=True) from exc
    return paths


def metadata_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MemoriesBackendError(_os_error_message(exc), fatal=True) from exc
