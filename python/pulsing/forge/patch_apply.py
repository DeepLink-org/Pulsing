# SPDX-License-Identifier: Apache-2.0
"""Minimal Codex apply_patch format support (local fs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class _UpdateChunk:
    old_lines: list[str]
    new_lines: list[str]
    change_context: str | None = None
    is_end_of_file: bool = False


def apply_patch_to_fs(patch: str, cwd: Path, *, root: Path | None = None) -> str:
    boundary = root or cwd
    hunks = _parse_patch(patch)
    if not hunks:
        raise ValueError("No files were modified.")
    added: list[Path] = []
    modified: list[Path] = []
    deleted: list[Path] = []
    for hunk in hunks:
        if hunk[0] == "add":
            path = _resolve_patch_path(hunk[1], cwd, boundary)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(hunk[2], encoding="utf-8")
            added.append(path)
        elif hunk[0] == "delete":
            path = _resolve_patch_path(hunk[1], cwd, boundary)
            if path.is_dir():
                raise ValueError(f"Refusing to delete directory {path}")
            path.unlink(missing_ok=False)
            deleted.append(path)
        elif hunk[0] == "update":
            path = _resolve_patch_path(hunk[1], cwd, boundary)
            text = path.read_text(encoding="utf-8")
            new_text = _apply_update_chunks(text, hunk[2])
            move_to = hunk[3]
            if move_to is not None:
                dest = _resolve_patch_path(move_to, cwd, boundary)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(new_text, encoding="utf-8")
                path.unlink(missing_ok=False)
                modified.append(dest)
            else:
                path.write_text(new_text, encoding="utf-8")
                modified.append(path)
    lines: list[str] = []
    for p in added:
        lines.append(f"A {p}")
    for p in modified:
        lines.append(f"M {p}")
    for p in deleted:
        lines.append(f"D {p}")
    return "\n".join(lines)


def _resolve_patch_path(rel: str | Path, base: Path, root: Path) -> Path:
    p = Path(rel)
    joined = p if p.is_absolute() else base / p
    target = _normalize_lexically(joined)
    boundary = _normalize_lexically(root)
    if target != boundary and boundary not in target.parents:
        raise ValueError(
            f"refusing to apply patch outside working directory: {target} (cwd: {boundary})"
        )
    resolved = target.resolve()
    root_resolved = boundary.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(
            f"refusing to apply patch outside working directory: {target} (cwd: {boundary})"
        )
    return target


def _normalize_lexically(path: Path) -> Path:
    out = Path("")
    for part in path.parts:
        if part == ".":
            continue
        if part == "..":
            if out.parts and out.parts[-1] not in ("..", ""):
                out = out.parent
            else:
                out = out / part
            continue
        out = out / part
    return out


def _apply_update_chunks(text: str, chunks: list[_UpdateChunk]) -> str:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for chunk in chunks:
        if chunk.change_context:
            idx = _seek(lines, [chunk.change_context], 0, False)
            if idx is None:
                raise ValueError(f"Failed to find context '{chunk.change_context}'")
            start = idx + 1
        else:
            start = 0
        if not chunk.old_lines:
            insert_at = len(lines) if not lines or lines[-1] != "" else len(lines) - 1
            for i, nl in enumerate(chunk.new_lines):
                lines.insert(insert_at + i, nl)
            continue
        found = _seek(lines, chunk.old_lines, start, chunk.is_end_of_file)
        if found is None and chunk.old_lines and chunk.old_lines[-1] == "":
            found = _seek(lines, chunk.old_lines[:-1], start, chunk.is_end_of_file)
            new_lines = (
                chunk.new_lines[:-1]
                if chunk.new_lines and chunk.new_lines[-1] == ""
                else chunk.new_lines
            )
        else:
            new_lines = chunk.new_lines
        if found is None:
            raise ValueError("Failed to find expected lines in update hunk")
        del lines[found : found + len(chunk.old_lines)]
        for i, nl in enumerate(new_lines):
            lines.insert(found + i, nl)
    if not lines or lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


def _seek(lines: list[str], pattern: list[str], start: int, eof: bool) -> int | None:
    if not pattern:
        return start
    search_start = (
        max(0, len(lines) - len(pattern))
        if eof and len(lines) >= len(pattern)
        else start
    )
    for i in range(search_start, len(lines) - len(pattern) + 1):
        if lines[i : i + len(pattern)] == pattern:
            return i
        if all(
            lines[i + j].rstrip() == pattern[j].rstrip() for j in range(len(pattern))
        ):
            return i
        if all(lines[i + j].strip() == pattern[j].strip() for j in range(len(pattern))):
            return i
    return None


def _parse_patch(patch: str) -> list[tuple]:
    lines = patch.strip().splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("The first line of the patch must be '*** Begin Patch'")
    if lines[-1].strip() != "*** End Patch":
        raise ValueError("The last line of the patch must be '*** End Patch'")
    hunks: list[tuple] = []
    i = 1
    while i < len(lines) - 1:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("*** Add File: "):
            path = stripped[len("*** Add File: ") :].strip()
            i += 1
            content_lines: list[str] = []
            while i < len(lines) - 1 and lines[i].startswith("+"):
                content_lines.append(lines[i][1:])
                i += 1
            content = "\n".join(content_lines)
            if content and not content.endswith("\n"):
                content += "\n"
            hunks.append(("add", path, content))
            continue
        if stripped.startswith("*** Delete File: "):
            path = stripped[len("*** Delete File: ") :].strip()
            hunks.append(("delete", path))
            i += 1
            continue
        if stripped.startswith("*** Update File: "):
            path = stripped[len("*** Update File: ") :].strip()
            i += 1
            move_to: str | None = None
            if i < len(lines) - 1 and lines[i].strip().startswith("*** Move to: "):
                move_to = lines[i].strip()[len("*** Move to: ") :].strip()
                i += 1
            chunks: list[_UpdateChunk] = []
            while i < len(lines) - 1 and not lines[i].strip().startswith("***"):
                line = lines[i]
                if line.strip() == "*** End of File":
                    if chunks:
                        chunks[-1].is_end_of_file = True
                    i += 1
                    continue
                if line.startswith("@@"):
                    ctx = line[2:].strip() or None
                    chunks.append(
                        _UpdateChunk(old_lines=[], new_lines=[], change_context=ctx)
                    )
                    i += 1
                    continue
                if not chunks:
                    chunks.append(_UpdateChunk(old_lines=[], new_lines=[]))
                if line.startswith(" "):
                    chunks[-1].old_lines.append(line[1:])
                    chunks[-1].new_lines.append(line[1:])
                elif line.startswith("-"):
                    chunks[-1].old_lines.append(line[1:])
                elif line.startswith("+"):
                    chunks[-1].new_lines.append(line[1:])
                i += 1
            hunks.append(("update", path, chunks, move_to))
            continue
        i += 1
    return hunks
