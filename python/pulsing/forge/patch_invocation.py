# SPDX-License-Identifier: Apache-2.0
"""apply_patch argv detection and pre-apply verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pulsing.forge.patch_apply import (
    _normalize_lexically,
    _resolve_patch_path,
    apply_patch_to_fs,
)

APPLY_PATCH_COMMANDS = frozenset({"apply_patch", "applypatch"})


class MaybeApplyPatch(str, Enum):
    BODY = "body"
    IMPLICIT = "implicit"
    NOT = "not"


@dataclass
class ParsedPatch:
    patch: str
    workdir: str | None = None


def maybe_parse_apply_patch(
    argv: list[str],
) -> tuple[MaybeApplyPatch, ParsedPatch | None]:
    if len(argv) == 1 and _looks_like_patch(argv[0]):
        return MaybeApplyPatch.IMPLICIT, None
    if len(argv) == 2 and argv[0] in APPLY_PATCH_COMMANDS:
        return MaybeApplyPatch.BODY, ParsedPatch(patch=argv[1])
    if len(argv) == 3 and _is_shell(argv[0], argv[1]):
        if _looks_like_patch(argv[2]):
            return MaybeApplyPatch.IMPLICIT, None
        parsed = _extract_from_script(argv[2])
        if parsed:
            return MaybeApplyPatch.BODY, parsed
    return MaybeApplyPatch.NOT, None


def apply_parsed_patch(parsed: ParsedPatch, cwd: Path) -> str:
    effective = _resolve_effective_cwd(cwd, parsed.workdir)
    _verify_patch_text(parsed.patch, effective, cwd)
    return apply_patch_to_fs(parsed.patch, effective, root=cwd)


def _resolve_effective_cwd(cwd: Path, workdir: str | None) -> Path:
    if not workdir:
        return _normalize_lexically(cwd)
    return _resolve_patch_path(workdir, cwd, cwd)


def _verify_patch_text(patch: str, base: Path, root: Path) -> None:
    for line in patch.splitlines():
        stripped = line.strip()
        if stripped.startswith("*** Add File: "):
            rel = stripped.removeprefix("*** Add File: ").strip()
            target = _resolve_patch_path(rel, base, root)
            if target.exists():
                raise ValueError(f"add file blocked: {target} already exists")
        if stripped.startswith("*** Delete File: "):
            rel = stripped.removeprefix("*** Delete File: ").strip()
            target = _resolve_patch_path(rel, base, root)
            if not target.is_file():
                raise ValueError(f"file not found: {target}")
        if stripped.startswith("*** Update File: "):
            rel = stripped.removeprefix("*** Update File: ").strip()
            target = _resolve_patch_path(rel, base, root)
            if not target.is_file():
                raise ValueError(f"file not found: {target}")
        if stripped.startswith("*** Move to: "):
            rel = stripped.removeprefix("*** Move to: ").strip()
            _resolve_patch_path(rel, base, root)


def _looks_like_patch(text: str) -> bool:
    return "*** Begin Patch" in text and "*** End Patch" in text


def _is_shell(shell: str, flag: str) -> bool:
    name = Path(shell).stem.lower()
    return name in {"sh", "bash", "zsh"} and flag in {"-c", "-lc"}


def _extract_from_script(script: str) -> ParsedPatch | None:
    trimmed = script.strip()
    idx = trimmed.find("apply_patch")
    if idx < 0:
        idx = trimmed.find("applypatch")
    if idx < 0:
        return None
    prefix, rest = trimmed[:idx], trimmed[idx:]
    workdir = None
    pfx = prefix.strip()
    if pfx.startswith("cd ") and pfx.endswith("&&"):
        workdir = pfx[3:-2].strip().strip("'\"")
    if "<<" not in rest:
        return None
    after = rest.split("<<", 1)[1].lstrip()
    if after.startswith("'"):
        end = after.find("'", 1)
        delim = after[1:end]
        body_start = after.find("\n", end) + 1
    elif after.startswith('"'):
        end = after.find('"', 1)
        delim = after[1:end]
        body_start = after.find("\n", end) + 1
    else:
        end = next((i for i, c in enumerate(after) if c.isspace()), len(after))
        delim = after[:end]
        body_start = after.find("\n", end) + 1
    region = after[body_start:]
    marker = f"\n{delim}"
    end_idx = region.rfind(marker)
    if end_idx < 0:
        return None
    return ParsedPatch(patch=region[:end_idx], workdir=workdir)
