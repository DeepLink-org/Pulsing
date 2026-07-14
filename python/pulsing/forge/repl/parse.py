# SPDX-License-Identifier: Apache-2.0
"""REPL line parser — nu-style ``call Tool {json}`` and bare tool invocations."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from pulsing.forge.integrated import FORGE_TOOL_NAMES

_META_ALIASES = {
    "help": "help",
    "?": "help",
    "h": "help",
    "quit": "quit",
    "exit": "quit",
    "q": "quit",
    "tools": "tools",
    "ls": "tools",
    "plan": "plan",
    "session": "session",
    "events": "events",
    "approve": "approve",
    "replay": "replay",
    "trace": "trace",
    "call": "call",
}


def parse_flags(tokens: list[str]) -> dict[str, Any]:
    """Parse ``--key value`` / ``--key=value`` (Nushell-style flags, MVP subset)."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].split("=", 1)[0]
        if "=" in tok:
            out[key] = _coerce(tok.split("=", 1)[1])
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[key] = _coerce(tokens[i + 1])
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def _coerce(raw: str) -> Any:
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_json_or_flags(rest: str) -> dict[str, Any]:
    rest = rest.strip()
    if not rest:
        return {}
    if rest.startswith("{"):
        return dict(json.loads(rest))
    return parse_flags(shlex.split(rest))


def _split_tool_rest(line: str) -> tuple[str, str]:
    line = line.strip()
    if not line:
        return "", ""
    tool, _, rest = line.partition(" ")
    return tool, rest.strip()


def parse_line(line: str) -> tuple[str, dict[str, Any]]:
    """Return ``(command, args)`` where command is meta name or ``call``."""
    line = line.strip()
    if not line or line.startswith("#"):
        return ("noop", {})
    if line.startswith(":"):
        parts = shlex.split(line[1:])
        if not parts:
            return ("help", {})
        cmd = _META_ALIASES.get(parts[0].lower(), parts[0].lower())
        return (cmd, {"rest": parts[1:]})

    lower = line.lower()
    if lower in _META_ALIASES:
        return (_META_ALIASES[lower], {"rest": []})

    if lower.startswith("approve "):
        return ("approve", {"rest": shlex.split(line)[1:]})

    if lower.startswith("replay"):
        return ("replay", {"rest": shlex.split(line)[1:]})

    if lower.startswith("trace "):
        return ("trace", {"rest": shlex.split(line)[1:]})

    if line.lower().startswith("call "):
        tool, rest = _split_tool_rest(line[5:])
        if not tool:
            return ("help", {})
        args = parse_json_or_flags(rest)
        return ("call", {"tool": tool, "arguments": args})

    tool, rest = _split_tool_rest(line)
    if tool in FORGE_TOOL_NAMES:
        args = parse_json_or_flags(rest)
        return ("call", {"tool": tool, "arguments": args})

    # ``tool Read ...`` alias
    m = re.match(r"tool\s+(\S+)(?:\s+(.*))?$", line, re.I)
    if m:
        args = parse_json_or_flags(m.group(2) or "")
        return ("call", {"tool": m.group(1), "arguments": args})

    return ("unknown", {"line": line})
