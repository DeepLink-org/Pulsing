# SPDX-License-Identifier: Apache-2.0
"""Rewrite ``pulsing actor`` argv: constructor kwargs after ``--``."""

from __future__ import annotations

import json


def _collect_key_value_pairs(tokens: list[str]) -> dict:
    extra: dict = {}
    i = 0
    while i < len(tokens):
        a = tokens[i]
        if a.startswith("--") and not a.startswith("---") and len(a) > 2:
            key = a[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                extra[key] = tokens[i + 1]
                i += 2
                continue
        i += 1
    return extra


def rewrite_actor_argv(argv: list[str]) -> list[str]:
    if len(argv) < 2 or argv[1] != "actor":
        return argv
    rest = argv[2:]
    if "--" not in rest:
        return argv
    dash_idx = rest.index("--")
    before, after = rest[:dash_idx], rest[dash_idx + 1 :]
    extra = _collect_key_value_pairs(after)
    if not extra:
        return argv
    return (
        [argv[0], "actor"] + before + ["-D", f"actor.extra_kwargs={json.dumps(extra)}"]
    )
