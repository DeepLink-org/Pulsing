# SPDX-License-Identifier: Apache-2.0
"""JSON-schema helpers for Forge LLM tool definitions."""

from __future__ import annotations

from typing import Any


def json_schema_object(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
