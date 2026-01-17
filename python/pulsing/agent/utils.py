"""Agent development utility functions"""

from __future__ import annotations

import json
from typing import Any


def parse_json(content: str | None, fallback: Any = None) -> Any:
    """
    Parse JSON from LLM output.

    Supports:
    - Pure JSON strings
    - ```json ... ``` code block wrapped
    - Returns fallback on parse failure

    Example:
        from pulsing.agent import parse_json

        # Pure JSON
        data = parse_json('{"name": "test"}')

        # Code block wrapped
        data = parse_json('```json\\n{"name": "test"}\\n```')

        # Parse failure
        data = parse_json('invalid', fallback={})  # Returns {}
    """
    if content is None:
        return fallback

    text = str(content).strip()
    if not text:
        return fallback

    # Handle ```json ... ``` wrapping
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].lstrip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return fallback


def extract_field(content: str | None, field: str, fallback: Any = None) -> Any:
    """
    Extract specified field from JSON output.

    Example:
        response = '{"score": 8, "reason": "good"}'
        score = extract_field(response, "score", fallback=0)  # 8
    """
    data = parse_json(content, {})
    if isinstance(data, dict):
        return data.get(field, fallback)
    return fallback
