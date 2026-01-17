"""Agent 开发工具函数"""

from __future__ import annotations

import json
from typing import Any


def parse_json(content: str | None, fallback: Any = None) -> Any:
    """
    从 LLM 输出中解析 JSON。

    支持：
    - 纯 JSON 字符串
    - ```json ... ``` 代码块包裹
    - 解析失败时返回 fallback

    Example:
        from pulsing.agent import parse_json

        # 纯 JSON
        data = parse_json('{"name": "test"}')

        # 代码块包裹
        data = parse_json('```json\\n{"name": "test"}\\n```')

        # 解析失败
        data = parse_json('invalid', fallback={})  # 返回 {}
    """
    if content is None:
        return fallback

    text = str(content).strip()
    if not text:
        return fallback

    # 处理 ```json ... ``` 包裹
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
    从 JSON 输出中提取指定字段。

    Example:
        response = '{"score": 8, "reason": "good"}'
        score = extract_field(response, "score", fallback=0)  # 8
    """
    data = parse_json(content, {})
    if isinstance(data, dict):
        return data.get(field, fallback)
    return fallback
