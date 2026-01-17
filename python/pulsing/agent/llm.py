"""LLM client management"""

from __future__ import annotations

import os
from typing import Any

_default_llm = None


async def llm(
    *,
    model: str | None = None,
    temperature: float = 0.7,
    new: bool = False,
    **kwargs: Any,
):
    """
    Get LLM client (lazy loading).

    Args:
        model: Model name, defaults to LLM_MODEL environment variable
        temperature: Sampling temperature
        new: Whether to create a new instance (instead of reusing singleton)

    Environment:
        OPENAI_API_KEY: API key (required)
        OPENAI_BASE_URL: API base URL (optional)
        LLM_MODEL: Default model name (optional, default: gpt-4o-mini)

    Example:
        from pulsing.agent import llm

        client = await llm()
        resp = await client.ainvoke("Hello")
    """
    global _default_llm

    if not new and _default_llm is not None:
        return _default_llm

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("pip install langchain-openai") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set (or use --mock mode)")

    instance = ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
        **kwargs,
    )

    if not new:
        _default_llm = instance

    return instance


def reset_llm():
    """Reset LLM singleton"""
    global _default_llm
    _default_llm = None
