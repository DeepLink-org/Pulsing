"""LLM 客户端管理"""

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
    获取 LLM 客户端（懒加载）。

    Args:
        model: 模型名称，默认从 LLM_MODEL 环境变量读取
        temperature: 采样温度
        new: 是否创建新实例（而非复用单例）

    Environment:
        OPENAI_API_KEY: API 密钥（必需）
        OPENAI_BASE_URL: API 地址（可选）
        LLM_MODEL: 默认模型名（可选，默认 gpt-4o-mini）

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
        raise ValueError("未设置 OPENAI_API_KEY（或使用 --mock 模式）")

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
    """重置 LLM 单例"""
    global _default_llm
    _default_llm = None
