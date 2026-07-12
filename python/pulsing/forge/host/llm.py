# SPDX-License-Identifier: Apache-2.0
"""LLM clients for Host applications (ForgeAgent, workspace agents, demos)."""

from __future__ import annotations

import os
from typing import Any

from pulsing.agent.loop.deps import require_provider_deps
from pulsing.agent.loop.llm_client import LLMClient, LLMMessage, LLMUsage

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMUsage",
    "create_llm_client",
    "default_model",
    "llm_runtime_options",
]


def default_model(provider: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    p = (provider or "anthropic").strip().lower()
    if p == "demo":
        return "demo"
    if p == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o")
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def create_llm_client(
    *,
    provider: str = "demo",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 8192,
) -> LLMClient:
    """Build an :class:`LLMClient` after optional-dependency checks."""
    require_provider_deps(provider)
    return LLMClient(
        provider=provider,
        model=default_model(provider, model),
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
    )


def llm_runtime_options(
    *,
    provider: str = "demo",
    model: str | None = None,
    auto_approve: bool = True,
    sandbox: str = "off",
) -> dict[str, Any]:
    """Dict passed to workspace ``spawn_npc`` helpers."""
    return {
        "provider": provider,
        "model": default_model(provider, model),
        "auto_approve": auto_approve,
        "sandbox": sandbox,
    }
