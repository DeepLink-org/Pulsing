# SPDX-License-Identifier: Apache-2.0
"""LLM clients for Host applications — backed by Forge (Rust)."""

from __future__ import annotations

import os
from typing import Any

from pulsing.agent.loop.deps import require_provider_deps
from pulsing.forge.llm_client import LLMClient, LLMMessage, LLMUsage, RUST_LLM_AVAILABLE

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMUsage",
    "RUST_LLM_AVAILABLE",
    "create_llm_client",
    "default_model",
    "default_provider",
    "llm_runtime_options",
]


def default_provider() -> str:
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    return "demo"


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
    del max_tokens  # applied per stream_messages call
    require_provider_deps(provider)
    return LLMClient(provider=provider, api_key=api_key, base_url=base_url)


def llm_runtime_options(
    *,
    provider: str = "demo",
    model: str | None = None,
    auto_approve: bool = True,
    sandbox: str = "off",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": default_model(provider, model),
        "auto_approve": auto_approve,
        "sandbox": sandbox,
    }
