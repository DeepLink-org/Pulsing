# SPDX-License-Identifier: Apache-2.0
"""Forge-native LLM client (Rust ``pulsing-forge`` via ``pulsing._core``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

try:
    from pulsing._core import LlmClient as _RustLlmClient
    from pulsing._core import LlmStream as _RustLlmStream

    RUST_LLM_AVAILABLE = True
except ImportError:
    _RustLlmClient = None  # type: ignore[misc, assignment]
    _RustLlmStream = None  # type: ignore[misc, assignment]
    RUST_LLM_AVAILABLE = False

_VALID_PROVIDERS = frozenset({"anthropic", "openai", "demo"})


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMMessage:
    content: list[dict[str, Any]]
    usage: LLMUsage | None = None
    stop_reason: str | None = None


def _usage_from_dict(raw: Any) -> LLMUsage | None:
    if not raw:
        return None
    if isinstance(raw, LLMUsage):
        return raw
    if isinstance(raw, dict):
        return LLMUsage(
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
        )
    return None


def _message_from_dict(raw: Any) -> LLMMessage:
    if isinstance(raw, LLMMessage):
        return raw
    if not isinstance(raw, dict):
        return LLMMessage(content=[{"type": "text", "text": str(raw)}])
    content = raw.get("content") or []
    if not isinstance(content, list):
        content = [{"type": "text", "text": str(content)}]
    return LLMMessage(
        content=[dict(b) if isinstance(b, dict) else b for b in content],
        usage=_usage_from_dict(raw.get("usage")),
        stop_reason=raw.get("stop_reason"),
    )


class _RustStreamAdapter:
    """Adapt Rust ``LlmStream`` to the Python agent loop protocol."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.text_stream: Iterator[str] = iter(())

    def __enter__(self) -> _RustStreamAdapter:
        self._inner.__enter__()
        self.text_stream = iter(self._inner.text_stream)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._inner.__exit__(exc_type, exc, tb)

    def close(self) -> None:
        self._inner.close()

    def get_final_message(self) -> LLMMessage:
        return _message_from_dict(self._inner.get_final_message())


class LLMClient:
    """Anthropic / OpenAI / demo LLM client backed by ``pulsing-forge``."""

    def __init__(
        self,
        *,
        provider: str = "anthropic",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        p = (provider or "anthropic").strip().lower()
        if p not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported provider {provider!r}; use anthropic, openai, or demo",
            )
        if not RUST_LLM_AVAILABLE:
            raise RuntimeError(
                "Rust LLM client unavailable — rebuild with: maturin develop",
            )
        self.provider = p
        self._client = _RustLlmClient(
            provider=p,
            api_key=api_key,
            base_url=base_url,
        )

    def stream_messages(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> _RustStreamAdapter:
        stream = self._client.stream_messages(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            tools=tools or [],
        )
        return _RustStreamAdapter(stream)

    @staticmethod
    def error_message(exc: Exception) -> str:
        return str(getattr(exc, "message", None) or exc)

    def is_authentication_error(self, exc: Exception) -> bool:
        if not RUST_LLM_AVAILABLE:
            return False
        return bool(self._client.is_authentication_error(exc))

    def is_retryable_error(self, exc: Exception) -> bool:
        if not RUST_LLM_AVAILABLE:
            return False
        return bool(self._client.is_retryable_error(exc))

    def is_api_error(self, exc: Exception) -> bool:
        if not RUST_LLM_AVAILABLE:
            return False
        return bool(self._client.is_api_error(exc))
