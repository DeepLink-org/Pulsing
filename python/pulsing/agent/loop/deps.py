# SPDX-License-Identifier: Apache-2.0
"""Optional dependency checks for Craft LLM providers."""

from __future__ import annotations

import sys


def _require_anthropic() -> None:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        print(
            'Missing optional dependency. Install with: pip install "pulsing[agent]"',
            file=sys.stderr,
        )
        sys.exit(1)


def _require_openai() -> None:
    try:
        import openai  # noqa: F401
    except ImportError:
        print(
            'OpenAI provider requires: pip install "pulsing[agent]"',
            file=sys.stderr,
        )
        sys.exit(1)


def require_provider_deps(provider: str) -> None:
    p = (provider or "anthropic").strip().lower()
    if p == "demo":
        return
    if p == "openai":
        _require_openai()
    else:
        _require_anthropic()
