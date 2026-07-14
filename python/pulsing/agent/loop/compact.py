# SPDX-License-Identifier: Apache-2.0
"""MVP transcript compaction by message count (not token-aware)."""

from __future__ import annotations


def maybe_compact(messages: list[dict], max_messages: int = 40) -> list[dict]:
    """Return a shallow-truncated copy when ``len(messages) > max_messages``.

    **MVP:** keep the first two entries (warm-up / system-like prefix)
    and the most recent tail; drop a contiguous slice from the middle so the
    result length is at most ``max_messages``. This preserves a short prefix and
    fresh context without token-aware summarization.
    """
    if len(messages) <= max_messages:
        return messages
    head_keep = min(2, max_messages)
    tail_keep = max_messages - head_keep
    if tail_keep < 1:
        return messages[-max_messages:]
    return list(messages[:head_keep]) + list(messages[-tail_keep:])
