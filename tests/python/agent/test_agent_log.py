# SPDX-License-Identifier: Apache-2.0
"""Per-agent log ring buffer."""

from __future__ import annotations

from pulsing.agent.actors.log import append_log, get_logs, init_log


class _Fake:
    pass


def test_log_ring_buffer_and_since() -> None:
    agent = _Fake()
    init_log(agent)
    append_log(agent, "hello")
    append_log(agent, "world")
    chunk = get_logs(agent, since=0)
    assert chunk["next"] == 2
    assert len(chunk["lines"]) == 2
    assert "hello" in chunk["lines"][0]
    tail = get_logs(agent, since=1)
    assert len(tail["lines"]) == 1
    assert "world" in tail["lines"][0]
