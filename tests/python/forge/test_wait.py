# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``wait`` Forge tool (Codex code_mode exec/wait pairing)."""

from __future__ import annotations

import pytest

from pulsing.forge.code_mode.protocol import WaitArgs

pytestmark = pytest.mark.forge


def test_wait_returns_result_for_completed_cell(local_forge) -> None:
    exec_out = local_forge.call_tool("exec", {"source": "text('hi')"})
    cell_id = exec_out.structured["cell_id"]

    out = local_forge.call_tool("wait", {"cell_id": cell_id})

    assert not out.is_error
    assert out.structured["kind"] == "result"
    assert out.structured["content_items"][0]["text"] == "hi"


def test_wait_unknown_cell_id_is_a_clear_error(local_forge) -> None:
    out = local_forge.call_tool("wait", {"cell_id": "cell-does-not-exist"})

    assert out.is_error
    assert "cell-does-not-exist" in out.content


def test_wait_missing_cell_id_is_rejected(local_forge) -> None:
    out = local_forge.call_tool("wait", {})

    assert out.is_error
    assert "cell_id" in out.content


def test_wait_terminate_on_finished_cell_does_not_clobber_result(local_forge) -> None:
    exec_out = local_forge.call_tool("exec", {"source": "text('done')"})
    cell_id = exec_out.structured["cell_id"]

    out = local_forge.call_tool("wait", {"cell_id": cell_id, "terminate": True})

    assert not out.is_error
    assert out.structured["kind"] == "result"
    assert out.structured["content_items"][0]["text"] == "done"


def test_wait_terminate_on_yielded_cell_marks_terminated(local_forge) -> None:
    exec_out = local_forge.call_tool("exec", {"source": "yield_control()"})
    cell_id = exec_out.structured["cell_id"]
    assert exec_out.structured["kind"] == "yielded"

    out = local_forge.call_tool("wait", {"cell_id": cell_id, "terminate": True})

    assert not out.is_error
    assert out.structured["kind"] == "terminated"


def test_wait_max_tokens_trims_content(local_forge) -> None:
    exec_out = local_forge.call_tool("exec", {"source": "text('hello world')"})
    cell_id = exec_out.structured["cell_id"]

    out = local_forge.call_tool("wait", {"cell_id": cell_id, "max_tokens": 5})

    assert not out.is_error
    assert out.structured["content_items"][0]["text"] == "hello"


def test_wait_args_rejects_negative_yield_time_ms() -> None:
    with pytest.raises(ValueError):
        WaitArgs.from_dict({"cell_id": "x", "yield_time_ms": -1})


def test_wait_args_rejects_non_positive_max_tokens() -> None:
    with pytest.raises(ValueError):
        WaitArgs.from_dict({"cell_id": "x", "max_tokens": 0})


def test_wait_args_preserves_explicit_zero_yield_time_ms() -> None:
    args = WaitArgs.from_dict({"cell_id": "x", "yield_time_ms": 0})
    assert args.yield_time_ms == 0


def test_wait_invalid_max_tokens_type_is_a_clean_tool_error(local_forge) -> None:
    out = local_forge.call_tool("wait", {"cell_id": "x", "max_tokens": "not-a-number"})

    assert out.is_error
    assert "max_tokens" in out.content or "invalid literal" in out.content


def test_wait_on_exec_error_without_output_is_structured_not_tool_error(
    local_forge,
) -> None:
    exec_out = local_forge.call_tool("exec", {"source": "1 / 0"})
    cell_id = exec_out.structured["cell_id"]

    out = local_forge.call_tool("wait", {"cell_id": cell_id})

    assert not out.is_error
    assert out.structured["kind"] == "result"
    assert out.structured["error_text"].startswith("ZeroDivisionError:")
    assert "Error:" in out.content
