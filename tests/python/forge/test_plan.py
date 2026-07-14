# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``update_plan`` Forge tool (Codex plan/todo parity).

Reference: vendor/codex-rs/protocol/src/plan_tool.rs (``UpdatePlanArgs``,
``PlanItemArg``, ``StepStatus``) - the closest Codex protocol type, since the
`update_plan` tool handler itself (codex-rs/core/src/tools/handlers/plan.rs)
does no validation beyond parsing.
"""

from __future__ import annotations

import pytest

from pulsing.forge.session import PlanItem, StepStatus, UpdatePlanArgs

pytestmark = pytest.mark.forge


def test_update_plan_valid_call_updates_session(local_forge) -> None:
    out = local_forge.call_tool(
        "update_plan",
        {
            "plan": [
                {"step": "one", "status": "pending"},
                {"step": "two", "status": "in_progress"},
            ]
        },
    )

    assert not out.is_error
    assert local_forge.context.session.plan == [
        PlanItem(step="one", status=StepStatus.PENDING),
        PlanItem(step="two", status=StepStatus.IN_PROGRESS),
    ]


def test_update_plan_accepts_empty_plan(local_forge) -> None:
    out = local_forge.call_tool("update_plan", {"plan": []})

    assert not out.is_error
    assert local_forge.context.session.plan == []


def test_update_plan_missing_plan_is_rejected(local_forge) -> None:
    out = local_forge.call_tool("update_plan", {})

    assert out.is_error
    assert "plan" in out.content


def test_update_plan_invalid_status_is_a_clear_error(local_forge) -> None:
    out = local_forge.call_tool(
        "update_plan",
        {"plan": [{"step": "one", "status": "doing"}]},
    )

    assert out.is_error
    assert "status" in out.content
    assert "doing" in out.content


def test_update_plan_missing_step_is_a_clear_error(local_forge) -> None:
    out = local_forge.call_tool(
        "update_plan",
        {"plan": [{"status": "pending"}]},
    )

    assert out.is_error
    assert "step" in out.content


def test_update_plan_multiple_in_progress_is_rejected(local_forge) -> None:
    out = local_forge.call_tool(
        "update_plan",
        {
            "plan": [
                {"step": "one", "status": "in_progress"},
                {"step": "two", "status": "in_progress"},
            ]
        },
    )

    assert out.is_error
    assert "in_progress" in out.content


def test_update_plan_args_from_dict_rejects_multiple_in_progress() -> None:
    with pytest.raises(ValueError, match="in_progress"):
        UpdatePlanArgs.from_dict(
            {
                "plan": [
                    {"step": "one", "status": "in_progress"},
                    {"step": "two", "status": "in_progress"},
                ]
            }
        )


def test_update_plan_args_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        UpdatePlanArgs.from_dict({"plan": [], "extra": 1})


def test_update_plan_args_from_dict_rejects_non_list_plan() -> None:
    with pytest.raises(ValueError):
        UpdatePlanArgs.from_dict({"plan": "not-a-list"})


def test_update_plan_args_from_dict_preserves_explanation() -> None:
    args = UpdatePlanArgs.from_dict(
        {"plan": [{"step": "one", "status": "completed"}], "explanation": "why"}
    )
    assert args.explanation == "why"
    assert args.plan == [PlanItem(step="one", status=StepStatus.COMPLETED)]
