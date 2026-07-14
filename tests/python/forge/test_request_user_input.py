# SPDX-License-Identifier: Apache-2.0
"""`request_user_input` validation, timeout-defaults, and dispatch behavior."""

from __future__ import annotations

import json

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.handlers import dispatch_tool
from pulsing.forge.session import LocalToolSession
from pulsing.forge.session_input import (
    RequestUserInputArgs,
    RequestUserInputQuestion,
    default_auto_answers,
    resolve_user_input,
    validate_request_user_input,
)

pytestmark = pytest.mark.forge


def _payload(**overrides):
    base = {
        "questions": [
            {
                "id": "q1",
                "header": "Pick",
                "question": "Which one?",
                "options": [{"label": "A", "description": "first"}, {"label": "B"}],
            }
        ]
    }
    base.update(overrides)
    return base


def test_validate_accepts_minimal_question():
    args = validate_request_user_input({"questions": [{"id": "q1", "question": "Go?"}]})
    assert args.questions[0].id == "q1"
    assert args.questions[0].options is None
    assert args.auto_resolution_ms is None


def test_validate_accepts_full_payload_and_clamps_timeout():
    args = validate_request_user_input(_payload(autoResolutionMs=90_000))
    assert args.questions[0].options[0].label == "A"
    assert args.auto_resolution_ms == 90_000


@pytest.mark.parametrize(
    "raw",
    [
        {"questions": []},
        {"questions": "not-a-list"},
        {"questions": [{"question": "Go?"}]},
        {"questions": [{"id": "  ", "question": "Go?"}]},
        {"questions": [{"id": "q1", "question": ""}]},
        {
            "questions": [
                {"id": "q1", "question": "Go?"},
                {"id": "q1", "question": "Again?"},
            ]
        },
        {"questions": [{"id": "q1", "question": "Go?", "options": []}]},
        {"questions": [{"id": "q1", "question": "Go?", "options": [{"label": " "}]}]},
        {"questions": [{"id": "q1", "question": "Go?"}], "autoResolutionMs": "soon"},
        {"questions": [{"id": "q1", "question": "Go?"}], "autoResolutionMs": [1, 2]},
    ],
    ids=[
        "empty-questions",
        "questions-not-list",
        "missing-id",
        "blank-id",
        "blank-question-text",
        "duplicate-id",
        "empty-options-array",
        "blank-option-label",
        "auto_resolution_ms-not-numeric",
        "auto_resolution_ms-wrong-type",
    ],
)
def test_validate_rejects_invalid_payloads(raw):
    with pytest.raises(ValueError):
        validate_request_user_input(raw)


def test_validate_clamps_auto_resolution_ms_to_bounds():
    import pulsing.forge.session_input as session_input

    lo = validate_request_user_input(_payload(autoResolutionMs=1)).auto_resolution_ms
    hi = validate_request_user_input(
        _payload(autoResolutionMs=999_999_999)
    ).auto_resolution_ms
    assert lo == session_input.MIN_AUTO_RESOLUTION_MS
    assert hi == session_input.MAX_AUTO_RESOLUTION_MS


def test_default_auto_answers_picks_first_option():
    args = validate_request_user_input(_payload())
    answers = default_auto_answers(args)
    assert answers["answers"]["q1"]["answers"] == ["A"]


def test_default_auto_answers_empty_string_without_options():
    args = validate_request_user_input({"questions": [{"id": "q1", "question": "Go?"}]})
    answers = default_auto_answers(args)
    assert answers["answers"]["q1"]["answers"] == [""]


def test_resolve_user_input_prefers_callback_over_timeout():
    args = RequestUserInputArgs(
        questions=[RequestUserInputQuestion(id="q1", header="H", question="Go?")]
    )
    seen = {}

    def callback(payload):
        seen["payload"] = payload
        return {"answers": {"q1": {"answers": ["yes"]}}}

    out = resolve_user_input(args, user_input_callback=callback)
    assert out["answers"]["q1"]["answers"] == ["yes"]
    assert seen["payload"]["questions"][0]["id"] == "q1"


def test_resolve_user_input_falls_back_to_defaults_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    """When autoResolutionMs elapses, resolution must degrade to the first option per question."""
    import pulsing.forge.session_input as session_input

    monkeypatch.setattr(session_input, "MIN_AUTO_RESOLUTION_MS", 1)
    args = session_input.validate_request_user_input(_payload(autoResolutionMs=5))
    out = session_input.resolve_user_input(
        args, auto_approve=False, user_input_callback=None, prompt_callback=None
    )
    assert out["answers"]["q1"]["answers"] == ["A"]


def test_resolve_user_input_callback_timeout_falls_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    import time

    import pulsing.forge.session_input as session_input

    monkeypatch.setattr(session_input, "MIN_AUTO_RESOLUTION_MS", 1)
    args = session_input.validate_request_user_input(_payload(autoResolutionMs=5))

    def slow_callback(_payload):
        time.sleep(0.05)
        return {"answers": {"q1": {"answers": ["late"]}}}

    out = session_input.resolve_user_input(args, user_input_callback=slow_callback)
    assert out["answers"]["q1"]["answers"] == ["A"]


def test_resolve_user_input_auto_approve_skips_callback():
    args = RequestUserInputArgs(
        questions=[RequestUserInputQuestion(id="q1", header="H", question="Go?")]
    )
    out = resolve_user_input(
        args,
        auto_approve=True,
        user_input_callback=lambda _: pytest.fail("must not be called"),
    )
    assert out["answers"]["q1"]["answers"] == [""]


def test_resolve_user_input_raises_without_any_configured_channel():
    args = RequestUserInputArgs(
        questions=[RequestUserInputQuestion(id="q1", header="H", question="Go?")]
    )
    with pytest.raises(RuntimeError):
        resolve_user_input(args)


def test_dispatch_reports_invalid_arguments_as_tool_error(tmp_path):
    ctx = ToolCallContext(cwd=str(tmp_path), session=LocalToolSession())
    out = dispatch_tool("request_user_input", {"questions": []}, ctx=ctx)
    assert out.is_error
    assert "question" in out.content.lower()


def test_dispatch_reports_malformed_auto_resolution_ms(tmp_path):
    ctx = ToolCallContext(cwd=str(tmp_path), session=LocalToolSession())
    out = dispatch_tool(
        "request_user_input",
        _payload(autoResolutionMs={"bad": True}),
        ctx=ctx,
    )
    assert out.is_error
    assert "autoresolutionms" in out.content.lower()


def test_dispatch_without_configured_session_reports_runtime_error(tmp_path):
    ctx = ToolCallContext(cwd=str(tmp_path), session=LocalToolSession())
    out = dispatch_tool("request_user_input", _payload(), ctx=ctx)
    assert out.is_error
    assert "not configured" in out.content.lower()


def test_dispatch_returns_session_answers_on_success(tmp_path):
    session = LocalToolSession(
        user_input=lambda _args: {"answers": {"q1": {"answers": ["A"]}}}
    )
    ctx = ToolCallContext(cwd=str(tmp_path), session=session)
    out = dispatch_tool("request_user_input", _payload(), ctx=ctx)
    assert not out.is_error
    assert json.loads(out.content)["answers"]["q1"]["answers"] == ["A"]
