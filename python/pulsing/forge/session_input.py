# SPDX-License-Identifier: Apache-2.0
"""Codex-compatible `request_user_input` validation and resolution."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable

# Codex: core/src/tools/handlers/request_user_input_spec.rs
MIN_AUTO_RESOLUTION_MS = 60_000
MAX_AUTO_RESOLUTION_MS = 240_000


@dataclass
class RequestUserInputQuestionOption:
    label: str
    description: str = ""


@dataclass
class RequestUserInputQuestion:
    id: str
    header: str
    question: str
    is_other: bool = False
    is_secret: bool = False
    options: list[RequestUserInputQuestionOption] | None = None


@dataclass
class RequestUserInputArgs:
    questions: list[RequestUserInputQuestion]
    auto_resolution_ms: int | None = None


def validate_request_user_input(raw: dict[str, Any]) -> RequestUserInputArgs:
    questions_raw = raw.get("questions")
    if questions_raw is None:
        raise ValueError("request_user_input requires at least one question")
    if not isinstance(questions_raw, list):
        raise ValueError("field 'questions' must be a list")
    if not questions_raw:
        raise ValueError("request_user_input requires at least one question")

    questions: list[RequestUserInputQuestion] = []
    seen: set[str] = set()
    for item in questions_raw:
        if not isinstance(item, dict):
            raise ValueError("each question must be an object")
        qid = str(item.get("id", "")).strip()
        if not qid:
            raise ValueError("each question requires a non-empty id")
        if qid in seen:
            raise ValueError(f"duplicate question id {qid!r}")
        seen.add(qid)
        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"question {qid!r} requires non-empty question text")
        opts_raw = item.get("options")
        options: list[RequestUserInputQuestionOption] | None = None
        if opts_raw is not None:
            if not isinstance(opts_raw, list) or not opts_raw:
                raise ValueError(f"question {qid!r} has empty options array")
            options = []
            for opt in opts_raw:
                if not isinstance(opt, dict):
                    raise ValueError(f"question {qid!r} option must be an object")
                label = str(opt.get("label", "")).strip()
                if not label:
                    raise ValueError(f"question {qid!r} has option with empty label")
                options.append(
                    RequestUserInputQuestionOption(
                        label=label,
                        description=str(opt.get("description", "")),
                    )
                )
        questions.append(
            RequestUserInputQuestion(
                id=qid,
                header=str(item.get("header", "")),
                question=question,
                is_other=bool(item.get("isOther") or item.get("is_other")),
                is_secret=bool(item.get("isSecret") or item.get("is_secret")),
                options=options,
            )
        )

    auto_ms_raw = raw.get("autoResolutionMs", raw.get("auto_resolution_ms"))
    if auto_ms_raw is None:
        auto_resolution_ms = None
    elif isinstance(auto_ms_raw, bool) or not isinstance(auto_ms_raw, int):
        raise ValueError(f"autoResolutionMs must be an integer, got {auto_ms_raw!r}")
    else:
        auto_resolution_ms = normalize_auto_resolution_ms(auto_ms_raw)
    return RequestUserInputArgs(
        questions=questions, auto_resolution_ms=auto_resolution_ms
    )


def normalize_auto_resolution_ms(value: int) -> int:
    """Clamp to Codex's [MIN, MAX] auto-resolution window (core/src/tools/handlers/request_user_input_spec.rs)."""
    return max(MIN_AUTO_RESOLUTION_MS, min(MAX_AUTO_RESOLUTION_MS, value))


def args_to_payload(args: RequestUserInputArgs) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "questions": [
            {
                "id": q.id,
                "header": q.header,
                "question": q.question,
                "isOther": q.is_other,
                "isSecret": q.is_secret,
                **(
                    {
                        "options": [
                            {"label": o.label, "description": o.description}
                            for o in q.options
                        ]
                    }
                    if q.options
                    else {}
                ),
            }
            for q in args.questions
        ],
    }
    if args.auto_resolution_ms is not None:
        payload["autoResolutionMs"] = args.auto_resolution_ms
    return payload


def default_auto_answers(args: RequestUserInputArgs) -> dict[str, Any]:
    """Pick first option per question (Codex: recommended option should be first)."""
    answers: dict[str, Any] = {}
    for q in args.questions:
        default = q.options[0].label if q.options else ""
        answers[q.id] = {"answers": [default]}
    return {"answers": answers}


def resolve_user_input(
    args: RequestUserInputArgs,
    *,
    auto_approve: bool = False,
    user_input_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    prompt_callback: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    """Resolve `request_user_input` with optional autoResolutionMs timeout → defaults."""
    if auto_approve:
        return default_auto_answers(args)

    payload = args_to_payload(args)
    ms = args.auto_resolution_ms

    if user_input_callback is not None:
        if ms is None:
            return user_input_callback(payload)
        return _run_with_deadline(
            lambda: user_input_callback(payload),
            ms,
            default=lambda: default_auto_answers(args),
        )

    if ms is not None:
        if prompt_callback is not None:
            prompt_callback("request_user_input", format_questions_summary(args))
        _run_with_deadline(lambda: _wait_ms(ms), ms + 1000, default=lambda: None)
        return default_auto_answers(args)

    if prompt_callback is not None:
        summary = format_questions_summary(args)
        choice = prompt_callback("request_user_input", summary)
        if choice in ("allow", "once"):
            return default_auto_answers(args)
        raise RuntimeError("user denied request_user_input")

    raise RuntimeError("request_user_input is not configured on this ToolSession")


def _run_with_deadline(
    fn: Callable[[], Any], timeout_ms: int, *, default: Callable[[], Any]
) -> Any:
    """Run `fn` in a worker thread, falling back to `default()` once `timeout_ms` elapses."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_ms / 1000.0)
        except concurrent.futures.TimeoutError:
            return default()


def format_questions_summary(args: RequestUserInputArgs) -> str:
    lines = ["request_user_input"]
    for q in args.questions:
        opts = ""
        if q.options:
            labels = ", ".join(o.label for o in q.options)
            opts = f" [{labels}]"
        lines.append(f"- {q.header}: {q.question}{opts}")
    if args.auto_resolution_ms is not None:
        lines.append(
            f"(auto-resolves in {args.auto_resolution_ms}ms with recommended defaults)"
        )
    return "\n".join(lines)


def _wait_ms(ms: int) -> None:
    import time

    time.sleep(ms / 1000.0)
