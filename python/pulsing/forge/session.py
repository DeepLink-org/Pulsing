# SPDX-License-Identifier: Apache-2.0
"""Session / plan abstractions for Pulsing Forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class PlanItem:
    step: str
    status: StepStatus | str

    def to_dict(self) -> dict[str, str]:
        status = (
            self.status.value
            if isinstance(self.status, StepStatus)
            else str(self.status)
        )
        return {"step": self.step, "status": status}


@dataclass
class UpdatePlanArgs:
    plan: list[PlanItem]
    explanation: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> UpdatePlanArgs:
        """Parse `update_plan` arguments, mirroring Codex's `UpdatePlanArgs`
        (vendor/codex-rs/protocol/src/plan_tool.rs): `plan` and each item's
        `step`/`status` are required; `status` must be a known enum value.
        """
        if not isinstance(raw, dict):
            raise TypeError("update_plan arguments must be an object")
        unknown = set(raw) - {"plan", "explanation"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown field(s): {names}")

        if "plan" not in raw:
            raise ValueError("missing required field 'plan'")
        plan_raw = raw["plan"]
        if not isinstance(plan_raw, list):
            raise ValueError("field 'plan' must be a list")

        items: list[PlanItem] = []
        for idx, item in enumerate(plan_raw):
            if not isinstance(item, dict):
                raise ValueError(f"plan[{idx}] must be an object")
            item_unknown = set(item) - {"step", "status"}
            if item_unknown:
                names = ", ".join(sorted(item_unknown))
                raise ValueError(f"plan[{idx}] has unknown field(s): {names}")
            if "step" not in item:
                raise ValueError(f"plan[{idx}] missing required field 'step'")
            step = item["step"]
            if not isinstance(step, str):
                raise ValueError(f"plan[{idx}].step must be a string")
            if "status" not in item:
                raise ValueError(f"plan[{idx}] missing required field 'status'")
            try:
                status = StepStatus(item["status"])
            except ValueError:
                valid = ", ".join(s.value for s in StepStatus)
                raise ValueError(
                    f"plan[{idx}].status must be one of [{valid}], got {item['status']!r}"
                ) from None
            items.append(PlanItem(step=step, status=status))

        in_progress = sum(1 for item in items if item.status == StepStatus.IN_PROGRESS)
        if in_progress > 1:
            raise ValueError(
                'update_plan allows at most one step with status "in_progress"'
            )

        explanation = raw.get("explanation")
        if explanation is not None and not isinstance(explanation, str):
            raise ValueError("field 'explanation' must be a string")
        return cls(plan=items, explanation=explanation)


class ToolSession(Protocol):
    def update_plan(self, args: UpdatePlanArgs) -> None: ...

    def request_new_context(self) -> None: ...

    def tokens_remaining(self) -> int | None: ...

    def request_user_input(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def request_plugin_install(self, args: dict[str, Any]) -> bool: ...

    def on_exec_output_delta(self, delta: Any) -> None: ...


@dataclass
class LocalToolSession:
    """Default in-process session store for local tool runs."""

    token_budget: int | None = None
    plan: list[PlanItem] = field(default_factory=list)
    new_context_requested: bool = False
    user_input: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    plugin_install: Callable[[dict[str, Any]], bool] | None = None
    exec_deltas: list[Any] = field(default_factory=list)

    def update_plan(self, args: UpdatePlanArgs) -> None:
        self.plan = list(args.plan)

    def request_new_context(self) -> None:
        self.new_context_requested = True

    def tokens_remaining(self) -> int | None:
        return self.token_budget

    def request_user_input(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.user_input is None:
            raise RuntimeError(
                "request_user_input is not configured on this ToolSession"
            )
        return self.user_input(arguments)

    def request_plugin_install(self, args: dict[str, Any]) -> bool:
        if self.plugin_install is None:
            raise RuntimeError(
                "request_plugin_install is not configured on this ToolSession"
            )
        return self.plugin_install(args)

    def on_exec_output_delta(self, delta: Any) -> None:
        self.exec_deltas.append(delta)


class NullToolSession:
    def update_plan(self, args: UpdatePlanArgs) -> None:
        del args

    def request_new_context(self) -> None:
        return None

    def tokens_remaining(self) -> int | None:
        return None

    def request_user_input(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        raise RuntimeError("request_user_input is not available in this runtime")

    def request_plugin_install(self, args: dict[str, Any]) -> bool:
        del args
        raise RuntimeError("request_plugin_install is not available in this runtime")

    def on_exec_output_delta(self, delta: Any) -> None:
        del delta
