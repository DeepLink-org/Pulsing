# SPDX-License-Identifier: Apache-2.0
"""Pulsing Forge — agent tool & environment runtime (``pulsing.forge``).

Provides a sandboxed execution environment for AI agents: shell, filesystem,
and session collaboration tools. See ``docs/src/forge/`` (MkDocs **Pulsing Forge** chapter).

Heavy modules (backend, worker, host) are loaded lazily via ``__getattr__`` so
that leaf imports like ``pulsing.forge.naming`` do not pull in ``@remote`` actors
during package initialization.
"""

from __future__ import annotations

from typing import Any

from pulsing.forge.config import ToolWorkerConfig
from pulsing.forge.context import ToolCallContext
from pulsing.forge.environment import ForgeEnvironment
from pulsing.forge.events import ForgeEvent, ForgeEventKind
from pulsing.forge.integrated import (
    FORGE_HOST_TOOL_NAMES,
    FORGE_ISOLATED_TOOL_NAMES,
    FORGE_TOOL_NAMES,
    ForgeHostLink,
)
from pulsing.forge.naming import shared_tool_worker_name
from pulsing.forge.result import ToolResult
from pulsing.forge.runtime import LocalToolRuntime
from pulsing.forge.session import (
    LocalToolSession,
    NullToolSession,
    PlanItem,
    StepStatus,
    ToolSession,
    UpdatePlanArgs,
)
from pulsing.forge.tool_calls import (
    OpenAIToolCallAccumulator,
    ParsedToolCall,
    anthropic_tool_result_block,
    anthropic_tool_results_message,
    extract_tool_calls,
    forge_tool_definitions,
    openai_tool_message,
    parse_tool_arguments,
    to_anthropic_tools,
    to_openai_tools,
)

__all__ = [
    "CliEventSink",
    "ForgeAgent",
    "ForgeBackend",
    "ForgeBackendMode",
    "ForgeEnvironment",
    "ForgeEvent",
    "ForgeEventKind",
    "ForgeEventPump",
    "ForgeHostConfig",
    "ForgeHostLink",
    "ForgeIsolatedWorker",
    "FORGE_HOST_TOOL_NAMES",
    "FORGE_ISOLATED_TOOL_NAMES",
    "FORGE_TOOL_NAMES",
    "HybridForgeRuntime",
    "LocalToolRuntime",
    "LocalToolSession",
    "NullToolSession",
    "OpenAIToolCallAccumulator",
    "P2PToolSession",
    "ParsedToolCall",
    "PlanItem",
    "StepStatus",
    "ToolCallContext",
    "ToolResult",
    "ToolSession",
    "ToolWorkerActor",
    "ToolWorkerConfig",
    "UpdatePlanArgs",
    "anthropic_tool_result_block",
    "anthropic_tool_results_message",
    "create_host_runtime",
    "extract_tool_calls",
    "forge_tool_definitions",
    "openai_tool_message",
    "parse_tool_arguments",
    "resolve_shared_tool_worker",
    "shared_tool_worker_name",
    "spawn_shared_tool_worker",
    "tell_forge_event",
    "to_anthropic_tools",
    "to_openai_tools",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "CliEventSink": ("pulsing.forge.host", "CliEventSink"),
    "ForgeAgent": ("pulsing.forge.host", "ForgeAgent"),
    "ForgeBackend": ("pulsing.forge.backend", "ForgeBackend"),
    "ForgeBackendMode": ("pulsing.forge.backend", "ForgeBackendMode"),
    "ForgeEventPump": ("pulsing.forge.p2p_transport", "ForgeEventPump"),
    "ForgeHostConfig": ("pulsing.forge.backend", "ForgeHostConfig"),
    "ForgeIsolatedWorker": ("pulsing.forge.backend", "ForgeIsolatedWorker"),
    "HybridForgeRuntime": ("pulsing.forge.hybrid_runtime", "HybridForgeRuntime"),
    "P2PToolSession": ("pulsing.forge.p2p_session", "P2PToolSession"),
    "ToolWorkerActor": ("pulsing.forge.worker", "ToolWorkerActor"),
    "create_host_runtime": ("pulsing.forge.backend", "create_host_runtime"),
    "resolve_shared_tool_worker": (
        "pulsing.forge.backend",
        "resolve_shared_tool_worker",
    ),
    "spawn_shared_tool_worker": ("pulsing.forge.backend", "spawn_shared_tool_worker"),
    "tell_forge_event": ("pulsing.forge.p2p_transport", "tell_forge_event"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value
