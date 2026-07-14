# SPDX-License-Identifier: Apache-2.0
"""Pulsing Forge — agent tool & environment runtime (``pulsing.forge``).

Provides a sandboxed execution environment for AI agents: shell, filesystem,
and session collaboration tools. See ``docs/src/forge/`` (MkDocs **Pulsing Forge** chapter).
"""

from __future__ import annotations

from pulsing.forge.backend import (
    ForgeBackend,
    ForgeBackendMode,
    ForgeHostConfig,
    ForgeIsolatedWorker,
    create_host_runtime,
    resolve_shared_tool_worker,
    spawn_shared_tool_worker,
)
from pulsing.forge.config import ToolWorkerConfig
from pulsing.forge.context import ToolCallContext
from pulsing.forge.environment import ForgeEnvironment
from pulsing.forge.events import ForgeEvent, ForgeEventKind
from pulsing.forge.host import CliEventSink, ForgeAgent
from pulsing.forge.hybrid_runtime import HybridForgeRuntime
from pulsing.forge.integrated import (
    FORGE_HOST_TOOL_NAMES,
    FORGE_ISOLATED_TOOL_NAMES,
    FORGE_TOOL_NAMES,
    ForgeHostLink,
)
from pulsing.forge.naming import shared_tool_worker_name
from pulsing.forge.p2p_session import P2PToolSession
from pulsing.forge.p2p_transport import ForgeEventPump, tell_forge_event
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
from pulsing.forge.worker import ToolWorkerActor

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
