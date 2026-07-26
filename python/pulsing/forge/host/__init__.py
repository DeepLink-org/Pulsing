# SPDX-License-Identifier: Apache-2.0
"""High-level Forge client facade plus explicit legacy host adapters."""

from pulsing.forge.host.agent import ForgeAgent, DEFAULT_TOOL_NAMES
from pulsing.forge.host.legacy_agent import LegacyPythonForgeAgent
from pulsing.forge.host.cli_events import CliEventSink
from pulsing.forge.host.llm import (
    LLMClient,
    LLMMessage,
    LLMUsage,
    RUST_LLM_AVAILABLE,
    create_llm_client,
    default_model,
    llm_runtime_options,
)

__all__ = [
    "CliEventSink",
    "DEFAULT_TOOL_NAMES",
    "ForgeAgent",
    "LLMClient",
    "LLMMessage",
    "LLMUsage",
    "LegacyPythonForgeAgent",
    "RUST_LLM_AVAILABLE",
    "create_llm_client",
    "default_model",
    "llm_runtime_options",
]
