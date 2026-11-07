"""Actor lifecycle and metadata helpers.

LLM clients and the Forge-backed coding-agent SDK are no longer bundled.
"""

from .base import AgentMeta, agent, clear_agent_registry, get_agent_meta, list_agents
from .runtime import runtime
from .utils import extract_field, parse_json


def cleanup():
    """Clear actor metadata between runtime sessions."""
    clear_agent_registry()


__all__ = [
    "runtime",
    "cleanup",
    "agent",
    "AgentMeta",
    "get_agent_meta",
    "list_agents",
    "clear_agent_registry",
    "parse_json",
    "extract_field",
]
