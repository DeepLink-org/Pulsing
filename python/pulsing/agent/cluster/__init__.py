# SPDX-License-Identifier: Apache-2.0
"""Local agent cluster: gossip discovery and inter-agent messaging."""

from pulsing.agent.cluster.constants import (
    full_agent_name,
    short_agent_name,
    workspace_agent_name,
)
from pulsing.agent.cluster.discovery import list_cluster_agents
from pulsing.agent.cluster.resolve import (
    message_cluster_agent,
    resolve_agent,
    resolve_craft_agent,
)

__all__ = [
    "full_agent_name",
    "short_agent_name",
    "workspace_agent_name",
    "list_cluster_agents",
    "message_cluster_agent",
    "resolve_agent",
    "resolve_craft_agent",
]
