"""
Pulsing Agent Toolbox

Lightweight multi-agent development tools, fully compatible with pulsing.actor.

Core APIs:
- runtime(): Actor system lifecycle management
- agent(): @remote with metadata (for visualization)
- llm(): LLM client
- parse_json(): JSON parsing

Example:
    from pulsing.actor import remote, resolve
    from pulsing.agent import agent, runtime, llm, get_agent_meta

    # @remote: Basic Actor
    @remote
    class Worker:
        async def work(self): ...

    # @agent: Actor with metadata (for visualization/debugging)
    @agent(role="Researcher", goal="Deep analysis")
    class Researcher:
        async def analyze(self, topic: str) -> str:
            client = await llm()
            return await client.ainvoke(f"Analyze: {topic}")

    async def main():
        async with runtime():
            r = await Researcher.spawn(name="researcher")
            result = await r.analyze("AI")

            # Get metadata
            meta = get_agent_meta("researcher")
            print(meta.role)  # "Researcher"
"""

# Runtime
from .runtime import runtime

# Agent decorator (@remote with metadata)
from .base import agent, AgentMeta, get_agent_meta, list_agents, clear_agent_registry

# LLM
from .llm import llm, reset_llm

# Utility functions
from .utils import parse_json, extract_field


def cleanup():
    """
    Clean up all agent-related global state.

    Includes:
    - Agent metadata registry
    - LLM singleton

    Recommended to call when repeatedly creating/destroying runtime to avoid memory leaks.

    Example:
        from pulsing.agent import runtime, cleanup

        try:
            async with runtime():
                agent = await MyAgent.spawn(name="agent")
                await agent.work()
        finally:
            cleanup()  # Clean up all global state
    """
    clear_agent_registry()
    reset_llm()


__all__ = [
    # Runtime
    "runtime",
    "cleanup",
    # Agent
    "agent",
    "AgentMeta",
    "get_agent_meta",
    "list_agents",
    "clear_agent_registry",
    # LLM
    "llm",
    "reset_llm",
    # Utility functions
    "parse_json",
    "extract_field",
]
