"""
Agent decorator - @remote with metadata

Functionally equivalent to @remote, but with additional metadata:
- role: Role name
- goal: Goal description
- backstory: Background story
- tags: Custom tags

This metadata can be used for:
1. Runtime visualization (Agent topology graph)
2. Debugging and logging
3. Automatic documentation generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from pulsing.actor import remote

T = TypeVar("T")

# Global metadata registry: actor_name -> AgentMeta
_agent_meta_registry: dict[str, "AgentMeta"] = {}


@dataclass
class AgentMeta:
    """Agent metadata"""

    role: str = ""
    goal: str = ""
    backstory: str = ""
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tags": self.tags,
        }

    def __str__(self) -> str:
        return f"{self.role}" if self.role else "Agent"


def agent(
    role: str = "",
    goal: str = "",
    backstory: str = "",
    **tags: Any,
) -> Callable[[type[T]], type[T]]:
    """
    Agent decorator - equivalent to @remote, but with additional metadata

    Metadata can be retrieved via get_agent_meta(name).

    Example:
        @agent(role="Researcher", goal="Deep analysis")
        class Researcher:
            async def analyze(self, topic: str) -> str:
                ...

        async with runtime():
            r = await Researcher.spawn(name="researcher")

            # Get metadata
            meta = get_agent_meta("researcher")
            print(meta.role)  # "Researcher"
    """

    def decorator(cls: type[T]) -> type[T]:
        # Create metadata
        meta = AgentMeta(role=role, goal=goal, backstory=backstory, tags=tags)

        # Store on class (for use during spawn)
        cls._agent_meta_template = meta  # type: ignore

        # Wrap spawn method to register metadata
        actor_cls = remote(cls)
        original_spawn = actor_cls.spawn

        async def spawn_with_meta(*args: Any, name: str | None = None, **kwargs: Any):
            proxy = await original_spawn(*args, name=name, **kwargs)
            # Register metadata
            if name:
                _agent_meta_registry[name] = meta
            return proxy

        actor_cls.spawn = spawn_with_meta
        actor_cls.__agent_meta__ = meta  # Also accessible at class level

        return actor_cls

    return decorator


def get_agent_meta(name: str) -> AgentMeta | None:
    """Get metadata by actor name"""
    return _agent_meta_registry.get(name)


def list_agents() -> dict[str, AgentMeta]:
    """List all registered agents and their metadata"""
    return _agent_meta_registry.copy()


def clear_agent_registry() -> None:
    """Clear registry (for testing)"""
    _agent_meta_registry.clear()
