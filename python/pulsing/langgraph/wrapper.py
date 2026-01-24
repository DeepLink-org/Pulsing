"""
PulsingGraphWrapper - One line of code to give LangGraph distributed capabilities
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Optional, Union

from pulsing.actor import ActorSystem, SystemConfig
from pulsing.actor.remote import PYTHON_ACTOR_SERVICE_NAME, PythonActorService
from .executor import NodeExecutorPool

logger = logging.getLogger("pulsing.langgraph")


def with_pulsing(
    compiled_graph,
    *,
    node_mapping: Dict[str, str] | None = None,
    addr: str | None = None,
    seeds: list[str] | None = None,
) -> "PulsingGraphWrapper":
    """
    Wraps LangGraph CompiledGraph to enable distributed execution

    Args:
        compiled_graph: LangGraph compiled graph (return value of graph.compile())
        node_mapping: Mapping from nodes to remote Actors
        addr: Local node address (optional)
        seeds: List of seed node addresses

    Example:
        distributed_app = with_pulsing(
            app,
            node_mapping={"llm": "langgraph_node_llm"},
            seeds=["gpu-server:8001"],
        )
        result = await distributed_app.ainvoke({"messages": []})
    """
    return PulsingGraphWrapper(
        compiled_graph,
        node_mapping=node_mapping or {},
        addr=addr,
        seeds=seeds or [],
    )


class PulsingGraphWrapper:
    """Distributed wrapper for LangGraph CompiledGraph"""

    def __init__(
        self,
        graph,
        node_mapping: Dict[str, str],
        addr: str | None,
        seeds: list[str],
    ):
        self._graph = graph
        self._node_mapping = node_mapping
        self._addr = addr
        self._seeds = seeds
        self._system = None
        self._executor: NodeExecutorPool | None = None
        self._connected = False

    async def _ensure_connected(self):
        """Ensure connection to Pulsing cluster"""
        if self._connected:
            return

        if self._addr:
            config = SystemConfig.with_addr(self._addr)
            if self._seeds:
                config = config.with_seeds(self._seeds)
        elif self._seeds:
            config = SystemConfig.with_addr("0.0.0.0:0").with_seeds(self._seeds)
        else:
            config = SystemConfig.standalone()

        loop = asyncio.get_running_loop()
        self._system = await ActorSystem.create(config, loop)
        # Register PythonActorService for remote actor creation
        service = PythonActorService(self._system)
        await self._system.spawn(service, name=PYTHON_ACTOR_SERVICE_NAME, public=True)
        self._executor = NodeExecutorPool(self._system, self._node_mapping)
        self._connected = True

    async def ainvoke(
        self,
        input: Union[dict, Any],
        config: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """Execute graph asynchronously (API compatible with LangGraph)"""
        await self._ensure_connected()

        if not self._node_mapping:
            return await self._graph.ainvoke(input, config, **kwargs)

        return await self._distributed_execute(input, config)

    def invoke(
        self,
        input: Union[dict, Any],
        config: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """Execute graph synchronously"""
        return asyncio.run(self.ainvoke(input, config, **kwargs))

    async def astream(
        self,
        input: Union[dict, Any],
        config: Optional[dict] = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Execute graph in streaming mode"""
        await self._ensure_connected()

        if not self._node_mapping:
            async for chunk in self._graph.astream(input, config, **kwargs):
                yield chunk
        else:
            result = await self._distributed_execute(input, config)
            yield result

    async def _distributed_execute(self, input: dict, config: Optional[dict]) -> dict:
        """Distributed execution - remote node calls via Pulsing"""
        state = input.copy()
        max_steps = 25
        step = 0

        # Get entry point
        current_node = (
            getattr(self._graph, "_first_node", None)
            or getattr(self._graph, "entry_point", None)
            or (list(self._node_mapping.keys())[0] if self._node_mapping else None)
        )

        while current_node and current_node != "__end__" and step < max_steps:
            step += 1

            if self._executor.is_distributed_node(current_node):
                result = await self._executor.execute(current_node, state, config)
                if result.get("success"):
                    new_state = result.get("state", {})
                    for key, value in new_state.items():
                        if (
                            key in state
                            and isinstance(state[key], list)
                            and isinstance(value, list)
                        ):
                            state[key] = state[key] + value
                        else:
                            state[key] = value
                else:
                    raise RuntimeError(
                        f"Node '{current_node}' failed: {result.get('error')}"
                    )

            # Determine next node based on state
            next_step = state.get("next_step", "end")
            current_node = next_step if next_step in self._node_mapping else "__end__"

        return state

    def __getattr__(self, name: str) -> Any:
        """Proxy undefined attributes to original graph"""
        return getattr(self._graph, name)

    async def close(self):
        """Close connection"""
        if self._system:
            await self._system.shutdown()
            self._connected = False
