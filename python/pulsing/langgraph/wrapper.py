"""
PulsingGraphWrapper - 一行代码让 LangGraph 获得分布式能力
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Optional, Union

from pulsing.actor import SystemConfig, create_actor_system
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
    包装 LangGraph CompiledGraph，使其支持分布式执行

    Args:
        compiled_graph: LangGraph 编译后的图 (graph.compile() 的返回值)
        node_mapping: 节点到远程 Actor 的映射
        addr: 本节点地址 (可选)
        seeds: 种子节点地址列表

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
    """LangGraph CompiledGraph 的分布式包装器"""

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
        """确保已连接到 Pulsing 集群"""
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

        self._system = await create_actor_system(config)
        self._executor = NodeExecutorPool(self._system, self._node_mapping)
        self._connected = True

    async def ainvoke(
        self,
        input: Union[dict, Any],
        config: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """异步执行图 (API 兼容 LangGraph)"""
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
        """同步执行图"""
        return asyncio.run(self.ainvoke(input, config, **kwargs))

    async def astream(
        self,
        input: Union[dict, Any],
        config: Optional[dict] = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """流式执行图"""
        await self._ensure_connected()

        if not self._node_mapping:
            async for chunk in self._graph.astream(input, config, **kwargs):
                yield chunk
        else:
            result = await self._distributed_execute(input, config)
            yield result

    async def _distributed_execute(self, input: dict, config: Optional[dict]) -> dict:
        """分布式执行 - 通过 Pulsing 远程调用节点"""
        state = input.copy()
        max_steps = 25
        step = 0

        # 获取入口点
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

            # 根据状态决定下一个节点
            next_step = state.get("next_step", "end")
            current_node = next_step if next_step in self._node_mapping else "__end__"

        return state

    def __getattr__(self, name: str) -> Any:
        """代理未定义的属性到原始图"""
        return getattr(self._graph, name)

    async def close(self):
        """关闭连接"""
        if self._system:
            await self._system.shutdown()
            self._connected = False
