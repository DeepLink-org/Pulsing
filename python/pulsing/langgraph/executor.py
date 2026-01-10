"""
LangGraph Node Executor - 在 Pulsing Actor 中执行 LangGraph 节点
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import Any, Callable, Dict

from pulsing.actor import Actor, ActorId, ActorRef, SystemConfig, create_actor_system

logger = logging.getLogger("pulsing.langgraph")


class LangGraphNodeActor(Actor):
    """将 LangGraph 节点函数包装为 Pulsing Actor"""

    def __init__(self, node_name: str, node_func: Callable):
        self.node_name = node_name
        self.node_func = node_func

    def on_start(self, actor_id: ActorId) -> None:
        logger.info(f"LangGraph node '{self.node_name}' started")

    def metadata(self) -> dict[str, str]:
        return {"type": "langgraph_node", "node_name": self.node_name}

    async def receive(self, msg: Any) -> Any:
        if not isinstance(msg, dict) or msg.get("type") != "execute":
            return {"success": False, "error": "Invalid message"}

        try:
            state = msg.get("state", {})
            if asyncio.iscoroutinefunction(self.node_func):
                result = await self.node_func(state)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.node_func, state)

            return {"success": True, "state": result, "node": self.node_name}
        except Exception as e:
            logger.exception(f"Node '{self.node_name}' failed: {e}")
            return {"success": False, "error": str(e), "node": self.node_name}


class NodeExecutorPool:
    """管理到远程节点的连接"""

    def __init__(self, system, node_mapping: Dict[str, str]):
        self._system = system
        self._node_mapping = node_mapping
        self._node_refs: Dict[str, ActorRef] = {}

    async def execute(
        self, node_name: str, state: dict, config: dict | None = None
    ) -> dict:
        """执行节点（可能是远程的）"""
        actor_ref = await self._get_node_ref(node_name)

        if actor_ref is None:
            return {
                "success": False,
                "error": f"Node '{node_name}' not found",
                "node": node_name,
            }

        result = await actor_ref.ask(
            {"type": "execute", "state": state, "config": config}
        )
        return self._deserialize(result)

    def _deserialize(self, response) -> dict:
        """反序列化响应 (处理 pickle 序列化的 Python 对象)"""
        if hasattr(response, "msg_type") and hasattr(response, "payload"):
            if not response.msg_type and isinstance(response.payload, bytes):
                try:
                    return pickle.loads(response.payload)
                except Exception:
                    pass
        return (
            response
            if isinstance(response, dict)
            else {"success": False, "error": "Bad response"}
        )

    async def _get_node_ref(self, node_name: str) -> ActorRef | None:
        """获取节点的 ActorRef"""
        if node_name in self._node_refs:
            return self._node_refs[node_name]

        actor_name = self._node_mapping.get(node_name, f"langgraph_node_{node_name}")
        try:
            ref = await self._system.resolve_named(actor_name)
            self._node_refs[node_name] = ref
            return ref
        except Exception as e:
            logger.warning(f"Failed to resolve '{node_name}' -> '{actor_name}': {e}")
            return None

    def is_distributed_node(self, node_name: str) -> bool:
        return node_name in self._node_mapping


async def start_worker(
    node_name: str,
    node_func: Callable,
    *,
    addr: str,
    seeds: list[str] | None = None,
    actor_name: str | None = None,
):
    """
    启动 LangGraph 节点 Worker

    Example:
        await start_worker("llm", llm_node, addr="0.0.0.0:8001")
    """
    config = SystemConfig.with_addr(addr)
    if seeds:
        config = config.with_seeds(seeds)
    system = await create_actor_system(config)

    actor = LangGraphNodeActor(node_name, node_func)
    name = actor_name or f"langgraph_node_{node_name}"

    await system.spawn(name, actor, public=True)
    logger.info(f"Worker started: {name} @ {addr}")

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        await system.shutdown()
