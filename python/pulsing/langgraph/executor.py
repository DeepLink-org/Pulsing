"""
LangGraph Node Executor - Execute LangGraph nodes in Pulsing Actor

Features:
- Configurable thread pool size
- Bounded queue + backpressure support
- Graceful overload handling
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict

from pulsing.actor import Actor, ActorId, ActorRef, ActorSystem, SystemConfig
from pulsing.actor.remote import PYTHON_ACTOR_SERVICE_NAME, PythonActorService

logger = logging.getLogger("pulsing.langgraph")

# Default configuration
DEFAULT_MAX_WORKERS = 4
DEFAULT_QUEUE_SIZE = 100


class ExecutorConfig:
    """Executor configuration"""

    def __init__(
        self,
        max_workers: int = DEFAULT_MAX_WORKERS,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        queue_timeout: float = 30.0,
    ):
        """
        Args:
            max_workers: Maximum number of worker threads in thread pool
            queue_size: Maximum capacity of bounded queue
            queue_timeout: Timeout (seconds) when queue is full
        """
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.queue_timeout = queue_timeout


class BoundedExecutor:
    """Thread pool executor with bounded queue and backpressure"""

    def __init__(self, config: ExecutorConfig | None = None):
        self._config = config or ExecutorConfig()
        self._executor = ThreadPoolExecutor(max_workers=self._config.max_workers)
        self._semaphore = asyncio.Semaphore(self._config.queue_size)
        self._pending_count = 0
        self._lock = asyncio.Lock()

    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """
        Submit task to thread pool

        If queue is full, waits for queue_timeout seconds.
        Raises asyncio.TimeoutError after timeout.

        Returns:
            Task execution result
        """
        # Wait to acquire semaphore (backpressure)
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._config.queue_timeout
            )
            if not acquired:
                raise RuntimeError("Failed to acquire semaphore")
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Executor queue full ({self._config.queue_size} pending tasks), "
                f"timed out after {self._config.queue_timeout}s"
            )

        async with self._lock:
            self._pending_count += 1

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, func, *args, **kwargs)
            return result
        finally:
            self._semaphore.release()
            async with self._lock:
                self._pending_count -= 1

    @property
    def pending_count(self) -> int:
        """Number of tasks currently waiting to execute"""
        return self._pending_count

    @property
    def queue_capacity(self) -> int:
        """Queue capacity"""
        return self._config.queue_size

    @property
    def available_slots(self) -> int:
        """Number of available queue slots"""
        return self._config.queue_size - self._pending_count

    def shutdown(self, wait: bool = True):
        """Shutdown executor"""
        self._executor.shutdown(wait=wait)


class LangGraphNodeActor(Actor):
    """Wrap LangGraph node function as Pulsing Actor"""

    def __init__(
        self,
        node_name: str,
        node_func: Callable,
        executor_config: ExecutorConfig | None = None,
    ):
        self.node_name = node_name
        self.node_func = node_func
        self._executor = BoundedExecutor(executor_config)

    def on_start(self, actor_id: ActorId) -> None:
        logger.info(
            f"LangGraph node '{self.node_name}' started "
            f"(workers={self._executor._config.max_workers}, "
            f"queue={self._executor._config.queue_size})"
        )

    def on_stop(self) -> None:
        self._executor.shutdown(wait=True)
        logger.info(f"LangGraph node '{self.node_name}' stopped")

    def metadata(self) -> dict[str, str]:
        return {
            "type": "langgraph_node",
            "node_name": self.node_name,
            "pending_tasks": str(self._executor.pending_count),
            "queue_capacity": str(self._executor.queue_capacity),
        }

    async def receive(self, msg: Any) -> Any:
        if not isinstance(msg, dict):
            return {"success": False, "error": "Invalid message format"}

        msg_type = msg.get("type")

        # Health check / status query
        if msg_type == "status":
            return {
                "success": True,
                "node": self.node_name,
                "pending_tasks": self._executor.pending_count,
                "available_slots": self._executor.available_slots,
                "queue_capacity": self._executor.queue_capacity,
            }

        # Execute node
        if msg_type != "execute":
            return {"success": False, "error": f"Unknown message type: {msg_type}"}

        try:
            state = msg.get("state", {})

            if asyncio.iscoroutinefunction(self.node_func):
                result = await self.node_func(state)
            else:
                # Use bounded executor (supports backpressure)
                result = await self._executor.submit(self.node_func, state)

            return {"success": True, "state": result, "node": self.node_name}
        except asyncio.TimeoutError as e:
            logger.warning(f"Node '{self.node_name}' backpressure triggered: {e}")
            return {
                "success": False,
                "error": str(e),
                "node": self.node_name,
                "backpressure": True,
            }
        except Exception as e:
            logger.exception(f"Node '{self.node_name}' failed: {e}")
            return {"success": False, "error": str(e), "node": self.node_name}


class NodeExecutorPool:
    """Manage connections to remote nodes"""

    def __init__(self, system, node_mapping: Dict[str, str]):
        self._system = system
        self._node_mapping = node_mapping
        self._node_refs: Dict[str, ActorRef] = {}

    async def execute(
        self, node_name: str, state: dict, config: dict | None = None
    ) -> dict:
        """Execute node (may be remote)"""
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

    async def get_status(self, node_name: str) -> dict:
        """Get node status (including backpressure information)"""
        actor_ref = await self._get_node_ref(node_name)

        if actor_ref is None:
            return {"success": False, "error": f"Node '{node_name}' not found"}

        result = await actor_ref.ask({"type": "status"})
        return self._deserialize(result)

    def _deserialize(self, response) -> dict:
        """Deserialize response (handles pickle-serialized Python objects)"""
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
        """Get node's ActorRef"""
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
    max_workers: int = DEFAULT_MAX_WORKERS,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    queue_timeout: float = 30.0,
):
    """
    Start LangGraph node worker

    Args:
        node_name: Node name
        node_func: Node processing function
        addr: Bind address
        seeds: Cluster seed nodes
        actor_name: Actor name (default: langgraph_node_{node_name})
        max_workers: Maximum worker threads in thread pool
        queue_size: Bounded queue capacity
        queue_timeout: Wait timeout when queue is full

    Example:
        await start_worker(
            "llm",
            llm_node,
            addr="0.0.0.0:8001",
            max_workers=8,
            queue_size=200,
        )
    """
    config = SystemConfig.with_addr(addr)
    if seeds:
        config = config.with_seeds(seeds)
    loop = asyncio.get_running_loop()
    system = await ActorSystem.create(config, loop)
    # Register PythonActorService for remote actor creation
    service = PythonActorService(system)
    await system.spawn(service, name=PYTHON_ACTOR_SERVICE_NAME, public=True)

    executor_config = ExecutorConfig(
        max_workers=max_workers,
        queue_size=queue_size,
        queue_timeout=queue_timeout,
    )

    actor = LangGraphNodeActor(node_name, node_func, executor_config)
    name = actor_name or f"langgraph_node_{node_name}"

    await system.spawn(actor, name=name, public=True)
    logger.info(
        f"Worker started: {name} @ {addr} (workers={max_workers}, queue={queue_size})"
    )

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        await system.shutdown()
