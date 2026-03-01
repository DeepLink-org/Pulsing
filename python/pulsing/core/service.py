"""System-level actor services that communicate with the Rust runtime via Message protocol.

- PythonActorService: per-node service for remote Python actor creation
- SystemActorProxy / PythonActorServiceProxy: typed proxies for system actors
"""

import inspect
import logging

from pulsing._core import ActorRef, ActorSystem, Message
from pulsing.exceptions import PulsingRuntimeError

from .remote import (
    Actor,
    _WrappedActor,
    _actor_class_registry,
    _extract_methods,
    _register_actor_metadata,
)

logger = logging.getLogger(__name__)

PYTHON_ACTOR_SERVICE_NAME = "system/python_actor_service"


class PythonActorService(Actor):
    """Python Actor creation service - one per node, handles Python actor creation requests.

    Note: Rust SystemActor (path "system/core") handles system-level operations,
    this service specifically handles Python actor creation.
    """

    def __init__(self, system: ActorSystem):
        self.system = system

    async def receive(self, msg: Message) -> Message | None:
        data = msg.to_json()

        if msg.msg_type == "CreateActor":
            return await self._create_actor(data)
        elif msg.msg_type == "ListRegistry":
            return Message.from_json(
                "Registry",
                {"classes": list(_actor_class_registry.keys())},
            )
        return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})

    async def _create_actor(self, data: dict) -> Message:
        class_name = data.get("class_name")
        actor_name = data.get("actor_name")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})
        public = data.get("public", True)

        restart_policy = data.get("restart_policy", "never")
        max_restarts = data.get("max_restarts", 3)
        min_backoff = data.get("min_backoff", 0.1)
        max_backoff = data.get("max_backoff", 30.0)

        cls = _actor_class_registry.get(class_name)
        if cls is None:
            return Message.from_json(
                "Error", {"error": f"Class '{class_name}' not found"}
            )

        try:
            if restart_policy != "never":

                def factory():
                    instance = cls(*args, **kwargs)
                    return _WrappedActor(instance)

                actor_ref = await self.system.spawn(
                    factory,
                    name=actor_name,
                    public=public,
                    restart_policy=restart_policy,
                    max_restarts=max_restarts,
                    min_backoff=min_backoff,
                    max_backoff=max_backoff,
                )
            else:
                instance = cls(*args, **kwargs)
                actor = _WrappedActor(instance)
                actor_ref = await self.system.spawn(
                    actor, name=actor_name, public=public
                )

            _register_actor_metadata(actor_name, cls)
            methods, _ = _extract_methods(cls)

            return Message.from_json(
                "Created",
                {
                    "actor_id": str(actor_ref.actor_id.id),
                    "node_id": str(self.system.node_id.id),
                    "methods": methods,
                },
            )
        except Exception as e:
            logger.exception(f"Create actor failed: {e}")
            return Message.from_json("Error", {"error": str(e)})


class SystemActorProxy:
    """Proxy for SystemActor with direct method calls.

    Example:
        system_proxy = await get_system_actor(system)
        actors = await system_proxy.list_actors()
        metrics = await system_proxy.get_metrics()
        await system_proxy.ping()
    """

    def __init__(self, actor_ref: ActorRef):
        self._ref = actor_ref

    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef."""
        return self._ref

    async def _ask(self, msg_type: str) -> dict:
        """Send SystemMessage and return response."""
        resp = await self._ref.ask(
            Message.from_json("SystemMessage", {"type": msg_type}),
        )
        return resp.to_json()

    async def list_actors(self) -> list[dict]:
        """List all actors on this node."""
        data = await self._ask("ListActors")
        if data.get("type") == "Error":
            raise PulsingRuntimeError(data.get("message"))
        return data.get("actors", [])

    async def get_metrics(self) -> dict:
        """Get system metrics."""
        return await self._ask("GetMetrics")

    async def get_node_info(self) -> dict:
        """Get node info."""
        return await self._ask("GetNodeInfo")

    async def health_check(self) -> dict:
        """Health check."""
        return await self._ask("HealthCheck")

    async def ping(self) -> dict:
        """Ping this node."""
        return await self._ask("Ping")


async def get_system_actor(
    system: ActorSystem, node_id: int | None = None
) -> SystemActorProxy:
    """Get SystemActorProxy for direct method calls."""
    if node_id is None:
        actor_ref = await system.system()
    else:
        actor_ref = await system.remote_system(node_id)
    return SystemActorProxy(actor_ref)


class PythonActorServiceProxy:
    """Proxy for PythonActorService with direct method calls."""

    def __init__(self, actor_ref: ActorRef):
        self._ref = actor_ref

    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef."""
        return self._ref

    async def list_registry(self) -> list[str]:
        """List registered actor classes."""
        resp = await self._ref.ask(Message.from_json("ListRegistry", {}))
        data = resp.to_json()
        return data.get("classes", [])

    async def create_actor(
        self,
        class_name: str,
        *args,
        name: str | None = None,
        public: bool = True,
        restart_policy: str = "never",
        max_restarts: int = 3,
        min_backoff: float = 0.1,
        max_backoff: float = 30.0,
        **kwargs,
    ) -> dict:
        """Create a Python actor on a remote node."""
        resp = await self._ref.ask(
            Message.from_json(
                "CreateActor",
                {
                    "class_name": class_name,
                    "actor_name": name,
                    "args": args,
                    "kwargs": kwargs,
                    "public": public,
                    "restart_policy": restart_policy,
                    "max_restarts": max_restarts,
                    "min_backoff": min_backoff,
                    "max_backoff": max_backoff,
                },
            ),
        )
        data = resp.to_json()
        if resp.msg_type == "Error" or data.get("error"):
            raise PulsingRuntimeError(data.get("error", "Unknown error"))
        return data


async def get_python_actor_service(
    system: ActorSystem, node_id: int | None = None
) -> PythonActorServiceProxy:
    """Get PythonActorServiceProxy for direct method calls."""
    service_ref = await system.resolve_named(PYTHON_ACTOR_SERVICE_NAME, node_id=node_id)
    return PythonActorServiceProxy(service_ref)
