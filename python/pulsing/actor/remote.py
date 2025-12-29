"""
@as_actor decorator - Ray-like distributed object wrapper

Usage:
    from pulsing.actor import as_actor, create_actor_system, SystemConfig

    @as_actor
    class Counter:
        def __init__(self, init_value=0):
            self.value = init_value

        def increment(self, n=1):
            self.value += n
            return self.value

    system = await create_actor_system(config)

    # Local creation
    counter = await Counter.local(system, init_value=10)

    # Remote creation (randomly selects a remote node)
    counter = await Counter.remote(system, init_value=10)

    # Call methods (automatically converted to actor messages)
    result = await counter.increment(5)  # Returns 15
"""

import asyncio
import inspect
import logging
import random
import uuid
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pulsing._core import ActorRef, ActorSystem, Message, StreamMessage

logger = logging.getLogger(__name__)


class _ActorBase(ABC):
    """Actor base class (avoids circular imports)"""

    def on_start(self, actor_id) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def metadata(self) -> dict[str, str]:
        return {}

    @abstractmethod
    async def receive(self, msg: Message) -> Message | StreamMessage | None:
        pass


T = TypeVar("T")

# Global class registry
_actor_class_registry: dict[str, type] = {}

# Python actor service name (different from Rust SystemActor "system/core")
PYTHON_ACTOR_SERVICE_NAME = "_python_actor_service"


class ActorProxy:
    """Actor proxy: automatically converts method calls to ask messages"""

    def __init__(self, actor_ref: ActorRef, method_names: list[str]):
        self._ref = actor_ref
        self._method_names = set(method_names)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"Cannot access private attribute: {name}")
        if name not in self._method_names:
            raise AttributeError(f"No method '{name}'")
        return _MethodCaller(self._ref, name)

    @property
    def ref(self) -> ActorRef:
        """Get underlying ActorRef"""
        return self._ref


class _MethodCaller:
    """Method caller: executes remote method calls"""

    def __init__(self, actor_ref: ActorRef, method_name: str):
        self._ref = actor_ref
        self._method = method_name

    async def __call__(self, *args, **kwargs) -> Any:
        msg = Message.from_json(
            "Call",
            {"method": self._method, "args": list(args), "kwargs": kwargs},
        )
        resp = await self._ref.ask(msg)
        data = resp.to_json()

        if resp.msg_type == "Error":
            raise RuntimeError(data.get("error", "Remote call failed"))
        return data.get("result")


class _WrappedActor(_ActorBase):
    """Wraps user class as an Actor"""

    def __init__(self, instance: Any):
        self._instance = instance

    def on_start(self, actor_id) -> None:
        if hasattr(self._instance, "on_start"):
            self._instance.on_start(actor_id)

    def on_stop(self) -> None:
        if hasattr(self._instance, "on_stop"):
            self._instance.on_stop()

    async def receive(self, msg: Message) -> Message | StreamMessage | None:
        if msg.msg_type != "Call":
            return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})

        data = msg.to_json()
        method = data.get("method")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})

        if not method or method.startswith("_"):
            return Message.from_json("Error", {"error": f"Invalid method: {method}"})

        func = getattr(self._instance, method, None)
        if func is None or not callable(func):
            return Message.from_json("Error", {"error": f"Not found: {method}"})

        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return Message.from_json("Result", {"result": result})
        except Exception as e:
            return Message.from_json("Error", {"error": str(e)})


class PythonActorService(_ActorBase):
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
            # List registered actor classes
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

        cls = _actor_class_registry.get(class_name)
        if cls is None:
            return Message.from_json(
                "Error", {"error": f"Class '{class_name}' not found"}
            )

        try:
            instance = cls(*args, **kwargs)
            actor = _WrappedActor(instance)
            actor_ref = await self.system.spawn(actor_name, actor, public=public)

            method_names = [
                n
                for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
                if not n.startswith("_")
            ]

            return Message.from_json(
                "Created",
                {
                    "actor_id": actor_ref.actor_id.local_id,
                    "node_id": self.system.node_id.id,
                    "methods": method_names,
                },
            )
        except Exception as e:
            logger.exception(f"Create actor failed: {e}")
            return Message.from_json("Error", {"error": str(e)})


class ActorClass:
    """Actor class wrapper"""

    def __init__(self, cls: type):
        self._cls = cls
        self._class_name = f"{cls.__module__}.{cls.__name__}"
        self._methods = [
            n
            for n, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not n.startswith("_")
        ]
        # Register class
        _actor_class_registry[self._class_name] = cls

    async def local(
        self,
        system: ActorSystem,
        *args,
        name: str | None = None,
        **kwargs,
    ) -> ActorProxy:
        """Create actor locally.

        Note: Use create_actor_system() to create ActorSystem,
        which automatically registers PythonActorService.
        """
        instance = self._cls(*args, **kwargs)
        actor = _WrappedActor(instance)
        actor_name = name or f"{self._cls.__name__}_{uuid.uuid4().hex[:8]}"
        actor_ref = await system.spawn(actor_name, actor, public=True)

        return ActorProxy(actor_ref, self._methods)

    async def remote(
        self,
        system: ActorSystem,
        *args,
        name: str | None = None,
        **kwargs,
    ) -> ActorProxy:
        """Create actor remotely (randomly selects a remote node).

        Note: Use create_actor_system() to create ActorSystem,
        which automatically registers PythonActorService.
        """

        members = await system.members()
        local_id = system.node_id.id

        # Filter out remote nodes
        remote_nodes = [m for m in members if int(m["node_id"]) != local_id]

        if not remote_nodes:
            # No remote nodes, fallback to local creation
            logger.warning("No remote nodes, fallback to local")
            return await self.local(system, *args, name=name, **kwargs)

        # Randomly select one
        target = random.choice(remote_nodes)
        target_id = int(target["node_id"])

        # Get target node's Python actor creation service
        service_ref = await system.resolve_named(
            PYTHON_ACTOR_SERVICE_NAME, node_id=target_id
        )

        actor_name = name or f"{self._cls.__name__}_{uuid.uuid4().hex[:8]}"

        # Send creation request
        resp = await service_ref.ask(
            Message.from_json(
                "CreateActor",
                {
                    "class_name": self._class_name,
                    "actor_name": actor_name,
                    "args": list(args),
                    "kwargs": kwargs,
                    "public": True,
                },
            )
        )

        data = resp.to_json()
        if resp.msg_type == "Error":
            raise RuntimeError(f"Remote create failed: {data.get('error')}")

        # Build remote ActorRef
        from pulsing._core import ActorId, NodeId

        remote_id = ActorId(data["actor_id"], NodeId(data["node_id"]))
        actor_ref = await system.actor_ref(remote_id)

        return ActorProxy(actor_ref, data.get("methods", self._methods))

    def __call__(self, *args, **kwargs):
        """Direct call returns local instance (not an Actor)"""
        return self._cls(*args, **kwargs)


def as_actor(cls: type[T]) -> ActorClass:
    """@as_actor decorator

    Converts a regular class into a distributed deployable Actor.

    Example:
        @as_actor
        class Counter:
            def __init__(self, init_value=0):
                self.value = init_value

            def increment(self, n=1):
                self.value += n
                return self.value

        # Local creation
        counter = await Counter.local(system, init_value=10)

        # Remote creation
        counter = await Counter.remote(system, init_value=10)

        # Call
        result = await counter.increment(5)
    """
    return ActorClass(cls)


# ============================================================================
# System operation helper functions (calls Rust SystemActor)
# ============================================================================


async def list_actors(system: ActorSystem) -> list[dict]:
    """List all actors on the current node."""
    sys_actor = await system.system()
    # SystemMessage uses serde tag format
    resp = await sys_actor.ask(
        Message.from_json("SystemMessage", {"type": "ListActors"})
    )
    data = resp.to_json()
    if data.get("type") == "Error":
        raise RuntimeError(data.get("message"))
    return data.get("actors", [])


async def get_metrics(system: ActorSystem) -> dict:
    """Get system metrics."""
    sys_actor = await system.system()
    resp = await sys_actor.ask(
        Message.from_json("SystemMessage", {"type": "GetMetrics"})
    )
    return resp.to_json()


async def get_node_info(system: ActorSystem) -> dict:
    """Get node info."""
    sys_actor = await system.system()
    resp = await sys_actor.ask(
        Message.from_json("SystemMessage", {"type": "GetNodeInfo"})
    )
    return resp.to_json()


async def health_check(system: ActorSystem) -> dict:
    """Health check."""
    sys_actor = await system.system()
    resp = await sys_actor.ask(
        Message.from_json("SystemMessage", {"type": "HealthCheck"})
    )
    return resp.to_json()


async def ping(system: ActorSystem, node_id: int | None = None) -> dict:
    """Ping node.

    Args:
        system: ActorSystem instance
        node_id: Target node ID (None means local node)
    """
    if node_id is None:
        sys_actor = await system.system()
    else:
        sys_actor = await system.remote_system(node_id)
    resp = await sys_actor.ask(Message.from_json("SystemMessage", {"type": "Ping"}))
    return resp.to_json()


# Keep `remote` as alias (backward compatibility)
remote = as_actor
RemoteClass = ActorClass
# Keep old name as alias (backward compatibility)
SystemActor = PythonActorService
