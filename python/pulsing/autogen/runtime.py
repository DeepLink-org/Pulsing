"""
PulsingRuntime - AutoGen Compatible Runtime Based on Pulsing

Features:
- Unified API for standalone/distributed modes
- No central coordinator, Gossip auto-discovery
- Built-in load balancing and failover
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import signal
import uuid
from asyncio import Future, Task
from collections import defaultdict
from typing import (
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    Dict,
    Mapping,
    Sequence,
    Set,
    Type,
    TypeVar,
    cast,
)

from pulsing.actor import (
    Actor,
    ActorRef,
    ActorSystem,
    Message,
    SystemConfig,
)
from pulsing.actor.remote import PYTHON_ACTOR_SERVICE_NAME, PythonActorService

logger = logging.getLogger("pulsing.autogen")
T = TypeVar("T")


class PulsingRuntime:
    """AutoGen Compatible Runtime Based on Pulsing

    Features:
    - Unified API for standalone/distributed modes
    - No central coordinator
    - Gossip auto-discovery
    - Built-in load balancing and failover

    Usage:
        # Standalone mode
        runtime = PulsingRuntime()

        # Distributed mode - first node
        runtime = PulsingRuntime(addr="0.0.0.0:8000")

        # Distributed mode - join cluster
        runtime = PulsingRuntime(
            addr="0.0.0.0:8000",
            seeds=["first-node:8000"]
        )
    """

    def __init__(
        self,
        *,
        addr: str | None = None,
        seeds: list[str] | None = None,
    ) -> None:
        """
        Args:
            addr: Local node listening address, None means standalone mode
            seeds: List of seed node addresses for joining cluster
        """
        self._addr = addr
        self._seeds = seeds or []

        # Pulsing ActorSystem
        self._system: ActorSystem | None = None

        # Agent management
        self._agent_factories: Dict[str, Callable[[], Any | Awaitable[Any]]] = {}
        self._instantiated_agents: Dict[str, Any] = {}  # agent_key -> Agent instance
        self._agent_refs: Dict[str, ActorRef] = {}  # agent_key -> ActorRef

        # Subscription management: topic_key -> set of agent_types
        self._subscriptions: DefaultDict[str, Set[str]] = defaultdict(set)

        # Running state
        self._running = False
        self._run_task: Task | None = None
        self._pending_requests: Dict[str, Future[Any]] = {}
        self._background_tasks: Set[Task[Any]] = set()

    @property
    def is_distributed(self) -> bool:
        """Whether in distributed mode"""
        return self._addr is not None

    async def start(self) -> None:
        """Start runtime"""
        if self._running:
            raise RuntimeError("Runtime is already running")

        # Create Pulsing ActorSystem
        if self._addr:
            # Distributed mode
            config = SystemConfig.with_addr(self._addr)
            if self._seeds:
                config = config.with_seeds(self._seeds)
        else:
            # Standalone mode
            config = SystemConfig.standalone()

        loop = asyncio.get_running_loop()
        self._system = await ActorSystem.create(config, loop)
        # Register PythonActorService for remote actor creation
        service = PythonActorService(self._system)
        await self._system.spawn(service, name=PYTHON_ACTOR_SERVICE_NAME, public=True)
        self._running = True

        mode = "distributed" if self.is_distributed else "standalone"
        logger.info(f"PulsingRuntime started in {mode} mode")
        if self._addr:
            logger.info(f"Listening on {self._addr}, seeds: {self._seeds}")

    async def stop(self) -> None:
        """Stop runtime"""
        if not self._running:
            raise RuntimeError("Runtime is not running")

        self._running = False

        # Wait for all background tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("PulsingRuntime stopped")

    async def stop_when_idle(self) -> None:
        """Stop runtime when idle"""
        # Wait for all pending requests to complete
        while self._pending_requests:
            await asyncio.sleep(0.1)
        await self.stop()

    async def stop_when_signal(
        self, signals: Sequence[signal.Signals] = (signal.SIGTERM, signal.SIGINT)
    ) -> None:
        """Stop when signal is received"""
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def signal_handler():
            logger.info("Received exit signal, shutting down...")
            shutdown_event.set()

        for sig in signals:
            loop.add_signal_handler(sig, signal_handler)

        await shutdown_event.wait()
        await self.stop()

    # ========================================================================
    # AutoGen AgentRuntime Protocol Implementation
    # ========================================================================

    async def send_message(
        self,
        message: Any,
        recipient: Any,  # AgentId
        *,
        sender: Any | None = None,
        cancellation_token: Any | None = None,
        message_id: str | None = None,
    ) -> Any:
        """Send message to specified Agent (RPC mode)"""
        if not self._running:
            raise RuntimeError("Runtime is not running")

        # Parse recipient
        agent_type = recipient.type if hasattr(recipient, "type") else str(recipient)
        agent_key = recipient.key if hasattr(recipient, "key") else "default"
        full_key = f"{agent_type}/{agent_key}"

        # Ensure Agent is created
        await self._ensure_agent(agent_type, agent_key)

        # Get ActorRef
        actor_ref = self._agent_refs.get(full_key)
        if actor_ref is None:
            raise LookupError(f"Agent '{full_key}' not found")

        # Construct message
        msg_id = message_id or str(uuid.uuid4())
        envelope = {
            "__autogen_msg__": True,
            "payload": message,
            "sender": {
                "type": sender.type if sender else None,
                "key": sender.key if sender else None,
            }
            if sender
            else None,
            "topic_id": None,
            "is_rpc": True,
            "message_id": msg_id,
        }

        # Send and wait for response
        response = await actor_ref.ask(envelope)

        # Process response
        response = self._deserialize_response(response)

        # If it's our AutoGen response format
        if isinstance(response, dict) and "__autogen_response__" in response:
            if "__error__" in response:
                raise RuntimeError(response["__error__"])
            return response.get("result")

        # Direct return
        return response

    def _deserialize_response(self, response: Any) -> Any:
        """Deserialize response"""
        import pickle

        # 1. If it's a Pulsing Message object
        if hasattr(response, "msg_type") and hasattr(response, "payload"):
            msg_type = response.msg_type
            payload = response.payload

            # If msg_type is empty, it's a pickled Python object
            if not msg_type and isinstance(payload, bytes):
                try:
                    return pickle.loads(payload)
                except Exception:
                    pass

            # If msg_type exists, try to_json
            if hasattr(response, "to_json"):
                try:
                    return response.to_json()
                except Exception:
                    pass

        # 2. If it's bytes, try unpickle
        if isinstance(response, bytes):
            try:
                return pickle.loads(response)
            except Exception:
                pass

        # 3. Direct return
        return response

    async def publish_message(
        self,
        message: Any,
        topic_id: Any,  # TopicId
        *,
        sender: Any | None = None,
        cancellation_token: Any | None = None,
        message_id: str | None = None,
    ) -> None:
        """Publish message to Topic (Pub/Sub mode)"""
        if not self._running:
            raise RuntimeError("Runtime is not running")

        # Parse topic
        topic_type = topic_id.type if hasattr(topic_id, "type") else str(topic_id)
        topic_source = topic_id.source if hasattr(topic_id, "source") else "default"
        topic_key = f"{topic_type}/{topic_source}"

        # Get subscribers
        subscriber_types = self._subscriptions.get(topic_key, set())

        # Construct message
        msg_id = message_id or str(uuid.uuid4())
        envelope = {
            "__autogen_msg__": True,
            "payload": message,
            "sender": {
                "type": sender.type if sender else None,
                "key": sender.key if sender else None,
            }
            if sender
            else None,
            "topic_id": {
                "type": topic_type,
                "source": topic_source,
            },
            "is_rpc": False,
            "message_id": msg_id,
        }

        # Send to all subscribers
        tasks = []
        sender_key = f"{sender.type}/{sender.key}" if sender else None

        for agent_type in subscriber_types:
            agent_key = "default"  # Default key
            full_key = f"{agent_type}/{agent_key}"

            # Don't send to self
            if full_key == sender_key:
                continue

            # Ensure Agent is created
            await self._ensure_agent(agent_type, agent_key)

            actor_ref = self._agent_refs.get(full_key)
            if actor_ref:
                # Use tell (don't wait for response)
                task = asyncio.create_task(actor_ref.ask(envelope))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def register_factory(
        self,
        type: str | Any,  # str | AgentType
        agent_factory: Callable[[], T | Awaitable[T]],
        *,
        expected_class: type[T] | None = None,
        eager: bool = True,  # Immediately create Agent instance
    ) -> Any:  # AgentType
        """Register Agent factory

        Args:
            type: Agent type name
            agent_factory: Factory function to create Agent
            expected_class: Expected Agent type (for validation)
            eager: Whether to immediately create Agent instance (True required for distributed mode)
        """
        agent_type = type.type if hasattr(type, "type") else str(type)

        if agent_type in self._agent_factories:
            raise ValueError(f"Agent type '{agent_type}' already registered")

        self._agent_factories[agent_type] = agent_factory
        logger.debug(f"Registered agent factory: {agent_type}")

        # Immediately create instance (required in distributed mode, otherwise other nodes cannot discover)
        if eager and self._running:
            await self._ensure_agent(agent_type, "default")

        # Return AgentType (compatible with AutoGen)
        try:
            from autogen_core import AgentType as AutoGenAgentType

            return AutoGenAgentType(agent_type)
        except ImportError:
            return agent_type

    async def register_agent_instance(
        self,
        agent_instance: Any,  # Agent
        agent_id: Any,  # AgentId
    ) -> Any:  # AgentId
        """Register Agent instance"""
        agent_type = agent_id.type if hasattr(agent_id, "type") else str(agent_id)
        agent_key = agent_id.key if hasattr(agent_id, "key") else "default"
        full_key = f"{agent_type}/{agent_key}"

        if full_key in self._instantiated_agents:
            raise ValueError(f"Agent '{full_key}' already exists")

        # Bind runtime
        if hasattr(agent_instance, "bind_id_and_runtime"):
            await agent_instance.bind_id_and_runtime(id=agent_id, runtime=self)

        # Create wrapper and spawn
        wrapper = AutoGenAgentWrapper(agent_instance, self)

        actor_ref = await self._system.spawn(wrapper, name=full_key, public=True)

        self._instantiated_agents[full_key] = agent_instance
        self._agent_refs[full_key] = actor_ref

        logger.debug(f"Registered agent instance: {full_key}")
        return agent_id

    async def add_subscription(self, subscription: Any) -> None:
        """Add subscription"""
        # Parse subscription
        if hasattr(subscription, "topic_type"):
            topic_type = subscription.topic_type
        elif hasattr(subscription, "id"):
            topic_type = subscription.id
        else:
            topic_type = "default"

        if hasattr(subscription, "agent_type"):
            agent_type = subscription.agent_type
        else:
            agent_type = str(subscription)

        # Use "default" as default source
        topic_key = f"{topic_type}:default"
        self._subscriptions[topic_key].add(agent_type)

        logger.debug(f"Added subscription: {agent_type} -> {topic_key}")

    async def remove_subscription(self, id: str) -> None:
        """Remove subscription"""
        # Simplified implementation: iterate to find
        for topic_key, agent_types in self._subscriptions.items():
            agent_types.discard(id)

    async def get(
        self,
        id_or_type: Any,  # AgentId | AgentType | str
        /,
        key: str = "default",
        *,
        lazy: bool = True,
    ) -> Any:  # AgentId
        """Get Agent ID"""
        if hasattr(id_or_type, "type") and hasattr(id_or_type, "key"):
            # Already an AgentId
            return id_or_type

        agent_type = id_or_type.type if hasattr(id_or_type, "type") else str(id_or_type)

        if not lazy:
            await self._ensure_agent(agent_type, key)

        try:
            from autogen_core import AgentId as AutoGenAgentId

            return AutoGenAgentId(agent_type, key)
        except ImportError:
            return f"{agent_type}:{key}"

    async def try_get_underlying_agent_instance(
        self,
        id: Any,  # AgentId
        type: Type[T] = object,  # type: ignore
    ) -> T:
        """Get underlying Agent instance"""
        agent_type = id.type if hasattr(id, "type") else str(id)
        agent_key = id.key if hasattr(id, "key") else "default"
        full_key = f"{agent_type}/{agent_key}"

        if full_key not in self._instantiated_agents:
            await self._ensure_agent(agent_type, agent_key)

        instance = self._instantiated_agents.get(full_key)
        if instance is None:
            raise LookupError(f"Agent '{full_key}' not found")

        if not isinstance(instance, type):
            raise TypeError(f"Agent is not of type {type.__name__}")

        return cast(T, instance)

    async def agent_metadata(self, agent: Any) -> Any:
        """Get Agent metadata"""
        instance = await self.try_get_underlying_agent_instance(agent)
        if hasattr(instance, "metadata"):
            return instance.metadata
        return {}

    async def save_state(self) -> Mapping[str, Any]:
        """Save runtime state"""
        state = {}
        for key, agent in self._instantiated_agents.items():
            if hasattr(agent, "save_state"):
                state[key] = dict(await agent.save_state())
        return state

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load runtime state"""
        for key, agent_state in state.items():
            if key in self._instantiated_agents:
                agent = self._instantiated_agents[key]
                if hasattr(agent, "load_state"):
                    await agent.load_state(agent_state)

    async def agent_save_state(self, agent: Any) -> Mapping[str, Any]:
        """Save single Agent state"""
        instance = await self.try_get_underlying_agent_instance(agent)
        if hasattr(instance, "save_state"):
            return await instance.save_state()
        return {}

    async def agent_load_state(self, agent: Any, state: Mapping[str, Any]) -> None:
        """Load single Agent state"""
        instance = await self.try_get_underlying_agent_instance(agent)
        if hasattr(instance, "load_state"):
            await instance.load_state(state)

    def add_message_serializer(self, serializer: Any) -> None:
        """Add message serializer (Pulsing uses pickle, this method is kept for compatibility)"""
        pass

    # ========================================================================
    # Internal Methods
    # ========================================================================

    async def _ensure_agent(self, agent_type: str, agent_key: str = "default") -> None:
        """Ensure Agent is available (local or remote)"""
        full_key = f"{agent_type}/{agent_key}"

        # 1. Check if local reference already exists
        if full_key in self._agent_refs:
            return

        # 2. Try to find remote Agent in cluster
        if self.is_distributed:
            try:
                actor_ref = await self._system.resolve_named(full_key)
                if actor_ref:
                    self._agent_refs[full_key] = actor_ref
                    logger.debug(f"Found remote agent: {full_key}")
                    return
            except Exception as e:
                logger.debug(f"Agent '{full_key}' not found in cluster: {e}")

        # 3. If local factory exists, create local instance
        if agent_type not in self._agent_factories:
            raise LookupError(f"Agent type '{agent_type}' not registered")

        # Call factory to create Agent
        factory = self._agent_factories[agent_type]
        agent = factory()
        if inspect.isawaitable(agent):
            agent = await agent

        # Create AgentId
        try:
            from autogen_core import AgentId as AutoGenAgentId

            agent_id = AutoGenAgentId(agent_type, agent_key)
        except ImportError:
            agent_id = full_key

        # Bind runtime
        if hasattr(agent, "bind_id_and_runtime"):
            await agent.bind_id_and_runtime(id=agent_id, runtime=self)

        # Create wrapper and spawn
        wrapper = AutoGenAgentWrapper(agent, self)

        actor_ref = await self._system.spawn(wrapper, name=full_key, public=True)

        self._instantiated_agents[full_key] = agent
        self._agent_refs[full_key] = actor_ref

        logger.debug(f"Created local agent: {full_key}")


# Import wrapper
from .agent_wrapper import AutoGenAgentWrapper
