"""Topic API - Pub/Sub high-level interface"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from pulsing.actor import ActorRef
    from pulsing.actor.remote import ActorProxy

from pulsing.actor import Actor, ActorId, ActorSystem, Message

logger = logging.getLogger(__name__)

# Default timeout for publish operations (seconds)
DEFAULT_PUBLISH_TIMEOUT = 30.0


class PublishMode(Enum):
    """Publish mode"""

    FIRE_AND_FORGET = "fire_and_forget"  # Return immediately after sending
    WAIT_ALL_ACKS = "wait_all_acks"  # Wait for all subscribers to respond
    WAIT_ANY_ACK = "wait_any_ack"  # Wait for any subscriber to respond
    BEST_EFFORT = "best_effort"  # Try to send, record failures


@dataclass
class PublishResult:
    """Publish result"""

    success: bool
    delivered: int
    failed: int
    subscriber_count: int
    failed_subscribers: list[str] | None = None


# Callback type
MessageCallback = Callable[[Any], Coroutine[Any, Any, Any] | Any]


async def _get_broker(system: ActorSystem, topic: str) -> "ActorProxy":
    """Get topic broker proxy (reuses queue/manager infrastructure)"""
    from pulsing.queue.manager import get_topic_broker

    # get_topic_broker already returns ActorProxy (via TopicBroker.resolve)
    return await get_topic_broker(system, topic)


async def subscribe_to_topic(
    system: ActorSystem,
    topic: str,
    subscriber_id: str,
    actor_name: str,
    node_id: int | None = None,
) -> dict:
    """Subscribe an actor to a topic.

    This is a helper function for manually registering subscribers with a topic broker.
    For normal usage, prefer using TopicReader which handles this automatically.

    Args:
        system: ActorSystem instance
        topic: Topic name
        subscriber_id: Unique subscriber identifier
        actor_name: Name of the actor to receive messages
        node_id: Optional node ID (defaults to local node)

    Returns:
        Response dict from broker

    Raises:
        RuntimeError: If subscription fails
    """
    broker = await _get_broker(system, topic)
    # Direct method call on broker proxy
    return await broker.subscribe(subscriber_id, actor_name, node_id)


class TopicWriter:
    """Topic write handle"""

    def __init__(self, system: ActorSystem, topic: str, writer_id: str | None = None):
        self._system = system
        self._topic = topic
        self._writer_id = writer_id or f"writer_{uuid.uuid4().hex[:8]}"
        self._broker: "ActorProxy | None" = None

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def writer_id(self) -> str:
        return self._writer_id

    async def _broker_ref(self) -> "ActorProxy":
        if self._broker is None:
            self._broker = await _get_broker(self._system, self._topic)
        return self._broker

    async def publish(
        self,
        message: Any,
        mode: PublishMode = PublishMode.FIRE_AND_FORGET,
        timeout: float | None = None,
    ) -> PublishResult:
        """Publish message

        Args:
            message: Message to publish (any Python object)
            mode: Publish mode
            timeout: Timeout in seconds. None means use default timeout.
                     For WAIT_ANY_ACK and WAIT_ALL_ACKS modes, local task will be cancelled after timeout,
                     but remote handler may still be executing (relies on HTTP/2 RST_STREAM to propagate cancellation).

        Returns:
            PublishResult: Publish result

        Raises:
            asyncio.TimeoutError: Timeout
            RuntimeError: Other errors
        """
        broker = await self._broker_ref()

        # Determine timeout value
        effective_timeout = timeout if timeout is not None else DEFAULT_PUBLISH_TIMEOUT

        async def _do_publish():
            # Direct method call on broker proxy
            return await broker.publish(
                message,
                mode=mode.value,
                sender_id=self._writer_id,
                timeout=effective_timeout,
            )

        data = await asyncio.wait_for(_do_publish(), timeout=effective_timeout)

        return PublishResult(
            success=data.get("success", False),
            delivered=data.get("delivered", 0),
            failed=data.get("failed", 0),
            subscriber_count=data.get("subscriber_count", 0),
            failed_subscribers=data.get("failed_subscribers"),
        )

    async def stats(self) -> dict[str, Any]:
        """Get topic statistics"""
        broker = await self._broker_ref()
        # Direct method call on broker proxy
        return await broker.get_stats()


class _SubscriberActor(Actor):
    """Subscriber Actor (internal use)"""

    def __init__(self, callbacks: list[MessageCallback]):
        self._callbacks = callbacks

    def on_start(self, actor_id: ActorId) -> None:
        pass

    def on_stop(self) -> None:
        pass

    async def receive(self, msg: Any) -> Any:
        # Extract payload
        if isinstance(msg, dict):
            payload = msg.get("payload", msg)
        elif isinstance(msg, Message):
            payload = msg.to_json().get("payload", msg.to_json())
        else:
            payload = msg

        # Call callbacks
        for callback in self._callbacks:
            try:
                result = callback(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.exception(f"Callback error: {e}")

        return {"ack": True}


class TopicReader:
    """Topic read handle

    Example:
        reader = await read_topic(system, "events")

        @reader.on_message
        async def handle(msg):
            print(f"Received: {msg}")

        await reader.start()
    """

    def __init__(self, system: ActorSystem, topic: str, reader_id: str | None = None):
        self._system = system
        self._topic = topic
        self._reader_id = reader_id or f"reader_{uuid.uuid4().hex[:8]}"
        self._callbacks: list[MessageCallback] = []
        self._subscriber_ref: "ActorRef | None" = None
        self._started = False

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def reader_id(self) -> str:
        return self._reader_id

    @property
    def is_started(self) -> bool:
        return self._started

    def on_message(self, callback: MessageCallback) -> MessageCallback:
        """Register message callback (decorator style)

        Example:
            @reader.on_message
            async def handle(msg):
                print(f"Received: {msg}")
        """
        self._callbacks.append(callback)
        return callback

    def add_callback(self, callback: MessageCallback) -> None:
        """Add message callback"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: MessageCallback) -> bool:
        """Remove message callback"""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    async def start(self) -> None:
        """Start receiving messages"""
        if self._started:
            return

        if not self._callbacks:
            logger.warning(f"TopicReader[{self._reader_id}] has no callbacks")

        # Create subscriber Actor
        actor_name = f"_topic_sub_{self._topic}_{self._reader_id}"
        subscriber = _SubscriberActor(self._callbacks)
        self._subscriber_ref = await self._system.spawn(
            subscriber, name=actor_name, public=True
        )

        # Register with broker using direct method call
        broker = await _get_broker(self._system, self._topic)
        await broker.subscribe(
            self._reader_id,
            actor_name,
            node_id=self._system.node_id.id,
        )

        self._started = True
        logger.debug(f"TopicReader[{self._reader_id}] started for topic: {self._topic}")

    async def stop(self) -> None:
        """Stop receiving messages"""
        if not self._started:
            return

        # Unsubscribe from broker using direct method call
        try:
            broker = await _get_broker(self._system, self._topic)
            await broker.unsubscribe(self._reader_id)
        except Exception as e:
            logger.warning(f"Unsubscribe error: {e}")

        # Stop subscriber Actor
        if self._subscriber_ref:
            try:
                actor_name = f"_topic_sub_{self._topic}_{self._reader_id}"
                await self._system.stop(actor_name)
            except Exception as e:
                logger.warning(f"Stop subscriber error: {e}")

        self._started = False
        self._subscriber_ref = None
        logger.debug(f"TopicReader[{self._reader_id}] stopped")

    async def stats(self) -> dict[str, Any]:
        """Get topic statistics"""
        broker = await _get_broker(self._system, self._topic)
        # Direct method call on broker proxy
        return await broker.get_stats()


async def write_topic(
    system: ActorSystem,
    topic: str,
    writer_id: str | None = None,
) -> TopicWriter:
    """Open topic for writing

    Args:
        system: Actor system
        topic: Topic name
        writer_id: Writer ID (optional)

    Returns:
        TopicWriter: Write handle

    Example:
        writer = await write_topic(system, "events")
        await writer.publish({"type": "user_login"})
    """
    return TopicWriter(system, topic, writer_id)


async def read_topic(
    system: ActorSystem,
    topic: str,
    reader_id: str | None = None,
    auto_start: bool = False,
) -> TopicReader:
    """Open topic for reading

    Args:
        system: Actor system
        topic: Topic name
        reader_id: Reader ID (optional)
        auto_start: Whether to automatically start receiving

    Returns:
        TopicReader: Read handle

    Example:
        reader = await read_topic(system, "events")

        @reader.on_message
        async def handle(msg):
            print(f"Received: {msg}")

        await reader.start()
    """
    reader = TopicReader(system, topic, reader_id)
    if auto_start:
        await reader.start()
    return reader
