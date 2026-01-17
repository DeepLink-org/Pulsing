"""TopicBroker - Lightweight Pub/Sub Broker Actor (internal implementation)

TopicBroker lifecycle is managed by StorageManager in queue/manager.py,
uses consistent hashing to ensure only one broker per topic in the cluster.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pulsing.actor import ActorRef, ActorSystem

from pulsing.actor import Actor, ActorId, Message

logger = logging.getLogger(__name__)

# Subscriber lifecycle management configuration
MAX_CONSECUTIVE_FAILURES = 3  # Consecutive failure threshold, evict after exceeding
REF_TTL_SECONDS = 60.0  # ActorRef cache TTL, re-resolve after expiration

# Timeout configuration (approach 2+3: timeout + idempotency)
DEFAULT_FANOUT_TIMEOUT = 30.0  # Default timeout for wait_any_ack / wait_all_acks


@dataclass
class _Subscriber:
    """Subscriber information (internal use)"""

    subscriber_id: str
    actor_name: str
    node_id: int | None = None
    subscribed_at: float = field(default_factory=time.time)
    _ref: "ActorRef | None" = field(default=None, repr=False)
    _ref_resolved_at: float = 0  # ActorRef resolution time, for TTL judgment
    messages_delivered: int = 0
    messages_failed: int = 0
    consecutive_failures: int = 0  # Consecutive failures, for eviction judgment


class TopicBroker(Actor):
    """Topic Broker Actor (internal implementation)

    Each topic corresponds to one broker, responsible for:
    1. Managing subscriber list
    2. Receiving publish requests and distributing to subscribers
    3. Deciding whether to wait for ack based on publish mode
    """

    def __init__(self, topic: str, system: "ActorSystem"):
        self.topic = topic
        self.system = system
        self._subscribers: dict[str, _Subscriber] = {}
        self._lock = asyncio.Lock()
        self._total_published = 0
        self._total_delivered = 0
        self._total_failed = 0

    def on_start(self, actor_id: ActorId) -> None:
        logger.debug(f"TopicBroker[{self.topic}] started")

    def on_stop(self) -> None:
        logger.debug(f"TopicBroker[{self.topic}] stopped")

    def metadata(self) -> dict[str, str]:
        return {
            "type": "topic_broker",
            "topic": self.topic,
            "subscriber_count": str(len(self._subscribers)),
        }

    async def receive(self, msg: Message) -> Message | None:
        try:
            return await self._handle(msg)
        except Exception as e:
            logger.exception(f"TopicBroker[{self.topic}] error: {e}")
            return Message.from_json("Error", {"error": str(e)})

    async def _handle(self, msg: Message) -> Message | None:
        data = msg.to_json()

        if msg.msg_type == "Subscribe":
            return await self._subscribe(data)
        elif msg.msg_type == "Unsubscribe":
            return await self._unsubscribe(data)
        elif msg.msg_type == "Publish":
            return await self._publish(data)
        elif msg.msg_type == "GetStats":
            return self._stats()
        else:
            return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})

    async def _subscribe(self, data: dict) -> Message:
        subscriber_id = data.get("subscriber_id")
        actor_name = data.get("actor_name")
        node_id = data.get("node_id")

        if not subscriber_id or not actor_name:
            return Message.from_json(
                "Error", {"error": "Missing subscriber_id or actor_name"}
            )

        async with self._lock:
            if subscriber_id in self._subscribers:
                return Message.from_json(
                    "SubscribeResult", {"success": True, "already": True}
                )

            self._subscribers[subscriber_id] = _Subscriber(
                subscriber_id=subscriber_id,
                actor_name=actor_name,
                node_id=node_id,
            )
            logger.debug(f"TopicBroker[{self.topic}] +subscriber: {subscriber_id}")
            return Message.from_json(
                "SubscribeResult", {"success": True, "topic": self.topic}
            )

    async def _unsubscribe(self, data: dict) -> Message:
        subscriber_id = data.get("subscriber_id")
        if not subscriber_id:
            return Message.from_json("Error", {"error": "Missing subscriber_id"})

        async with self._lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                logger.debug(f"TopicBroker[{self.topic}] -subscriber: {subscriber_id}")
                return Message.from_json("UnsubscribeResult", {"success": True})
            return Message.from_json("UnsubscribeResult", {"success": False})

    async def _resolve(self, sub: _Subscriber) -> "ActorRef | None":
        """Resolve subscriber ActorRef (with TTL cache)

        Cache strategy:
        - Cache ActorRef after first resolution
        - Re-resolve after TTL expiration to detect node failures
        - Clear cache on resolution failure, retry next time
        """
        now = time.time()

        # Check if cache is valid (exists and not expired)
        if sub._ref is not None and (now - sub._ref_resolved_at) < REF_TTL_SECONDS:
            return sub._ref

        # TTL expired or no cache, re-resolve
        try:
            sub._ref = await self.system.resolve_named(
                sub.actor_name, node_id=sub.node_id
            )
            sub._ref_resolved_at = now
            return sub._ref
        except Exception as e:
            logger.warning(f"Failed to resolve {sub.subscriber_id}: {e}")
            sub._ref = None  # Clear invalid cache
            sub._ref_resolved_at = 0
            return None

    async def _publish(self, data: dict) -> Message:
        payload = data.get("payload")
        mode = data.get("mode", "fire_and_forget")
        sender_id = data.get("sender_id")

        self._total_published += 1

        if not self._subscribers:
            return Message.from_json(
                "PublishResult",
                {"success": True, "delivered": 0, "failed": 0, "subscriber_count": 0},
            )

        envelope = {
            "topic": self.topic,
            "payload": payload,
            "sender_id": sender_id,
            "timestamp": time.time(),
        }

        if mode == "fire_and_forget":
            return await self._fanout_tell(envelope, sender_id)
        elif mode == "wait_all_acks":
            return await self._fanout_ask(envelope, sender_id, wait_all=True)
        elif mode == "wait_any_ack":
            return await self._fanout_ask(envelope, sender_id, wait_all=False)
        elif mode == "best_effort":
            return await self._fanout_best_effort(envelope, sender_id)
        else:
            return Message.from_json("Error", {"error": f"Unknown mode: {mode}"})

    def _record_success(self, sub: _Subscriber) -> None:
        """Record delivery success, reset consecutive failure count"""
        sub.messages_delivered += 1
        sub.consecutive_failures = 0

    def _record_failure(self, sub: _Subscriber) -> bool:
        """Record delivery failure, return whether should evict

        Returns:
            True if consecutive failures exceed threshold, should evict
        """
        sub.messages_failed += 1
        sub.consecutive_failures += 1
        return sub.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    async def _evict_zombies(self, zombie_ids: list[str]) -> None:
        """Evict zombie subscribers"""
        if not zombie_ids:
            return
        async with self._lock:
            for sub_id in zombie_ids:
                if sub_id in self._subscribers:
                    del self._subscribers[sub_id]
                    logger.warning(
                        f"TopicBroker[{self.topic}] evicted zombie subscriber: {sub_id}"
                    )

    async def _fanout_tell(self, envelope: dict, sender_id: str | None) -> Message:
        """Fire-and-forget: tell without waiting for response"""
        sent = 0
        failed = 0
        zombies: list[str] = []

        for sub_id, sub in list(self._subscribers.items()):
            if sender_id and sub_id == sender_id:
                continue
            try:
                ref = await self._resolve(sub)
                if ref:
                    await ref.tell(envelope)
                    sent += 1
                    self._record_success(sub)
                else:
                    failed += 1
                    if self._record_failure(sub):
                        zombies.append(sub_id)
            except Exception:
                failed += 1
                if self._record_failure(sub):
                    zombies.append(sub_id)

        # Evict zombie subscribers
        await self._evict_zombies(zombies)

        self._total_delivered += sent
        self._total_failed += failed

        return Message.from_json(
            "PublishResult",
            {
                "success": True,
                "delivered": sent,
                "failed": failed,
                "subscriber_count": len(self._subscribers),
            },
        )

    async def _fanout_ask(
        self,
        envelope: dict,
        sender_id: str | None,
        wait_all: bool,
        timeout: float = DEFAULT_FANOUT_TIMEOUT,
    ) -> Message:
        """Wait for ack mode

        Args:
            envelope: Message envelope
            sender_id: Sender ID (to exclude self)
            wait_all: True=wait for all responses, False=wait for any response (wait_any_ack)
            timeout: Timeout in seconds. Local task will be cancelled after timeout,
                     remote handler may still be executing (relies on HTTP/2 RST_STREAM propagation).

        Note (cancellation semantics - approach 2+3):
        - Local cancellation: via asyncio.wait timeout or task.cancel()
        - Remote cancellation: relies on HTTP/2 RST_STREAM auto-propagation (triggered when body read is interrupted)
        - Idempotency: Handler should implement idempotent operations to ensure repeated requests don't produce side effects
        """
        tasks = []
        sub_ids = []
        resolve_failed: list[str] = []  # Subscribers that failed to resolve

        for sub_id, sub in list(self._subscribers.items()):
            if sender_id and sub_id == sender_id:
                continue
            ref = await self._resolve(sub)
            if ref:
                tasks.append(ref.ask(envelope))
                sub_ids.append(sub_id)
            else:
                # Resolve failures also count as failures
                if self._record_failure(sub):
                    resolve_failed.append(sub_id)

        if not tasks:
            await self._evict_zombies(resolve_failed)
            return Message.from_json(
                "PublishResult",
                {"success": True, "delivered": 0, "failed": 0, "subscriber_count": 0},
            )

        delivered = 0
        failed = 0
        failed_ids = []
        zombies: list[str] = resolve_failed.copy()

        if wait_all:
            # wait_all_acks: wait for all responses with overall timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )
                for i, result in enumerate(results):
                    sub = self._subscribers.get(sub_ids[i])
                    if isinstance(result, Exception):
                        failed += 1
                        failed_ids.append(sub_ids[i])
                        if sub and self._record_failure(sub):
                            zombies.append(sub_ids[i])
                    else:
                        delivered += 1
                        if sub:
                            self._record_success(sub)
            except asyncio.TimeoutError:
                # Timeout: all tasks considered failed
                logger.warning(
                    f"TopicBroker[{self.topic}] wait_all_acks timeout after {timeout}s"
                )
                failed = len(tasks)
                failed_ids = sub_ids.copy()
                # Cancel all pending tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
        else:
            # wait_any_ack: wait for any response with overall timeout
            try:
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=timeout,
                )
                for task in done:
                    if not task.exception():
                        delivered = 1
                        break
                # Cancel other pending tasks (local cancellation, remote relies on RST_STREAM)
                for task in pending:
                    task.cancel()
            except asyncio.TimeoutError:
                # Timeout: no response
                logger.warning(
                    f"TopicBroker[{self.topic}] wait_any_ack timeout after {timeout}s"
                )
                # Cancel all tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()

        # Evict zombie subscribers
        await self._evict_zombies(zombies)

        self._total_delivered += delivered
        self._total_failed += failed

        return Message.from_json(
            "PublishResult",
            {
                "success": delivered > 0 or failed == 0,
                "delivered": delivered,
                "failed": failed,
                "failed_subscribers": failed_ids,
                "subscriber_count": len(self._subscribers),
            },
        )

    async def _fanout_best_effort(
        self, envelope: dict, sender_id: str | None
    ) -> Message:
        """Best-effort: try to send, record failures"""
        delivered = 0
        failed = 0
        failed_ids = []
        zombies: list[str] = []

        for sub_id, sub in list(self._subscribers.items()):
            if sender_id and sub_id == sender_id:
                continue
            try:
                ref = await self._resolve(sub)
                if ref:
                    await asyncio.wait_for(ref.ask(envelope), timeout=5.0)
                    delivered += 1
                    self._record_success(sub)
                else:
                    failed += 1
                    failed_ids.append(sub_id)
                    if self._record_failure(sub):
                        zombies.append(sub_id)
            except Exception:
                failed += 1
                failed_ids.append(sub_id)
                if self._record_failure(sub):
                    zombies.append(sub_id)

        # Evict zombie subscribers
        await self._evict_zombies(zombies)

        self._total_delivered += delivered
        self._total_failed += failed

        return Message.from_json(
            "PublishResult",
            {
                "success": True,
                "delivered": delivered,
                "failed": failed,
                "failed_subscribers": failed_ids,
                "subscriber_count": len(self._subscribers),
            },
        )

    def _stats(self) -> Message:
        return Message.from_json(
            "TopicStats",
            {
                "topic": self.topic,
                "subscriber_count": len(self._subscribers),
                "total_published": self._total_published,
                "total_delivered": self._total_delivered,
                "total_failed": self._total_failed,
            },
        )
