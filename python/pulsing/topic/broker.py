"""Topic broker (internal)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pulsing.actor import ActorRef, ActorSystem

from pulsing.actor import ActorId, remote

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3
REF_TTL_SECONDS = 60.0
DEFAULT_FANOUT_TIMEOUT = 30.0


@dataclass
class _Subscriber:
    """Subscriber information."""

    subscriber_id: str
    actor_name: str
    node_id: int | None = None
    subscribed_at: float = field(default_factory=time.time)
    _ref: "ActorRef | None" = field(default=None, repr=False)
    _ref_resolved_at: float = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    consecutive_failures: int = 0


@remote
class TopicBroker:
    """Topic broker actor with remote method support."""

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

    # ========== Public Remote Methods ==========

    async def subscribe(
        self,
        subscriber_id: str,
        actor_name: str,
        node_id: int | None = None,
    ) -> dict:
        """Subscribe an actor to this topic.

        Args:
            subscriber_id: Unique subscriber identifier
            actor_name: Name of the actor to receive messages
            node_id: Optional node ID (for cross-node subscriptions)

        Returns:
            {"success": True, "topic": "..."}
        """
        if not subscriber_id or not actor_name:
            raise ValueError("Missing subscriber_id or actor_name")

        async with self._lock:
            if subscriber_id in self._subscribers:
                return {"success": True, "already": True}

            self._subscribers[subscriber_id] = _Subscriber(
                subscriber_id=subscriber_id,
                actor_name=actor_name,
                node_id=node_id,
            )
            logger.debug(f"TopicBroker[{self.topic}] +subscriber: {subscriber_id}")
            return {"success": True, "topic": self.topic}

    async def unsubscribe(self, subscriber_id: str) -> dict:
        """Unsubscribe from this topic.

        Args:
            subscriber_id: Subscriber ID to remove

        Returns:
            {"success": True/False}
        """
        if not subscriber_id:
            raise ValueError("Missing subscriber_id")

        async with self._lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                logger.debug(f"TopicBroker[{self.topic}] -subscriber: {subscriber_id}")
                return {"success": True}
            return {"success": False}

    async def publish(
        self,
        payload: Any,
        mode: str = "fire_and_forget",
        sender_id: str | None = None,
        timeout: float = DEFAULT_FANOUT_TIMEOUT,
    ) -> dict:
        """Publish a message to all subscribers.

        Args:
            payload: Message payload
            mode: "fire_and_forget", "wait_all_acks", "wait_any_ack", "best_effort"
            sender_id: Optional sender ID (excluded from delivery)
            timeout: Timeout for ack modes

        Returns:
            {"success": True, "delivered": N, "failed": N, "subscriber_count": N}
        """
        self._total_published += 1

        if not self._subscribers:
            return {"success": True, "delivered": 0, "failed": 0, "subscriber_count": 0}

        envelope = {
            "topic": self.topic,
            "payload": payload,
            "sender_id": sender_id,
            "timestamp": time.time(),
        }

        if mode == "fire_and_forget":
            return await self._fanout_tell(envelope, sender_id)
        elif mode == "wait_all_acks":
            return await self._fanout_ask(
                envelope, sender_id, wait_all=True, timeout=timeout
            )
        elif mode == "wait_any_ack":
            return await self._fanout_ask(
                envelope, sender_id, wait_all=False, timeout=timeout
            )
        elif mode == "best_effort":
            return await self._fanout_best_effort(envelope, sender_id)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_stats(self) -> dict:
        """Get topic statistics.

        Returns:
            {"topic": "...", "subscriber_count": N, "total_published": N, ...}
        """
        return {
            "topic": self.topic,
            "subscriber_count": len(self._subscribers),
            "total_published": self._total_published,
            "total_delivered": self._total_delivered,
            "total_failed": self._total_failed,
        }

    # ========== Internal Methods ==========

    async def _resolve(self, sub: _Subscriber) -> "ActorRef | None":
        now = time.time()

        if sub._ref is not None and (now - sub._ref_resolved_at) < REF_TTL_SECONDS:
            return sub._ref

        try:
            sub._ref = await self.system.resolve_named(
                sub.actor_name, node_id=sub.node_id
            )
            sub._ref_resolved_at = now
            return sub._ref
        except Exception as e:
            logger.warning(f"Failed to resolve {sub.subscriber_id}: {e}")
            sub._ref = None
            sub._ref_resolved_at = 0
            return None

    def _record_success(self, sub: _Subscriber) -> None:
        sub.messages_delivered += 1
        sub.consecutive_failures = 0

    def _record_failure(self, sub: _Subscriber) -> bool:
        sub.messages_failed += 1
        sub.consecutive_failures += 1
        return sub.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    async def _evict_zombies(self, zombie_ids: list[str]) -> None:
        if not zombie_ids:
            return
        async with self._lock:
            for sub_id in zombie_ids:
                if sub_id in self._subscribers:
                    del self._subscribers[sub_id]
                    logger.warning(
                        f"TopicBroker[{self.topic}] evicted zombie subscriber: {sub_id}"
                    )

    async def _fanout_tell(self, envelope: dict, sender_id: str | None) -> dict:
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

        await self._evict_zombies(zombies)

        self._total_delivered += sent
        self._total_failed += failed

        return {
            "success": True,
            "delivered": sent,
            "failed": failed,
            "subscriber_count": len(self._subscribers),
        }

    async def _fanout_ask(
        self,
        envelope: dict,
        sender_id: str | None,
        wait_all: bool,
        timeout: float = DEFAULT_FANOUT_TIMEOUT,
    ) -> dict:
        """Wait for ack mode."""
        tasks = []
        sub_ids = []
        resolve_failed: list[str] = []

        for sub_id, sub in list(self._subscribers.items()):
            if sender_id and sub_id == sender_id:
                continue
            ref = await self._resolve(sub)
            if ref:
                tasks.append(ref.ask(envelope))
                sub_ids.append(sub_id)
            else:
                if self._record_failure(sub):
                    resolve_failed.append(sub_id)

        if not tasks:
            await self._evict_zombies(resolve_failed)
            return {"success": True, "delivered": 0, "failed": 0, "subscriber_count": 0}

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
                # Cancel other pending tasks
                for task in pending:
                    task.cancel()
            except asyncio.TimeoutError:
                logger.warning(
                    f"TopicBroker[{self.topic}] wait_any_ack timeout after {timeout}s"
                )
                for task in tasks:
                    if not task.done():
                        task.cancel()

        await self._evict_zombies(zombies)

        self._total_delivered += delivered
        self._total_failed += failed

        return {
            "success": delivered > 0 or failed == 0,
            "delivered": delivered,
            "failed": failed,
            "failed_subscribers": failed_ids,
            "subscriber_count": len(self._subscribers),
        }

    async def _fanout_best_effort(self, envelope: dict, sender_id: str | None) -> dict:
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

        await self._evict_zombies(zombies)

        self._total_delivered += delivered
        self._total_failed += failed

        return {
            "success": True,
            "delivered": delivered,
            "failed": failed,
            "failed_subscribers": failed_ids,
            "subscriber_count": len(self._subscribers),
        }
