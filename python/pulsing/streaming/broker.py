"""Topic broker (internal)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pulsing.core import ActorRef, ActorSystem

from pulsing.core import ActorId, remote

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
            return await self._fanout_sequential(envelope, sender_id)
        elif mode == "best_effort":
            return await self._fanout_sequential(
                envelope, sender_id, per_sub_timeout=5.0
            )
        elif mode == "wait_all_acks":
            return await self._fanout_concurrent(
                envelope, sender_id, wait_all=True, timeout=timeout
            )
        elif mode == "wait_any_ack":
            return await self._fanout_concurrent(
                envelope, sender_id, wait_all=False, timeout=timeout
            )
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

    async def _resolve_targets(
        self, sender_id: str | None
    ) -> tuple[list[tuple[str, "_Subscriber", "ActorRef"]], list[str]]:
        """Resolve all eligible subscriber refs.

        Returns (targets, zombie_ids) where targets is a list of
        (sub_id, sub, ref) triples and zombie_ids are subs that hit the
        consecutive-failure threshold during resolution.
        """
        targets: list[tuple[str, "_Subscriber", "ActorRef"]] = []
        zombies: list[str] = []
        for sub_id, sub in list(self._subscribers.items()):
            if sender_id and sub_id == sender_id:
                continue
            ref = await self._resolve(sub)
            if ref:
                targets.append((sub_id, sub, ref))
            else:
                if self._record_failure(sub):
                    zombies.append(sub_id)
        return targets, zombies

    def _fanout_result(
        self,
        delivered: int,
        failed: int,
        failed_ids: list[str] | None,
        always_success: bool = False,
        timed_out: bool = False,
    ) -> dict:
        """Build the standard publish result dict and update running totals."""
        self._total_delivered += delivered
        self._total_failed += failed
        result: dict = {
            "success": True if always_success else (delivered > 0 or failed == 0),
            "delivered": delivered,
            "failed": failed,
            "subscriber_count": len(self._subscribers),
        }
        if failed_ids is not None:
            result["failed_subscribers"] = failed_ids
        if timed_out:
            result["timed_out"] = True
        return result

    async def _fanout_sequential(
        self,
        envelope: dict,
        sender_id: str | None,
        per_sub_timeout: float | None = None,
    ) -> dict:
        """Sequential fanout engine used by fire_and_forget and best_effort.

        When *per_sub_timeout* is None the message is sent fire-and-forget via
        ``ref.tell``; otherwise each subscriber is awaited via ``ref.ask`` with
        the given per-subscriber timeout.
        """
        targets, zombies = await self._resolve_targets(sender_id)
        delivered = failed = 0
        failed_ids: list[str] | None = [] if per_sub_timeout is not None else None

        for sub_id, sub, ref in targets:
            try:
                if per_sub_timeout is not None:
                    await asyncio.wait_for(ref.ask(envelope), timeout=per_sub_timeout)
                else:
                    await ref.tell(envelope)
                delivered += 1
                self._record_success(sub)
            except Exception:
                failed += 1
                if failed_ids is not None:
                    failed_ids.append(sub_id)
                if self._record_failure(sub):
                    zombies.append(sub_id)

        await self._evict_zombies(zombies)
        return self._fanout_result(delivered, failed, failed_ids, always_success=True)

    async def _fanout_concurrent(
        self,
        envelope: dict,
        sender_id: str | None,
        wait_all: bool,
        timeout: float = DEFAULT_FANOUT_TIMEOUT,
    ) -> dict:
        """Concurrent fanout engine used by wait_all_acks and wait_any_ack.

        All subscriber asks are launched as Tasks simultaneously; *wait_all*
        controls whether we wait for every ack or only the first.
        """
        targets, zombies = await self._resolve_targets(sender_id)

        if not targets:
            await self._evict_zombies(zombies)
            return {"success": True, "delivered": 0, "failed": 0, "subscriber_count": 0}

        sub_ids = [sub_id for sub_id, _, _ in targets]
        subs = {sub_id: sub for sub_id, sub, _ in targets}
        # ref.ask() returns a pyo3 Future (not a native coroutine); ensure_future
        # handles both Futures and coroutines, unlike create_task which rejects Futures.
        tasks = [asyncio.ensure_future(ref.ask(envelope)) for _, _, ref in targets]

        delivered = failed = 0
        failed_ids: list[str] = []

        try:
            if wait_all:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout,
                    )
                    for i, result in enumerate(results):
                        sub = subs.get(sub_ids[i])
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
                    logger.warning(
                        f"TopicBroker[{self.topic}] wait_all_acks timeout after {timeout}s"
                    )
                    await self._evict_zombies(zombies)
                    return self._fanout_result(
                        0, len(tasks), sub_ids.copy(), timed_out=True
                    )
            else:
                # asyncio.wait with timeout returns empty done-set on timeout (no exception)
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=timeout,
                )
                if done:
                    for task in done:
                        try:
                            task.result()
                            delivered = 1
                            break
                        except Exception:
                            failed += 1
                    if not delivered:
                        failed_ids = sub_ids[: len(done)]
                else:
                    logger.warning(
                        f"TopicBroker[{self.topic}] wait_any_ack timeout after {timeout}s"
                    )
                    failed = len(tasks)
                    failed_ids = sub_ids.copy()
        finally:
            pending = [t for t in tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        await self._evict_zombies(zombies)
        return self._fanout_result(delivered, failed, failed_ids)
