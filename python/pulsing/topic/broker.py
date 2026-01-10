"""TopicBroker - 轻量级 Pub/Sub Broker Actor（内部实现）

TopicBroker 的生命周期由 queue/manager.py 的 StorageManager 管理，
使用一致性哈希保证每个 topic 在集群中只有一个 broker。
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


@dataclass
class _Subscriber:
    """订阅者信息（内部使用）"""

    subscriber_id: str
    actor_name: str
    node_id: int | None = None
    subscribed_at: float = field(default_factory=time.time)
    _ref: "ActorRef | None" = field(default=None, repr=False)
    messages_delivered: int = 0
    messages_failed: int = 0


class TopicBroker(Actor):
    """Topic Broker Actor（内部实现）

    每个 topic 对应一个 broker，负责：
    1. 管理订阅者列表
    2. 接收发布请求并分发给订阅者
    3. 根据发布模式决定是否等待 ack
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
            return Message.from_json("Error", {"error": "Missing subscriber_id or actor_name"})

        async with self._lock:
            if subscriber_id in self._subscribers:
                return Message.from_json("SubscribeResult", {"success": True, "already": True})

            self._subscribers[subscriber_id] = _Subscriber(
                subscriber_id=subscriber_id,
                actor_name=actor_name,
                node_id=node_id,
            )
            logger.debug(f"TopicBroker[{self.topic}] +subscriber: {subscriber_id}")
            return Message.from_json("SubscribeResult", {"success": True, "topic": self.topic})

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
        """解析订阅者 ActorRef（带缓存）"""
        if sub._ref is not None:
            return sub._ref
        try:
            sub._ref = await self.system.resolve_named(sub.actor_name, node_id=sub.node_id)
            return sub._ref
        except Exception as e:
            logger.warning(f"Failed to resolve {sub.subscriber_id}: {e}")
            return None

    async def _publish(self, data: dict) -> Message:
        payload = data.get("payload")
        mode = data.get("mode", "fire_and_forget")
        sender_id = data.get("sender_id")

        self._total_published += 1

        if not self._subscribers:
            return Message.from_json("PublishResult", {
                "success": True, "delivered": 0, "failed": 0, "subscriber_count": 0
            })

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

    async def _fanout_tell(self, envelope: dict, sender_id: str | None) -> Message:
        """Fire-and-forget: tell 不等响应"""
        sent = 0
        failed = 0

        for sub_id, sub in self._subscribers.items():
            if sender_id and sub_id == sender_id:
                continue
            try:
                ref = await self._resolve(sub)
                if ref:
                    await ref.tell(envelope)
                    sent += 1
                    sub.messages_delivered += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
                sub.messages_failed += 1

        self._total_delivered += sent
        self._total_failed += failed

        return Message.from_json("PublishResult", {
            "success": True, "delivered": sent, "failed": failed,
            "subscriber_count": len(self._subscribers)
        })

    async def _fanout_ask(self, envelope: dict, sender_id: str | None, wait_all: bool) -> Message:
        """等待 ack 模式"""
        tasks = []
        sub_ids = []

        for sub_id, sub in self._subscribers.items():
            if sender_id and sub_id == sender_id:
                continue
            ref = await self._resolve(sub)
            if ref:
                tasks.append(ref.ask(envelope))
                sub_ids.append(sub_id)

        if not tasks:
            return Message.from_json("PublishResult", {
                "success": True, "delivered": 0, "failed": 0, "subscriber_count": 0
            })

        delivered = 0
        failed = 0
        failed_ids = []

        if wait_all:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                sub = self._subscribers.get(sub_ids[i])
                if isinstance(result, Exception):
                    failed += 1
                    failed_ids.append(sub_ids[i])
                    if sub:
                        sub.messages_failed += 1
                else:
                    delivered += 1
                    if sub:
                        sub.messages_delivered += 1
        else:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not task.exception():
                    delivered = 1
                    break
            for task in pending:
                task.cancel()

        self._total_delivered += delivered
        self._total_failed += failed

        return Message.from_json("PublishResult", {
            "success": delivered > 0 or failed == 0,
            "delivered": delivered, "failed": failed,
            "failed_subscribers": failed_ids,
            "subscriber_count": len(self._subscribers)
        })

    async def _fanout_best_effort(self, envelope: dict, sender_id: str | None) -> Message:
        """Best-effort: 尝试发送，记录失败"""
        delivered = 0
        failed = 0
        failed_ids = []

        for sub_id, sub in self._subscribers.items():
            if sender_id and sub_id == sender_id:
                continue
            try:
                ref = await self._resolve(sub)
                if ref:
                    await asyncio.wait_for(ref.ask(envelope), timeout=5.0)
                    delivered += 1
                    sub.messages_delivered += 1
                else:
                    failed += 1
                    failed_ids.append(sub_id)
            except Exception:
                failed += 1
                failed_ids.append(sub_id)
                sub.messages_failed += 1

        self._total_delivered += delivered
        self._total_failed += failed

        return Message.from_json("PublishResult", {
            "success": True, "delivered": delivered, "failed": failed,
            "failed_subscribers": failed_ids,
            "subscriber_count": len(self._subscribers)
        })

    def _stats(self) -> Message:
        return Message.from_json("TopicStats", {
            "topic": self.topic,
            "subscriber_count": len(self._subscribers),
            "total_published": self._total_published,
            "total_delivered": self._total_delivered,
            "total_failed": self._total_failed,
        })
