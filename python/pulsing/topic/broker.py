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

# 订阅者生命周期管理配置
MAX_CONSECUTIVE_FAILURES = 3  # 连续失败次数阈值，超过后清退
REF_TTL_SECONDS = 60.0  # ActorRef 缓存 TTL，过期后重新解析

# 超时配置（方案 2+3：超时 + 幂等）
DEFAULT_FANOUT_TIMEOUT = 30.0  # wait_any_ack / wait_all_acks 的默认超时


@dataclass
class _Subscriber:
    """订阅者信息（内部使用）"""

    subscriber_id: str
    actor_name: str
    node_id: int | None = None
    subscribed_at: float = field(default_factory=time.time)
    _ref: "ActorRef | None" = field(default=None, repr=False)
    _ref_resolved_at: float = 0  # ActorRef 解析时间，用于 TTL 判断
    messages_delivered: int = 0
    messages_failed: int = 0
    consecutive_failures: int = 0  # 连续失败次数，用于清退判断


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
        """解析订阅者 ActorRef（带 TTL 缓存）

        缓存策略：
        - 首次解析后缓存 ActorRef
        - TTL 过期后重新解析，发现节点故障
        - 解析失败时清除缓存，下次重试
        """
        now = time.time()

        # 检查缓存是否有效（存在且未过期）
        if sub._ref is not None and (now - sub._ref_resolved_at) < REF_TTL_SECONDS:
            return sub._ref

        # TTL 过期或无缓存，重新解析
        try:
            sub._ref = await self.system.resolve_named(
                sub.actor_name, node_id=sub.node_id
            )
            sub._ref_resolved_at = now
            return sub._ref
        except Exception as e:
            logger.warning(f"Failed to resolve {sub.subscriber_id}: {e}")
            sub._ref = None  # 清除失效缓存
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
        """记录投递成功，重置连续失败计数"""
        sub.messages_delivered += 1
        sub.consecutive_failures = 0

    def _record_failure(self, sub: _Subscriber) -> bool:
        """记录投递失败，返回是否应该清退

        Returns:
            True 如果连续失败次数超过阈值，应该清退
        """
        sub.messages_failed += 1
        sub.consecutive_failures += 1
        return sub.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    async def _evict_zombies(self, zombie_ids: list[str]) -> None:
        """清退僵尸订阅者"""
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
        """Fire-and-forget: tell 不等响应"""
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

        # 清退僵尸订阅者
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
        """等待 ack 模式

        Args:
            envelope: 消息信封
            sender_id: 发送者 ID（用于排除自己）
            wait_all: True=等待所有响应，False=等待任一响应(wait_any_ack)
            timeout: 超时时间（秒）。超时后本地任务会被取消，
                     远端 handler 可能仍在执行（依赖 HTTP/2 RST_STREAM 传播）。

        注意（取消语义 - 方案 2+3）：
        - 本地取消：通过 asyncio.wait 的 timeout 或 task.cancel() 实现
        - 远端取消：依赖 HTTP/2 RST_STREAM 自动传播（当 body 读取中断时触发）
        - 幂等性：Handler 应该实现幂等操作，确保重复请求不产生副作用
        """
        tasks = []
        sub_ids = []
        resolve_failed: list[str] = []  # resolve 失败的订阅者

        for sub_id, sub in list(self._subscribers.items()):
            if sender_id and sub_id == sender_id:
                continue
            ref = await self._resolve(sub)
            if ref:
                tasks.append(ref.ask(envelope))
                sub_ids.append(sub_id)
            else:
                # resolve 失败也计入失败
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
            # wait_all_acks: 等待所有响应，带整体超时
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
                # 超时：所有任务视为失败
                logger.warning(
                    f"TopicBroker[{self.topic}] wait_all_acks timeout after {timeout}s"
                )
                failed = len(tasks)
                failed_ids = sub_ids.copy()
                # 取消所有 pending tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
        else:
            # wait_any_ack: 等待任一响应，带整体超时
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
                # 取消其他 pending tasks（本地取消，远端依赖 RST_STREAM）
                for task in pending:
                    task.cancel()
            except asyncio.TimeoutError:
                # 超时：没有任何响应
                logger.warning(
                    f"TopicBroker[{self.topic}] wait_any_ack timeout after {timeout}s"
                )
                # 取消所有任务
                for task in tasks:
                    if not task.done():
                        task.cancel()

        # 清退僵尸订阅者
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
        """Best-effort: 尝试发送，记录失败"""
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

        # 清退僵尸订阅者
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
