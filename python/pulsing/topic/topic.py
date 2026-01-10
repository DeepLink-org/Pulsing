"""Topic API - Pub/Sub 高级接口"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from pulsing.actor import ActorRef

from pulsing.actor import Actor, ActorId, ActorSystem, Message

logger = logging.getLogger(__name__)


class PublishMode(Enum):
    """发布模式"""

    FIRE_AND_FORGET = "fire_and_forget"  # 发送后立即返回
    WAIT_ALL_ACKS = "wait_all_acks"  # 等待所有订阅者响应
    WAIT_ANY_ACK = "wait_any_ack"  # 等待任一订阅者响应
    BEST_EFFORT = "best_effort"  # 尝试发送，记录失败


@dataclass
class PublishResult:
    """发布结果"""

    success: bool
    delivered: int
    failed: int
    subscriber_count: int
    failed_subscribers: list[str] | None = None


# 回调类型
MessageCallback = Callable[[Any], Coroutine[Any, Any, Any] | Any]


async def _get_broker(system: ActorSystem, topic: str) -> "ActorRef":
    """获取 topic broker（复用 queue/manager 的基础设施）"""
    from pulsing.queue.manager import get_topic_broker

    return await get_topic_broker(system, topic)


class TopicWriter:
    """Topic 写入句柄"""

    def __init__(self, system: ActorSystem, topic: str, writer_id: str | None = None):
        self._system = system
        self._topic = topic
        self._writer_id = writer_id or f"writer_{uuid.uuid4().hex[:8]}"
        self._broker: "ActorRef | None" = None

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def writer_id(self) -> str:
        return self._writer_id

    async def _broker_ref(self) -> "ActorRef":
        if self._broker is None:
            self._broker = await _get_broker(self._system, self._topic)
        return self._broker

    async def publish(
        self,
        message: Any,
        mode: PublishMode = PublishMode.FIRE_AND_FORGET,
    ) -> PublishResult:
        """发布消息

        Args:
            message: 要发布的消息（任意 Python 对象）
            mode: 发布模式

        Returns:
            PublishResult: 发布结果
        """
        broker = await self._broker_ref()
        response = await broker.ask(
            Message.from_json(
                "Publish",
                {
                    "payload": message,
                    "mode": mode.value,
                    "sender_id": self._writer_id,
                },
            )
        )

        if response.msg_type == "Error":
            raise RuntimeError(response.to_json().get("error"))

        data = response.to_json()
        return PublishResult(
            success=data.get("success", False),
            delivered=data.get("delivered", 0),
            failed=data.get("failed", 0),
            subscriber_count=data.get("subscriber_count", 0),
            failed_subscribers=data.get("failed_subscribers"),
        )

    async def stats(self) -> dict[str, Any]:
        """获取 topic 统计信息"""
        broker = await self._broker_ref()
        response = await broker.ask(Message.from_json("GetStats", {}))
        return response.to_json()


class _SubscriberActor(Actor):
    """订阅者 Actor（内部使用）"""

    def __init__(self, callbacks: list[MessageCallback]):
        self._callbacks = callbacks

    def on_start(self, actor_id: ActorId) -> None:
        pass

    def on_stop(self) -> None:
        pass

    async def receive(self, msg: Any) -> Any:
        # 提取 payload
        if isinstance(msg, dict):
            payload = msg.get("payload", msg)
        elif isinstance(msg, Message):
            payload = msg.to_json().get("payload", msg.to_json())
        else:
            payload = msg

        # 调用回调
        for callback in self._callbacks:
            try:
                result = callback(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.exception(f"Callback error: {e}")

        return {"ack": True}


class TopicReader:
    """Topic 读取句柄

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
        """注册消息回调（装饰器风格）

        Example:
            @reader.on_message
            async def handle(msg):
                print(f"Received: {msg}")
        """
        self._callbacks.append(callback)
        return callback

    def add_callback(self, callback: MessageCallback) -> None:
        """添加消息回调"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: MessageCallback) -> bool:
        """移除消息回调"""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    async def start(self) -> None:
        """开始接收消息"""
        if self._started:
            return

        if not self._callbacks:
            logger.warning(f"TopicReader[{self._reader_id}] has no callbacks")

        # 创建订阅者 Actor
        actor_name = f"_topic_sub_{self._topic}_{self._reader_id}"
        subscriber = _SubscriberActor(self._callbacks)
        self._subscriber_ref = await self._system.spawn(
            actor_name, subscriber, public=True
        )

        # 向 broker 注册
        broker = await _get_broker(self._system, self._topic)
        response = await broker.ask(
            Message.from_json(
                "Subscribe",
                {
                    "subscriber_id": self._reader_id,
                    "actor_name": actor_name,
                    "node_id": self._system.node_id.id,
                },
            )
        )

        if response.msg_type == "Error":
            raise RuntimeError(f"Subscribe failed: {response.to_json().get('error')}")

        self._started = True
        logger.debug(f"TopicReader[{self._reader_id}] started for topic: {self._topic}")

    async def stop(self) -> None:
        """停止接收消息"""
        if not self._started:
            return

        # 从 broker 取消订阅
        try:
            broker = await _get_broker(self._system, self._topic)
            await broker.ask(
                Message.from_json("Unsubscribe", {"subscriber_id": self._reader_id})
            )
        except Exception as e:
            logger.warning(f"Unsubscribe error: {e}")

        # 停止订阅者 Actor
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
        """获取 topic 统计信息"""
        broker = await _get_broker(self._system, self._topic)
        response = await broker.ask(Message.from_json("GetStats", {}))
        return response.to_json()


async def write_topic(
    system: ActorSystem,
    topic: str,
    writer_id: str | None = None,
) -> TopicWriter:
    """打开 topic 用于写入

    Args:
        system: Actor 系统
        topic: Topic 名称
        writer_id: 写入者 ID（可选）

    Returns:
        TopicWriter: 写入句柄

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
    """打开 topic 用于读取

    Args:
        system: Actor 系统
        topic: Topic 名称
        reader_id: 读取者 ID（可选）
        auto_start: 是否自动开始接收

    Returns:
        TopicReader: 读取句柄

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
