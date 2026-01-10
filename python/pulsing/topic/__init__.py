"""Topic - 轻量级 Pub/Sub 模块

复用 queue/manager 的 StorageManager 进行一致性哈希和重定向，
确保每个 topic 在集群中只有一个 broker。

Usage:
    from pulsing.topic import write_topic, read_topic

    # 发布消息
    writer = await write_topic(system, "events")
    await writer.publish({"type": "user_login"})

    # 订阅消息
    reader = await read_topic(system, "events")

    @reader.on_message
    async def handle(msg):
        print(f"Received: {msg}")

    await reader.start()
"""

from pulsing.topic.topic import (
    PublishMode,
    PublishResult,
    TopicReader,
    TopicWriter,
    read_topic,
    write_topic,
)

__all__ = [
    "write_topic",
    "read_topic",
    "TopicWriter",
    "TopicReader",
    "PublishMode",
    "PublishResult",
]
