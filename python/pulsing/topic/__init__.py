"""Topic - Lightweight Pub/Sub Module

Reuses queue/manager's StorageManager for consistent hashing and redirection,
ensuring only one broker per topic in the cluster.

Usage:
    from pulsing.topic import write_topic, read_topic

    # Publish message
    writer = await write_topic(system, "events")
    await writer.publish({"type": "user_login"})

    # Subscribe to messages
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
    subscribe_to_topic,
    write_topic,
)

__all__ = [
    "write_topic",
    "read_topic",
    "subscribe_to_topic",
    "TopicWriter",
    "TopicReader",
    "PublishMode",
    "PublishResult",
]
