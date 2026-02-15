"""Topic - Lightweight Pub/Sub Module

Reuses queue/manager's StorageManager for consistent hashing and redirection,
ensuring only one broker per topic in the cluster.

Usage:
    import pulsing as pul

    await pul.init()

    writer = await pul.topic.write("events")
    await writer.publish({"type": "user_login"})

    reader = await pul.topic.read("events")

    @reader.on_message
    async def handle(msg):
        print(f"Received: {msg}")

    await reader.start()
"""

from typing import TYPE_CHECKING

from pulsing.topic.topic import (
    PublishMode,
    PublishResult,
    TopicReader,
    TopicWriter,
    read_topic,
    subscribe_to_topic,
    write_topic,
)

if TYPE_CHECKING:
    from pulsing._core import ActorSystem


class TopicAPI:
    """Topic API entry point via system.topic or pul.topic

    Example:
        writer = await pul.topic.write("events")
        await writer.publish({"type": "user_login"})

        reader = await pul.topic.read("events")
    """

    def __init__(self, system: "ActorSystem"):
        self._system = system

    async def write(
        self,
        topic: str,
        *,
        writer_id: str | None = None,
    ) -> TopicWriter:
        """Open topic for writing

        Args:
            topic: Topic name
            writer_id: Writer ID (optional)

        Returns:
            TopicWriter for publish operations
        """
        return await write_topic(self._system, topic, writer_id=writer_id)

    async def read(
        self,
        topic: str,
        *,
        reader_id: str | None = None,
        auto_start: bool = False,
    ) -> TopicReader:
        """Open topic for reading

        Args:
            topic: Topic name
            reader_id: Reader ID (optional)
            auto_start: Whether to automatically start receiving

        Returns:
            TopicReader for subscribing to messages
        """
        return await read_topic(
            self._system, topic, reader_id=reader_id, auto_start=auto_start
        )


__all__ = [
    # High-level API
    "TopicAPI",
    # Async API
    "write_topic",
    "read_topic",
    "subscribe_to_topic",
    "TopicWriter",
    "TopicReader",
    "PublishMode",
    "PublishResult",
]
