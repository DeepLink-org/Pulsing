"""StorageManager - One per node, manages all BucketStorage Actors on this node"""

import asyncio
import hashlib
import logging
from typing import Any

from pulsing.actor import Actor, ActorId, ActorRef, ActorSystem, Message

from .storage import BucketStorage

logger = logging.getLogger(__name__)

# StorageManager fixed service name
STORAGE_MANAGER_NAME = "queue_storage_manager"


def _compute_owner(bucket_key: str, nodes: list[dict]) -> int | None:
    """Compute owner node ID based on bucket key

    Uses Rendezvous Hashing (highest random weight hashing) to ensure:
    1. Same bucket is always handled by the same node
    2. Node changes only affect ~1/N keys (minimal migration)
    3. Natural uniform load distribution

    Algorithm: Calculate score for each (key, node) combination, select node with highest score
    """
    if not nodes:
        return None

    # Only select nodes in Alive state
    alive_nodes = [n for n in nodes if n.get("state") == "Alive"]
    if not alive_nodes:
        # If no Alive nodes, fallback to all nodes
        alive_nodes = nodes

    best_score = -1
    best_node_id = None

    for node in alive_nodes:
        node_id = node.get("node_id")
        if node_id is None:
            continue
        node_id = int(node_id)
        # Combine key and node_id to calculate hash score
        combined = f"{bucket_key}:{node_id}"
        score = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        if score > best_score:
            best_score = score
            best_node_id = node_id

    return best_node_id


class StorageManager(Actor):
    """Storage manager Actor

    One instance per node, responsible for:
    1. Receiving GetBucket/GetTopic requests
    2. Determining if resource belongs to this node (consistent hashing)
    3. If belongs to this node: create/return corresponding Actor
    4. If not belongs to this node: return Redirect response pointing to correct node

    Supported resource types:
    - Queue Bucket: GetBucket -> BucketStorage Actor
    - Topic Broker: GetTopic -> TopicBroker Actor
    """

    def __init__(
        self,
        system: ActorSystem,
        base_storage_path: str = "./queue_storage",
        default_backend: str | type = "memory",
    ):
        self.system = system
        self.base_storage_path = base_storage_path
        self.default_backend = default_backend

        # Buckets managed by this node: {(topic, bucket_id): ActorRef}
        self._buckets: dict[tuple[str, int], ActorRef] = {}
        # Topic brokers managed by this node: {topic_name: ActorRef}
        self._topics: dict[str, ActorRef] = {}
        self._lock = asyncio.Lock()

        # Cached cluster member information
        self._members: list[dict] = []
        self._members_updated_at: float = 0

    def on_start(self, actor_id: ActorId) -> None:
        logger.info(f"StorageManager started on node {self.system.node_id}")

    def on_stop(self) -> None:
        logger.info("StorageManager stopping")

    async def _refresh_members(self) -> list[dict]:
        """Refresh cluster member list (with cache)"""
        import time

        now = time.time()
        if now - self._members_updated_at > 1.0:  # 1 second cache
            self._members = await self.system.members()
            self._members_updated_at = now
        return self._members

    def _bucket_key(self, topic: str, bucket_id: int) -> str:
        """Generate unique key for bucket"""
        return f"{topic}:bucket_{bucket_id}"

    def _topic_key(self, topic_name: str) -> str:
        """Generate unique key for topic"""
        return f"topic:{topic_name}"

    async def _get_or_create_bucket(
        self,
        topic: str,
        bucket_id: int,
        batch_size: int,
        storage_path: str | None = None,
        backend: str | type | None = None,
        backend_options: dict | None = None,
    ) -> ActorRef:
        """Get or create local BucketStorage Actor"""
        key = (topic, bucket_id)

        if key in self._buckets:
            return self._buckets[key]

        async with self._lock:
            if key in self._buckets:
                return self._buckets[key]

            # Create BucketStorage Actor
            actor_name = f"bucket_{topic}_{bucket_id}"
            # Use provided storage_path or default path
            if storage_path:
                bucket_storage_path = f"{storage_path}/bucket_{bucket_id}"
            else:
                bucket_storage_path = (
                    f"{self.base_storage_path}/{topic}/bucket_{bucket_id}"
                )

            try:
                # Try to resolve existing
                self._buckets[key] = await self.system.resolve_named(actor_name)
                logger.debug(f"Resolved existing bucket: {actor_name}")
            except Exception:
                # Create new, use specified backend or default backend
                storage = BucketStorage(
                    bucket_id=bucket_id,
                    storage_path=bucket_storage_path,
                    batch_size=batch_size,
                    backend=backend or self.default_backend,
                    backend_options=backend_options,
                )
                self._buckets[key] = await self.system.spawn(
                    actor_name, storage, public=True
                )
                logger.info(f"Created bucket: {actor_name} at {bucket_storage_path}")

            return self._buckets[key]

    async def _get_or_create_topic_broker(self, topic_name: str) -> ActorRef:
        """Get or create local TopicBroker Actor"""
        if topic_name in self._topics:
            return self._topics[topic_name]

        async with self._lock:
            if topic_name in self._topics:
                return self._topics[topic_name]

            actor_name = f"_topic_broker_{topic_name}"
            try:
                self._topics[topic_name] = await self.system.resolve_named(actor_name)
                logger.debug(f"Resolved existing topic broker: {actor_name}")
            except Exception:
                # Lazy import to avoid circular dependency
                from pulsing.topic.broker import TopicBroker

                broker = TopicBroker(topic_name, self.system)
                self._topics[topic_name] = await self.system.spawn(
                    actor_name, broker, public=True
                )
                logger.info(f"Created topic broker: {actor_name}")

            return self._topics[topic_name]

    async def receive(self, msg: Message) -> Message | None:
        try:
            return await self._handle_message(msg)
        except Exception as e:
            logger.exception(f"Error handling message: {e}")
            return Message.from_json("Error", {"error": str(e)})

    async def _handle_message(self, msg: Message) -> Message | None:
        msg_type = msg.msg_type
        data = msg.to_json()

        if msg_type == "GetBucket":
            # Request bucket reference
            topic = data.get("topic")
            bucket_id = data.get("bucket_id")
            batch_size = data.get("batch_size", 100)
            storage_path = data.get("storage_path")  # Optional custom storage path
            backend = data.get("backend")  # Optional backend name
            backend_options = data.get("backend_options")  # Optional backend options

            if topic is None or bucket_id is None:
                return Message.from_json(
                    "Error", {"error": "Missing 'topic' or 'bucket_id'"}
                )

            # Compute owner
            bucket_key = self._bucket_key(topic, bucket_id)
            members = await self._refresh_members()
            owner_node_id = _compute_owner(bucket_key, members)
            local_node_id = self.system.node_id.id

            # Determine if belongs to this node
            if owner_node_id is None or owner_node_id == local_node_id:
                # This node is responsible, create/return bucket
                bucket_ref = await self._get_or_create_bucket(
                    topic, bucket_id, batch_size, storage_path, backend, backend_options
                )
                return Message.from_json(
                    "BucketReady",
                    {
                        "_type": "BucketReady",  # Fallback: msg_type may be lost across nodes
                        "topic": topic,
                        "bucket_id": bucket_id,
                        "actor_id": bucket_ref.actor_id.local_id,
                        # Use hex string to transmit node_id, avoid JSON big integer precision loss
                        "node_id_hex": hex(local_node_id),
                    },
                )
            else:
                # Not owned by this node, return redirect
                # Find owner node address
                owner_addr = None
                for m in members:
                    # node_id might be string, convert to int for comparison
                    m_node_id = m.get("node_id")
                    if m_node_id is not None and int(m_node_id) == owner_node_id:
                        owner_addr = m.get("addr")
                        break

                return Message.from_json(
                    "Redirect",
                    {
                        "_type": "Redirect",  # Fallback: msg_type may be lost across nodes
                        "topic": topic,
                        "bucket_id": bucket_id,
                        # Use hex string to transmit node_id, avoid JSON big integer precision loss
                        "owner_node_id_hex": hex(owner_node_id),
                        "owner_addr": owner_addr,
                    },
                )

        elif msg_type == "GetTopic":
            # Request topic broker reference
            topic_name = data.get("topic")
            if not topic_name:
                return Message.from_json("Error", {"error": "Missing 'topic'"})

            # Compute owner
            topic_key = self._topic_key(topic_name)
            members = await self._refresh_members()
            owner_node_id = _compute_owner(topic_key, members)
            local_node_id = self.system.node_id.id

            if owner_node_id is None or owner_node_id == local_node_id:
                # This node is responsible, create/return topic broker
                broker_ref = await self._get_or_create_topic_broker(topic_name)
                return Message.from_json(
                    "TopicReady",
                    {
                        "_type": "TopicReady",
                        "topic": topic_name,
                        "actor_id": broker_ref.actor_id.local_id,
                        "node_id_hex": hex(local_node_id),
                    },
                )
            else:
                # Not owned by this node, return redirect
                owner_addr = None
                for m in members:
                    m_node_id = m.get("node_id")
                    if m_node_id is not None and int(m_node_id) == owner_node_id:
                        owner_addr = m.get("addr")
                        break

                return Message.from_json(
                    "Redirect",
                    {
                        "_type": "Redirect",
                        "topic": topic_name,
                        "owner_node_id_hex": hex(owner_node_id),
                        "owner_addr": owner_addr,
                    },
                )

        elif msg_type == "ListBuckets":
            # List all buckets managed by this node
            buckets = [
                {"topic": topic, "bucket_id": bid}
                for (topic, bid) in self._buckets.keys()
            ]
            return Message.from_json("BucketList", {"buckets": buckets})

        elif msg_type == "ListTopics":
            # List all topics managed by this node
            return Message.from_json("TopicList", {"topics": list(self._topics.keys())})

        elif msg_type == "GetStats":
            # Get statistics
            return Message.from_json(
                "Stats",
                {
                    "node_id": self.system.node_id.id,
                    "bucket_count": len(self._buckets),
                    "topic_count": len(self._topics),
                    "buckets": [
                        {"topic": t, "bucket_id": b} for (t, b) in self._buckets.keys()
                    ],
                    "topics": list(self._topics.keys()),
                },
            )

        else:
            return Message.from_json(
                "Error", {"error": f"Unknown message type: {msg_type}"}
            )


# Lock to prevent concurrent creation of StorageManager
_manager_lock = asyncio.Lock()


async def get_storage_manager(system: ActorSystem) -> ActorRef:
    """Get StorageManager for this node, create if not exists"""
    local_node_id = system.node_id.id

    # Try to resolve local node's StorageManager
    try:
        return await system.resolve_named(STORAGE_MANAGER_NAME, node_id=local_node_id)
    except Exception:
        pass

    async with _manager_lock:
        # Check local node again
        try:
            return await system.resolve_named(
                STORAGE_MANAGER_NAME, node_id=local_node_id
            )
        except Exception:
            pass

        # Create new StorageManager
        try:
            manager = StorageManager(system)
            return await system.spawn(STORAGE_MANAGER_NAME, manager, public=True)
        except Exception as e:
            if "already exists" in str(e).lower():
                return await system.resolve_named(
                    STORAGE_MANAGER_NAME, node_id=local_node_id
                )
            raise


async def ensure_storage_managers(system: ActorSystem) -> None:
    """Ensure local node has StorageManager

    Each node needs its own StorageManager to handle bucket requests.
    This function ensures the local node has created StorageManager.
    Remote nodes' StorageManagers will be automatically created when they call write_queue or read_queue.
    """
    await get_storage_manager(system)
    logger.debug(f"Local StorageManager ensured on node {system.node_id.id}")


async def get_bucket_ref(
    system: ActorSystem,
    topic: str,
    bucket_id: int,
    batch_size: int = 100,
    storage_path: str | None = None,
    backend: str | type | None = None,
    backend_options: dict | None = None,
    max_redirects: int = 3,
) -> ActorRef:
    """Get ActorRef for specified bucket

    Automatically handles redirects to ensure getting the bucket on the correct node.

    Args:
        system: Actor system
        topic: Queue topic
        bucket_id: Bucket ID
        batch_size: Batch size
        storage_path: Custom storage path (optional)
        backend: Storage backend name or class (optional)
        backend_options: Additional backend options (optional)
        max_redirects: Maximum redirect count
    """
    from pulsing.actor import ActorId, NodeId

    # Request from local StorageManager first
    manager = await get_storage_manager(system)

    for redirect_count in range(max_redirects + 1):
        msg_data = {
            "topic": topic,
            "bucket_id": bucket_id,
            "batch_size": batch_size,
        }
        if storage_path:
            msg_data["storage_path"] = storage_path
        if backend:
            # If it's a class, pass class name (classes cannot be serialized across nodes)
            msg_data["backend"] = (
                backend if isinstance(backend, str) else backend.__name__
            )
        if backend_options:
            msg_data["backend_options"] = backend_options

        response = await manager.ask(Message.from_json("GetBucket", msg_data))

        resp_data = response.to_json()
        # msg_type may be lost across nodes, use _type field as fallback
        msg_type = response.msg_type or resp_data.get("_type", "")

        if msg_type == "BucketReady":
            # Successfully got bucket
            actor_id = resp_data["actor_id"]
            # node_id transmitted as hex string, convert to int
            node_id = int(resp_data["node_id_hex"], 16)

            bucket_actor_id = ActorId(actor_id, NodeId(node_id))
            return await system.actor_ref(bucket_actor_id)

        elif msg_type == "Redirect":
            # Need to redirect to other node
            # owner_node_id transmitted as hex string, convert to int
            hex_str = resp_data.get("owner_node_id_hex")
            owner_node_id = int(hex_str, 16)
            owner_addr = resp_data.get("owner_addr")

            logger.debug(
                f"Redirecting bucket {topic}:{bucket_id} to node {owner_node_id} @ {owner_addr}"
            )

            if redirect_count >= max_redirects:
                raise RuntimeError(f"Too many redirects for bucket {topic}:{bucket_id}")

            # Check if redirecting to self (avoid infinite loop)
            if owner_node_id == system.node_id.id:
                raise RuntimeError(
                    f"Redirect loop detected for bucket {topic}:{bucket_id}"
                )

            # Get owner node's StorageManager (with retry, wait for remote node initialization)
            max_resolve_retries = 10
            for resolve_retry in range(max_resolve_retries):
                try:
                    manager = await system.resolve_named(
                        STORAGE_MANAGER_NAME, node_id=owner_node_id
                    )
                    break
                except Exception as e:
                    if resolve_retry < max_resolve_retries - 1:
                        logger.debug(
                            f"StorageManager not found on node {owner_node_id}, "
                            f"retry {resolve_retry + 1}/{max_resolve_retries}"
                        )
                        await asyncio.sleep(0.5)
                    else:
                        raise RuntimeError(
                            f"StorageManager not found on node {owner_node_id} after "
                            f"{max_resolve_retries} retries: {e}"
                        ) from e

        elif msg_type == "Error":
            raise RuntimeError(f"GetBucket failed: {resp_data.get('error')}")

        else:
            raise RuntimeError(f"Unexpected response: {msg_type}")

    raise RuntimeError(f"Failed to get bucket {topic}:{bucket_id}")


async def get_topic_broker(
    system: ActorSystem,
    topic: str,
    max_redirects: int = 3,
) -> ActorRef:
    """Get broker ActorRef for specified topic

    Automatically handles redirects to ensure getting the broker on the correct node.

    Args:
        system: Actor system
        topic: Topic name
        max_redirects: Maximum redirect count
    """
    from pulsing.actor import ActorId, NodeId

    manager = await get_storage_manager(system)

    for redirect_count in range(max_redirects + 1):
        response = await manager.ask(Message.from_json("GetTopic", {"topic": topic}))

        resp_data = response.to_json()
        msg_type = response.msg_type or resp_data.get("_type", "")

        if msg_type == "TopicReady":
            actor_id = resp_data["actor_id"]
            node_id = int(resp_data["node_id_hex"], 16)
            broker_actor_id = ActorId(actor_id, NodeId(node_id))
            return await system.actor_ref(broker_actor_id)

        elif msg_type == "Redirect":
            owner_node_id = int(resp_data["owner_node_id_hex"], 16)

            logger.debug(f"Redirecting topic {topic} to node {owner_node_id}")

            if redirect_count >= max_redirects:
                raise RuntimeError(f"Too many redirects for topic: {topic}")

            if owner_node_id == system.node_id.id:
                raise RuntimeError(f"Redirect loop for topic: {topic}")

            # Get owner node's StorageManager
            for retry in range(10):
                try:
                    manager = await system.resolve_named(
                        STORAGE_MANAGER_NAME, node_id=owner_node_id
                    )
                    break
                except Exception as e:
                    if retry < 9:
                        await asyncio.sleep(0.5)
                    else:
                        raise RuntimeError(
                            f"StorageManager not found on node {owner_node_id}: {e}"
                        ) from e

        elif msg_type == "Error":
            raise RuntimeError(f"GetTopic failed: {resp_data.get('error')}")

        else:
            raise RuntimeError(f"Unexpected response: {msg_type}")

    raise RuntimeError(f"Failed to get topic broker: {topic}")
