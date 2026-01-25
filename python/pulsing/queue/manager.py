"""StorageManager - One per node, manages all BucketStorage Actors on this node"""

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any

from pulsing.actor import ActorId, ActorRef, ActorSystem, remote

from .storage import BucketStorage

if TYPE_CHECKING:
    from pulsing.actor.remote import ActorProxy

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

    # Only select nodes in Alive state.
    #
    # Note: `ActorSystem.members()` (Rust binding) currently returns `status`
    # (e.g. "Alive"/"Suspect"/"Dead") rather than `state`, while some older
    # callers used `state`. Support both to stay backward compatible.
    alive_nodes = [n for n in nodes if (n.get("state") or n.get("status")) == "Alive"]
    if not alive_nodes:
        # If no Alive nodes, fallback to all nodes
        alive_nodes = nodes

    best_score = -1
    best_node_id = None

    for node in alive_nodes:
        node_id = node.get("node_id")
        if node_id is None:
            continue
        # node_id is u128 integer, convert to string for consistent hashing
        node_id_str = str(node_id)
        # Combine key and node_id to calculate hash score
        combined = f"{bucket_key}:{node_id_str}"
        score = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        if score > best_score:
            best_score = score
            best_node_id = node_id  # Keep as integer

    return best_node_id


@remote
class StorageManager:
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
                # Create new using BucketStorage.local() for proper @remote wrapping
                proxy = await BucketStorage.local(
                    self.system,
                    bucket_id=bucket_id,
                    storage_path=bucket_storage_path,
                    batch_size=batch_size,
                    backend=backend or self.default_backend,
                    backend_options=backend_options,
                    name=actor_name,
                    public=True,
                )
                self._buckets[key] = proxy.ref
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

                # Use TopicBroker.local() to create properly wrapped actor
                proxy = await TopicBroker.local(
                    self.system, topic_name, self.system, name=actor_name, public=True
                )
                self._topics[topic_name] = proxy.ref
                logger.info(f"Created topic broker: {actor_name}")

            return self._topics[topic_name]

    # ========== Public Remote Methods ==========

    async def get_bucket(
        self,
        topic: str,
        bucket_id: int,
        batch_size: int = 100,
        storage_path: str | None = None,
        backend: str | None = None,
        backend_options: dict | None = None,
    ) -> dict:
        """Get bucket reference.

        Returns:
            - {"_type": "BucketReady", "topic": ..., "bucket_id": ..., "actor_id": ..., "node_id": ...}
            - {"_type": "Redirect", "topic": ..., "bucket_id": ..., "owner_node_id": ..., "owner_addr": ...}
        """
        # Compute owner
        bucket_key = self._bucket_key(topic, bucket_id)
        members = await self._refresh_members()
        owner_node_id = _compute_owner(bucket_key, members)
        local_node_id = str(self.system.node_id.id)

        if owner_node_id is None or owner_node_id == local_node_id:
            # This node is responsible, create/return bucket
            bucket_ref = await self._get_or_create_bucket(
                topic, bucket_id, batch_size, storage_path, backend, backend_options
            )
            return {
                "_type": "BucketReady",
                "topic": topic,
                "bucket_id": bucket_id,
                "actor_id": str(bucket_ref.actor_id.id),
                "node_id": str(local_node_id),
            }
        else:
            # Not owned by this node, return redirect
            owner_addr = None
            for m in members:
                m_node_id = m.get("node_id")
                if m_node_id is not None and m_node_id == owner_node_id:
                    owner_addr = m.get("addr")
                    break

            return {
                "_type": "Redirect",
                "topic": topic,
                "bucket_id": bucket_id,
                "owner_node_id": str(owner_node_id),
                "owner_addr": owner_addr,
            }

    async def get_topic(self, topic: str) -> dict:
        """Get topic broker reference.

        Returns:
            - {"_type": "TopicReady", "topic": ..., "actor_id": ..., "node_id": ...}
            - {"_type": "Redirect", "topic": ..., "owner_node_id": ..., "owner_addr": ...}
        """
        # Compute owner
        topic_key = self._topic_key(topic)
        members = await self._refresh_members()
        owner_node_id = _compute_owner(topic_key, members)
        local_node_id = str(self.system.node_id.id)

        if owner_node_id is None or owner_node_id == local_node_id:
            # This node is responsible, create/return topic broker
            broker_ref = await self._get_or_create_topic_broker(topic)
            return {
                "_type": "TopicReady",
                "topic": topic,
                "actor_id": str(broker_ref.actor_id.id),
                "node_id": str(local_node_id),
            }
        else:
            # Not owned by this node, return redirect
            owner_addr = None
            for m in members:
                m_node_id = m.get("node_id")
                if m_node_id is not None and m_node_id == owner_node_id:
                    owner_addr = m.get("addr")
                    break

            return {
                "_type": "Redirect",
                "topic": topic,
                "owner_node_id": str(owner_node_id),
                "owner_addr": owner_addr,
            }

    async def list_buckets(self) -> list[dict]:
        """List all buckets managed by this node.

        Returns:
            List of {"topic": ..., "bucket_id": ...}
        """
        return [
            {"topic": topic, "bucket_id": bid} for (topic, bid) in self._buckets.keys()
        ]

    async def list_topics(self) -> list[str]:
        """List all topics managed by this node.

        Returns:
            List of topic names
        """
        return list(self._topics.keys())

    async def get_stats(self) -> dict:
        """Get storage manager statistics.

        Returns:
            {"node_id": ..., "bucket_count": ..., "topic_count": ..., "buckets": [...], "topics": [...]}
        """
        return {
            "node_id": str(self.system.node_id.id),
            "bucket_count": len(self._buckets),
            "topic_count": len(self._topics),
            "buckets": [
                {"topic": t, "bucket_id": b} for (t, b) in self._buckets.keys()
            ],
            "topics": list(self._topics.keys()),
        }


# Lock to prevent concurrent creation of StorageManager
_manager_lock = asyncio.Lock()


async def get_storage_manager(system: ActorSystem) -> "ActorProxy":
    """Get StorageManager proxy for this node, create if not exists.

    Returns:
        ActorProxy for direct method calls on StorageManager
    """
    local_node_id = system.node_id.id

    # Try to resolve local node's StorageManager
    try:
        return await StorageManager.resolve(
            STORAGE_MANAGER_NAME, system=system, node_id=local_node_id
        )
    except Exception:
        pass

    async with _manager_lock:
        # Check local node again
        try:
            return await StorageManager.resolve(
                STORAGE_MANAGER_NAME, system=system, node_id=local_node_id
            )
        except Exception:
            pass

        # Create new StorageManager using .local()
        try:
            return await StorageManager.local(
                system, system, name=STORAGE_MANAGER_NAME, public=True
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                return await StorageManager.resolve(
                    STORAGE_MANAGER_NAME, system=system, node_id=local_node_id
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
) -> "ActorProxy":
    """Get ActorProxy for specified bucket

    Automatically handles redirects to ensure getting the bucket on the correct node.
    Returns ActorProxy for direct method calls on BucketStorage.

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
    # Request from local StorageManager first
    manager = await get_storage_manager(system)

    # Convert backend class to name if needed
    backend_name = None
    if backend:
        backend_name = backend if isinstance(backend, str) else backend.__name__

    for redirect_count in range(max_redirects + 1):
        # Call manager.get_bucket() via proxy
        resp_data = await manager.get_bucket(
            topic=topic,
            bucket_id=bucket_id,
            batch_size=batch_size,
            storage_path=storage_path,
            backend=backend_name,
            backend_options=backend_options,
        )

        msg_type = resp_data.get("_type", "")

        if msg_type == "BucketReady":
            # Successfully got bucket - resolve by actor name for typed proxy
            actor_name = f"bucket_{topic}_{bucket_id}"
            # Use BucketStorage.resolve to get typed ActorProxy
            return await BucketStorage.resolve(actor_name, system=system)

        elif msg_type == "Redirect":
            # Need to redirect to other node
            # owner_node_id transmitted as string, convert to int
            owner_node_id_str = resp_data.get("owner_node_id")
            owner_node_id = int(owner_node_id_str)
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
                    manager = await StorageManager.resolve(
                        STORAGE_MANAGER_NAME, system=system, node_id=owner_node_id
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

        else:
            raise RuntimeError(f"Unexpected response: {msg_type}")

    raise RuntimeError(f"Failed to get bucket {topic}:{bucket_id}")


async def get_topic_broker(
    system: ActorSystem,
    topic: str,
    max_redirects: int = 3,
) -> "ActorProxy":
    """Get broker ActorProxy for specified topic

    Automatically handles redirects to ensure getting the broker on the correct node.
    Returns ActorProxy for direct method calls on TopicBroker.

    Args:
        system: Actor system
        topic: Topic name
        max_redirects: Maximum redirect count
    """
    from pulsing.topic.broker import TopicBroker

    manager = await get_storage_manager(system)

    for redirect_count in range(max_redirects + 1):
        # Call manager.get_topic() via proxy
        resp_data = await manager.get_topic(topic=topic)
        msg_type = resp_data.get("_type", "")

        if msg_type == "TopicReady":
            # Successfully got topic - resolve by actor name for typed proxy
            actor_name = f"_topic_broker_{topic}"
            return await TopicBroker.resolve(actor_name, system=system)

        elif msg_type == "Redirect":
            # owner_node_id transmitted as string, convert to int
            owner_node_id = int(resp_data["owner_node_id"])

            logger.debug(f"Redirecting topic {topic} to node {owner_node_id}")

            if redirect_count >= max_redirects:
                raise RuntimeError(f"Too many redirects for topic: {topic}")

            if owner_node_id == system.node_id.id:
                raise RuntimeError(f"Redirect loop for topic: {topic}")

            # Get owner node's StorageManager via proxy
            for retry in range(10):
                try:
                    manager = await StorageManager.resolve(
                        STORAGE_MANAGER_NAME, system=system, node_id=owner_node_id
                    )
                    break
                except Exception as e:
                    if retry < 9:
                        await asyncio.sleep(0.5)
                    else:
                        raise RuntimeError(
                            f"StorageManager not found on node {owner_node_id}: {e}"
                        ) from e

        else:
            raise RuntimeError(f"Unexpected response: {msg_type}")

    raise RuntimeError(f"Failed to get topic broker: {topic}")
