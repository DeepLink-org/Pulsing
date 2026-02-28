"""StorageManager - One per node, manages all BucketStorage Actors on this node"""

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any

from pulsing.core import ActorId, ActorRef, ActorSystem, remote

from .storage import BucketStorage

if TYPE_CHECKING:
    from pulsing.core.remote import ActorProxy

logger = logging.getLogger(__name__)

# StorageManager fixed service name
STORAGE_MANAGER_NAME = "queue_storage_manager"


def _compute_owner(bucket_key: str, nodes: list[dict]) -> str | None:
    """Compute owner node ID based on bucket key

    Uses Rendezvous Hashing (highest random weight hashing) to ensure:
    1. Same bucket is always handled by the same node
    2. Node changes only affect ~1/N keys (minimal migration)
    3. Natural uniform load distribution

    Algorithm: Calculate score for each (key, node) combination, select node with highest score

    Returns:
        Node ID as string (unified representation for consistent comparison)
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
        # Normalize node_id to string for consistent hashing and comparison
        node_id_str = str(node_id)
        # Combine key and node_id to calculate hash score
        combined = f"{bucket_key}:{node_id_str}"
        score = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        if score > best_score:
            best_score = score
            best_node_id = node_id_str  # Keep as string for unified comparison

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
        # Per-resource locks so different buckets/topics can be created in parallel
        self._bucket_locks: dict[tuple[str, int], asyncio.Lock] = {}
        self._topic_locks: dict[str, asyncio.Lock] = {}
        self._locks_meta = asyncio.Lock()

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
        """Get or create local BucketStorage Actor. Per-key lock allows parallel creation."""
        key = (topic, bucket_id)
        if key in self._buckets:
            return self._buckets[key]

        async with self._locks_meta:
            if key not in self._bucket_locks:
                self._bucket_locks[key] = asyncio.Lock()
            lock = self._bucket_locks[key]

        async with lock:
            if key in self._buckets:
                return self._buckets[key]
            actor_name = f"bucket_{topic}_{bucket_id}"
            if storage_path:
                bucket_storage_path = f"{storage_path}/bucket_{bucket_id}"
            else:
                bucket_storage_path = (
                    f"{self.base_storage_path}/{topic}/bucket_{bucket_id}"
                )
            try:
                self._buckets[key] = await self.system.resolve_named(actor_name)
                logger.debug(f"Resolved existing bucket: {actor_name}")
            except Exception:
                proxy = await BucketStorage.spawn(
                    bucket_id=bucket_id,
                    storage_path=bucket_storage_path,
                    batch_size=batch_size,
                    backend=backend or self.default_backend,
                    backend_options=backend_options,
                    system=self.system,
                    name=actor_name,
                    public=True,
                )
                self._buckets[key] = proxy.ref
                logger.info(f"Created bucket: {actor_name} at {bucket_storage_path}")
            return self._buckets[key]

    async def _get_or_create_topic_broker(self, topic_name: str) -> ActorRef:
        """Get or create local TopicBroker Actor. Per-topic lock allows parallel creation."""
        if topic_name in self._topics:
            return self._topics[topic_name]

        async with self._locks_meta:
            if topic_name not in self._topic_locks:
                self._topic_locks[topic_name] = asyncio.Lock()
            lock = self._topic_locks[topic_name]

        async with lock:
            if topic_name in self._topics:
                return self._topics[topic_name]
            actor_name = f"_topic_broker_{topic_name}"
            try:
                self._topics[topic_name] = await self.system.resolve_named(actor_name)
                logger.debug(f"Resolved existing topic broker: {actor_name}")
            except Exception:
                from pulsing.streaming.broker import TopicBroker

                proxy = await TopicBroker.spawn(
                    topic_name,
                    self.system,
                    system=self.system,
                    name=actor_name,
                    public=True,
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

        if owner_node_id is None or str(owner_node_id) == local_node_id:
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
                if m_node_id is not None and str(m_node_id) == str(owner_node_id):
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

        if owner_node_id is None or str(owner_node_id) == local_node_id:
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
                if m_node_id is not None and str(m_node_id) == str(owner_node_id):
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


# Per-event-loop lock to prevent concurrent creation of StorageManager.
# Lazy init so the lock is bound to the current loop (avoids "bound to a different event loop" in tests).
_manager_lock: asyncio.Lock | None = None
_manager_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_manager_lock() -> asyncio.Lock:
    global _manager_lock, _manager_lock_loop
    loop = asyncio.get_running_loop()
    if _manager_lock is None or _manager_lock_loop is not loop:
        _manager_lock = asyncio.Lock()
        _manager_lock_loop = loop
    return _manager_lock


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

    async with _get_manager_lock():
        # Check local node again
        try:
            return await StorageManager.resolve(
                STORAGE_MANAGER_NAME, system=system, node_id=local_node_id
            )
        except Exception:
            pass

        try:
            return await StorageManager.spawn(
                system, system=system, name=STORAGE_MANAGER_NAME, public=True
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


async def _get_remote_manager(
    system: ActorSystem,
    owner_node_id_str: str,
    retries: int = 10,
) -> "ActorProxy":
    """Resolve StorageManager on a remote node, retrying until it appears."""
    owner_node_id_int = int(owner_node_id_str)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return await StorageManager.resolve(
                STORAGE_MANAGER_NAME, system=system, node_id=owner_node_id_int
            )
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                logger.debug(
                    f"StorageManager not on node {owner_node_id_str}, "
                    f"retry {attempt + 1}/{retries}"
                )
                await asyncio.sleep(0.5)
    raise RuntimeError(
        f"StorageManager not found on node {owner_node_id_str} after {retries} retries: {last_exc}"
    ) from last_exc


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
    """Get ActorProxy for the specified bucket, following redirects automatically."""
    manager = await get_storage_manager(system)
    backend_name = (
        (backend if isinstance(backend, str) else backend.__name__) if backend else None
    )

    for redirect_count in range(max_redirects + 1):
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
            return await BucketStorage.resolve(
                f"bucket_{topic}_{bucket_id}", system=system
            )

        if msg_type == "Redirect":
            owner_node_id_str = str(resp_data.get("owner_node_id"))
            if redirect_count >= max_redirects:
                raise RuntimeError(f"Too many redirects for bucket {topic}:{bucket_id}")
            if owner_node_id_str == str(system.node_id.id):
                raise RuntimeError(
                    f"Redirect loop detected for bucket {topic}:{bucket_id}"
                )
            logger.debug(
                f"Redirecting bucket {topic}:{bucket_id} to node {owner_node_id_str}"
            )
            manager = await _get_remote_manager(system, owner_node_id_str)
            continue

        raise RuntimeError(f"Unexpected response type: {msg_type}")

    raise RuntimeError(f"Failed to get bucket {topic}:{bucket_id}")


async def get_topic_broker(
    system: ActorSystem,
    topic: str,
    max_redirects: int = 3,
) -> "ActorProxy":
    """Get broker ActorProxy for the specified topic, following redirects automatically."""
    from pulsing.streaming.broker import TopicBroker

    manager = await get_storage_manager(system)

    for redirect_count in range(max_redirects + 1):
        resp_data = await manager.get_topic(topic=topic)
        msg_type = resp_data.get("_type", "")

        if msg_type == "TopicReady":
            return await TopicBroker.resolve(f"_topic_broker_{topic}", system=system)

        if msg_type == "Redirect":
            owner_node_id_str = str(resp_data["owner_node_id"])
            if redirect_count >= max_redirects:
                raise RuntimeError(f"Too many redirects for topic: {topic}")
            if owner_node_id_str == str(system.node_id.id):
                raise RuntimeError(f"Redirect loop for topic: {topic}")
            logger.debug(f"Redirecting topic {topic} to node {owner_node_id_str}")
            manager = await _get_remote_manager(system, owner_node_id_str)
            continue

        raise RuntimeError(f"Unexpected response type: {msg_type}")

    raise RuntimeError(f"Failed to get topic broker: {topic}")
