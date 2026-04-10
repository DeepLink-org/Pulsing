"""StorageManager for transfer queue - one per node, routes to StorageUnit via Rendezvous Hashing."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import TYPE_CHECKING

from pulsing.core import ActorId, ActorRef, ActorSystem, remote

from .storage import StorageUnit

if TYPE_CHECKING:
    from pulsing.core.remote import ActorProxy

logger = logging.getLogger(__name__)

STORAGE_MANAGER_NAME = "transfer_queue_storage_manager"


def _compute_owner(bucket_key: str, nodes: list[dict]) -> str | None:
    """Rendezvous Hashing: pick the node with the highest hash score for *bucket_key*."""
    if not nodes:
        return None

    alive_nodes = [n for n in nodes if (n.get("state") or n.get("status")) == "Alive"]
    if not alive_nodes:
        alive_nodes = nodes

    best_score = -1
    best_node_id = None

    for node in alive_nodes:
        node_id = node.get("node_id")
        if node_id is None:
            continue
        node_id_str = str(node_id)
        combined = f"{bucket_key}:{node_id_str}"
        score = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        if score > best_score:
            best_score = score
            best_node_id = node_id_str

    return best_node_id


@remote
class StorageManager:
    """Transfer queue storage manager - one instance per node.

    Routes get_unit requests to the correct node using Rendezvous Hashing.
    Creates StorageUnit actors locally when this node owns the bucket.
    """

    def __init__(self, system: ActorSystem):
        self.system = system
        self._units: dict[tuple[str, int], ActorRef] = {}
        self._unit_locks: dict[tuple[str, int], asyncio.Lock] = {}
        self._locks_meta = asyncio.Lock()
        self._members: list[dict] = []
        self._members_updated_at: float = 0

    def on_start(self, actor_id: ActorId) -> None:
        logger.info(
            f"TransferQueue StorageManager started on node {self.system.node_id}"
        )

    def on_stop(self) -> None:
        logger.info("TransferQueue StorageManager stopping")

    async def _refresh_members(self) -> list[dict]:
        now = time.time()
        if now - self._members_updated_at > 1.0:
            self._members = await self.system.members()
            self._members_updated_at = now
        return self._members

    def _unit_key(self, topic: str, bucket_id: int) -> str:
        return f"tq:{topic}:bucket_{bucket_id}"

    async def _get_or_create_unit(
        self,
        topic: str,
        bucket_id: int,
        batch_size: int,
    ) -> ActorRef:
        key = (topic, bucket_id)
        if key in self._units:
            return self._units[key]

        async with self._locks_meta:
            if key not in self._unit_locks:
                self._unit_locks[key] = asyncio.Lock()
            lock = self._unit_locks[key]

        async with lock:
            if key in self._units:
                return self._units[key]

            actor_name = f"tq_unit_{topic}_{bucket_id}"
            try:
                self._units[key] = await self.system.resolve_named(actor_name)
                logger.debug(f"Resolved existing unit: {actor_name}")
            except Exception:
                proxy = await StorageUnit.spawn(
                    bucket_id=bucket_id,
                    batch_size=batch_size,
                    system=self.system,
                    name=actor_name,
                    public=True,
                )
                self._units[key] = proxy.ref
                logger.info(f"Created transfer queue unit: {actor_name}")
            return self._units[key]

    # ========== Public Remote Methods ==========

    async def get_unit(
        self,
        topic: str,
        bucket_id: int,
        batch_size: int = 10,
    ) -> dict:
        """Get a StorageUnit reference.

        Returns:
            - {"_type": "UnitReady", ...} if this node owns the bucket
            - {"_type": "Redirect", ...} if another node owns it
        """
        bucket_key = self._unit_key(topic, bucket_id)
        members = await self._refresh_members()
        owner_node_id = _compute_owner(bucket_key, members)
        local_node_id = str(self.system.node_id.id)

        if owner_node_id is None or str(owner_node_id) == local_node_id:
            unit_ref = await self._get_or_create_unit(topic, bucket_id, batch_size)
            return {
                "_type": "UnitReady",
                "topic": topic,
                "bucket_id": bucket_id,
                "actor_id": str(unit_ref.actor_id.id),
                "node_id": local_node_id,
            }
        else:
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


# ---------------------------------------------------------------------------
# Module-level helpers (same pattern as pulsing.streaming.manager)
# ---------------------------------------------------------------------------

_manager_lock: asyncio.Lock | None = None
_manager_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_manager_lock() -> asyncio.Lock:
    global _manager_lock, _manager_lock_loop
    loop = asyncio.get_running_loop()
    if _manager_lock is None or _manager_lock_loop is not loop:
        _manager_lock = asyncio.Lock()
        _manager_lock_loop = loop
    return _manager_lock


async def get_storage_manager(system: ActorSystem) -> ActorProxy:
    """Get or create the local TransferQueue StorageManager."""
    local_node_id = system.node_id.id

    try:
        return await StorageManager.resolve(
            STORAGE_MANAGER_NAME, system=system, node_id=local_node_id
        )
    except Exception:
        pass

    async with _get_manager_lock():
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
    """Ensure the local node has a TransferQueue StorageManager."""
    await get_storage_manager(system)
    logger.debug(f"TransferQueue StorageManager ensured on node {system.node_id.id}")


async def get_unit_ref(
    system: ActorSystem,
    topic: str,
    bucket_id: int,
    batch_size: int = 10,
    max_redirects: int = 3,
) -> ActorProxy:
    """Resolve StorageUnit proxy, following redirects across nodes."""
    manager = await get_storage_manager(system)

    for redirect_count in range(max_redirects + 1):
        resp = await manager.get_unit(
            topic=topic,
            bucket_id=bucket_id,
            batch_size=batch_size,
        )

        msg_type = resp.get("_type", "")

        if msg_type == "UnitReady":
            actor_name = f"tq_unit_{topic}_{bucket_id}"
            return await StorageUnit.resolve(actor_name, system=system)

        elif msg_type == "Redirect":
            owner_node_id_str = str(resp.get("owner_node_id"))

            logger.debug(
                f"Redirecting transfer queue unit {topic}:{bucket_id} to node {owner_node_id_str}"
            )

            if redirect_count >= max_redirects:
                raise RuntimeError(
                    f"Too many redirects for transfer queue unit {topic}:{bucket_id}"
                )

            if str(owner_node_id_str) == str(system.node_id.id):
                raise RuntimeError(
                    f"Redirect loop for transfer queue unit {topic}:{bucket_id}"
                )

            owner_node_id_int = int(owner_node_id_str)
            max_resolve_retries = 10
            for resolve_retry in range(max_resolve_retries):
                try:
                    manager = await StorageManager.resolve(
                        STORAGE_MANAGER_NAME, system=system, node_id=owner_node_id_int
                    )
                    break
                except Exception as e:
                    if resolve_retry < max_resolve_retries - 1:
                        logger.debug(
                            f"TransferQueue StorageManager not found on node {owner_node_id_str}, "
                            f"retry {resolve_retry + 1}/{max_resolve_retries}"
                        )
                        await asyncio.sleep(0.5)
                    else:
                        raise RuntimeError(
                            f"TransferQueue StorageManager not found on node {owner_node_id_str} "
                            f"after {max_resolve_retries} retries: {e}"
                        ) from e
        else:
            raise RuntimeError(f"Unexpected response: {msg_type}")

    raise RuntimeError(f"Failed to get transfer queue unit {topic}:{bucket_id}")
