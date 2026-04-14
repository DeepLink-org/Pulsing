"""StorageManager for transfer_queue bucket placement and lookup."""

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
    """Pick the node with the highest rendezvous hash score for *bucket_key*."""
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

        combined = f"{bucket_key}:{node_id}"
        score = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        if score > best_score:
            best_score = score
            best_node_id = str(node_id)

    return best_node_id


def _unit_key(topic: str, bucket_id: int) -> tuple[str, int]:
    return (topic, bucket_id)


def _bucket_key(topic: str, bucket_id: int) -> str:
    return f"tq:{topic}:bucket_{bucket_id}"


def _actor_name(topic: str, bucket_id: int) -> str:
    return f"tq_unit_{topic}_{bucket_id}"


@remote
class StorageManager:
    """One per node; routes transfer queue buckets via rendezvous hashing."""

    def __init__(self, system: ActorSystem):
        self.system = system
        self._units: dict[tuple[str, int], ActorRef] = {}
        self._unit_capacities: dict[tuple[str, int], int] = {}
        self._unit_locks: dict[tuple[str, int], asyncio.Lock] = {}
        self._locks_meta = asyncio.Lock()
        self._members: list[dict] = []
        self._members_updated_at = 0.0

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

    def _validate_capacity(
        self,
        topic: str,
        bucket_id: int,
        requested_capacity: int,
        actual_capacity: int,
    ) -> None:
        if requested_capacity != actual_capacity:
            raise ValueError(
                "transfer_queue bucket already exists with a different "
                f"bucket_capacity for topic={topic!r}, bucket_id={bucket_id}: "
                f"requested {requested_capacity}, existing {actual_capacity}"
            )

    async def _resolve_existing_unit(
        self,
        topic: str,
        bucket_id: int,
        bucket_capacity: int,
    ) -> ActorRef:
        actor_name = _actor_name(topic, bucket_id)
        proxy = await StorageUnit.resolve(actor_name, system=self.system)
        config = await proxy.get_config()
        actual_capacity = int(config["bucket_capacity"])
        self._validate_capacity(topic, bucket_id, bucket_capacity, actual_capacity)
        key = _unit_key(topic, bucket_id)
        self._unit_capacities[key] = actual_capacity
        return proxy.ref

    async def _get_or_create_unit(
        self,
        topic: str,
        bucket_id: int,
        bucket_capacity: int,
    ) -> ActorRef:
        key = _unit_key(topic, bucket_id)
        cached = self._units.get(key)
        if cached is not None:
            self._validate_capacity(
                topic,
                bucket_id,
                bucket_capacity,
                self._unit_capacities[key],
            )
            return cached

        async with self._locks_meta:
            if key not in self._unit_locks:
                self._unit_locks[key] = asyncio.Lock()
            lock = self._unit_locks[key]

        async with lock:
            cached = self._units.get(key)
            if cached is not None:
                self._validate_capacity(
                    topic,
                    bucket_id,
                    bucket_capacity,
                    self._unit_capacities[key],
                )
                return cached

            actor_name = _actor_name(topic, bucket_id)
            try:
                ref = await self._resolve_existing_unit(
                    topic=topic,
                    bucket_id=bucket_id,
                    bucket_capacity=bucket_capacity,
                )
                logger.debug(f"Resolved existing unit: {actor_name}")
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                proxy = await StorageUnit.spawn(
                    bucket_id=bucket_id,
                    bucket_capacity=bucket_capacity,
                    system=self.system,
                    name=actor_name,
                    public=True,
                )
                ref = proxy.ref
                logger.info(f"Created transfer queue unit: {actor_name}")

            self._units[key] = ref
            self._unit_capacities[key] = bucket_capacity
            return ref

    async def get_unit(
        self,
        topic: str,
        bucket_id: int,
        bucket_capacity: int = 10,
    ) -> dict:
        """Return local ready info or a redirect for the requested bucket."""
        bucket_key = _bucket_key(topic, bucket_id)
        members = await self._refresh_members()
        owner_node_id = _compute_owner(bucket_key, members)
        local_node_id = str(self.system.node_id.id)

        if owner_node_id is None or owner_node_id == local_node_id:
            unit_ref = await self._get_or_create_unit(topic, bucket_id, bucket_capacity)
            return {
                "_type": "UnitReady",
                "topic": topic,
                "bucket_id": bucket_id,
                "actor_id": str(unit_ref.actor_id.id),
                "node_id": local_node_id,
            }

        owner_addr = None
        for member in members:
            member_node_id = member.get("node_id")
            if member_node_id is not None and str(member_node_id) == owner_node_id:
                owner_addr = member.get("addr")
                break

        return {
            "_type": "Redirect",
            "topic": topic,
            "bucket_id": bucket_id,
            "owner_node_id": owner_node_id,
            "owner_addr": owner_addr,
        }


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
    """Get or create the local transfer_queue StorageManager."""
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
        except Exception as exc:
            if "already exists" in str(exc).lower():
                return await StorageManager.resolve(
                    STORAGE_MANAGER_NAME, system=system, node_id=local_node_id
                )
            raise


async def ensure_storage_managers(system: ActorSystem) -> None:
    """Ensure the local node has a transfer_queue StorageManager."""
    await get_storage_manager(system)
    logger.debug(f"TransferQueue StorageManager ensured on node {system.node_id.id}")


async def _resolve_unit_proxy(
    system: ActorSystem,
    topic: str,
    bucket_id: int,
    owner_node_id: str,
    retries: int = 10,
) -> ActorProxy:
    """Resolve a named StorageUnit on a specific node, retrying for visibility."""
    actor_name = _actor_name(topic, bucket_id)
    owner_node_id_int = int(owner_node_id)
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            return await StorageUnit.resolve(
                actor_name,
                system=system,
                node_id=owner_node_id_int,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                await asyncio.sleep(0.2)

    raise RuntimeError(
        f"TransferQueue unit {topic}:{bucket_id} not found on node "
        f"{owner_node_id} after {retries} retries: {last_exc}"
    ) from last_exc


async def get_unit_ref(
    system: ActorSystem,
    topic: str,
    bucket_id: int,
    bucket_capacity: int = 10,
    max_redirects: int = 3,
) -> ActorProxy:
    """Resolve the StorageUnit proxy for one bucket, following redirects."""
    manager = await get_storage_manager(system)

    for redirect_count in range(max_redirects + 1):
        resp = await manager.get_unit(
            topic=topic,
            bucket_id=bucket_id,
            bucket_capacity=bucket_capacity,
        )
        msg_type = resp.get("_type", "")

        if msg_type == "UnitReady":
            owner_node_id = str(resp.get("node_id"))
            return await _resolve_unit_proxy(
                system=system,
                topic=topic,
                bucket_id=bucket_id,
                owner_node_id=owner_node_id,
            )

        if msg_type != "Redirect":
            raise RuntimeError(f"Unexpected response: {msg_type}")

        owner_node_id_str = str(resp.get("owner_node_id"))
        logger.debug(
            f"Redirecting transfer queue unit {topic}:{bucket_id} to node {owner_node_id_str}"
        )

        if redirect_count >= max_redirects:
            raise RuntimeError(
                f"Too many redirects for transfer queue unit {topic}:{bucket_id}"
            )

        if owner_node_id_str == str(system.node_id.id):
            raise RuntimeError(
                f"Redirect loop for transfer queue unit {topic}:{bucket_id}"
            )

        owner_node_id_int = int(owner_node_id_str)
        max_resolve_retries = 10
        for resolve_retry in range(max_resolve_retries):
            try:
                manager = await StorageManager.resolve(
                    STORAGE_MANAGER_NAME,
                    system=system,
                    node_id=owner_node_id_int,
                )
                break
            except Exception as exc:
                if resolve_retry < max_resolve_retries - 1:
                    logger.debug(
                        "TransferQueue StorageManager not found on node %s, retry %s/%s",
                        owner_node_id_str,
                        resolve_retry + 1,
                        max_resolve_retries,
                    )
                    await asyncio.sleep(0.5)
                else:
                    raise RuntimeError(
                        "TransferQueue StorageManager not found on node "
                        f"{owner_node_id_str} after {max_resolve_retries} retries: {exc}"
                    ) from exc

    raise RuntimeError(f"Failed to get transfer queue unit {topic}:{bucket_id}")
