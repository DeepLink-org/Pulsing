"""Load information subscription based on streaming requests

Architecture:
    Router sends SubscribeLoad request → Worker returns StreamMessage
    Worker continuously pushes load updates in stream → Router reads asynchronously

    ┌─────────┐                      ┌─────────┐
    │ Router  │ ─── SubscribeLoad ─► │ Worker  │
    │         │                      │         │
    │         │ ◄─── Stream ──────── │         │
    │         │      {load: 5}       │         │
    │         │ ◄─── Stream ──────── │         │
    │         │      {load: 3}       │         │
    └─────────┘                      └─────────┘

Usage:
    # Router side
    scheduler = StreamLoadScheduler(actor_system, "worker")
    await scheduler.start()
    worker_ref = await scheduler.select_worker()
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from pulsing.actor import ActorRef, Message


@dataclass
class LoadSnapshot:
    """Load snapshot"""

    worker_id: str
    node_id: str
    load: int
    capacity: int
    processed: int
    timestamp: float

    @property
    def load_ratio(self) -> float:
        return self.load / max(1, self.capacity)

    @classmethod
    def from_dict(cls, data: dict) -> "LoadSnapshot":
        return cls(
            worker_id=data.get("worker_id", ""),
            node_id=data.get("node_id", ""),
            load=data.get("load", 0),
            capacity=data.get("capacity", 100),
            processed=data.get("processed", 0),
            timestamp=data.get("timestamp", time.time()),
        )


class LoadStreamConsumer:
    """Router side: Load stream consumer

    Subscribes to load streams from multiple Workers, aggregates and maintains load snapshots.
    """

    def __init__(self, stale_timeout: float = 10.0):
        self._stale_timeout = stale_timeout
        self._loads: dict[str, LoadSnapshot] = {}
        self._subscriptions: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._on_update: Callable[[LoadSnapshot], None] | None = None
        self._on_disconnect: Callable[[str], None] | None = None

    async def subscribe(self, worker_ref: ActorRef, worker_id: str = None):
        """Subscribe to Worker's load stream"""
        wid = worker_id or str(worker_ref.actor_id)
        await self.unsubscribe(wid)

        async def consume():
            try:
                stream_msg = await worker_ref.ask(
                    Message.from_json("SubscribeLoad", {})
                )
                async for chunk in stream_msg:
                    data = chunk.to_json()
                    snapshot = LoadSnapshot.from_dict(data)
                    async with self._lock:
                        self._loads[snapshot.worker_id] = snapshot
                    if self._on_update:
                        try:
                            self._on_update(snapshot)
                        except Exception:
                            pass
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            finally:
                async with self._lock:
                    self._loads.pop(wid, None)
                if self._on_disconnect:
                    try:
                        self._on_disconnect(wid)
                    except Exception:
                        pass

        task = asyncio.create_task(consume())
        self._subscriptions[wid] = task

    def on_disconnect(self, callback: Callable[[str], None]):
        self._on_disconnect = callback

    async def unsubscribe(self, worker_id: str):
        if worker_id in self._subscriptions:
            self._subscriptions[worker_id].cancel()
            try:
                await self._subscriptions[worker_id]
            except asyncio.CancelledError:
                pass
            del self._subscriptions[worker_id]
        async with self._lock:
            self._loads.pop(worker_id, None)

    async def unsubscribe_all(self):
        for wid in list(self._subscriptions.keys()):
            await self.unsubscribe(wid)

    def get_load(self, worker_id: str) -> LoadSnapshot | None:
        snapshot = self._loads.get(worker_id)
        if snapshot and time.time() - snapshot.timestamp <= self._stale_timeout:
            return snapshot
        return None

    def get_all_loads(self) -> dict[str, LoadSnapshot]:
        now = time.time()
        return {
            wid: snap
            for wid, snap in self._loads.items()
            if now - snap.timestamp <= self._stale_timeout
        }

    def get_lowest_load_worker(self) -> str | None:
        valid = self.get_all_loads()
        return min(valid.keys(), key=lambda w: valid[w].load) if valid else None

    def on_update(self, callback: Callable[[LoadSnapshot], None]):
        self._on_update = callback


class StreamLoadScheduler:
    """Load-aware scheduler based on stream subscription

    - Automatically discovers new Workers and subscribes
    - Detects Worker offline and cleans up
    - Selects Worker with lowest load
    """

    def __init__(
        self,
        actor_system,
        worker_name: str = "worker",
        auto_discover: bool = True,
        discover_interval: float = 10.0,
    ):
        self._system = actor_system
        self._worker_name = worker_name
        self._auto_discover = auto_discover
        self._discover_interval = discover_interval

        self._consumer = LoadStreamConsumer()
        self._worker_refs: dict[str, ActorRef] = {}
        self._subscribed_workers: set = set()
        self._running = False
        self._discover_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        self._on_worker_added: Callable[[str], None] | None = None
        self._on_worker_removed: Callable[[str], None] | None = None

    async def start(self):
        """Start scheduler"""
        self._running = True
        self._consumer.on_disconnect(self._handle_worker_disconnect)
        await self._discover_and_subscribe()
        if self._auto_discover:
            self._discover_task = asyncio.create_task(self._auto_discover_loop())

    def _handle_worker_disconnect(self, node_id: str):
        self._subscribed_workers.discard(node_id)
        self._worker_refs.pop(node_id, None)
        if self._on_worker_removed:
            try:
                self._on_worker_removed(node_id)
            except Exception:
                pass

    async def stop(self):
        """Stop scheduler"""
        self._running = False
        if self._discover_task:
            self._discover_task.cancel()
            try:
                await self._discover_task
            except asyncio.CancelledError:
                pass
        await self._consumer.unsubscribe_all()
        self._subscribed_workers.clear()
        self._worker_refs.clear()

    async def _discover_and_subscribe(self):
        """Discover and subscribe to Workers"""
        try:
            # Use get_named_instances instead of unbound lookup_named_actor
            workers = await self._system.get_named_instances(self._worker_name)
            current = {w.get("node_id") for w in workers if w.get("node_id")}

            async with self._lock:
                # New Workers
                for node_id in current - self._subscribed_workers:
                    await self._subscribe_worker(node_id)
                # Offline Workers
                for node_id in self._subscribed_workers - current:
                    await self._unsubscribe_worker(node_id)
        except Exception as e:
            print(f"[StreamLoadScheduler] Discover error: {e}")
            pass

    async def _subscribe_worker(self, node_id: str):
        if node_id in self._subscribed_workers:
            return
        try:
            # Use resolve_named instead of unbound get_actor_ref
            # node_id is string from members(), convert to int for resolve_named
            worker_ref = await self._system.resolve_named(
                self._worker_name, node_id=int(node_id)
            )
            if worker_ref:
                self._worker_refs[node_id] = worker_ref
                await self._consumer.subscribe(worker_ref, node_id)
                self._subscribed_workers.add(node_id)
                if self._on_worker_added:
                    try:
                        self._on_worker_added(node_id)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[StreamLoadScheduler] Subscribe error for node {node_id}: {e}")
            pass

    async def _unsubscribe_worker(self, node_id: str):
        if node_id not in self._subscribed_workers:
            return
        await self._consumer.unsubscribe(node_id)
        self._subscribed_workers.discard(node_id)
        self._worker_refs.pop(node_id, None)
        if self._on_worker_removed:
            try:
                self._on_worker_removed(node_id)
            except Exception:
                pass

    async def _auto_discover_loop(self):
        while self._running:
            await asyncio.sleep(self._discover_interval)
            await self._discover_and_subscribe()

    async def select_worker(
        self,
        request_text: str = None,
        headers: dict[str, str] = None,
    ) -> ActorRef | None:
        """Select Worker with lowest load"""
        worker_id = self._consumer.get_lowest_load_worker()
        if worker_id and worker_id in self._worker_refs:
            return self._worker_refs[worker_id]
        if self._worker_refs:
            import random

            return random.choice(list(self._worker_refs.values()))
        return None

    def get_all_loads(self) -> dict[str, LoadSnapshot]:
        return self._consumer.get_all_loads()

    def get_worker_count(self) -> int:
        return len(self._subscribed_workers)

    def get_subscribed_workers(self) -> set:
        return self._subscribed_workers.copy()

    def on_load_update(self, callback: Callable[[LoadSnapshot], None]):
        self._consumer.on_update(callback)

    def on_worker_added(self, callback: Callable[[str], None]):
        self._on_worker_added = callback

    def on_worker_removed(self, callback: Callable[[str], None]):
        self._on_worker_removed = callback
