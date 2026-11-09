"""Process-local delivery edges. Wait outside serial mailbox actors, never in them."""

from __future__ import annotations

import asyncio
import threading


class DeliverySignals:
    """Small thread-safe cursors, not a second queue or a message store."""

    def __init__(self, max_waiters: int = 1024):
        self._lock = threading.Lock()
        self._versions: dict[str, tuple[str, int]] = {}
        self._waiters: dict[
            str, set[tuple[asyncio.AbstractEventLoop, asyncio.Future]]
        ] = {}
        self._scheduled: set[asyncio.Future] = set()
        self._count = 0
        self._maximum = max_waiters

    @staticmethod
    def _resolve(future):
        if not future.done():
            future.set_result(None)

    def _wake(self, identity):
        # Called under the lock. Never execute recipient code on the actor thread.
        for loop, future in self._waiters.get(identity, ()):
            if future.done() or future in self._scheduled:
                continue
            try:
                self._scheduled.add(future)
                loop.call_soon_threadsafe(self._resolve, future)
            except RuntimeError:
                self._scheduled.discard(future)
                # The owning event loop already closed; its finally owns removal.
                continue

    def start(self, identity: str, epoch: str):
        with self._lock:
            self._versions[identity] = (epoch, 0)
            self._wake(identity)

    def publish(self, identity: str, epoch: str, revision: int):
        with self._lock:
            if self._versions.get(identity, (None,))[0] == epoch:
                self._versions[identity] = (epoch, revision)
                self._wake(identity)

    def stop(self, identity: str, epoch: str):
        with self._lock:
            if self._versions.get(identity, (None,))[0] == epoch:
                del self._versions[identity]
                self._wake(identity)

    async def wait(self, identity: str, epoch: str, revision: int, timeout: float):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        entry = (loop, future)
        with self._lock:
            # Snapshot and subscription are atomic relative to publish: a
            # delivery between the actor snapshot and this call is not lost.
            if self._versions.get(identity) != (epoch, revision):
                return
            waiters = self._waiters.get(identity, set())
            if self._count >= self._maximum or len(waiters) >= 32:
                raise ValueError("too many mailbox delivery waiters")
            self._waiters[identity] = waiters
            waiters.add(entry)
            self._count += 1
        try:
            await asyncio.wait_for(future, timeout)
        except TimeoutError:
            # A heartbeat is not an empty-inbox model turn.
            pass
        finally:
            with self._lock:
                waiters = self._waiters[identity]
                waiters.remove(entry)
                self._scheduled.discard(future)
                self._count -= 1
                if not waiters:
                    del self._waiters[identity]


delivery_signals = DeliverySignals()
