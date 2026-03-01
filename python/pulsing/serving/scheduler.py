"""Worker scheduler - Load balancing strategies

Supports the following scheduling strategies (all Rust-implemented):
- RustRandomScheduler: Random selection
- RustRoundRobinScheduler: Round-robin selection
- RustPowerOfTwoScheduler: Power-of-Two Choices
- RustConsistentHashScheduler: Consistent hashing (session affinity)
- RustCacheAwareScheduler: Cache-aware routing (Radix Tree prefix matching)
- LeastConnectionScheduler: Least connections (Python)

For load-aware scheduling, use StreamLoadScheduler from load_stream module.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import pulsing

# Import Rust policies if available
try:
    from pulsing._core import (
        CacheAwareConfig,
        CacheAwarePolicy,
        ConsistentHashPolicy,
        PowerOfTwoPolicy,
        RandomPolicy,
        RoundRobinPolicy,
        WorkerInfo,
    )

    RUST_POLICIES_AVAILABLE = True
except ImportError:
    RUST_POLICIES_AVAILABLE = False
    # Placeholder types when Rust not available
    WorkerInfo = Any
    RandomPolicy = None
    RoundRobinPolicy = None
    PowerOfTwoPolicy = None
    ConsistentHashPolicy = None
    CacheAwarePolicy = None
    CacheAwareConfig = None


class Scheduler(ABC):
    """Scheduler base class

    All scheduler implementations must inherit from this class.
    Provides default lifecycle (start/stop) and health query methods.
    """

    def __init__(self, actor_system, worker_name: str = "worker"):
        self._system = actor_system
        self._worker_name = worker_name
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the scheduler. Override for schedulers that need background tasks."""

    async def stop(self):
        """Stop the scheduler. Override for schedulers that need cleanup."""

    async def get_available_workers(self):
        try:
            return await self._system.get_named_instances(self._worker_name)
        except Exception:
            return []

    async def get_worker_count(self) -> int:
        return len(await self.get_available_workers())

    async def get_healthy_worker_count(self) -> int:
        workers = await self.get_available_workers()
        return sum(1 for w in workers if w.get("status") == "Alive")

    async def _resolve_worker(self, node_id: int | None = None):
        try:
            return await self._system.resolve_named(self._worker_name, node_id=node_id)
        except Exception:
            return None

    @abstractmethod
    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        """Select a worker, returns ActorRef or None

        Args:
            request_text: Request text (for cache-aware and consistent hash routing)
            headers: HTTP request headers (for consistent hash routing)
        """
        pass


# ============================================================================
# Python-implemented schedulers
# ============================================================================


class LeastConnectionScheduler(Scheduler):
    """Least connections scheduler (Python implementation)"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._request_counts = {}

    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        workers = await self.get_available_workers()
        if not workers:
            return None

        async with self._lock:
            selected_worker = min(
                workers, key=lambda w: self._request_counts.get(w.get("node_id"), 0)
            )
            node_id = selected_worker.get("node_id")
            self._request_counts[node_id] = self._request_counts.get(node_id, 0) + 1

        return await pulsing.refer(selected_worker.get("actor_id"))


# ============================================================================
# Rust-implemented schedulers (high performance)
# ============================================================================


class RustSchedulerBase(Scheduler):
    """Rust scheduler base class — provides shared select_worker implementation."""

    def __init__(self, actor_system, worker_name: str = "worker"):
        if not RUST_POLICIES_AVAILABLE:
            raise ImportError("Rust policies not available. Rebuild with maturin.")
        super().__init__(actor_system, worker_name)
        self._worker_info_cache: dict[str, WorkerInfo] = {}
        self._policy = None  # subclasses set this

    def _get_worker_info(self, worker_data: dict) -> WorkerInfo:
        node_id = worker_data.get("node_id", "")
        if node_id not in self._worker_info_cache:
            url = worker_data.get("addr", f"http://{node_id}")
            model_id = worker_data.get("model_id", "default")
            self._worker_info_cache[node_id] = WorkerInfo(url, model_id)
        worker_info = self._worker_info_cache[node_id]
        worker_info.is_healthy = worker_data.get("status") == "Alive"
        return worker_info

    def _workers_to_info_list(self, workers: list) -> list:
        return [self._get_worker_info(w) for w in workers]

    def _pre_select(self, worker_infos: list) -> None:
        """Hook for subclasses that need setup before selection (e.g. CacheAware)."""

    def _do_select(self, worker_infos, request_text, headers):
        """Invoke the Rust policy. Override for policies that need extra args (e.g. headers)."""
        return self._policy.select_worker(worker_infos, request_text)

    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        workers = await self.get_available_workers()
        if not workers:
            return None

        worker_infos = self._workers_to_info_list(workers)
        self._pre_select(worker_infos)
        selected_idx = self._do_select(worker_infos, request_text, headers)

        if selected_idx is None:
            return None
        return await pulsing.refer(workers[selected_idx].get("actor_id"))


class RustRandomScheduler(RustSchedulerBase):
    """Random scheduler (Rust implementation, high performance)"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._policy = RandomPolicy()


class RustRoundRobinScheduler(RustSchedulerBase):
    """Round-robin scheduler (Rust implementation)"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._policy = RoundRobinPolicy()

    def reset(self):
        self._policy.reset()


class RustPowerOfTwoScheduler(RustSchedulerBase):
    """Power-of-Two Choices scheduler (Rust implementation)"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._policy = PowerOfTwoPolicy()

    def update_loads(self, loads: dict[str, int]):
        self._policy.update_loads(loads)


class RustConsistentHashScheduler(RustSchedulerBase):
    """Consistent hash scheduler (Rust implementation)

    Routes based on session ID or user ID, ensuring requests from the same user
    are always routed to the same worker.
    """

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._policy = ConsistentHashPolicy()

    def _do_select(self, worker_infos, request_text, headers):
        return self._policy.select_worker(worker_infos, request_text, headers)

    def reset(self):
        self._policy.reset()


class RustCacheAwareScheduler(RustSchedulerBase):
    """Cache-aware scheduler (Rust implementation)

    Combines cache affinity and load balancing. Suitable for LLM inference
    scenarios to improve KV Cache hit rate.
    """

    def __init__(
        self,
        actor_system,
        worker_name: str = "worker",
        cache_threshold: float = 0.5,
        balance_abs_threshold: int = 32,
        balance_rel_threshold: float = 1.0001,
        eviction_interval_secs: int = 60,
        max_tree_size: int = 100000,
    ):
        super().__init__(actor_system, worker_name)
        config = CacheAwareConfig(
            cache_threshold=cache_threshold,
            balance_abs_threshold=balance_abs_threshold,
            balance_rel_threshold=balance_rel_threshold,
            eviction_interval_secs=eviction_interval_secs,
            max_tree_size=max_tree_size,
        )
        self._policy = CacheAwarePolicy(config)

    def _pre_select(self, worker_infos: list) -> None:
        self._policy.init_workers(worker_infos)

    def add_worker(self, url: str, model_id: str = "default"):
        self._policy.add_worker(url, model_id)

    def remove_worker(self, url: str):
        self._policy.remove_worker(url)

    def evict_cache(self, max_size: int):
        self._policy.evict_cache(max_size)


# ============================================================================
# Factory function
# ============================================================================


def get_scheduler(
    policy_name: str, actor_system, worker_name: str = "worker", **kwargs
) -> Scheduler:
    """Get scheduler instance

    Args:
        policy_name: Policy name, supports:
            - "random": Random
            - "round_robin": Round robin
            - "power_of_two": Power-of-Two Choices
            - "consistent_hash": Consistent hash
            - "cache_aware": Cache-aware
            - "least_connection": Least connections (Python)
        actor_system: Actor system instance
        worker_name: Worker actor name
        **kwargs: Policy-specific parameters (e.g., cache_threshold, etc.)

    Returns:
        Scheduler instance

    Note:
        For load-aware scheduling, recommend using StreamLoadScheduler from load_stream module

    Examples:
        # Use cache-aware scheduler
        scheduler = get_scheduler("cache_aware", actor_system, "worker",
                                  cache_threshold=0.5)

        # Use round-robin scheduler
        scheduler = get_scheduler("round_robin", actor_system, "worker")
    """
    policy_map = {
        "random": RustRandomScheduler,
        "round_robin": RustRoundRobinScheduler,
        "power_of_two": RustPowerOfTwoScheduler,
        "consistent_hash": RustConsistentHashScheduler,
        "cache_aware": RustCacheAwareScheduler,
        "least_connection": LeastConnectionScheduler,
    }

    scheduler_class = policy_map.get(policy_name)
    if scheduler_class is None:
        raise ValueError(
            f"Unknown policy: {policy_name}. Available: {list(policy_map.keys())}"
        )

    # Policies that support kwargs
    if policy_name == "cache_aware":
        return scheduler_class(actor_system, worker_name, **kwargs)
    else:
        return scheduler_class(actor_system, worker_name)


# Exports
__all__ = [
    "Scheduler",
    "LeastConnectionScheduler",
    "RustRandomScheduler",
    "RustRoundRobinScheduler",
    "RustPowerOfTwoScheduler",
    "RustConsistentHashScheduler",
    "RustCacheAwareScheduler",
    "get_scheduler",
    "RUST_POLICIES_AVAILABLE",
]
