"""Worker scheduler - Load balancing strategies

Supports the following scheduling strategies:
- RandomScheduler: Random selection (Python implementation)
- RoundRobinScheduler: Round-robin selection (Python implementation)
- LeastConnectionScheduler: Least connections (Python implementation)
- RustRandomScheduler: Random selection (Rust implementation, high performance)
- RustRoundRobinScheduler: Round-robin selection (Rust implementation)
- RustPowerOfTwoScheduler: Power-of-Two Choices (Rust implementation)
- RustConsistentHashScheduler: Consistent hashing (Rust implementation, supports session affinity)
- RustCacheAwareScheduler: Cache-aware routing (Rust implementation, supports Radix Tree prefix matching)

For load-aware scheduling, recommend using StreamLoadScheduler from load_stream module
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

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
    """Scheduler base class"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        self._system = actor_system
        self._worker_name = worker_name
        self._lock = asyncio.Lock()

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
            # node_id is now u128 integer from members()
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


class RoundRobinScheduler(Scheduler):
    """Round-robin scheduler (Python implementation)"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._index = 0

    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        workers = await self.get_available_workers()
        if not workers:
            return None

        async with self._lock:
            self._index = (self._index + 1) % len(workers)
            selected_worker = workers[self._index]

        return await self._resolve_worker(node_id=selected_worker.get("node_id"))


class RandomScheduler(Scheduler):
    """Random scheduler (Python implementation)"""

    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        import random

        workers = await self.get_available_workers()
        if not workers:
            return None

        selected_worker = random.choice(workers)
        return await self._resolve_worker(node_id=selected_worker.get("node_id"))


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

        return await self._resolve_worker(node_id=node_id)


# ============================================================================
# Rust-implemented schedulers (high performance)
# ============================================================================


class RustSchedulerBase(Scheduler):
    """Rust scheduler base class"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._worker_info_cache: dict[str, WorkerInfo] = {}

    def _get_worker_info(self, worker_data: dict) -> WorkerInfo:
        """Get or create WorkerInfo object"""
        node_id = worker_data.get("node_id", "")

        if node_id not in self._worker_info_cache:
            url = worker_data.get("addr", f"http://{node_id}")
            model_id = worker_data.get("model_id", "default")
            self._worker_info_cache[node_id] = WorkerInfo(url, model_id)

        worker_info = self._worker_info_cache[node_id]

        # Update health status
        is_healthy = worker_data.get("status") == "Alive"
        worker_info.is_healthy = is_healthy

        return worker_info

    def _workers_to_info_list(self, workers: list) -> list:
        """Convert worker data to WorkerInfo list"""
        return [self._get_worker_info(w) for w in workers]

    @abstractmethod
    def _get_policy(self):
        """Get Rust policy object"""
        pass


class RustRandomScheduler(RustSchedulerBase):
    """Random scheduler (Rust implementation, high performance)"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        if not RUST_POLICIES_AVAILABLE:
            raise ImportError("Rust policies not available. Rebuild with maturin.")
        super().__init__(actor_system, worker_name)
        self._policy = RandomPolicy()

    def _get_policy(self):
        return self._policy

    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        workers = await self.get_available_workers()
        if not workers:
            return None

        worker_infos = self._workers_to_info_list(workers)
        selected_idx = self._policy.select_worker(worker_infos, request_text)

        if selected_idx is None:
            return None

        selected_worker = workers[selected_idx]
        return await self._resolve_worker(node_id=selected_worker.get("node_id"))


class RustRoundRobinScheduler(RustSchedulerBase):
    """Round-robin scheduler (Rust implementation)"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        if not RUST_POLICIES_AVAILABLE:
            raise ImportError("Rust policies not available. Rebuild with maturin.")
        super().__init__(actor_system, worker_name)
        self._policy = RoundRobinPolicy()

    def _get_policy(self):
        return self._policy

    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        workers = await self.get_available_workers()
        if not workers:
            return None

        worker_infos = self._workers_to_info_list(workers)
        selected_idx = self._policy.select_worker(worker_infos, request_text)

        if selected_idx is None:
            return None

        selected_worker = workers[selected_idx]
        return await self._resolve_worker(node_id=selected_worker.get("node_id"))

    def reset(self):
        """Reset round-robin counter"""
        self._policy.reset()


class RustPowerOfTwoScheduler(RustSchedulerBase):
    """Power-of-Two Choices scheduler (Rust implementation)

    Randomly selects two workers, then chooses the one with lower load.
    Provides near-optimal load balancing in large-scale clusters.
    """

    def __init__(self, actor_system, worker_name: str = "worker"):
        if not RUST_POLICIES_AVAILABLE:
            raise ImportError("Rust policies not available. Rebuild with maturin.")
        super().__init__(actor_system, worker_name)
        self._policy = PowerOfTwoPolicy()

    def _get_policy(self):
        return self._policy

    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        workers = await self.get_available_workers()
        if not workers:
            return None

        worker_infos = self._workers_to_info_list(workers)
        selected_idx = self._policy.select_worker(worker_infos, request_text)

        if selected_idx is None:
            return None

        selected_worker = workers[selected_idx]
        return await self._resolve_worker(node_id=selected_worker.get("node_id"))

    def update_loads(self, loads: dict[str, int]):
        """Update cached load information

        Args:
            loads: Mapping from worker URL to load value
        """
        self._policy.update_loads(loads)


class RustConsistentHashScheduler(RustSchedulerBase):
    """Consistent hash scheduler (Rust implementation)

    Routes based on session ID or user ID, ensuring requests from the same user are always routed to the same worker.
    Supports extracting routing key from HTTP headers or request body.

    HTTP header priority (checked in order):
    - x-session-id
    - x-user-id
    - x-tenant-id
    - x-request-id
    - x-correlation-id
    - x-trace-id

    Request body field priority:
    - session_params.session_id
    - user
    - session_id
    - user_id
    """

    def __init__(self, actor_system, worker_name: str = "worker"):
        if not RUST_POLICIES_AVAILABLE:
            raise ImportError("Rust policies not available. Rebuild with maturin.")
        super().__init__(actor_system, worker_name)
        self._policy = ConsistentHashPolicy()

    def _get_policy(self):
        return self._policy

    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        workers = await self.get_available_workers()
        if not workers:
            return None

        worker_infos = self._workers_to_info_list(workers)
        selected_idx = self._policy.select_worker(worker_infos, request_text, headers)

        if selected_idx is None:
            return None

        selected_worker = workers[selected_idx]
        return await self._resolve_worker(node_id=selected_worker.get("node_id"))

    def reset(self):
        """Reset hash ring"""
        self._policy.reset()


class RustCacheAwareScheduler(RustSchedulerBase):
    """Cache-aware scheduler (Rust implementation)

    Combines cache affinity and load balancing strategies:
    1. When system load is balanced, use cache-aware routing (based on Radix Tree prefix matching)
    2. When system load is unbalanced, switch to shortest queue routing

    Particularly suitable for LLM inference scenarios, can improve KV Cache hit rate.

    Args:
        cache_threshold: Prefix match threshold (0.0-1.0), use cache affinity routing when exceeded
        balance_abs_threshold: Absolute threshold for load imbalance
        balance_rel_threshold: Relative threshold for load imbalance
        eviction_interval_secs: Cache eviction interval (seconds)
        max_tree_size: Maximum number of Radix Tree nodes
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
        if not RUST_POLICIES_AVAILABLE:
            raise ImportError("Rust policies not available. Rebuild with maturin.")
        super().__init__(actor_system, worker_name)

        config = CacheAwareConfig(
            cache_threshold=cache_threshold,
            balance_abs_threshold=balance_abs_threshold,
            balance_rel_threshold=balance_rel_threshold,
            eviction_interval_secs=eviction_interval_secs,
            max_tree_size=max_tree_size,
        )
        self._policy = CacheAwarePolicy(config)

    def _get_policy(self):
        return self._policy

    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        workers = await self.get_available_workers()
        if not workers:
            return None

        worker_infos = self._workers_to_info_list(workers)

        # Initialize workers (if first call)
        self._policy.init_workers(worker_infos)

        selected_idx = self._policy.select_worker(worker_infos, request_text)

        if selected_idx is None:
            return None

        selected_worker = workers[selected_idx]
        return await self._resolve_worker(node_id=selected_worker.get("node_id"))

    def add_worker(self, url: str, model_id: str = "default"):
        """Add worker to cache tree"""
        self._policy.add_worker(url, model_id)

    def remove_worker(self, url: str):
        """Remove worker from cache tree"""
        self._policy.remove_worker(url)

    def evict_cache(self, max_size: int):
        """Manually trigger cache eviction"""
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
            - "random": Random (Rust implementation)
            - "round_robin": Round robin (Rust implementation)
            - "power_of_two": Power-of-Two Choices (Rust implementation)
            - "consistent_hash": Consistent hash (Rust implementation)
            - "cache_aware": Cache-aware (Rust implementation)
            - "py_random": Random (Python implementation)
            - "py_round_robin": Round robin (Python implementation)
            - "least_connection": Least connections (Python implementation)
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
        # Rust implementation (recommended)
        "random": RustRandomScheduler if RUST_POLICIES_AVAILABLE else RandomScheduler,
        "round_robin": (
            RustRoundRobinScheduler if RUST_POLICIES_AVAILABLE else RoundRobinScheduler
        ),
        "power_of_two": RustPowerOfTwoScheduler,
        "consistent_hash": RustConsistentHashScheduler,
        "cache_aware": RustCacheAwareScheduler,
        # Python implementation
        "py_random": RandomScheduler,
        "py_round_robin": RoundRobinScheduler,
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
    # Base class
    "Scheduler",
    # Python schedulers
    "RandomScheduler",
    "RoundRobinScheduler",
    "LeastConnectionScheduler",
    # Rust schedulers
    "RustRandomScheduler",
    "RustRoundRobinScheduler",
    "RustPowerOfTwoScheduler",
    "RustConsistentHashScheduler",
    "RustCacheAwareScheduler",
    # Factory
    "get_scheduler",
    # Constants
    "RUST_POLICIES_AVAILABLE",
]
