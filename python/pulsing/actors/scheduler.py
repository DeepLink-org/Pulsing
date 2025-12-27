"""Worker 调度器 - 负载均衡策略

支持以下调度策略:
- RandomScheduler: 随机选择 (Python实现)
- RoundRobinScheduler: 轮询选择 (Python实现)
- LeastConnectionScheduler: 最少连接 (Python实现)
- RustRandomScheduler: 随机选择 (Rust实现，高性能)
- RustRoundRobinScheduler: 轮询选择 (Rust实现)
- RustPowerOfTwoScheduler: Power-of-Two Choices (Rust实现)
- RustConsistentHashScheduler: 一致性哈希 (Rust实现，支持会话亲和性)
- RustCacheAwareScheduler: 缓存感知路由 (Rust实现，支持Radix Tree前缀匹配)

负载感知调度推荐使用 load_stream 模块中的 StreamLoadScheduler
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
    """调度器基类"""

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

    async def _resolve_worker(self, node_id: str | None = None):
        try:
            # node_id 在 MemberInfo 中被序列化为字符串，需要转回 int 以匹配 resolve_named
            nid_int = int(node_id) if node_id else None
            return await self._system.resolve_named(self._worker_name, node_id=nid_int)
        except Exception:
            return None

    @abstractmethod
    async def select_worker(
        self,
        request_text: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        """选择一个 worker，返回 ActorRef 或 None

        Args:
            request_text: 请求文本 (用于缓存感知和一致性哈希路由)
            headers: HTTP 请求头 (用于一致性哈希路由)
        """
        pass


# ============================================================================
# Python 实现的调度器
# ============================================================================


class RoundRobinScheduler(Scheduler):
    """轮询调度器 (Python实现)"""

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
    """随机调度器 (Python实现)"""

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
    """最少连接调度器 (Python实现)"""

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
# Rust 实现的调度器 (高性能)
# ============================================================================


class RustSchedulerBase(Scheduler):
    """Rust 调度器基类"""

    def __init__(self, actor_system, worker_name: str = "worker"):
        super().__init__(actor_system, worker_name)
        self._worker_info_cache: dict[str, WorkerInfo] = {}

    def _get_worker_info(self, worker_data: dict) -> WorkerInfo:
        """获取或创建 WorkerInfo 对象"""
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
        """将 worker 数据转换为 WorkerInfo 列表"""
        return [self._get_worker_info(w) for w in workers]

    @abstractmethod
    def _get_policy(self):
        """获取 Rust 策略对象"""
        pass


class RustRandomScheduler(RustSchedulerBase):
    """随机调度器 (Rust实现，高性能)"""

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
    """轮询调度器 (Rust实现)"""

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
        """重置轮询计数器"""
        self._policy.reset()


class RustPowerOfTwoScheduler(RustSchedulerBase):
    """Power-of-Two Choices 调度器 (Rust实现)

    随机选择两个 worker，然后选择负载较低的那个。
    在大规模集群中能够提供接近最优的负载均衡。
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
        """更新缓存的负载信息

        Args:
            loads: worker URL 到负载值的映射
        """
        self._policy.update_loads(loads)


class RustConsistentHashScheduler(RustSchedulerBase):
    """一致性哈希调度器 (Rust实现)

    基于会话ID或用户ID进行路由，确保同一用户的请求始终路由到同一 worker。
    支持从 HTTP 头部或请求体中提取路由键。

    HTTP 头部优先级 (按顺序检查):
    - x-session-id
    - x-user-id
    - x-tenant-id
    - x-request-id
    - x-correlation-id
    - x-trace-id

    请求体字段优先级:
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
        """重置哈希环"""
        self._policy.reset()


class RustCacheAwareScheduler(RustSchedulerBase):
    """缓存感知调度器 (Rust实现)

    结合缓存亲和性和负载均衡两种策略:
    1. 当系统负载均衡时，使用缓存感知路由 (基于 Radix Tree 前缀匹配)
    2. 当系统负载不均衡时，切换到最短队列路由

    特别适合 LLM 推理场景，可以提高 KV Cache 命中率。

    Args:
        cache_threshold: 前缀匹配阈值 (0.0-1.0)，超过此阈值时使用缓存亲和性路由
        balance_abs_threshold: 负载不均衡的绝对阈值
        balance_rel_threshold: 负载不均衡的相对阈值
        eviction_interval_secs: 缓存驱逐间隔 (秒)
        max_tree_size: Radix Tree 最大节点数
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

        # 初始化 workers (如果是第一次调用)
        self._policy.init_workers(worker_infos)

        selected_idx = self._policy.select_worker(worker_infos, request_text)

        if selected_idx is None:
            return None

        selected_worker = workers[selected_idx]
        return await self._resolve_worker(node_id=selected_worker.get("node_id"))

    def add_worker(self, url: str, model_id: str = "default"):
        """添加 worker 到缓存树"""
        self._policy.add_worker(url, model_id)

    def remove_worker(self, url: str):
        """从缓存树移除 worker"""
        self._policy.remove_worker(url)

    def evict_cache(self, max_size: int):
        """手动触发缓存驱逐"""
        self._policy.evict_cache(max_size)


# ============================================================================
# 工厂函数
# ============================================================================


def get_scheduler(
    policy_name: str, actor_system, worker_name: str = "worker", **kwargs
) -> Scheduler:
    """获取调度器实例

    Args:
        policy_name: 策略名称，支持:
            - "random": 随机 (Rust实现)
            - "round_robin": 轮询 (Rust实现)
            - "power_of_two": Power-of-Two Choices (Rust实现)
            - "consistent_hash": 一致性哈希 (Rust实现)
            - "cache_aware": 缓存感知 (Rust实现)
            - "py_random": 随机 (Python实现)
            - "py_round_robin": 轮询 (Python实现)
            - "least_connection": 最少连接 (Python实现)
        actor_system: Actor 系统实例
        worker_name: Worker actor 名称
        **kwargs: 策略特定参数 (如 cache_threshold 等)

    Returns:
        Scheduler 实例

    Note:
        负载感知调度推荐使用 load_stream 模块中的 StreamLoadScheduler

    Examples:
        # 使用缓存感知调度
        scheduler = get_scheduler("cache_aware", actor_system, "worker",
                                  cache_threshold=0.5)

        # 使用轮询调度
        scheduler = get_scheduler("round_robin", actor_system, "worker")
    """
    policy_map = {
        # Rust 实现 (推荐)
        "random": RustRandomScheduler if RUST_POLICIES_AVAILABLE else RandomScheduler,
        "round_robin": (
            RustRoundRobinScheduler if RUST_POLICIES_AVAILABLE else RoundRobinScheduler
        ),
        "power_of_two": RustPowerOfTwoScheduler,
        "consistent_hash": RustConsistentHashScheduler,
        "cache_aware": RustCacheAwareScheduler,
        # Python 实现
        "py_random": RandomScheduler,
        "py_round_robin": RoundRobinScheduler,
        "least_connection": LeastConnectionScheduler,
    }

    scheduler_class = policy_map.get(policy_name)
    if scheduler_class is None:
        raise ValueError(
            f"Unknown policy: {policy_name}. Available: {list(policy_map.keys())}"
        )

    # 支持 kwargs 的策略
    if policy_name == "cache_aware":
        return scheduler_class(actor_system, worker_name, **kwargs)
    else:
        return scheduler_class(actor_system, worker_name)


# 导出
__all__ = [
    # 基类
    "Scheduler",
    # Python 调度器
    "RandomScheduler",
    "RoundRobinScheduler",
    "LeastConnectionScheduler",
    # Rust 调度器
    "RustRandomScheduler",
    "RustRoundRobinScheduler",
    "RustPowerOfTwoScheduler",
    "RustConsistentHashScheduler",
    "RustCacheAwareScheduler",
    # 工厂
    "get_scheduler",
    # 常量
    "RUST_POLICIES_AVAILABLE",
]
