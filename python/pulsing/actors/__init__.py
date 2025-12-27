"""Pulsing Actors - 分布式 LLM 推理组件"""

# Worker
# Router
# 流式负载订阅
from .load_stream import LoadSnapshot, LoadStreamConsumer, StreamLoadScheduler
from .router import start_router, stop_router

# Scheduler
from .scheduler import (  # 基类; Python 调度器; Rust 高性能调度器; 工厂函数
    RUST_POLICIES_AVAILABLE,
    LeastConnectionScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    RustCacheAwareScheduler,
    RustConsistentHashScheduler,
    RustPowerOfTwoScheduler,
    RustRandomScheduler,
    RustRoundRobinScheduler,
    Scheduler,
    get_scheduler,
)
from .vllm_worker import VllmWorker
from .worker import GenerationConfig, TransformersWorker

# 向后兼容别名
TransformersWorkerActor = TransformersWorker


__all__ = [
    # Core API
    "TransformersWorker",
    "VllmWorker",
    "GenerationConfig",
    "start_router",
    "stop_router",
    # Scheduler 基类和 Python 实现
    "Scheduler",
    "RoundRobinScheduler",
    "RandomScheduler",
    "LeastConnectionScheduler",
    # Rust 高性能调度器
    "RustRandomScheduler",
    "RustRoundRobinScheduler",
    "RustPowerOfTwoScheduler",
    "RustConsistentHashScheduler",
    "RustCacheAwareScheduler",
    # 流式负载订阅 (推荐用于负载感知调度)
    "LoadSnapshot",
    "LoadStreamConsumer",
    "StreamLoadScheduler",
    # 工厂函数
    "get_scheduler",
    "RUST_POLICIES_AVAILABLE",
    # Compatibility aliases
    "TransformersWorkerActor",
]
