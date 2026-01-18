"""Pulsing Actors - Distributed LLM inference components"""

# Worker
# Router
# Stream load subscription
from .load_stream import LoadSnapshot, LoadStreamConsumer, StreamLoadScheduler
from .router import Router, start_router, stop_router

# Scheduler
from .scheduler import (  # Base class; Python schedulers; Rust high-performance schedulers; Factory function
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
from .vllm import VllmWorker
from .worker import GenerationConfig, TransformersWorker

# Backward compatibility alias
TransformersWorkerActor = TransformersWorker


__all__ = [
    # Core API
    "Router",
    "TransformersWorker",
    "VllmWorker",
    "GenerationConfig",
    "start_router",
    "stop_router",
    # Scheduler base class and Python implementations
    "Scheduler",
    "RoundRobinScheduler",
    "RandomScheduler",
    "LeastConnectionScheduler",
    # Rust high-performance schedulers
    "RustRandomScheduler",
    "RustRoundRobinScheduler",
    "RustPowerOfTwoScheduler",
    "RustConsistentHashScheduler",
    "RustCacheAwareScheduler",
    # Stream load subscription (recommended for load-aware scheduling)
    "LoadSnapshot",
    "LoadStreamConsumer",
    "StreamLoadScheduler",
    # Factory function
    "get_scheduler",
    "RUST_POLICIES_AVAILABLE",
    # Compatibility aliases
    "TransformersWorkerActor",
]
