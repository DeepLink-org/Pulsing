"""
Pulsing Actors - 类 Dynamo 的分布式服务框架

提供：
- RouterActor: RoundRobin 负载均衡路由器，带 OpenAI 兼容 API
- TransformersWorkerActor: 基于 Transformers 的推理 Worker
- WorkerDiscovery: 自动 Worker 发现

使用示例:
    # 启动 Router
    pulsing actor --type router --http_port 8080
    
    # 启动 Worker
    pulsing actor --type transformers --model gpt2 --seeds <router地址>
    
    # 测试
    curl http://localhost:8080/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "my-llm", "messages": [{"role": "user", "content": "Hello"}]}'
"""

from .base import BaseServiceActor
from .router import RouterActor, RoundRobinScheduler, WorkerInfo
from .worker import TransformersWorkerActor, GenerationConfig
from .discovery import WorkerDiscovery, DiscoveredWorker

__all__ = [
    "BaseServiceActor",
    "RouterActor",
    "RoundRobinScheduler",
    "WorkerInfo",
    "TransformersWorkerActor",
    "GenerationConfig",
    "WorkerDiscovery",
    "DiscoveredWorker",
]
