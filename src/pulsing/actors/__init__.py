"""
Pulsing Actors - 类 Dynamo 的分布式服务框架

提供：
- RouterActor: 基于 Named Actor 的路由器，带 OpenAI 兼容 API
- TransformersWorkerActor: 基于 Transformers 的推理 Worker

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
from .router import RouterActor
from .worker import TransformersWorkerActor
from .openai_server import OpenAIServer
from .scheduler import Scheduler, RoundRobinScheduler, RandomScheduler, LeastConnectionScheduler

__all__ = [
    "BaseServiceActor",
    "RouterActor",
    "TransformersWorkerActor",
    "OpenAIServer",
    "Scheduler",
    "RoundRobinScheduler",
    "RandomScheduler",
    "LeastConnectionScheduler",
]
