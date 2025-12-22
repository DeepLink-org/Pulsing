# Pulsing Actor System 示例

本目录包含基于 Pulsing Actor System 的服务示例。

## Actor 类型

### 1. RouterActor

基于 RoundRobin 调度算法的路由器。

```bash
# 启动 Router
pulsing actor --type router --namespace myapp --addr 0.0.0.0:8000
```

### 2. TransformersWorkerActor

基于 HuggingFace Transformers 的推理 Worker。

```bash
# 启动 Worker (GPU)
pulsing actor --type transformers \
    --model Qwen/Qwen3-0.6B \
    --namespace myapp \
    --addr 0.0.0.0:8001 \
    --seeds 192.168.1.100:8000

# 启动 Worker (CPU)
pulsing actor --type transformers \
    --model gpt2 \
    --device cpu \
    --namespace myapp
```

## 部署示例

### 单机部署

```bash
# 终端 1: 启动 Router
pulsing actor --type router --addr 0.0.0.0:8000

# 终端 2: 启动 Worker 1
pulsing actor --type transformers --model gpt2 --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000

# 终端 3: 启动 Worker 2
pulsing actor --type transformers --model gpt2 --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000
```

### 多机部署

```bash
# 机器 A (Router)
pulsing actor --type router --addr 0.0.0.0:8000

# 机器 B (Worker)
pulsing actor --type transformers --model Qwen/Qwen3-0.6B --addr 0.0.0.0:8000 --seeds 192.168.1.A:8000

# 机器 C (Worker)
pulsing actor --type transformers --model Qwen/Qwen3-0.6B --addr 0.0.0.0:8000 --seeds 192.168.1.A:8000
```

## 编程接口

### 直接使用 Actor 类

```python
import asyncio
from pulsing.actors import RouterActor, TransformersWorkerActor

async def main():
    # 启动 Router
    router = RouterActor(
        namespace="myapp",
        addr="0.0.0.0:8000"
    )
    await router.start()
    
    # 启动 Worker
    worker = TransformersWorkerActor(
        model="gpt2",
        namespace="myapp",
        addr="0.0.0.0:8001",
        seeds=["127.0.0.1:8000"],
        device="cpu"
    )
    await worker.start()
    
    # ... 业务逻辑 ...
    
    await worker.stop()
    await router.stop()

asyncio.run(main())
```

### 消息协议

#### Router 消息

| 消息类型 | 描述 | 请求字段 | 响应字段 |
|---------|------|---------|---------|
| RouteRequest | 请求路由 | - | worker_id, endpoint |
| RegisterWorker | 注册 Worker | worker_id, endpoint, metadata | message, total_workers |
| UnregisterWorker | 注销 Worker | worker_id | message, total_workers |
| HealthCheck | 健康检查 | - | status, total_workers, healthy_workers, workers |

#### Worker 消息

| 消息类型 | 描述 | 请求字段 | 响应字段 |
|---------|------|---------|---------|
| GenerateRequest | 生成请求 | prompt 或 token_ids, max_new_tokens | text, token_ids, finish_reason |
| WorkerStatus | 状态查询 | - | worker_id, model, is_loaded, request_count |
| HealthCheck | 健康检查 | - | status, worker_id, is_loaded |

## 架构图

```
                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Router    │
                    │ (RoundRobin)│
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Worker    │ │   Worker    │ │   Worker    │
    │(Transformers)│ │(Transformers)│ │(Transformers)│
    └─────────────┘ └─────────────┘ └─────────────┘
```
