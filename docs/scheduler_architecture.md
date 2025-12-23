# 调度器架构设计（v3 - 简化版）

## 设计理念

**按需查询，轻量级调度**

经过重新思考，我们采用了更简洁的架构：
- 不缓存 ActorRef（`resolve_named` 本身很快，Rust 层有缓存）
- 不需要后台循环（按需查询 Gossip）
- 调度器只负责选择策略（维护最小状态）

## 职责分离

### 1. **Rust Gossip 层** - 服务发现和解析

**职责**：
- 集群成员管理（通过 Gossip 协议）
- Named Actor 注册和追踪
- 节点健康状态监控（Alive/Suspect/Dead）
- 提供查询接口：`get_named_instances(name)` 
- 快速解析：`resolve_named(name)` 返回健康的 ActorRef（内部有缓存和负载均衡）

**优势**：
- 高性能 Rust 实现
- 分布式一致性保证
- 自动故障检测
- 零配置服务发现
- ActorRef 缓存在 Rust 层（更高效）

### 2. **Python 调度器层** - 轻量级路由选择

**职责**：
- **按需查询**：调用 `get_named_instances()` 获取 Worker 列表
- **路由选择**：根据策略（RoundRobin、Random、LeastConnection）选择
- **最小状态**：只维护调度器自己需要的状态（如 RoundRobin 的 index）

**优势**：
- 极简：~120 行代码
- 灵活：易于定制策略
- 实时：总是获取最新的 Worker 列表
- 无状态同步问题

## 工作流程

### 初始化流程

```
RouterActor.start()
    ↓
1. 创建 ActorSystem
    ↓
2. 创建 Scheduler 实例（轻量级，无后台任务）
    ↓
3. 启动 HTTP 服务器
```

### 请求流程（简洁高效）

```
HTTP 请求到达
    ↓
OpenAIServer.chat_completions()
    ↓
1. 调度器按需查询: worker_ref = await scheduler.select_worker()
   内部流程：
   a. 查询 Gossip: workers = get_named_instances("worker")
   b. 根据策略选择: selected = workers[index % len(workers)]  # RoundRobin 示例
   c. 解析 ActorRef: actor_ref = resolve_named("worker")
   （Rust 层会选择健康的实例并缓存 ActorRef）
    ↓
2. 发送请求: reader = await worker_ref.ask_stream(request_msg)
    ↓
3. 流式响应
```

**性能分析**：
- `get_named_instances()`: 本地内存查询，微秒级
- `resolve_named()`: Rust 层有 ActorRef 缓存，首次 resolve 后极快
- 总开销：每次请求增加 < 1ms

## 支持的调度策略

### RoundRobinScheduler
- **算法**：轮询每个 Alive 状态的 Worker
- **适用场景**：Worker 性能均衡
- **优点**：简单、公平
- **使用**：`--scheduler round_robin`（默认）

### RandomScheduler
- **算法**：随机选择一个 Alive 状态的 Worker
- **适用场景**：无状态请求
- **优点**：简单、无锁
- **使用**：`--scheduler random`

### LeastConnectionScheduler
- **算法**：选择当前连接数最少的 Worker
- **适用场景**：Worker 性能不均衡或请求耗时差异大
- **优点**：自适应负载
- **使用**：`--scheduler least_connection`

## 使用示例

### 启动 Router（使用不同调度器）

```bash
# 使用 RoundRobin（默认）
pulsing actor --type router --http-port 8080

# 使用随机调度
pulsing actor --type router --http-port 8080 --scheduler random

# 使用最少连接调度
pulsing actor --type router --http-port 8080 --scheduler least_connection
```

### 启动多个 Worker

```bash
# Worker 1
pulsing actor --type transformers --model gpt2 --addr 127.0.0.1:8001 --seeds 127.0.0.1:8000

# Worker 2
pulsing actor --type transformers --model gpt2 --addr 127.0.0.1:8002 --seeds 127.0.0.1:8000

# Worker 3
pulsing actor --type transformers --model gpt2 --addr 127.0.0.1:8003 --seeds 127.0.0.1:8000
```

### 测试

```bash
curl http://localhost:8080/health
# 输出: {"status": "healthy", "model": "pulsing-model", "healthy_workers": 3, ...}

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-llm", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## 自定义调度器

你可以轻松扩展自定义调度策略：

```python
from pulsing.actors import Scheduler

class WeightedScheduler(Scheduler):
    """基于权重的调度器"""
    
    def __init__(self, weights: dict):
        self.weights = weights  # {node_id: weight}
    
    async def select_worker(self, available_workers):
        import random
        alive = [w for w in available_workers if w.get("status") == "Alive"]
        if not alive:
            return None
        
        # 按权重选择
        total_weight = sum(self.weights.get(w["node_id"], 1.0) for w in alive)
        r = random.uniform(0, total_weight)
        
        cumsum = 0
        for w in alive:
            cumsum += self.weights.get(w["node_id"], 1.0)
            if r <= cumsum:
                return w
        return alive[-1]

# 使用
from pulsing.actors import RouterActor
router = RouterActor(
    scheduler=WeightedScheduler({"node1": 2.0, "node2": 1.0})
)
```

## 架构优势

1. **极简**：
   - 调度器只有 ~120 行代码
   - 无后台任务，无状态同步
   - 职责清晰：Gossip 发现+缓存，Scheduler 选择

2. **高性能**：
   - ActorRef 缓存在 Rust 层（C++ 级性能）
   - 按需查询 Gossip（本地内存，微秒级）
   - 每次请求开销 < 1ms

3. **实时性**：
   - 每次请求都获取最新的 Worker 列表
   - 无缓存一致性问题
   - Worker 上线/下线立即生效

4. **可靠**：
   - Gossip 层负责服务发现和健康检查
   - `resolve_named` 自动过滤不健康的 Worker
   - 自动故障转移

5. **灵活**：
   - 调度策略在 Python 层易于定制
   - 支持多种负载均衡算法
   - 可以基于业务指标（如请求数）进行调度

6. **易维护**：
   - 代码少，逻辑清晰
   - 无复杂的状态管理
   - 易于测试和调试

## 代码结构

```
src/pulsing/actors/
├── base.py          - BaseServiceActor
├── router.py        - RouterActor (HTTP 服务器)
├── worker.py        - TransformersWorkerActor
├── openai_server.py - OpenAI API 实现
└── scheduler.py     - 调度器（~120 行）
    ├── Scheduler                - 抽象基类
    │   ├── get_available_workers() - 查询 Gossip
    │   ├── get_worker_count()      - 统计方法
    │   └── select_worker()         - 选择并返回 ActorRef（抽象方法）
    ├── RoundRobinScheduler      - 轮询调度（维护 index）
    ├── RandomScheduler          - 随机调度（无状态）
    └── LeastConnectionScheduler - 最少连接（维护请求计数）
```

## 关键设计

### 按需查询

每次 `select_worker()` 调用时：
```python
async def select_worker(self):
    # 1. 查询 Gossip（本地内存，极快）
    workers = await self.get_available_workers()
    
    # 2. 根据策略选择
    selected = workers[self._index % len(workers)]  # RoundRobin 示例
    self._index += 1
    
    # 3. 解析 ActorRef（Rust 层缓存，首次后极快）
    actor_ref = await self._system.resolve_named("worker")
    
    return actor_ref
```

### 调度器状态

**RoundRobinScheduler**：只维护 `_index` 计数器

**RandomScheduler**：完全无状态

**LeastConnectionScheduler**：维护 `_request_counts: Dict[node_id, count]`

### Rust 层的职责

- **ActorRef 缓存**：`resolve_named` 内部缓存已解析的 ActorRef
- **健康检查**：只返回健康的 Worker
- **负载均衡**：当有多个健康 Worker 时，内部也会做负载均衡

