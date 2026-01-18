# CLI 命令

Pulsing 内置 CLI 工具，用于启动 actors、检查系统和基准测试分布式服务。

---

## 启动 Actor

`pulsing actor` 命令通过提供完整的类路径来启动 actors。CLI 会自动将命令行参数匹配到 Actor 的构造函数参数。

### 格式

Actor 类型必须是完整的类路径：
- 格式: `module.path.ClassName`
- 示例: `pulsing.actors.router.RouterActor`
- 示例: `pulsing.actors.worker.TransformersWorker`
- 示例: `pulsing.actors.vllm.VllmWorker`
- 示例: `my_module.my_actor.MyCustomActor`

### 示例

#### Router（OpenAI 兼容 HTTP API）

```bash
pulsing actor pulsing.actors.router.RouterActor \
  --addr 0.0.0.0:8000 \
  --http_host 0.0.0.0 \
  --http_port 8080 \
  --model_name my-llm \
  --worker_name worker \
  --scheduler stream_load
```

#### Transformers Worker

```bash
pulsing actor pulsing.actors.worker.TransformersWorker \
  --model_name gpt2 \
  --device cpu \
  --addr 0.0.0.0:8001 \
  --seeds 127.0.0.1:8000 \
  --name worker
```

#### vLLM Worker

```bash
pulsing actor pulsing.actors.vllm.VllmWorker \
  --model Qwen/Qwen2 \
  --addr 0.0.0.0:8002 \
  --seeds 127.0.0.1:8000 \
  --name worker \
  --role aggregated \
  --max_new_tokens 512
```

#### 多个 Worker

```bash
# 启动多个不同名称的 worker
pulsing actor pulsing.actors.worker.TransformersWorker \
  --model_name gpt2 \
  --name worker-1 \
  --seeds 127.0.0.1:8000

pulsing actor pulsing.actors.worker.TransformersWorker \
  --model_name gpt2 \
  --name worker-2 \
  --seeds 127.0.0.1:8000

# Router 路由到特定 worker 名称
pulsing actor pulsing.actors.router.RouterActor \
  --worker_name worker-1 \
  --seeds 127.0.0.1:8000
```

### 通用选项

- `--name NAME`: Actor 名称（默认: "worker"）
- `--addr ADDR`: Actor System 绑定地址
- `--seeds SEEDS`: 逗号分隔的种子节点列表
- 任何其他 `--param value` 参数对，匹配 Actor 的构造函数签名

### 工作原理

CLI 会检查 Actor 类的构造函数签名，并自动从命令行参数中提取匹配的参数。可以使用 `--help` 查看可用参数，或查看 Actor 类的文档。

Actor 类必须：
- 可以从指定的模块路径导入
- 继承自 `pulsing.actor.Actor`
- 具有带命名参数的构造函数（CLI 会自动将参数匹配到构造函数参数）

---

## Inspect

`pulsing inspect` 是轻量级 **观察者**工具，通过 HTTP 查询 actor 系统（**无需加入集群**）。它提供多个子命令用于不同的检查需求。

### 子命令

#### 集群状态

检查集群成员及其状态：

```bash
pulsing inspect cluster --seeds 127.0.0.1:8000
```

输出包括：
- 总节点数和存活节点数
- 状态摘要（Alive、Suspect、Failed 等）
- 详细的成员列表，包含节点 ID、地址和状态

#### Actor 分布

检查命名 actors 在集群中的分布：

```bash
pulsing inspect actors --seeds 127.0.0.1:8000
```

选项：
- `--endpoint ADDR`: 查询单个节点（例如：`127.0.0.1:8000`）
- `--top N`: 显示实例数最多的前 N 个 actors
- `--filter STR`: 按子字符串过滤 actor 名称
- `--all_actors True`: 包含内部/系统 actors
- `--json_output True`: JSON 格式输出
- `--detailed True`: 显示详细信息（类、模块等）

示例：
```bash
# 查询单个节点
pulsing inspect actors --endpoint 127.0.0.1:8000

# 显示前 10 个 actors
pulsing inspect actors --seeds 127.0.0.1:8000 --top 10

# 按名称过滤 actors
pulsing inspect actors --seeds 127.0.0.1:8000 --filter worker

# 显示详细信息
pulsing inspect actors --endpoint 127.0.0.1:8000 --detailed
```

#### 指标

检查集群节点的 Prometheus 指标：

```bash
pulsing inspect metrics --seeds 127.0.0.1:8000
```

选项：
- `--raw True`: 输出原始指标（默认）
- `--raw False`: 仅显示摘要（关键指标）

#### 监视模式

实时监视集群状态变化：

```bash
pulsing inspect watch --seeds 127.0.0.1:8000
```

选项：
- `--interval 1.0`: 刷新间隔（秒，默认: 1.0）
- `--kind all`: 监视内容：`cluster`、`actors`、`metrics` 或 `all`（默认: `all`）
- `--max_rounds N`: 最大刷新轮数（None = 无限）

示例：
```bash
# 监视集群成员变化
pulsing inspect watch --seeds 127.0.0.1:8000 --kind cluster --interval 2.0

# 监视 actor 变化
pulsing inspect watch --seeds 127.0.0.1:8000 --kind actors
```

### 通用选项

所有子命令支持：

- `--timeout 10.0`: 请求超时（秒，默认: 10.0）
- `--best_effort True`: 即使某些节点失败也继续（默认: False）

!!! note
    观察者模式使用 HTTP/2 (h2c)，**不会**加入 gossip 集群，使其轻量级且适合生产环境监控。

---

## Bench

`pulsing bench` 对 OpenAI 兼容推理端点进行负载测试。

```bash
pulsing bench gpt2 --url http://localhost:8080
```

!!! note "可选扩展"
    若提示 `pulsing._bench module not found`：

    ```bash
    maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
    ```

---

## 快速参考

| 任务 | 命令 |
|------|------|
| 启动 router | `pulsing actor pulsing.actors.router.RouterActor --addr 0.0.0.0:8000 --http_port 8080` |
| 启动 worker | `pulsing actor pulsing.actors.worker.TransformersWorker --model_name gpt2 --seeds ...` |
| 启动多个 worker | `pulsing actor pulsing.actors.worker.TransformersWorker --model_name gpt2 --name worker-1 --seeds ...` |
| Router 指定 worker | `pulsing actor pulsing.actors.router.RouterActor --worker_name worker-1 --seeds ...` |
| 列出 actors | `pulsing inspect actors --endpoint 127.0.0.1:8000` |
| 检查集群 | `pulsing inspect cluster --seeds 127.0.0.1:8000` |
| 检查 actors | `pulsing inspect actors --seeds 127.0.0.1:8000 --top 10` |
| 检查指标 | `pulsing inspect metrics --seeds 127.0.0.1:8000` |
| 监视集群 | `pulsing inspect watch --seeds 127.0.0.1:8000` |
| 基准测试 | `pulsing bench gpt2 --url http://localhost:8080` |

---

## 下一步

- [LLM 推理](../examples/llm_inference.zh.md) — 可运行的端到端教程
- [安全](security.zh.md) — mTLS 和集群隔离
