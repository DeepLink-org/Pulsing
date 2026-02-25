# 教程：LLM 推理服务

10 分钟内用 Pulsing 构建一个**可扩展的 LLM 推理后端**。

**前后对比：**

| | 之前（单进程或临时脚本） | 之后（Pulsing） |
|---|--------------------------|-----------------|
| **API** | 自建 HTTP 或仅进程内 | OpenAI 兼容 HTTP API（`/v1/chat/completions`） |
| **扩展** | 单进程、单模型 | Router + N 个 Worker；按需增加节点与 Worker |
| **流式** | 若有则手写 | Router 到客户端的原生流式 |

你会得到一个 **Router**（HTTP API + 负载均衡）和若干 **Worker**（模型后端）。同一套 Actor 模型；增加 Worker 或节点无需改客户端代码。

**你将构建：**

- 一个暴露 **OpenAI 兼容 HTTP API** 的 Router
- 一个或多个托管模型副本的 Worker
- 流式 token 生成

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   客户端    │────▶│   Router    │────▶│   Worker    │
│  (curl/SDK) │     │  :8080 HTTP │     │  (gpt2/vLLM)│
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Worker    │
                    │   (副本)    │
                    └─────────────┘
```

---

## 前置条件

```bash
pip install pulsing
```

选择后端：

| 后端 | 安装 | 适用场景 |
|------|------|----------|
| **Transformers** | `pip install torch transformers` | 快速测试，CPU |
| **vLLM** | `pip install vllm` | 生产环境，GPU |

---

## 步骤 1：启动 Router

打开**终端 A**：

```bash
pulsing actor pulsing.serving.Router \
  --addr 0.0.0.0:8000 \
  --name my-llm \
  -- \
  --http_port 8080 \
  --model_name gpt2 \
  --worker_name worker
```

| 参数 | 说明 |
|------|------|
| `--addr`、`--name`（`--` 前） | Actor 系统地址、Router 名称 |
| `--http_port`、`--model_name`、`--worker_name`（`--` 后） | Router 构造参数：HTTP 端口、API 模型名、目标 worker 名 |

---

## 步骤 2：启动 Worker

打开**终端 B**：

=== "Transformers (CPU)"

    ```bash
    pulsing actor pulsing.serving.TransformersWorker \
      --addr 0.0.0.0:8001 \
      --seeds 127.0.0.1:8000 \
      --name worker \
      -- \
      --model_name gpt2
    ```

=== "vLLM (GPU)"

    ```bash
    pulsing actor pulsing.serving.VllmWorker \
      --addr 0.0.0.0:8002 \
      --seeds 127.0.0.1:8000 \
      --name worker \
      -- \
      --model Qwen/Qwen2.5-0.5B
    ```

| 参数 | 说明 |
|------|------|
| `--addr`、`--seeds`（`--` 前） | actor 级：绑定地址、种子节点 |
| `--model` / `--model_name`（`--` 后） | 构造参数：模型名称/路径 |

---

## 步骤 3：验证集群

```bash
# 列出 actor
pulsing inspect actors --endpoint 127.0.0.1:8000

# 检查集群状态
pulsing inspect cluster --seeds 127.0.0.1:8000
```

你应该能看到 `router` 和 `worker` actor。

---

## 步骤 4：发送请求

### 非流式

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

### 流式 (SSE)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "messages": [{"role": "user", "content": "讲个笑话"}], "stream": true}'
```

---

## 扩展

添加更多 Worker 处理更大负载：

```bash
# 终端 C
pulsing actor pulsing.serving.TransformersWorker --addr 0.0.0.0:8003 --seeds 127.0.0.1:8000 -- --model_name gpt2

# 终端 D
pulsing actor pulsing.serving.TransformersWorker --addr 0.0.0.0:8004 --seeds 127.0.0.1:8000 -- --model_name gpt2
```

Router 会自动在所有 Worker 间负载均衡。

---

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| `No available workers` | Router 默认按名字 `worker` 查找。需 (1) Worker 用 `--name worker`（在 `--` 前）启动，或 (2) Router 用 `--worker_name <名字>`（在 `--` 后）与 Worker 一致；(3) Worker 必须 `--seeds <router_addr>`。可执行 `pulsing inspect actors --seeds 127.0.0.1:8000` 确认是否有名为 `worker` 的 actor。 |
| 连接被拒绝 | 检查 Router 是否以 `--addr` 启动 |
| 启动慢 | 首次请求需要加载模型权重 |

---

## 下一步

- [指南：运维操作](../guide/operations.zh.md) — 深入 CLI 工具
- [指南：安全](../guide/security.zh.md) — 用 mTLS 保护集群
- [设计：负载同步](../design/load_sync.md) — 负载均衡原理
