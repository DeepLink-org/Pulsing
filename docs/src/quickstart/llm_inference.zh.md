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
  --http_port 8080 \
  --model_name my-llm
```

| 参数 | 说明 |
|------|------|
| `--addr` | Actor 系统地址（Worker 加入此地址） |
| `--http_port` | OpenAI 兼容 HTTP 端点 |
| `--model_name` | API 响应中的模型名称 |

---

## 步骤 2：启动 Worker

打开**终端 B**：

=== "Transformers (CPU)"

    ```bash
    pulsing actor pulsing.serving.TransformersWorker \
      --model_name gpt2 \
      --device cpu \
      --addr 0.0.0.0:8001 \
      --seeds 127.0.0.1:8000
    ```

=== "vLLM (GPU)"

    ```bash
    pulsing actor pulsing.serving.VllmWorker \
      --model Qwen/Qwen2.5-0.5B \
      --addr 0.0.0.0:8002 \
      --seeds 127.0.0.1:8000
    ```

| 参数 | 说明 |
|------|------|
| `--model` / `--model_name` | 模型名称/路径（TransformersWorker 用 `--model_name`，VllmWorker 用 `--model`） |
| `--seeds` | 加入集群的 Router 地址 |

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
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-llm",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

### 流式 (SSE)

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-llm",
    "messages": [{"role": "user", "content": "讲个笑话"}],
    "stream": true
  }'
```

---

## 扩展

添加更多 Worker 处理更大负载：

```bash
# 终端 C
pulsing actor pulsing.serving.TransformersWorker --model_name gpt2 --addr 0.0.0.0:8003 --seeds 127.0.0.1:8000

# 终端 D
pulsing actor pulsing.serving.TransformersWorker --model_name gpt2 --addr 0.0.0.0:8004 --seeds 127.0.0.1:8000
```

Router 会自动在所有 Worker 间负载均衡。

---

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| `No available workers` | 确保 Worker 使用 `--seeds <router_addr>` |
| 连接被拒绝 | 检查 Router 是否以 `--addr` 启动 |
| 启动慢 | 首次请求需要加载模型权重 |

---

## 下一步

- [指南：运维操作](../guide/operations.zh.md) — 深入 CLI 工具
- [指南：安全](../guide/security.zh.md) — 用 mTLS 保护集群
- [设计：负载同步](../design/load_sync.md) — 负载均衡原理
