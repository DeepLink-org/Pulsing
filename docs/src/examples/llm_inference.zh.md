# LLM 推理（可运行）

本指南展示如何用 Pulsing 跑通 **router + worker** 架构，并对外暴露 **OpenAI 兼容 HTTP API**。

## 推荐架构

- **Router**：接入 HTTP 请求，选择 worker，转发 `GenerateRequest` / `GenerateStreamRequest`
- **Worker**：承载模型副本

## 0）前置条件

- `pip install pulsing`
- 选择一种或两种后端：
  - **Transformers**：安装 `torch` + `transformers`
  - **vLLM**：安装 `vllm`

## 1）启动 Router（终端 A）

Router 需要指定 **actor system 地址**，以便其它进程启动的 workers 加入同一集群：

```bash
pulsing actor pulsing.serving.Router \
  --addr 0.0.0.0:8000 \
  --name my-llm \
  -- \
  --http_port 8080 \
  --model_name gpt2 \
  --worker_name worker
```

## 2）启动 Worker

你可以启动 **一个或多个** worker。每个 worker 通过 `--seeds` 加入 Router 节点。

### 方案 A：Transformers Worker（终端 B）

```bash
pulsing actor pulsing.serving.TransformersWorker \
  --addr 0.0.0.0:8001 \
  --seeds 127.0.0.1:8000 \
  --name worker \
  -- \
  --model_name gpt2
```

### 方案 B：vLLM Worker（终端 C）

```bash
pulsing actor pulsing.serving.vllm.VllmWorker \
  --addr 0.0.0.0:8002 \
  --seeds 127.0.0.1:8000 \
  --name worker \
  -- \
  --model Qwen/Qwen2.5-0.5B
```

## 3）验证集群与 worker

### 列出 actors（观察者模式）

```bash
pulsing inspect actors --endpoint 127.0.0.1:8000
```

### 巡检集群

```bash
pulsing inspect cluster --seeds 127.0.0.1:8000
```

## 4）调用 OpenAI 兼容 API

### 非流式

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

### 流式（SSE）

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "messages": [{"role": "user", "content": "讲个笑话"}], "stream": true}'
```

## 排障

- 如果出现 `No available workers`，请检查：
  - Router 已用 `--addr` 启动，Worker 已用 `--seeds <router_addr>` 加入
  - **名字一致**：Worker 用 `--name worker`（`--` 前）启动，或 Router 用 `--worker_name <名字>`（`--` 后）与 Worker 一致
  - 执行 `pulsing inspect actors --seeds 127.0.0.1:8000`，确认能看到 Router 在找的名字（默认 `worker`）

更多：

- [运维（CLI）](../guide/operations.zh.md)
- [HTTP2 传输（设计）](../design/http2-transport.zh.md)
- [负载同步（设计）](../design/load_sync.zh.md)
