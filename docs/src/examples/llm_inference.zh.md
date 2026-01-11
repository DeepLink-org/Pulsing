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
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm
```

## 2）启动 Worker

你可以启动 **一个或多个** worker。每个 worker 通过 `--seeds` 加入 Router 节点。

### 方案 A：Transformers Worker（终端 B）

```bash
pulsing actor transformers --model gpt2 --device cpu --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000
```

### 方案 B：vLLM Worker（终端 C）

```bash
pulsing actor vllm --model Qwen/Qwen2.5-0.5B --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000
```

## 3）验证集群与 worker

### 列出 actors（观察者模式）

```bash
pulsing actor list --endpoint 127.0.0.1:8000
```

### 巡检集群

```bash
pulsing inspect --seeds 127.0.0.1:8000
```

## 4）调用 OpenAI 兼容 API

### 非流式

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-llm","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

### 流式（SSE）

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-llm","messages":[{"role":"user","content":"Tell me a joke"}],"stream":true}'
```

## 排障

- 如果出现 `No available workers`，请检查：
  - router 是否带了 `--addr`
  - worker 是否通过 `--seeds <router_addr>` 加入
  - worker actor 名称是否为 `worker`（默认）

更多：

- [运维（CLI）](../guide/operations.zh.md)
- [HTTP2 传输（设计）](../design/http2-transport.zh.md)
- [负载同步（设计）](../design/load_sync.zh.md)
