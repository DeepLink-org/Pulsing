# LLM Inference (runnable)

This guide shows how to run a **router + worker** LLM service with Pulsing, and expose an **OpenAI-compatible HTTP API**.

## Architecture

- **Router**: accepts HTTP requests, selects a worker, forwards `GenerateRequest` / `GenerateStreamRequest`
- **Workers**: host model replicas

## 0) Prerequisites

- `pip install pulsing`
- Choose one backend:
  - **Transformers**: install `torch` + `transformers`
  - **vLLM**: install `vllm`

## 1) Start the Router (Terminal A)

The router needs an **actor system address** so workers can join the same cluster:

```bash
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm
```

## 2) Start workers

You can run **one or more** workers. Each worker should join the router node via `--seeds`.

### Option A: Transformers worker (Terminal B)

```bash
pulsing actor transformers --model gpt2 --device cpu --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000
```

### Option B: vLLM worker (Terminal C)

```bash
pulsing actor vllm --model Qwen/Qwen2.5-0.5B --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000
```

## 3) Verify cluster + workers

### List actors (observer mode)

```bash
pulsing actor list --endpoint 127.0.0.1:8000
```

### Inspect cluster

```bash
pulsing inspect --seeds 127.0.0.1:8000
```

## 4) Call the OpenAI-compatible API

### Non-streaming

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-llm","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

### Streaming (SSE)

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-llm","messages":[{"role":"user","content":"Tell me a joke"}],"stream":true}'
```

## Troubleshooting

- If you see `No available workers`, ensure:
  - router is started with `--addr`
  - workers join via `--seeds <router_addr>`
  - the worker actor name is `worker` (default)

See also:

- [Operations (CLI)](../guide/operations.md)
- [HTTP2 Transport (design)](../design/http2-transport.md)
- [Load Sync (design)](../design/load_sync.md)
