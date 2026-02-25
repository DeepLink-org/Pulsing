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
pulsing actor pulsing.serving.Router \
  --addr 0.0.0.0:8000 \
  --name my-llm \
  -- \
  --http_port 8080 \
  --model_name gpt2 \
  --worker_name worker
```

## 2) Start workers

You can run **one or more** workers. Each worker should join the router node via `--seeds`.

### Option A: Transformers worker (Terminal B)

```bash
pulsing actor pulsing.serving.TransformersWorker \
  --addr 0.0.0.0:8001 \
  --seeds 127.0.0.1:8000 \
  --name worker \
  -- \
  --model_name gpt2
```

### Option B: vLLM worker (Terminal C)

```bash
pulsing actor pulsing.serving.vllm.VllmWorker \
  --addr 0.0.0.0:8002 \
  --seeds 127.0.0.1:8000 \
  --name worker \
  -- \
  --model Qwen/Qwen2.5-0.5B
```

## 3) Verify cluster + workers

### List actors (observer mode)

```bash
pulsing inspect actors --endpoint 127.0.0.1:8000
```

### Inspect cluster

```bash
pulsing inspect cluster --seeds 127.0.0.1:8000
```

## 4) Call the OpenAI-compatible API

### Non-streaming

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

### Streaming (SSE)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "messages": [{"role": "user", "content": "Tell me a joke"}], "stream": true}'
```

## Troubleshooting

- If you see `No available workers`, ensure:
  - router is started with `--addr` and workers join via `--seeds <router_addr>`
  - the worker actor **name** matches: workers started with `--name worker` (before `--`), or start the router with `--worker_name <name>` (after `--`) to match your worker name
  - check: `pulsing inspect actors --seeds 127.0.0.1:8000` — you should see an actor with the name the router is looking for (default `worker`)

See also:

- [Operations (CLI)](../guide/operations.md)
- [HTTP2 Transport (design)](../design/http2-transport.md)
- [Load Sync (design)](../design/load_sync.md)
