# Tutorial: LLM Inference Service

Build a **scalable LLM inference backend** with Pulsing in 10 minutes.

**Before / After:**

| | Before (single process or ad‑hoc script) | After (Pulsing) |
|---|------------------------------------------|-----------------|
| **API** | Your own HTTP or in-process only | OpenAI-compatible HTTP API (`/v1/chat/completions`) |
| **Scaling** | One process, one model | Router + N workers; add nodes and workers as needed |
| **Streaming** | Hand-rolled if any | Native streaming from Router to client |

You get a **Router** (HTTP API + load balancing) and **Workers** (model backends). Same Actor model; add more workers or nodes without changing client code.

**What you'll build:**

- A Router that exposes an **OpenAI-compatible HTTP API**
- One or more Workers that host model replicas
- Streaming token generation

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Router    │────▶│   Worker    │
│  (curl/SDK) │     │  :8080 HTTP │     │  (gpt2/vLLM)│
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Worker    │
                    │  (replica)  │
                    └─────────────┘
```

---

## Prerequisites

```bash
pip install pulsing
```

Choose a backend:

| Backend | Install | Best for |
|---------|---------|----------|
| **Transformers** | `pip install torch transformers` | Quick testing, CPU |
| **vLLM** | `pip install vllm` | Production, GPU |

---

## Step 1: Start the Router

Open **Terminal A**:

```bash
pulsing actor pulsing.serving.Router \
  --addr 0.0.0.0:8000 \
  -- \
  --http_port 8080 \
  --model_name my-llm
```

| Flag | Description |
|------|-------------|
| `--addr` (before `--`) | Actor system address (workers join here) |
| `--http_port`, `--model_name` (after `--`) | Router constructor: HTTP port, model name in API responses |

---

## Step 2: Start a Worker

Open **Terminal B**:

=== "Transformers (CPU)"

    ```bash
    pulsing actor pulsing.serving.TransformersWorker \
      --addr 0.0.0.0:8001 \
      --seeds 127.0.0.1:8000 \
      -- \
      --model_name gpt2 \
      --device cpu
    ```

=== "vLLM (GPU)"

    ```bash
    pulsing actor pulsing.serving.VllmWorker \
      --addr 0.0.0.0:8002 \
      --seeds 127.0.0.1:8000 \
      -- \
      --model Qwen/Qwen2.5-0.5B
    ```

| Flag | Description |
|------|-------------|
| `--addr`, `--seeds` (before `--`) | Actor-level: bind address, seed nodes |
| `--model` / `--model_name` (after `--`) | Constructor: model name/path |

---

## Step 3: Verify the Cluster

```bash
# List actors
pulsing inspect actors --endpoint 127.0.0.1:8000

# Inspect cluster state
pulsing inspect cluster --seeds 127.0.0.1:8000
```

You should see the `router` and `worker` actors.

---

## Step 4: Make Requests

### Non-streaming

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-llm",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

### Streaming (SSE)

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-llm",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

---

## Scaling Out

Add more workers to handle more load:

```bash
# Terminal C
pulsing actor pulsing.serving.TransformersWorker --addr 0.0.0.0:8003 --seeds 127.0.0.1:8000 -- --model_name gpt2

# Terminal D
pulsing actor pulsing.serving.TransformersWorker --addr 0.0.0.0:8004 --seeds 127.0.0.1:8000 -- --model_name gpt2
```

The Router automatically load-balances across all workers.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No available workers` | Router looks for actors named `worker` by default. (1) Start workers with `--name worker` (before `--`). (2) Or start Router with `--worker_name <name>` (after `--`) to match. (3) Workers must use `--seeds <router_addr>`. Check: `pulsing inspect actors --seeds 127.0.0.1:8000` and ensure a `worker` (or your custom name) appears. |
| Connection refused | Check router started with `--addr` |
| Slow startup | First request loads model weights |

---

## What's Next?

- [Guide: Operations](../guide/operations.md) — CLI tools in depth
- [Guide: Security](../guide/security.md) — secure your cluster with mTLS
- [Design: Load Sync](../design/load_sync.md) — how load balancing works
