# Tutorial: LLM Inference Service

Build a **scalable LLM inference backend** with Pulsing in 10 minutes.

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
pulsing actor router \
  --addr 0.0.0.0:8000 \
  --http_port 8080 \
  --model_name my-llm
```

| Flag | Description |
|------|-------------|
| `--addr` | Actor system address (workers join here) |
| `--http_port` | OpenAI-compatible HTTP endpoint |
| `--model_name` | Model name in API responses |

---

## Step 2: Start a Worker

Open **Terminal B**:

=== "Transformers (CPU)"

    ```bash
    pulsing actor transformers \
      --model gpt2 \
      --device cpu \
      --addr 0.0.0.0:8001 \
      --seeds 127.0.0.1:8000
    ```

=== "vLLM (GPU)"

    ```bash
    pulsing actor vllm \
      --model Qwen/Qwen2.5-0.5B \
      --addr 0.0.0.0:8002 \
      --seeds 127.0.0.1:8000
    ```

| Flag | Description |
|------|-------------|
| `--model` | Model name/path |
| `--seeds` | Router address to join cluster |

---

## Step 3: Verify the Cluster

```bash
# List actors
pulsing actor list --endpoint 127.0.0.1:8000

# Inspect cluster state
pulsing inspect --seeds 127.0.0.1:8000
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
pulsing actor transformers --model gpt2 --addr 0.0.0.0:8003 --seeds 127.0.0.1:8000

# Terminal D
pulsing actor transformers --model gpt2 --addr 0.0.0.0:8004 --seeds 127.0.0.1:8000
```

The Router automatically load-balances across all workers.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No available workers` | Ensure workers use `--seeds <router_addr>` |
| Connection refused | Check router started with `--addr` |
| Slow startup | First request loads model weights |

---

## What's Next?

- [Guide: Operations](../guide/operations.md) — CLI tools in depth
- [Guide: Security](../guide/security.md) — secure your cluster with mTLS
- [Design: Load Sync](../design/load_sync.md) — how load balancing works
