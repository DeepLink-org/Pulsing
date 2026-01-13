# CLI Operations

Pulsing ships with built-in CLI tools for running, inspecting, and benchmarking distributed systems.

---

## Running Services

### Router (OpenAI-compatible HTTP API)

```bash
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm
```

### Transformers Worker

```bash
pulsing actor transformers --model gpt2 --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000
```

### vLLM Worker

```bash
pulsing actor vllm --model Qwen/Qwen2 --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000
```

---

## Actor List

`pulsing actor list` is a lightweight **observer** that queries actors via HTTP (no cluster join required).

### Single Node

```bash
pulsing actor list --endpoint 127.0.0.1:8000
```

### Cluster (via Seeds)

```bash
pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001
```

### Options

| Flag | Description |
|------|-------------|
| `--all_actors True` | Include internal/system actors |
| `--json True` | Output as JSON |

!!! note
    Uses HTTP/2 (h2c). Node must expose HTTP endpoints.

---

## Inspect

`pulsing inspect` joins a cluster (via seeds) and prints a human-friendly snapshot of members and actors.

```bash
pulsing inspect --seeds 127.0.0.1:8000
```

Output includes:

- **Cluster members**: node id, addr, status
- **Named actors**: distribution across nodes

!!! tip
    For local seeds (`127.0.0.1`), the CLI auto-binds to `127.0.0.1:0`.

---

## Bench

`pulsing bench` runs load tests against an OpenAI-compatible inference endpoint.

```bash
pulsing bench gpt2 --url http://localhost:8080
```

!!! note "Optional Extension"
    If you see `pulsing._bench module not found`:

    ```bash
    maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml
    ```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start router | `pulsing actor router --addr 0.0.0.0:8000 --http_port 8080` |
| Start worker | `pulsing actor transformers --model gpt2 --seeds ...` |
| List actors | `pulsing actor list --endpoint 127.0.0.1:8000` |
| Inspect cluster | `pulsing inspect --seeds 127.0.0.1:8000` |
| Benchmark | `pulsing bench gpt2 --url http://localhost:8080` |

---

## Next Steps

- [LLM Inference](../examples/llm_inference.md) - runnable end-to-end tutorial
- [Security](security.md) - mTLS and cluster isolation
