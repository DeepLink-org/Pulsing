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

---

## Inspect

`pulsing inspect` is a lightweight **observer** tool that queries actor systems via HTTP (no cluster join required). It provides multiple subcommands for different inspection needs.

### Subcommands

#### Cluster Status

Inspect cluster members and their status:

```bash
pulsing inspect cluster --seeds 127.0.0.1:8000
```

Output includes:
- Total nodes and alive count
- Status summary (Alive, Suspect, Failed, etc.)
- Detailed member list with node ID, address, and status

#### Actors Distribution

Inspect named actors distribution across the cluster:

```bash
pulsing inspect actors --seeds 127.0.0.1:8000
```

Options:
- `--top N`: Show top N actors by instance count
- `--filter STR`: Filter actor names by substring
- `--all_actors True`: Include internal/system actors

Examples:
```bash
# Show top 10 actors
pulsing inspect actors --seeds 127.0.0.1:8000 --top 10

# Filter actors by name
pulsing inspect actors --seeds 127.0.0.1:8000 --filter worker
```

#### Metrics

Inspect Prometheus metrics from cluster nodes:

```bash
pulsing inspect metrics --seeds 127.0.0.1:8000
```

Options:
- `--raw True`: Output raw metrics (default)
- `--raw False`: Show summary only (key metrics)

#### Watch Mode

Watch cluster state changes in real-time:

```bash
pulsing inspect watch --seeds 127.0.0.1:8000
```

Options:
- `--interval 1.0`: Refresh interval in seconds (default: 1.0)
- `--kind all`: What to watch: `cluster`, `actors`, `metrics`, or `all` (default: `all`)
- `--max_rounds N`: Maximum number of refresh rounds (None = infinite)

Examples:
```bash
# Watch cluster member changes
pulsing inspect watch --seeds 127.0.0.1:8000 --kind cluster --interval 2.0

# Watch actor changes
pulsing inspect watch --seeds 127.0.0.1:8000 --kind actors
```

### Common Options

All subcommands support:

- `--timeout 10.0`: Request timeout in seconds (default: 10.0)
- `--best_effort True`: Continue even if some nodes fail (default: False)

### Legacy Mode

The old join-based mode is still available for backward compatibility:

```bash
pulsing inspect --seeds 127.0.0.1:8000
```

This defaults to `pulsing inspect cluster --seeds 127.0.0.1:8000` but uses the join-based implementation.

!!! note
    Observer mode (default) uses HTTP/2 (h2c) and does NOT join the gossip cluster, making it lightweight and suitable for production monitoring.

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
| List actors | `pulsing inspect actors --endpoint 127.0.0.1:8000` |
| Inspect cluster | `pulsing inspect cluster --seeds 127.0.0.1:8000` |
| Inspect actors | `pulsing inspect actors --seeds 127.0.0.1:8000 --top 10` |
| Inspect metrics | `pulsing inspect metrics --seeds 127.0.0.1:8000` |
| Watch cluster | `pulsing inspect watch --seeds 127.0.0.1:8000` |
| Benchmark | `pulsing bench gpt2 --url http://localhost:8080` |

---

## Next Steps

- [LLM Inference](../examples/llm_inference.md) - runnable end-to-end tutorial
- [Security](security.md) - mTLS and cluster isolation
