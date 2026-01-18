# CLI Commands

Pulsing ships with built-in CLI tools for starting actors, inspecting systems, and benchmarking distributed services.

---

## Starting Actors

The `pulsing actor` command starts actors by providing their full class path. The CLI automatically matches command-line arguments to the Actor's constructor parameters.

### Format

Actor type must be a full class path:
- Format: `module.path.ClassName`
- Example: `pulsing.actors.Router`
- Example: `pulsing.actors.TransformersWorker`
- Example: `pulsing.actors.VllmWorker`
- Example: `my_module.my_actor.MyCustomActor`

### Examples

#### Router (OpenAI-compatible HTTP API)

```bash
pulsing actor pulsing.actors.Router \
  --addr 0.0.0.0:8000 \
  --http_host 0.0.0.0 \
  --http_port 8080 \
  --model_name my-llm \
  --worker_name worker \
  --scheduler_type stream_load
```

#### Transformers Worker

```bash
pulsing actor pulsing.actors.worker.TransformersWorker \
  --model_name gpt2 \
  --device cpu \
  --addr 0.0.0.0:8001 \
  --seeds 127.0.0.1:8000 \
  --name worker
```

#### vLLM Worker

```bash
pulsing actor pulsing.actors.vllm.VllmWorker \
  --model Qwen/Qwen2 \
  --addr 0.0.0.0:8002 \
  --seeds 127.0.0.1:8000 \
  --name worker \
  --role aggregated \
  --max_new_tokens 512
```

#### Multiple Workers

```bash
# Start multiple workers with different names
pulsing actor pulsing.actors.worker.TransformersWorker \
  --model_name gpt2 \
  --name worker-1 \
  --seeds 127.0.0.1:8000

pulsing actor pulsing.actors.worker.TransformersWorker \
  --model_name gpt2 \
  --name worker-2 \
  --seeds 127.0.0.1:8000

# Router targeting specific worker name
pulsing actor pulsing.actors.Router \
  --worker_name worker-1 \
  --seeds 127.0.0.1:8000
```

### Common Options

- `--name NAME`: Actor name (default: "worker")
- `--addr ADDR`: Actor System bind address
- `--seeds SEEDS`: Comma-separated list of seed nodes
- Any other `--param value` pairs matching the Actor's constructor signature

### How It Works

```bash
# Pass parameters directly as command-line arguments
pulsing actor pulsing.actors.worker.TransformersWorker \
  --model_name gpt2 \
  --device cpu \
  --preload true \
  --name my-worker \
  --seeds 127.0.0.1:8000

# Start vLLM worker with all parameters
pulsing actor pulsing.actors.vllm.VllmWorker \
  --model Qwen/Qwen2 \
  --role aggregated \
  --max_new_tokens 512 \
  --name vllm-worker \
  --seeds 127.0.0.1:8000
```

Options:
- `--name NAME`: Actor name (default: "worker")
- `--addr ADDR`: Actor System bind address
- `--seeds SEEDS`: Comma-separated list of seed nodes
- Any other `--param value` pairs matching the Actor's constructor signature

The Actor class must:
- Be importable from the specified module path
- Inherit from `pulsing.actor.Actor`
- Have a constructor with named parameters (the CLI automatically matches arguments to constructor parameters)

**How it works:**
The CLI inspects the Actor class constructor signature and automatically extracts matching parameters from command-line arguments. You can use `--help` to see available parameters, or check the Actor class documentation.

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

!!! note
    Observer mode uses HTTP/2 (h2c) and does NOT join the gossip cluster, making it lightweight and suitable for production monitoring.

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
| Start router | `pulsing actor pulsing.actors.Router --addr 0.0.0.0:8000 --http_port 8080` |
| Start worker | `pulsing actor pulsing.actors.TransformersWorker --model_name gpt2 --seeds ...` |
| Start multiple workers | `pulsing actor pulsing.actors.TransformersWorker --model_name gpt2 --name worker-1 --seeds ...` |
| Router with custom worker | `pulsing actor pulsing.actors.Router --worker_name worker-1 --seeds ...` |
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
