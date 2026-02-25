# CLI Commands

Pulsing ships with built-in CLI tools for starting actors, inspecting systems, and benchmarking distributed services.

---

## Starting Actors

The `pulsing actor` command starts actors by providing their full class path. Arguments are split by `--` so that **actor-level options** (e.g. `--addr`, `--seeds`, `--name`) and **Actor constructor arguments** never collide.

### Parameter separation (`--`)

- **Before `--`**: All arguments are passed to the `actor` subcommand as-is (positional actor type + any options such as `--addr`, `--seeds`, `--name`).
- **After `--`**: All `--key value` pairs are collected and passed to the Actor's constructor. Use this to avoid name collision with actor-level options.

If you omit `--`, only the arguments recognized by the `actor` subcommand are used; constructor args must then be passed via `-D actor.extra_kwargs='{"key":"value"}'` or by using `--` when you need to pass both.

### Format

Actor type must be a full class path:
- Format: `module.path.ClassName`
- Example: `pulsing.serving.Router`
- Example: `pulsing.serving.TransformersWorker`
- Example: `pulsing.serving.VllmWorker`
- Example: `my_module.my_actor.MyCustomActor`

### Examples

#### Router (OpenAI-compatible HTTP API)

```bash
# Actor-level (addr, name) before --; Router constructor args after --
pulsing actor pulsing.serving.Router \
  --addr 0.0.0.0:8000 \
  --name my-llm \
  -- \
  --http_port 8080 \
  --model_name gpt2 \
  --worker_name worker
```

#### Transformers Worker

```bash
pulsing actor pulsing.serving.TransformersWorker \
  --addr 0.0.0.0:8001 \
  --seeds 127.0.0.1:8000 \
  --name worker \
  -- \
  --model_name gpt2
```

#### vLLM Worker

```bash
pulsing actor pulsing.serving.vllm.VllmWorker \
  --addr 0.0.0.0:8002 \
  --seeds 127.0.0.1:8000 \
  --name worker \
  -- \
  --model Qwen/Qwen2 \
  --role aggregated \
  --max_new_tokens 512
```

#### Multiple Workers

```bash
# Start multiple workers with different names
pulsing actor pulsing.serving.TransformersWorker \
  --name worker-1 \
  --seeds 127.0.0.1:8000 \
  -- --model_name gpt2

pulsing actor pulsing.serving.TransformersWorker \
  --name worker-2 \
  --seeds 127.0.0.1:8000 \
  -- --model_name gpt2

# Router targeting specific worker name
pulsing actor pulsing.serving.Router \
  --addr 0.0.0.0:8000 \
  --seeds 127.0.0.1:8000 \
  -- --worker_name worker-1
```

### Common options (before `--`)

- `--name NAME`: Actor name (default: "worker")
- `--addr ADDR`: Actor System bind address
- `--seeds SEEDS`: Comma-separated list of seed nodes

Arguments after `--` are passed to the Actor's constructor as `--key value` pairs.

The Actor class must:
- Be importable from the specified module path
- Be an `pulsing.core.Actor` subclass or a `@pulsing.remote` class
- Have a constructor with named parameters (arguments after `--` are matched to constructor parameters)

**How it works:** The CLI passes everything before `--` to the actor subcommand, and collects every `--key value` after `--` into the Actor constructor. Use `pulsing actor <class> --help` to see actor-level options; for constructor parameters, see the Actor class documentation.

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
| Start router | `pulsing actor pulsing.serving.Router --addr 0.0.0.0:8000 --name my-llm -- --http_port 8080 --model_name gpt2 --worker_name worker` |
| Start worker | `pulsing actor pulsing.serving.TransformersWorker --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000 --name worker -- --model_name gpt2` |
| Chat completions | `curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"gpt2","messages":[{"role":"user","content":"Hello"}],"stream":false}'` |
| Start multiple workers | `pulsing actor ... --name worker-1 --seeds ... -- --model_name gpt2` |
| Router with custom worker | `pulsing actor pulsing.serving.Router --addr 0.0.0.0:8000 -- --worker_name worker-1` |
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
