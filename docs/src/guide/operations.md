# Operations (CLI)

This page is a practical entry point for operating and inspecting Pulsing systems using the built-in CLI.

## What you can do

- **Run services**: start a router or inference workers
- **Inspect a cluster**: view nodes + named actors
- **List actors**: query actors via HTTP (observer mode)
- **Benchmark**: run load tests against an OpenAI-compatible endpoint

## Commands

## Quick links

- [Actor List](actor_list.md)
- [Inspect](inspect.md)
- [Bench](bench.md)

### Start services (router / workers)

- Router (OpenAI-compatible HTTP API):

```bash
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm
```

- Transformers worker:

```bash
pulsing actor transformers --model gpt2 --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000
```

- vLLM worker:

```bash
pulsing actor vllm --model Qwen/Qwen2 --addr 0.0.0.0:8002 --seeds 127.0.0.1:8000
```

### Inspect cluster

```bash
pulsing inspect --seeds 127.0.0.1:8000
```

### List actors (observer mode)

```bash
# single node
pulsing actor list --endpoint 127.0.0.1:8000

# cluster (via seeds)
pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001
```

### Benchmark an endpoint

```bash
pulsing bench gpt2 --url http://localhost:8080
```

## Next

- For a runnable end-to-end guide, see [LLM Inference](../examples/llm_inference.md).

