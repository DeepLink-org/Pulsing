# Terminology & Style Guide

This page defines terminology and style conventions for Pulsing documentation and code to ensure consistency.

## Core Terminology

| Term | Usage | Description |
|------|-------|-------------|
| `ActorSystem` | Code symbol | Rust/Python class name, use in code references |
| actor system | Conceptual | General description, lowercase |
| `Actor` | Code symbol | Base class name |
| actor | Conceptual | General description |
| `ActorRef` | Code symbol | Low-level actor reference |
| `ActorProxy` | Code symbol | High-level proxy returned by `@remote` decorator |

## Component Naming

| Component | CLI Actor Class Path | Description |
|-----------|---------------------|-------------|
| Router | `pulsing.serving.Router` | OpenAI-compatible HTTP router |
| TransformersWorker | `pulsing.serving.TransformersWorker` | Transformers inference worker |
| VllmWorker | `pulsing.serving.VllmWorker` | vLLM inference worker |

**Note**: When documentation mentions "Router", it typically refers to the HTTP routing component for LLM inference services. Example code requiring task dispatch logic should use names like `Dispatcher` to avoid confusion.

## CLI Command Format

### Starting Actors

```bash
pulsing actor <full.class.Path> [options]

# Examples
pulsing actor pulsing.serving.Router --http_port 8080 --model_name my-llm
pulsing actor pulsing.serving.TransformersWorker --model_name gpt2 --device cpu
```

### Inspect Commands (Observer Mode)

`pulsing inspect` uses a subcommand structure:

```bash
# Cluster status
pulsing inspect cluster --seeds <address>

# Actor distribution
pulsing inspect actors --seeds <address> [--top N] [--filter ...]
pulsing inspect actors --endpoint <address> [--detailed]

# Metrics
pulsing inspect metrics --seeds <address> [--raw]

# Live watch
pulsing inspect watch --seeds <address> [--interval ...] [--kind ...]
```

**Removed**: `pulsing actor list` → Use `pulsing inspect actors` instead

## Code Style

- **Python**: Follow PEP 8, prefer type annotations
- **Rust**: Use `cargo fmt` and `cargo clippy`
- **Documentation**: Bilingual (`.zh.md` suffix for Chinese)

## Documentation References

- Use backticks for code symbols: `` `ActorSystem` ``
- Use code blocks for commands
- Use backticks for file paths: `` `python/pulsing/actor/` ``
