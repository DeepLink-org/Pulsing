# Pulsing Actor System Examples

This directory contains examples of using the Pulsing Actor System for distributed LLM inference.

## Example Files

### 1. ping_pong.py - Basic Actor Communication

Simple example demonstrating actor message passing patterns.

```bash
python examples/actor/ping_pong.py
```

**Features:**
- Synchronous and asynchronous actors
- `ask` (request-response) pattern
- `tell` (fire-and-forget) pattern
- Actor lifecycle (`on_start`, `on_stop`)

### 2. cluster.py - Distributed Cluster

Multi-node actor cluster with gossip-based discovery.

```bash
# Terminal 1 - Start first node
python examples/actor/cluster.py --port 8000

# Terminal 2 - Start second node and join cluster
python examples/actor/cluster.py --port 8001 --seed 127.0.0.1:8000
```

**Features:**
- Multi-node cluster formation
- Service discovery via gossip protocol
- Cross-node actor communication
- Cluster membership monitoring

## CLI Usage

### 1. Router

HTTP server with OpenAI-compatible API and load balancing (RoundRobin scheduler).

```bash
# Start Router
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm
```

### 2. Worker

HuggingFace Transformers-based inference worker with streaming support.

```bash
# Start Worker (GPU)
pulsing actor transformers --model gpt2 --addr 127.0.0.1:8001 --seeds 127.0.0.1:8000

# Start Worker (CPU)
pulsing actor transformers --model gpt2 --device cpu --addr 127.0.0.1:8001 --seeds 127.0.0.1:8000

# Start Worker with larger model
pulsing actor transformers --model Qwen/Qwen2.5-0.5B --addr 127.0.0.1:8001 --seeds 127.0.0.1:8000
```

## Deployment Examples

### Single Machine

```bash
# Terminal 1: Start Router
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm

# Terminal 2: Start Worker 1
pulsing actor transformers --model gpt2 --addr 127.0.0.1:8001 --seeds 127.0.0.1:8000

# Terminal 3: Start Worker 2
pulsing actor transformers --model gpt2 --addr 127.0.0.1:8002 --seeds 127.0.0.1:8000
```

### Multi-Machine Cluster

```bash
# Machine A (Router)
pulsing actor router --addr 0.0.0.0:8000 --http_port 8080 --model_name my-llm

# Machine B (Worker 1)
pulsing actor transformers --model gpt2 --addr 0.0.0.0:8001 --seeds 192.168.1.A:8000

# Machine C (Worker 2)
pulsing actor transformers --model gpt2 --addr 0.0.0.0:8001 --seeds 192.168.1.A:8000
```

## Testing

### Chat Completions (Non-streaming)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-llm",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "stream": false
  }'
```

### Chat Completions (Streaming)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-llm",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Completions API

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-llm",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "stream": true
  }'
```

### Health Check

```bash
curl http://localhost:8080/health
```

## Programming API

### Basic Actor (from ping_pong.py)

```python
import asyncio
from pulsing.actor import create_actor_system, SystemConfig, Message, Actor, ActorId

class PingPongActor(Actor):
    def __init__(self):
        self.count = 0
    
    def on_start(self, actor_id: ActorId):
        print(f"Actor started: {actor_id.name}")
    
    def receive(self, msg: Message) -> Message:
        if msg.msg_type == "Ping":
            self.count += 1
            return Message.from_json("Pong", {"count": self.count})
        return Message.empty()

async def main():
    config = SystemConfig.standalone()
    system = await create_actor_system(config)
    actor = await system.spawn("counter", PingPongActor())
    
    # Request-response
    response = await actor.ask_json("Ping", {})
    print(f"Count: {response['count']}")
    
    await system.shutdown()

asyncio.run(main())
```

### LLM Inference (Router + Worker)

```python
import asyncio
from pulsing.actors import TransformersWorker, start_router, stop_router
from pulsing.actor.helpers import run_until_signal
from pulsing.actor import create_actor_system, SystemConfig

async def main():
    # Start Router
    router_config = SystemConfig.with_addr("0.0.0.0:8000")
    router_system = await create_actor_system(router_config)
    runner = await start_router(
        router_system,
        http_host="0.0.0.0",
        http_port=8080,
        model_name="my-llm"
    )
    
    # Start Worker
    worker_config = SystemConfig.with_addr("0.0.0.0:8001").with_seeds(["127.0.0.1:8000"])
    worker_system = await create_actor_system(worker_config)
    worker = TransformersWorker(model_name="gpt2", device="cpu")
    worker_ref = await worker_system.spawn("worker", worker, public=True)
    
    # Run until signal
    await run_until_signal(worker_system, "worker")
    
    # Cleanup
    await stop_router(runner)
    await router_system.shutdown()
    await worker_system.shutdown()

asyncio.run(main())
```

### Message Protocol

#### Worker Messages

| Message Type | Description | Request Fields | Response Fields |
|-------------|-------------|----------------|-----------------|
| GenerateRequest | Synchronous generation | prompt, max_new_tokens | text, prompt_tokens, completion_tokens |
| GenerateStreamRequest | Streaming generation | prompt, max_new_tokens | Stream of JSON chunks with text, finish_reason |
| HealthCheck | Health status | - | status, worker_id, is_loaded |

**GenerateStreamRequest Response Format:**

Each chunk is a JSON object:
```json
{"text": "token", "finish_reason": null}
{"text": "", "finish_reason": "length"}
```

## Architecture

```
                    ┌──────────────┐
                    │    Client    │
                    └──────┬───────┘
                           │ HTTP/1.1 (OpenAI API)
                           ▼
                    ┌──────────────┐
                    │    Router    │
                    │  (HTTP/SSE)  │
                    └──────┬───────┘
                           │ HTTP/2 (Actor System)
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Worker 1  │ │   Worker 2  │ │   Worker 3  │
    │(Transformers)│ │(Transformers)│ │(Transformers)│
    └─────────────┘ └─────────────┘ └─────────────┘
```

**Key Features:**
- **Load Balancing**: RoundRobin scheduler selects workers on demand
- **Service Discovery**: Gossip-based cluster membership via Rust Actor System
- **Streaming**: HTTP/2 for inter-actor streaming, SSE for client-facing API
- **Dynamic Scaling**: Workers can join/leave cluster without restart

## Learn More

- See `ping_pong.py` for basic actor patterns
- See `cluster.py` for distributed cluster setup
- See `../cli/README.md` for more CLI options
