# Pulsing Examples

## Rust Examples

```bash
# Basic actor communication
cargo run --example ping_pong -p pulsing-actor

# Message patterns (RPC, streaming)
cargo run --example message_patterns -p pulsing-actor

# Named actors and service discovery
cargo run --example named_actors -p pulsing-actor

# Multi-node cluster (run in two terminals)
cargo run --example cluster -p pulsing-actor -- --node 1
cargo run --example cluster -p pulsing-actor -- --node 2
```

## Python Examples

```bash
# Basic actor communication
python examples/python/ping_pong.py

# Message patterns (RPC, streaming)
python examples/python/message_patterns.py

# Named actors and service discovery
python examples/python/named_actors.py

# Multi-node cluster (run in two terminals)
python examples/python/cluster.py --port 8000
python examples/python/cluster.py --port 8001 --seed 127.0.0.1:8000
```

## Example Overview

| Example | Features |
|---------|----------|
| `ping_pong` | Actor lifecycle, ask/tell patterns |
| `message_patterns` | RPC, server streaming, client streaming |
| `named_actors` | Service discovery, ActorPath, ActorAddress |
| `cluster` | Multi-node, gossip, remote messaging |
