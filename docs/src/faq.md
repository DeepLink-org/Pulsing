# Frequently Asked Questions

This page addresses common questions and issues users encounter when working with Pulsing.

## General Questions

### What is Pulsing?

Pulsing is a distributed actor framework that provides a communication backbone for building distributed systems, with specialized support for AI applications.

### How does Pulsing differ from Ray?

While Ray focuses on general distributed computing with task-based parallelism, Pulsing specializes in the Actor model with:

- **Location transparency**: Same API for local and remote actors
- **True actor semantics**: Actors process messages one at a time
- **Zero external dependencies**: Pure Rust + Tokio implementation
- **Streaming support**: Native support for streaming responses

### When should I use Pulsing vs Ray?

Choose Pulsing if you need:

- Actor-based programming with location transparency
- Streaming responses (LLM applications)
- Minimal operational complexity (no external services)
- High-performance actor communication

Choose Ray if you need:

- General distributed computing tasks
- Complex dependency management
- Integration with existing Ray ecosystem

## Installation Issues

### ImportError: No module named 'pulsing'

**Problem**: Pulsing package is not installed or not in Python path.

**Solutions**:

1. **Install Pulsing**:
   ```bash
   pip install pulsing
   ```

2. **For development**:
   ```bash
   git clone https://github.com/DeepLink-org/pulsing
   cd pulsing
   pip install -e .
   ```

3. **Check Python path**:
   ```python
   import sys
   print(sys.path)
   ```

### Build failures on macOS/Linux

**Problem**: Rust compilation issues.

**Solutions**:

1. **Install Rust**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Install system dependencies** (Ubuntu/Debian):
   ```bash
   sudo apt-get install build-essential pkg-config libssl-dev
   ```

3. **Install system dependencies** (macOS):
   ```bash
   brew install openssl pkg-config
   ```

## Runtime Issues

### Actor not responding to messages

**Problem**: Actor appears to be stuck or not processing messages.

**Possible causes**:

1. **Blocking operations**: Actor is blocked on synchronous I/O
2. **Infinite loop**: Actor code contains an infinite loop
3. **Deadlock**: Actor is waiting for a message that will never arrive

**Solutions**:

```python
# ❌ Bad: Blocking I/O in actor
@pul.remote
class BadActor:
    def process(self, url):
        response = requests.get(url)  # Blocks the actor!
        return response.text

# ✅ Good: Use async I/O
@pul.remote
class GoodActor:
    async def process(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()
```

### Connection refused errors

**Problem**: Cannot connect to remote actors.

**Possible causes**:

1. **Wrong address**: Actor system listening on different address
2. **Firewall**: Network traffic blocked
3. **TLS issues**: Certificate validation failures

**Solutions**:

1. **Check actor system address**:
   ```python
   # Make sure addresses match
   system1 = await pul.actor_system(addr="0.0.0.0:8000")
   system2 = await pul.actor_system(addr="0.0.0.0:8001", seeds=["127.0.0.1:8000"])
   ```

2. **Disable TLS for testing**:
   ```python
   # For development only
   system = await pul.actor_system(addr="0.0.0.0:8000", passphrase=None)
   ```

### Memory leaks

**Problem**: Memory usage grows over time.

**Possible causes**:

1. **Message accumulation**: Messages not being processed fast enough
2. **Large message payloads**: Messages containing large data structures
3. **Actor leaks**: Actors not being properly cleaned up

**Solutions**:

1. **Monitor mailbox size**:
   ```python
   # Check actor mailbox size
   mailbox_size = await system.get_mailbox_size("actor_name")
   ```

2. **Use streaming for large data**:
   ```python
   @pul.remote
   class StreamingActor:
       async def process_large_data(self, data_stream):
           async for chunk in data_stream:
               # Process chunk by chunk
               yield self.process_chunk(chunk)
   ```

## Performance Issues

### High latency

**Problem**: Message round-trip takes too long.

**Optimizations**:

1. **Use local actors when possible**:
   ```python
   # Local actor (fast)
   local_actor = await MyActor.spawn()

   # Remote actor (slower)
   remote_actor = await MyActor.resolve("remote_actor")
   ```

2. **Batch messages**:
   ```python
   # Instead of multiple calls
   results = []
   for item in items:
       result = await actor.process(item)
       results.append(result)

   # Batch processing
   results = await actor.process_batch(items)
   ```

3. **Use tell() for fire-and-forget**:
   ```python
   # Don't wait for response if not needed
   await actor.log_event(event_data)  # Uses ask() internally
   await actor.tell({"action": "log", "data": event_data})  # Fire-and-forget
   ```

### Serialization overhead

**Problem**: Message serialization is slow.

**Solutions**:

1. **Use efficient data formats**:
   ```python
   # ✅ Good: Use simple types
   await actor.process({"numbers": [1, 2, 3], "text": "hello"})

   # ❌ Bad: Complex nested objects
   await actor.process({"data": very_complex_nested_object})
   ```

2. **Avoid sending large payloads**:
   ```python
   # Send references instead of data
   await actor.process_data(data_id)  # Send ID, not the data itself
   ```

## Deployment Issues

### Clustering not working

**Problem**: Multiple nodes cannot discover each other.

**Solutions**:

1. **Check seed node configuration**:
   ```python
   # Node 1 (seed)
   system1 = await pul.actor_system(addr="192.168.1.100:8000")

   # Node 2 (join cluster)
   system2 = await pul.actor_system(
       addr="192.168.1.101:8000",
       seeds=["192.168.1.100:8000"]
   )
   ```

2. **Verify network connectivity**:
   ```bash
   # Test if ports are open
   telnet 192.168.1.100 8000
   ```

3. **Check firewall settings**:
   ```bash
   # Linux
   sudo ufw status
   sudo ufw allow 8000

   # macOS
   sudo pfctl -s rules
   ```

### Load balancing issues

**Problem**: Requests not distributed evenly across cluster.

**Solutions**:

1. **Use round-robin resolution**:
   ```python
   # Default behavior distributes across instances
   actor = await MyActor.resolve("service_name")
   ```

2. **Check actor distribution**:
   ```python
   # Monitor cluster membership
   members = await system.members()
   print(f"Cluster has {len(members)} nodes")
   ```

## Migration Issues

### Migrating from Ray

**Common issues**:

1. **API differences**:
   ```python
   # Ray
   @ray.remote
   class MyActor:
       def __init__(self, value):
           self.value = value

   actor = MyActor.remote(42)
   result = ray.get(actor.method.remote())

   # Pulsing
   @pul.remote
   class MyActor:
       def __init__(self, value):
           self.value = value

   actor = await MyActor.spawn(value=42)
   result = await actor.method()
   ```

2. **Async/await everywhere**:
   ```python
   # Pulsing requires async/await
   async def main():
       await pul.init()
       actor = await MyActor.spawn()
       result = await actor.method()
       await pul.shutdown()

   asyncio.run(main())
   ```

## Getting Help

If you can't find the answer here:

1. **Check the documentation**: [User Guide](../guide/) and [API Reference](../api/overview.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/DeepLink-org/pulsing/issues)
3. **Ask the community**: [GitHub Discussions](https://github.com/DeepLink-org/pulsing/discussions)
4. **File a bug report**: If you found a bug, please [open an issue](https://github.com/DeepLink-org/pulsing/issues/new)

## Contributing

Found an issue with this FAQ? [Help improve it!](https://github.com/DeepLink-org/pulsing/blob/main/docs/src/faq.md)
