"""
Pulsing - Distributed Actor Framework

Two API styles:

1. Native async API (recommended):
    from pulsing.actor import init, shutdown, remote

    await init()

    @remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = await Counter.spawn(init=10)
    result = await counter.incr()

    await shutdown()

2. Ray-compatible sync API (for migration):
    from pulsing.compat import ray

    ray.init()

    @ray.remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = Counter.remote(init=10)
    result = ray.get(counter.incr.remote())

    ray.shutdown()

Submodules:
- pulsing.actor: Native async API (recommended)
- pulsing.compat.ray: Ray-compatible sync API (for migration)
"""

__version__ = "0.1.0"
