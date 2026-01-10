"""
Pulsing Compatibility Layer

Provides Ray-compatible API for easy migration.

Usage:
    from pulsing.compat import ray

    ray.init()

    @ray.remote
    class Counter:
        def __init__(self, init=0): self.value = init
        def incr(self): self.value += 1; return self.value

    counter = Counter.remote(init=10)
    result = ray.get(counter.incr.remote())

    ray.shutdown()
"""

from . import ray

__all__ = ["ray"]
