import asyncio


def test_ray_compat_init_inside_running_loop():
    """ray.init() should work even when called from within a running event loop.

    This covers environments like Jupyter or pytest-asyncio where an event loop
    is already running on the main thread.
    """
    from pulsing.integrations.ray_compat import ray

    async def main():
        ray.init()
        try:

            @ray.remote
            class Counter:
                def __init__(self, value=0):
                    self.value = value

                def inc(self, n=1):
                    self.value += n
                    return self.value

            c = Counter.remote(value=1)
            assert ray.get(c.inc.remote()) == 2
            assert ray.get(c.inc.remote(10)) == 12
        finally:
            ray.shutdown()

    asyncio.run(main())
