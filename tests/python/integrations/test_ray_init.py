"""
Tests for pulsing.ray - Pulsing initialization in Ray cluster

Tests:
- init_in_ray() basic behavior
- Seed registration via Ray KV store
- Multi-actor cluster formation
- async_init_in_ray()
- cleanup()
- Error cases
"""

import pytest

# Skip all tests if ray is not installed
ray = pytest.importorskip("ray")

from pulsing._async_bridge import (
    clear_pulsing_loop,
    get_shared_loop,
    stop_shared_loop,
    submit_on_shared_loop,
)


def _reset_pulsing_state():
    """Reset all Pulsing module state (system, background loop, KV)."""
    import pulsing.core as pc
    import pulsing.integrations.ray as pray

    # Shutdown Pulsing system via background loop
    if pc._global_system is not None and get_shared_loop() is not None:
        try:
            submit_on_shared_loop(pray._do_shutdown(), timeout=30)
        except Exception:
            pass

    # Force clear global system (safety net)
    pc._global_system = None

    stop_shared_loop()
    clear_pulsing_loop()

    # Clean KV store
    try:
        pray.cleanup()
    except Exception:
        pass


NUM_WORKERS = 20


@pytest.fixture
def ray_env():
    """Initialize local Ray cluster with clean Pulsing state."""
    ray.init(num_cpus=NUM_WORKERS + 1)
    _reset_pulsing_state()  # ensure clean state before test
    yield
    _reset_pulsing_state()  # cleanup after test
    ray.shutdown()


# ============================================================================
# Test: init_in_ray() basic
# ============================================================================


def test_init_returns_system(ray_env):
    """init_in_ray() returns a Pulsing ActorSystem."""
    from pulsing.integrations.ray import init_in_ray

    system = init_in_ray()
    assert system is not None
    assert system.addr is not None


def test_init_stores_seed_in_kv(ray_env):
    """First caller's address is stored as seed in Ray KV."""
    from pulsing.integrations.ray import _get_seed, init_in_ray

    system = init_in_ray()
    seed_addr = _get_seed()
    assert seed_addr is not None
    assert seed_addr == str(system.addr)


def test_init_sets_global_system(ray_env):
    """init_in_ray() sets pulsing.actor global system."""
    from pulsing.core import is_initialized
    from pulsing.integrations.ray import init_in_ray

    assert not is_initialized()
    init_in_ray()
    assert is_initialized()


# ============================================================================
# Test: error cases
# ============================================================================


def test_init_raises_without_ray():
    """init_in_ray() raises when Ray is not initialized."""
    from pulsing.integrations.ray import init_in_ray

    with pytest.raises(RuntimeError, match="Ray not initialized"):
        init_in_ray()


async def test_async_init_raises_without_ray():
    """async_init_in_ray() raises when Ray is not initialized."""
    from pulsing.integrations.ray import async_init_in_ray

    with pytest.raises(RuntimeError, match="Ray not initialized"):
        await async_init_in_ray()


# ============================================================================
# Test: cleanup()
# ============================================================================


def test_cleanup_clears_kv(ray_env):
    """cleanup() removes seed from KV store."""
    from pulsing.integrations.ray import _get_seed, cleanup, init_in_ray

    init_in_ray()
    assert _get_seed() is not None

    cleanup()
    assert _get_seed() is None


# ============================================================================
# Test: Ray actor integration
# ============================================================================


def test_init_in_ray_actor(ray_env):
    """init_in_ray() works inside a Ray actor."""

    @ray.remote
    class Worker:
        def setup(self):
            from pulsing.integrations.ray import init_in_ray

            system = init_in_ray()
            return str(system.addr)

        def ping(self):
            return "pong"

    worker = Worker.remote()
    addr = ray.get(worker.setup.remote())
    assert addr is not None
    assert ":" in addr

    result = ray.get(worker.ping.remote())
    assert result == "pong"


def test_multi_actor_same_seed(ray_env):
    """All workers in separate processes discover the same seed."""
    import os

    from pulsing.integrations.ray import _get_seed, init_in_ray

    driver_pid = os.getpid()

    # Driver becomes seed
    init_in_ray()
    seed_addr = _get_seed()

    @ray.remote
    class Worker:
        def setup(self):
            import os

            from pulsing.integrations.ray import init_in_ray

            init_in_ray()
            return os.getpid()

        def get_seed(self):
            from pulsing.integrations.ray import _get_seed

            return _get_seed()

    workers = [Worker.remote() for _ in range(NUM_WORKERS)]
    pids = ray.get([w.setup.remote() for w in workers])

    # Verify multi-process: all PIDs different from driver
    assert all(pid != driver_pid for pid in pids), (
        "Workers should run in separate processes"
    )

    # Verify multi-process: workers are in distinct processes
    unique_pids = set(pids)
    assert len(unique_pids) == NUM_WORKERS, (
        f"Expected {NUM_WORKERS} distinct processes, got {len(unique_pids)}"
    )

    # All workers see the same seed
    seeds = ray.get([w.get_seed.remote() for w in workers])
    assert all(s == seed_addr for s in seeds)


def test_concurrent_init_without_driver(ray_env):
    """20 processes concurrently call init_in_ray(), exactly one becomes seed."""
    import os

    @ray.remote
    class Worker:
        def setup(self):
            import os

            from pulsing.integrations.ray import init_in_ray

            system = init_in_ray()
            return os.getpid(), str(system.addr)

        def get_seed(self):
            from pulsing.integrations.ray import _get_seed

            return _get_seed()

    # Launch all workers at once — they race to become seed
    workers = [Worker.remote() for _ in range(NUM_WORKERS)]
    results = ray.get([w.setup.remote() for w in workers])
    pids = [r[0] for r in results]
    addrs = [r[1] for r in results]

    # Verify multi-process: all workers in distinct processes
    unique_pids = set(pids)
    assert len(unique_pids) == NUM_WORKERS, (
        f"Expected {NUM_WORKERS} distinct processes, got {len(unique_pids)}"
    )
    # None should be the driver
    assert os.getpid() not in unique_pids

    # All workers got a valid address
    assert len(addrs) == NUM_WORKERS
    assert all(a and ":" in a for a in addrs)

    # All workers see the same seed
    seeds = ray.get([w.get_seed.remote() for w in workers])
    unique_seeds = set(seeds)
    assert len(unique_seeds) == 1, (
        f"Expected 1 seed, got {len(unique_seeds)}: {unique_seeds}"
    )

    # The seed must be one of the workers' addresses
    seed = unique_seeds.pop()
    assert seed in addrs


def test_actor_becomes_seed_without_driver(ray_env):
    """When driver doesn't init, first actor becomes seed."""

    @ray.remote
    class Worker:
        def setup(self):
            from pulsing.integrations.ray import init_in_ray

            system = init_in_ray()
            return str(system.addr)

        def get_seed(self):
            from pulsing.integrations.ray import _get_seed

            return _get_seed()

    # First actor becomes seed
    w1 = Worker.remote()
    addr1 = ray.get(w1.setup.remote())
    seed = ray.get(w1.get_seed.remote())
    assert seed == addr1

    # Second actor joins
    w2 = Worker.remote()
    ray.get(w2.setup.remote())
    seed2 = ray.get(w2.get_seed.remote())
    assert seed2 == seed


# ============================================================================
# Test: async_init_in_ray()
# ============================================================================


async def test_async_init_returns_system(ray_env):
    """async_init_in_ray() returns a system."""
    from pulsing.integrations.ray import async_init_in_ray

    system = await async_init_in_ray()
    assert system is not None
    assert system.addr is not None


async def test_async_init_stores_seed(ray_env):
    """async_init_in_ray() stores seed in KV."""
    from pulsing.integrations.ray import _get_seed, async_init_in_ray

    system = await async_init_in_ray()
    assert _get_seed() == str(system.addr)


# ============================================================================
# Test: counting game (end-to-end Pulsing messaging across Ray workers)
# ============================================================================


def test_counting_game(ray_env):
    """20 processes play counting game via Pulsing actor (reuses pulsing.examples)."""
    from pulsing.examples.counting_game import run

    run(num_workers=NUM_WORKERS)
