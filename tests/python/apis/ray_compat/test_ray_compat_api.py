"""
Tests for Ray Compatible API (llms.binding.md)

Covers:
- ray.init() and ray.shutdown()
- ray.is_initialized()
- @ray.remote decorator
- MyActor.remote() -> _ActorHandle
- actor_handle.method.remote() -> ObjectRef
- ray.get() single and list
- ray.put()
- ray.wait()
"""

import pytest
import time

from pulsing.integrations.ray_compat import ray


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def initialized_ray():
    """Initialize ray-compatible system for testing."""
    ray.init()
    yield
    ray.shutdown()


# ============================================================================
# Test: ray.init() and ray.shutdown()
# ============================================================================


def test_init_and_shutdown():
    """Test ray.init() and ray.shutdown()."""
    ray.init()
    assert ray.is_initialized()
    ray.shutdown()
    assert not ray.is_initialized()


def test_init_ignore_reinit_error():
    """Test ray.init(ignore_reinit_error=True)."""
    ray.init()
    # Should not raise
    ray.init(ignore_reinit_error=True)
    ray.shutdown()


def test_init_raises_on_reinit():
    """Test ray.init() raises if already initialized."""
    ray.init()
    try:
        with pytest.raises(RuntimeError):
            ray.init()
    finally:
        ray.shutdown()


# ============================================================================
# Test: ray.is_initialized()
# ============================================================================


def test_is_initialized_false():
    """Test ray.is_initialized() returns False when not initialized."""
    assert not ray.is_initialized()


def test_is_initialized_true(initialized_ray):
    """Test ray.is_initialized() returns True when initialized."""
    assert ray.is_initialized()


# ============================================================================
# Test: @ray.remote decorator
# ============================================================================


@ray.remote
class Counter:
    """Counter actor for testing."""

    def __init__(self, init=0):
        self.value = init

    def incr(self):
        self.value += 1
        return self.value

    def decr(self):
        self.value -= 1
        return self.value

    def get(self):
        return self.value

    def add(self, n):
        self.value += n
        return self.value


def test_remote_decorator_class(initialized_ray):
    """Test @ray.remote decorator creates actor class wrapper."""
    # Counter should have .remote() method
    assert hasattr(Counter, "remote")


def test_remote_actor_creation(initialized_ray):
    """Test MyActor.remote() creates actor handle."""
    handle = Counter.remote(init=10)
    assert handle is not None


# ============================================================================
# Test: actor_handle.method.remote() -> ObjectRef
# ============================================================================


def test_method_remote_returns_objectref(initialized_ray):
    """Test actor_handle.method.remote() returns ObjectRef."""
    handle = Counter.remote(init=0)
    ref = handle.incr.remote()
    assert ref is not None
    # ObjectRef should have _get_sync method
    assert hasattr(ref, "_get_sync")


def test_method_with_args(initialized_ray):
    """Test calling method with arguments."""
    handle = Counter.remote(init=0)
    ref = handle.add.remote(5)
    result = ray.get(ref)
    assert result == 5


# ============================================================================
# Test: ray.get() - single and list
# ============================================================================


def test_get_single(initialized_ray):
    """Test ray.get() with single ObjectRef."""
    handle = Counter.remote(init=100)
    ref = handle.get.remote()
    result = ray.get(ref)
    assert result == 100


def test_get_list(initialized_ray):
    """Test ray.get() with list of ObjectRefs."""
    handle = Counter.remote(init=0)
    refs = [handle.incr.remote() for _ in range(5)]
    results = ray.get(refs)
    assert len(results) == 5
    # Last result should be 5 (incremented 5 times)
    assert results[-1] == 5


def test_get_with_timeout(initialized_ray):
    """Test ray.get() with timeout parameter."""
    handle = Counter.remote(init=0)
    ref = handle.get.remote()
    result = ray.get(ref, timeout=5.0)
    assert result == 0


def test_get_multiple_actors(initialized_ray):
    """Test ray.get() with refs from multiple actors."""
    h1 = Counter.remote(init=10)
    h2 = Counter.remote(init=20)
    h3 = Counter.remote(init=30)

    refs = [h1.get.remote(), h2.get.remote(), h3.get.remote()]
    results = ray.get(refs)
    assert results == [10, 20, 30]


# ============================================================================
# Test: ray.put()
# ============================================================================


def test_put_value(initialized_ray):
    """Test ray.put() wraps value as ObjectRef."""
    ref = ray.put(42)
    assert ref is not None
    result = ray.get(ref)
    assert result == 42


def test_put_complex_value(initialized_ray):
    """Test ray.put() with complex value."""
    data = {"key": "value", "numbers": [1, 2, 3]}
    ref = ray.put(data)
    result = ray.get(ref)
    assert result == data


def test_put_list_of_refs(initialized_ray):
    """Test ray.put() and ray.get() with list."""
    refs = [ray.put(i) for i in range(5)]
    results = ray.get(refs)
    assert results == [0, 1, 2, 3, 4]


# ============================================================================
# Test: ray.wait()
# ============================================================================


def test_wait_basic(initialized_ray):
    """Test ray.wait() returns ready and remaining."""
    handle = Counter.remote(init=0)
    refs = [handle.incr.remote() for _ in range(3)]

    ready, remaining = ray.wait(refs, num_returns=1)
    assert len(ready) >= 1
    assert len(ready) + len(remaining) == 3


def test_wait_num_returns(initialized_ray):
    """Test ray.wait() with num_returns parameter."""
    handle = Counter.remote(init=0)
    refs = [handle.incr.remote() for _ in range(5)]

    ready, remaining = ray.wait(refs, num_returns=3)
    # At most 3 ready (depends on timing)
    assert len(ready) <= 3


def test_wait_with_put_refs(initialized_ray):
    """Test ray.wait() with ray.put() refs (immediately ready)."""
    refs = [ray.put(i) for i in range(5)]

    ready, remaining = ray.wait(refs, num_returns=5)
    # put() refs are immediately ready
    assert len(ready) == 5
    assert len(remaining) == 0


# ============================================================================
# Test: Full workflow
# ============================================================================


def test_full_workflow(initialized_ray):
    """Test complete Ray-compatible workflow."""
    # Create actors
    c1 = Counter.remote(init=0)
    c2 = Counter.remote(init=100)

    # Call methods
    refs = [
        c1.incr.remote(),
        c1.incr.remote(),
        c2.decr.remote(),
        c2.get.remote(),
    ]

    # Get results
    results = ray.get(refs)
    assert results[0] == 1  # c1.incr() -> 1
    assert results[1] == 2  # c1.incr() -> 2
    assert results[2] == 99  # c2.decr() -> 99
    assert results[3] == 99  # c2.get() -> 99 (after decr)


def test_actor_state_persistence(initialized_ray):
    """Test actor maintains state across calls."""
    handle = Counter.remote(init=0)

    for i in range(10):
        ref = handle.incr.remote()
        result = ray.get(ref)
        assert result == i + 1

    final = ray.get(handle.get.remote())
    assert final == 10
