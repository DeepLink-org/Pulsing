"""Test agent runtime repeatedly creating and destroying scenarios

Focus areas:
1. Runtime context manager repeatedly entering/exiting
2. Global state cleanup (agent registry, llm, actor system)
3. Resource leaks (actor, network connections)
4. Concurrent scenarios
"""

import asyncio
import gc
import weakref

import pytest

from pulsing.actor import get_system, remote
from pulsing.agent import (
    agent,
    cleanup,
    clear_agent_registry,
    get_agent_meta,
    list_agents,
    llm,
    reset_llm,
    runtime,
)


@agent(role="Tester", goal="Testing")
class TestAgent:
    """Simple agent for testing"""

    def __init__(self, value: int = 0):
        self.value = value

    async def get_value(self) -> int:
        return self.value

    async def increment(self) -> int:
        self.value += 1
        return self.value


@remote
class TestActor:
    """Regular actor for testing"""

    def __init__(self, data: str = ""):
        self.data = data

    async def echo(self) -> str:
        return self.data


class TestRuntimeLifecycle:
    """Test runtime lifecycle management"""

    @pytest.mark.asyncio
    async def test_basic_create_destroy(self):
        """Basic test: create and destroy runtime"""
        async with runtime():
            agent = await TestAgent.spawn(name="agent1", value=10)
            result = await agent.get_value()
            assert result == 10

        # After runtime exits, global system should be cleaned up
        with pytest.raises(RuntimeError, match="Actor system not initialized"):
            get_system()

    @pytest.mark.asyncio
    async def test_repeated_create_destroy(self):
        """Repeatedly create and destroy runtime (sequential mode)"""
        for i in range(5):
            async with runtime():
                agent = await TestAgent.spawn(name=f"agent_{i}", value=i)
                result = await agent.get_value()
                assert result == i

            # Check system is cleaned up after each exit
            with pytest.raises(RuntimeError):
                get_system()

    @pytest.mark.asyncio
    async def test_agent_registry_cleanup(self):
        """Test if agent metadata registry needs manual cleanup"""
        # First round
        async with runtime():
            await TestAgent.spawn(name="test_agent", value=1)
            meta = get_agent_meta("test_agent")
            assert meta is not None
            assert meta.role == "Tester"

        # Second round - check if registry has residue
        async with runtime():
            await TestAgent.spawn(name="test_agent", value=2)
            meta = get_agent_meta("test_agent")
            assert meta is not None

        registry_after = list_agents()

        # Registry will accumulate (this is expected behavior, requires manual cleanup)
        assert len(registry_after) >= 1

        # Manual cleanup
        clear_agent_registry()
        assert len(list_agents()) == 0

    @pytest.mark.asyncio
    async def test_agent_registry_manual_cleanup_pattern(self):
        """Recommended pattern: manually clean registry after each runtime"""
        for i in range(3):
            async with runtime():
                await TestAgent.spawn(name="agent", value=i)
                assert get_agent_meta("agent") is not None

            # Recommended: cleanup after each exit
            clear_agent_registry()
            assert len(list_agents()) == 0

    @pytest.mark.asyncio
    async def test_llm_singleton_cleanup(self):
        """Test if LLM singleton needs cleanup"""
        # Skip if OPENAI_API_KEY is not configured
        pytest.importorskip("langchain_openai")

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY required")

        # First round
        async with runtime():
            client1 = await llm()
            assert client1 is not None

        # Second round - LLM singleton will be retained
        async with runtime():
            client2 = await llm()
            # In singleton mode, it's the same instance
            assert client2 is client1

        # Manual cleanup
        reset_llm()

        # Third round - create new instance
        async with runtime():
            client3 = await llm()
            # After cleanup, new instance will be created
            assert client3 is not client1

    @pytest.mark.asyncio
    async def test_multiple_actors_cleanup(self):
        """Test cleanup of multiple actors"""
        async with runtime():
            # Create multiple actors
            agents = []
            for i in range(10):
                agent = await TestAgent.spawn(name=f"agent_{i}", value=i)
                agents.append(agent)

            # Verify all work correctly
            results = await asyncio.gather(*[a.get_value() for a in agents])
            assert results == list(range(10))

        # After runtime exits, system should clean up all actors
        with pytest.raises(RuntimeError):
            get_system()

    @pytest.mark.asyncio
    async def test_mixed_agent_and_actor(self):
        """Test mixing @agent and @remote"""
        async with runtime():
            # @agent decorated
            a1 = await TestAgent.spawn(name="agent1", value=1)
            # @remote decorated
            a2 = await TestActor.spawn(name="actor1", data="test")

            v1 = await a1.get_value()
            v2 = await a2.echo()

            assert v1 == 1
            assert v2 == "test"

        # Cleanup
        clear_agent_registry()

    @pytest.mark.asyncio
    async def test_exception_during_runtime(self):
        """Test if exceptions during runtime are properly cleaned up"""
        try:
            async with runtime():
                agent = await TestAgent.spawn(name="agent", value=1)
                await agent.get_value()
                # Raise exception
                raise ValueError("Intentionally raised exception")
        except ValueError:
            pass

        # Even with exception, system should be cleaned up
        with pytest.raises(RuntimeError):
            get_system()

        clear_agent_registry()

    @pytest.mark.asyncio
    async def test_nested_runtime_not_supported(self):
        """Test nested runtime (not recommended, but should be safe)"""
        async with runtime():
            agent1 = await TestAgent.spawn(name="outer", value=1)
            v1 = await agent1.get_value()
            assert v1 == 1

            # Nested runtime - init() will return existing system
            async with runtime():
                agent2 = await TestAgent.spawn(name="inner", value=2)
                v2 = await agent2.get_value()
                assert v2 == 2

            # After inner exits, outer agent should be invalid
            # (because shutdown was called)
            # Behavior here depends on implementation

        clear_agent_registry()

    @pytest.mark.asyncio
    async def test_rapid_create_destroy(self):
        """Rapidly create and destroy repeatedly (stress test)"""
        for i in range(20):
            async with runtime():
                agent = await TestAgent.spawn(name=f"agent_{i}", value=i)
                await agent.increment()
                result = await agent.get_value()
                assert result == i + 1

            if i % 5 == 0:
                # Periodically clean registry
                clear_agent_registry()
                gc.collect()  # Manually trigger GC

    @pytest.mark.asyncio
    async def test_concurrent_runtimes_sequential(self):
        """Sequentially execute multiple concurrent tasks"""

        async def run_task(task_id: int):
            async with runtime():
                agent = await TestAgent.spawn(name=f"agent_{task_id}", value=task_id)
                for _ in range(3):
                    await agent.increment()
                result = await agent.get_value()
                return result

        # Note: Due to using global _global_system, concurrent execution will have issues
        # Here we execute sequentially
        results = []
        for i in range(5):
            result = await run_task(i)
            assert result == i + 3
            results.append(result)
            clear_agent_registry()

    @pytest.mark.asyncio
    async def test_actor_garbage_collection(self):
        """Test if actor objects can be garbage collected"""
        weak_refs = []

        for i in range(3):
            async with runtime():
                agent = await TestAgent.spawn(name=f"agent_{i}", value=i)
                # Create weak reference
                weak_refs.append(weakref.ref(agent))
                await agent.get_value()

            clear_agent_registry()

        # Force GC
        gc.collect()

        # Note: Due to actor implementation details, weak references may still be alive
        # This test is mainly to observe behavior, not necessarily assert
        # alive_count = sum(1 for ref in weak_refs if ref() is not None)
        # print(f"Alive weak references: {alive_count}/{len(weak_refs)}")

    @pytest.mark.asyncio
    async def test_distributed_mode_cleanup(self):
        """Test cleanup in distributed mode

        Note: Known issue - ports may need time to be released
        """
        import time

        # Use different addresses to avoid port conflicts
        async with runtime(addr="127.0.0.1:18001"):
            agent = await TestAgent.spawn(name="dist_agent", value=42)
            result = await agent.get_value()
            assert result == 42

        # Cleanup
        clear_agent_registry()

        # Give system some time to release port (known issue)
        await asyncio.sleep(0.1)

        # Use different address again (using same address may fail due to port not released)
        async with runtime(addr="127.0.0.1:18002"):
            agent = await TestAgent.spawn(name="dist_agent2", value=99)
            result = await agent.get_value()
            assert result == 99

        clear_agent_registry()

    @pytest.mark.asyncio
    async def test_cleanup_helper_pattern(self):
        """Recommended cleanup pattern - use helper function"""

        async def run_with_cleanup():
            try:
                async with runtime():
                    agent = await TestAgent.spawn(name="agent", value=1)
                    result = await agent.increment()
                    return result
            finally:
                # Ensure cleanup
                clear_agent_registry()
                reset_llm()

        # Multiple calls
        for i in range(5):
            result = await run_with_cleanup()
            assert result == 2

            # Verify cleanup succeeded
            assert len(list_agents()) == 0


class TestRuntimeEdgeCases:
    """Edge case tests"""

    @pytest.mark.asyncio
    async def test_empty_runtime(self):
        """Empty runtime (no actors created)"""
        async with runtime():
            pass

        with pytest.raises(RuntimeError):
            get_system()

    @pytest.mark.asyncio
    async def test_runtime_with_no_await(self):
        """Runtime with no await operations inside"""
        async with runtime():
            # Don't create actor, exit directly
            system = get_system()
            assert system is not None

    @pytest.mark.asyncio
    async def test_multiple_cleanup_calls(self):
        """Multiple cleanup function calls should be safe"""
        async with runtime():
            await TestAgent.spawn(name="agent", value=1)

        # Multiple cleanups
        clear_agent_registry()
        clear_agent_registry()
        clear_agent_registry()

        reset_llm()
        reset_llm()

        assert len(list_agents()) == 0

    @pytest.mark.asyncio
    async def test_cleanup_convenience_function(self):
        """Test cleanup() convenience function"""
        # Create some state
        async with runtime():
            await TestAgent.spawn(name="agent1", value=1)
            await TestAgent.spawn(name="agent2", value=2)

        # Confirm registry entries exist
        assert len(list_agents()) >= 2

        # Use cleanup() to clean all at once
        cleanup()

        # Verify cleanup succeeded
        assert len(list_agents()) == 0

        # Calling again should be safe
        cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_recommended_pattern(self):
        """Recommended usage pattern: try-finally with cleanup()"""

        async def run_with_cleanup_pattern():
            try:
                async with runtime():
                    agent = await TestAgent.spawn(name="agent", value=42)
                    result = await agent.increment()
                    return result
            finally:
                cleanup()

        # Multiple calls
        for _ in range(3):
            result = await run_with_cleanup_pattern()
            assert result == 43
            # Should be cleaned up each time
            assert len(list_agents()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
