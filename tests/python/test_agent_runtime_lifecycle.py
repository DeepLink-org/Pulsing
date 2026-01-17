"""测试 agent runtime 反复创建销毁的场景

关注点:
1. runtime 上下文管理器反复进入/退出
2. 全局状态清理 (agent registry, llm, actor system)
3. 资源泄漏 (actor, 网络连接)
4. 并发场景
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


@agent(role="测试员", goal="测试")
class TestAgent:
    """用于测试的简单 Agent"""

    def __init__(self, value: int = 0):
        self.value = value

    async def get_value(self) -> int:
        return self.value

    async def increment(self) -> int:
        self.value += 1
        return self.value


@remote
class TestActor:
    """用于测试的普通 Actor"""

    def __init__(self, data: str = ""):
        self.data = data

    async def echo(self) -> str:
        return self.data


class TestRuntimeLifecycle:
    """测试 runtime 生命周期管理"""

    @pytest.mark.asyncio
    async def test_basic_create_destroy(self):
        """基础测试：创建和销毁 runtime"""
        async with runtime():
            agent = await TestAgent.spawn(name="agent1", value=10)
            result = await agent.get_value()
            assert result == 10

        # runtime 退出后，全局系统应该被清理
        with pytest.raises(RuntimeError, match="Actor system not initialized"):
            get_system()

    @pytest.mark.asyncio
    async def test_repeated_create_destroy(self):
        """反复创建销毁 runtime (连续模式)"""
        for i in range(5):
            async with runtime():
                agent = await TestAgent.spawn(name=f"agent_{i}", value=i)
                result = await agent.get_value()
                assert result == i

            # 每次退出后检查系统已清理
            with pytest.raises(RuntimeError):
                get_system()

    @pytest.mark.asyncio
    async def test_agent_registry_cleanup(self):
        """测试 agent 元信息注册表是否需要手动清理"""
        # 第一轮
        async with runtime():
            await TestAgent.spawn(name="test_agent", value=1)
            meta = get_agent_meta("test_agent")
            assert meta is not None
            assert meta.role == "测试员"

        # 第二轮 - 检查注册表是否残留
        registry_before = list_agents()
        
        async with runtime():
            await TestAgent.spawn(name="test_agent", value=2)
            meta = get_agent_meta("test_agent")
            assert meta is not None
            
        registry_after = list_agents()
        
        # 注册表会积累 (这是预期行为，需要手动清理)
        assert len(registry_after) >= 1
        
        # 手动清理
        clear_agent_registry()
        assert len(list_agents()) == 0

    @pytest.mark.asyncio
    async def test_agent_registry_manual_cleanup_pattern(self):
        """推荐模式：每次 runtime 后手动清理注册表"""
        for i in range(3):
            async with runtime():
                await TestAgent.spawn(name="agent", value=i)
                assert get_agent_meta("agent") is not None
            
            # 推荐：每次退出后清理
            clear_agent_registry()
            assert len(list_agents()) == 0

    @pytest.mark.asyncio
    async def test_llm_singleton_cleanup(self):
        """测试 LLM 单例是否需要清理"""
        # 跳过如果没有配置 OPENAI_API_KEY
        pytest.importorskip("langchain_openai")
        
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("需要 OPENAI_API_KEY")
        
        # 第一轮
        async with runtime():
            client1 = await llm()
            assert client1 is not None
        
        # 第二轮 - LLM 单例会保留
        async with runtime():
            client2 = await llm()
            # 单例模式下是同一个实例
            assert client2 is client1
        
        # 手动清理
        reset_llm()
        
        # 第三轮 - 创建新实例
        async with runtime():
            client3 = await llm()
            # 清理后会创建新实例
            assert client3 is not client1

    @pytest.mark.asyncio
    async def test_multiple_actors_cleanup(self):
        """测试多个 actor 的清理"""
        async with runtime():
            # 创建多个 actor
            agents = []
            for i in range(10):
                agent = await TestAgent.spawn(name=f"agent_{i}", value=i)
                agents.append(agent)
            
            # 验证都能正常工作
            results = await asyncio.gather(*[a.get_value() for a in agents])
            assert results == list(range(10))
        
        # runtime 退出后，system 应该清理所有 actor
        with pytest.raises(RuntimeError):
            get_system()

    @pytest.mark.asyncio
    async def test_mixed_agent_and_actor(self):
        """测试混合使用 @agent 和 @remote"""
        async with runtime():
            # @agent 装饰的
            a1 = await TestAgent.spawn(name="agent1", value=1)
            # @remote 装饰的
            a2 = await TestActor.spawn(name="actor1", data="test")
            
            v1 = await a1.get_value()
            v2 = await a2.echo()
            
            assert v1 == 1
            assert v2 == "test"
        
        # 清理
        clear_agent_registry()

    @pytest.mark.asyncio
    async def test_exception_during_runtime(self):
        """测试 runtime 内部异常是否正确清理"""
        try:
            async with runtime():
                agent = await TestAgent.spawn(name="agent", value=1)
                await agent.get_value()
                # 抛出异常
                raise ValueError("故意抛出的异常")
        except ValueError:
            pass
        
        # 即使有异常，系统也应该被清理
        with pytest.raises(RuntimeError):
            get_system()
        
        clear_agent_registry()

    @pytest.mark.asyncio
    async def test_nested_runtime_not_supported(self):
        """测试嵌套 runtime (不推荐，但应该安全)"""
        async with runtime():
            agent1 = await TestAgent.spawn(name="outer", value=1)
            v1 = await agent1.get_value()
            assert v1 == 1
            
            # 嵌套 runtime - init() 会返回已存在的系统
            async with runtime():
                agent2 = await TestAgent.spawn(name="inner", value=2)
                v2 = await agent2.get_value()
                assert v2 == 2
            
            # 内层退出后，外层的 agent 应该已经失效
            # (因为 shutdown 被调用了)
            # 这里的行为取决于实现
        
        clear_agent_registry()

    @pytest.mark.asyncio
    async def test_rapid_create_destroy(self):
        """快速反复创建销毁 (压力测试)"""
        for i in range(20):
            async with runtime():
                agent = await TestAgent.spawn(name=f"agent_{i}", value=i)
                await agent.increment()
                result = await agent.get_value()
                assert result == i + 1
            
            if i % 5 == 0:
                # 定期清理注册表
                clear_agent_registry()
                gc.collect()  # 手动触发 GC

    @pytest.mark.asyncio
    async def test_concurrent_runtimes_sequential(self):
        """顺序执行多个并发任务"""
        async def run_task(task_id: int):
            async with runtime():
                agent = await TestAgent.spawn(name=f"agent_{task_id}", value=task_id)
                for _ in range(3):
                    await agent.increment()
                result = await agent.get_value()
                return result
        
        # 注意：由于使用全局 _global_system，并发运行会有问题
        # 这里顺序执行
        results = []
        for i in range(5):
            result = await run_task(i)
            assert result == i + 3
            results.append(result)
            clear_agent_registry()

    @pytest.mark.asyncio
    async def test_actor_garbage_collection(self):
        """测试 actor 对象是否能被 GC"""
        weak_refs = []
        
        for i in range(3):
            async with runtime():
                agent = await TestAgent.spawn(name=f"agent_{i}", value=i)
                # 创建弱引用
                weak_refs.append(weakref.ref(agent))
                await agent.get_value()
            
            clear_agent_registry()
        
        # 强制 GC
        gc.collect()
        
        # 注意：由于 actor 的实现细节，弱引用可能仍然存活
        # 这个测试主要是为了观察行为，不一定 assert
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        # print(f"存活的弱引用数: {alive_count}/{len(weak_refs)}")

    @pytest.mark.asyncio
    async def test_distributed_mode_cleanup(self):
        """测试分布式模式下的清理
        
        注意：发现的问题 - 端口可能需要时间释放
        """
        import time
        
        # 使用不同的地址避免端口冲突
        async with runtime(addr="127.0.0.1:18001"):
            agent = await TestAgent.spawn(name="dist_agent", value=42)
            result = await agent.get_value()
            assert result == 42
        
        # 清理
        clear_agent_registry()
        
        # 给系统一些时间释放端口 (已知问题)
        await asyncio.sleep(0.1)
        
        # 再次使用不同地址 (使用相同地址可能会因端口未释放而失败)
        async with runtime(addr="127.0.0.1:18002"):
            agent = await TestAgent.spawn(name="dist_agent2", value=99)
            result = await agent.get_value()
            assert result == 99
        
        clear_agent_registry()

    @pytest.mark.asyncio
    async def test_cleanup_helper_pattern(self):
        """推荐的清理模式 - 使用 helper 函数"""
        async def run_with_cleanup():
            try:
                async with runtime():
                    agent = await TestAgent.spawn(name="agent", value=1)
                    result = await agent.increment()
                    return result
            finally:
                # 确保清理
                clear_agent_registry()
                reset_llm()
        
        # 多次调用
        for i in range(5):
            result = await run_with_cleanup()
            assert result == 2
            
            # 验证清理成功
            assert len(list_agents()) == 0


class TestRuntimeEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_empty_runtime(self):
        """空 runtime (不创建任何 actor)"""
        async with runtime():
            pass
        
        with pytest.raises(RuntimeError):
            get_system()

    @pytest.mark.asyncio
    async def test_runtime_with_no_await(self):
        """runtime 内部没有 await 操作"""
        async with runtime():
            # 不创建 actor，直接退出
            system = get_system()
            assert system is not None

    @pytest.mark.asyncio
    async def test_multiple_cleanup_calls(self):
        """多次调用清理函数应该是安全的"""
        async with runtime():
            await TestAgent.spawn(name="agent", value=1)
        
        # 多次清理
        clear_agent_registry()
        clear_agent_registry()
        clear_agent_registry()
        
        reset_llm()
        reset_llm()
        
        assert len(list_agents()) == 0

    @pytest.mark.asyncio
    async def test_cleanup_convenience_function(self):
        """测试 cleanup() 便捷函数"""
        # 创建一些状态
        async with runtime():
            await TestAgent.spawn(name="agent1", value=1)
            await TestAgent.spawn(name="agent2", value=2)
        
        # 确认有注册表条目
        assert len(list_agents()) >= 2
        
        # 使用 cleanup() 一次清理所有
        cleanup()
        
        # 验证清理成功
        assert len(list_agents()) == 0
        
        # 再次调用应该是安全的
        cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_recommended_pattern(self):
        """推荐使用模式：try-finally 搭配 cleanup()"""
        async def run_with_cleanup_pattern():
            try:
                async with runtime():
                    agent = await TestAgent.spawn(name="agent", value=42)
                    result = await agent.increment()
                    return result
            finally:
                cleanup()
        
        # 多次调用
        for _ in range(3):
            result = await run_with_cleanup_pattern()
            assert result == 43
            # 每次都应该清理干净
            assert len(list_agents()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
