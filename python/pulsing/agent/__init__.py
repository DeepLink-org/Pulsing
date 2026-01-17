"""
Pulsing Agent 工具箱

轻量的 multi-agent 开发工具，与 pulsing.actor 完全兼容。

核心 API:
- runtime(): Actor 系统生命周期管理
- agent(): 带元信息的 @remote（用于可视化）
- llm(): LLM 客户端
- parse_json(): JSON 解析

Example:
    from pulsing.actor import remote, resolve
    from pulsing.agent import agent, runtime, llm, get_agent_meta

    # @remote: 基础 Actor
    @remote
    class Worker:
        async def work(self): ...

    # @agent: 带元信息的 Actor（用于可视化/调试）
    @agent(role="研究员", goal="深入分析")
    class Researcher:
        async def analyze(self, topic: str) -> str:
            client = await llm()
            return await client.ainvoke(f"分析: {topic}")

    async def main():
        async with runtime():
            r = await Researcher.spawn(name="researcher")
            result = await r.analyze("AI")

            # 获取元信息
            meta = get_agent_meta("researcher")
            print(meta.role)  # "研究员"
"""

# 运行时
from .runtime import runtime

# Agent 装饰器（带元信息的 @remote）
from .base import agent, AgentMeta, get_agent_meta, list_agents, clear_agent_registry

# LLM
from .llm import llm, reset_llm

# 工具函数
from .utils import parse_json, extract_field


def cleanup():
    """
    清理所有 agent 相关的全局状态。

    包括：
    - Agent 元信息注册表
    - LLM 单例

    推荐在反复创建销毁 runtime 时调用，避免内存泄漏。

    Example:
        from pulsing.agent import runtime, cleanup

        try:
            async with runtime():
                agent = await MyAgent.spawn(name="agent")
                await agent.work()
        finally:
            cleanup()  # 清理所有全局状态
    """
    clear_agent_registry()
    reset_llm()


__all__ = [
    # 运行时
    "runtime",
    "cleanup",
    # Agent
    "agent",
    "AgentMeta",
    "get_agent_meta",
    "list_agents",
    "clear_agent_registry",
    # LLM
    "llm",
    "reset_llm",
    # 工具函数
    "parse_json",
    "extract_field",
]
