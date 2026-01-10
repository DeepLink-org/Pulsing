"""
Pulsing AutoGen Runtime - 统一的单机/分布式运行时

Usage:
    from pulsing.autogen import PulsingRuntime

    # 单机模式
    runtime = PulsingRuntime()

    # 分布式模式
    runtime = PulsingRuntime(
        addr="0.0.0.0:8000",
        seeds=["other-node:8000"]
    )

    # 使用方式与 AutoGen 完全兼容
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent())
    runtime.start()
    await runtime.send_message(msg, recipient=AgentId("my_agent", "default"))
"""

from .runtime import PulsingRuntime
from .agent_wrapper import AutoGenAgentWrapper

__all__ = [
    "PulsingRuntime",
    "AutoGenAgentWrapper",
]
