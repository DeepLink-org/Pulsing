"""
Pulsing AutoGen Runtime - Unified Standalone/Distributed Runtime

Usage:
    from pulsing.autogen import PulsingRuntime

    # Standalone mode
    runtime = PulsingRuntime()

    # Distributed mode
    runtime = PulsingRuntime(
        addr="0.0.0.0:8000",
        seeds=["other-node:8000"]
    )

    # Usage is fully compatible with AutoGen
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
