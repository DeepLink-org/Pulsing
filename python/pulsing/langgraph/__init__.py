"""
Pulsing LangGraph Integration - 一行代码实现分布式 LangGraph

Usage:
    from langgraph.graph import StateGraph
    from pulsing.langgraph import with_pulsing

    # 原有 LangGraph 代码
    graph = StateGraph(MyState)
    graph.add_node("llm", llm_fn)
    app = graph.compile()

    # ✨ 一行代码实现分布式
    distributed_app = with_pulsing(
        app,
        node_mapping={"llm": "langgraph_node_llm"},
        seeds=["gpu-server:8001"],
    )

    result = await distributed_app.ainvoke({"messages": []})
"""

from .wrapper import with_pulsing, PulsingGraphWrapper
from .executor import LangGraphNodeActor, start_worker

__all__ = ["with_pulsing", "PulsingGraphWrapper", "LangGraphNodeActor", "start_worker"]
