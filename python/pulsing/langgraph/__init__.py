"""
Pulsing LangGraph Integration - One line of code to enable distributed LangGraph

Usage:
    from langgraph.graph import StateGraph
    from pulsing.langgraph import with_pulsing

    # Original LangGraph code
    graph = StateGraph(MyState)
    graph.add_node("llm", llm_fn)
    app = graph.compile()

    # ✨ One line of code to enable distribution
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
