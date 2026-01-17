"""
LangGraph + Pulsing Standalone Mode Example

Usage: python simple.py
"""

import asyncio
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    messages: Annotated[list, add]
    next_step: str


def llm_node(state: AgentState) -> AgentState:
    """Simulate LLM call"""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else {}
    content = (
        last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
    )
    role = last_msg.get("role", "") if isinstance(last_msg, dict) else ""

    if role == "tool":
        response, next_step = f"Weather: {content}", "end"
    elif "weather" in content.lower():
        response, next_step = "Let me check the weather.", "tool"
    else:
        response, next_step = f"Hello! You said: {content}", "end"

    print(f"[LLM] {content[:30]}... -> {response[:30]}...")
    return {
        "messages": [{"role": "assistant", "content": response}],
        "next_step": next_step,
    }


def tool_node(state: AgentState) -> AgentState:
    """Simulate tool call"""
    result = "Sunny, 25°C"
    print(f"[Tool] -> {result}")
    return {"messages": [{"role": "tool", "content": result}], "next_step": "llm"}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tool", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges(
        "llm", lambda s: s.get("next_step", "end"), {"tool": "tool", "end": END}
    )
    graph.add_edge("tool", "llm")
    return graph.compile()


async def main():
    from pulsing.langgraph import with_pulsing

    print("=" * 50)
    print("LangGraph + Pulsing Standalone Mode")
    print("=" * 50)

    app = with_pulsing(build_graph())  # Standalone mode, API fully compatible

    print("\n--- Test 1: Hello ---")
    await app.ainvoke(
        {"messages": [{"role": "user", "content": "Hello!"}], "next_step": ""}
    )

    print("\n--- Test 2: Weather ---")
    await app.ainvoke(
        {
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "next_step": "",
        }
    )

    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
