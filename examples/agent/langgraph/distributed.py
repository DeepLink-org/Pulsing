"""
LangGraph + Pulsing Distributed Mode Example

Usage:
    ./run_distributed.sh                          # One-click startup

    # Or start manually:
    python distributed.py worker llm 8001         # Terminal 1
    python distributed.py worker tool 8002 8001   # Terminal 2
    python distributed.py run                     # Terminal 3
"""

import asyncio
import sys
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    messages: Annotated[list, add]
    next_step: str


def llm_node(state: AgentState) -> AgentState:
    """LLM node - deployed on GPU server"""
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

    print(f"[LLM Worker] {content[:30]}... -> {response[:30]}...")
    return {
        "messages": [{"role": "assistant", "content": response}],
        "next_step": next_step,
    }


def tool_node(state: AgentState) -> AgentState:
    """Tool node - deployed on CPU server"""
    result = "Sunny, 25°C"
    print(f"[Tool Worker] -> {result}")
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


async def run_distributed():
    """Distributed main program"""
    from pulsing.integrations.langgraph import with_pulsing

    print("=" * 50)
    print("LangGraph + Pulsing Distributed Mode")
    print("=" * 50)

    # ✨ One line of code to enable distribution
    app = with_pulsing(
        build_graph(),
        node_mapping={
            "llm": "langgraph_node_llm",
            "tool": "langgraph_node_tool",
        },
        seeds=["127.0.0.1:8001", "127.0.0.1:8002"],
    )

    print("\nWaiting for connections...")
    await asyncio.sleep(2)

    print("\n--- Weather Query ---")
    result = await app.ainvoke(
        {
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "next_step": "",
        }
    )
    print(f"Messages: {len(result['messages'])}")

    await app.close()
    print("\n✅ Done!")


async def run_worker(node_name: str, port: int, seed_port: int | None = None):
    """Start Worker"""
    from pulsing.integrations.langgraph import start_worker

    nodes = {"llm": llm_node, "tool": tool_node}
    if node_name not in nodes:
        print(f"Unknown node: {node_name}")
        return

    seeds = [f"127.0.0.1:{seed_port}"] if seed_port else []
    print(f"Starting {node_name.upper()} Worker on :{port}...")
    await start_worker(node_name, nodes[node_name], addr=f"0.0.0.0:{port}", seeds=seeds)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python distributed.py run")
        print("  python distributed.py worker llm 8001")
        print("  python distributed.py worker tool 8002 8001")
        return

    cmd = sys.argv[1]
    if cmd == "run":
        asyncio.run(run_distributed())
    elif cmd == "worker" and len(sys.argv) >= 4:
        asyncio.run(
            run_worker(
                sys.argv[2],
                int(sys.argv[3]),
                int(sys.argv[4]) if len(sys.argv) > 4 else None,
            )
        )


if __name__ == "__main__":
    main()
