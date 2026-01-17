"""
LangGraph Parallel Multi-Agent Example (Each Agent is an Independent Node)

Architecture:
  dispatch ──┬── idea_agent (Send) ──┐
             ├── idea_agent (Send) ──┼── collect → critic → [refine loop] → judge
             └── idea_agent (Send) ──┘

Features:
- Each Idea Agent is an independent LangGraph node (can be distributed by Pulsing)
- Uses Send API for true fan-out/fan-in parallelism
- Supports multi-round review and improvement

Environment Variables:
  export OPENAI_API_KEY="sk-xxx"
  export OPENAI_BASE_URL="https://api.openai.com/v1"   # Optional
  export LLM_MODEL="gpt-4o-mini"                        # Optional

Usage:
  python parallel_ideas.py --mock                                    # Mock mode
  python parallel_ideas.py --question "How to improve API throughput?"  # Single round
  python parallel_ideas.py --question "xxx" --max-rounds 3           # Multi-round iteration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import operator
import os
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send


# ============================================================================
# LLM Client
# ============================================================================

_llm_client = None


def _get_llm():
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "Please install langchain-openai first: pip install langchain-openai"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set, please set it or use --mock mode")

    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    _llm_client = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
        max_tokens=1024,
    )
    print(f"[LLM] model={model}, base_url={base_url or '(default)'}")
    return _llm_client


# ============================================================================
# State Definition
# ============================================================================


class AgentState(TypedDict, total=False):
    question: str
    n_ideas: int
    max_rounds: int
    current_round: int
    mock: bool
    # Use Annotated + operator.add for automatic merging (fan-in)
    ideas: Annotated[list[dict[str, Any]], operator.add]
    critiques: list[dict[str, Any]]
    history: list[dict[str, Any]]
    final_answer: str
    review: dict[str, Any]


class IdeaAgentInput(TypedDict):
    """Input for a single Idea Agent"""

    question: str
    persona: str
    persona_idx: int
    mock: bool
    # For refine mode
    previous_idea: dict[str, Any] | None
    critique: dict[str, Any] | None
    is_refine: bool


@dataclass
class Idea:
    agent: str
    title: str
    proposal: str
    experiments: list[str]
    risks: list[str]
    version: int = 1


@dataclass
class Critique:
    idea_agent: str
    issues: list[str]
    improvements: list[str]
    score: int = 0


# ============================================================================
# Personas
# ============================================================================

PERSONAS = [
    "Performance Engineer",
    "Product Manager",
    "System Architect",
    "Security Expert",
    "Reliability/Ops",
    "Researcher",
    "Data Engineer",
    "Cost Controller",
    "User Experience",
    "Test Engineer",
]


def _get_personas(n: int) -> list[str]:
    if n <= len(PERSONAS):
        return PERSONAS[:n]
    return [
        PERSONAS[i % len(PERSONAS)]
        + (f"#{i // len(PERSONAS) + 1}" if i >= len(PERSONAS) else "")
        for i in range(n)
    ]


# ============================================================================
# Prompts
# ============================================================================

IDEA_PROMPT = """You are a senior {persona}.
User question: {question}

Please provide a solution from your professional perspective. Requirements:
1. Solution title (one sentence summary)
2. Detailed description (2-3 sentences)
3. Suggested experiments/validation steps (1-3 items)
4. Potential risks (1-2 items)

Please output strictly in the following JSON format (do not add markdown code blocks):
{{"title": "...", "proposal": "...", "experiments": ["...", "..."], "risks": ["...", "..."]}}
"""

REFINE_PROMPT = """You are a senior {persona}.
User question: {question}

Your previous solution:
{previous_idea}

Feedback from review experts:
- Issues: {issues}
- Improvement suggestions: {improvements}

Please improve your solution based on the review feedback.

Please output strictly in the following JSON format (do not add markdown code blocks):
{{"title": "...", "proposal": "...", "experiments": ["...", "..."], "risks": ["...", "..."]}}
"""

CRITIC_PROMPT = """You are a strict technical review expert.
Please review the following solutions:
1. Identify issues (1-3 items)
2. Provide improvement suggestions (1-3 items)
3. Score (0-10)

Solution list:
{ideas_json}

Please output strictly in JSON format:
[{{"idea_agent": "solution author", "issues": ["issue1"], "improvements": ["improvement1"], "score": 8}}, ...]
"""

JUDGE_PROMPT = """You are a senior technical decision maker.
After {n_rounds} rounds of review and improvement, please select the best solution.

Solution list:
{ideas_json}

Review comments:
{critiques_json}

Please output JSON:
{{"selected_agent": "...", "reason": "...", "final_recommendation": "..."}}
"""


# ============================================================================
# Helper
# ============================================================================


def _parse_json(content: str, fallback: Any = None) -> Any:
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            if content.startswith("json"):
                content = content[4:]
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        return fallback


# ============================================================================
# Dispatch Node: Decides which Idea Agents to dispatch in parallel
# ============================================================================


def dispatch(state: AgentState) -> list[Send]:
    """Fan-out: Create a parallel task for each persona"""
    question = state.get("question", "")
    n_ideas = max(3, min(10, int(state.get("n_ideas", 5))))
    mock = state.get("mock", False)

    personas = _get_personas(n_ideas)
    print(f"\n[Dispatch] Dispatching {n_ideas} Idea Agents to execute in parallel...")

    sends = []
    for idx, persona in enumerate(personas):
        sends.append(
            Send(
                "idea_agent",  # Target node
                IdeaAgentInput(
                    question=question,
                    persona=persona,
                    persona_idx=idx,
                    mock=mock,
                    previous_idea=None,
                    critique=None,
                    is_refine=False,
                ),
            )
        )
    return sends


# ============================================================================
# Idea Agent Node: Each Agent executes independently
# ============================================================================


async def idea_agent(input: IdeaAgentInput) -> dict[str, Any]:
    """
    Single Idea Agent node (can be distributed by Pulsing)

    This is an independent node that receives IdeaAgentInput and returns a single idea.
    LangGraph will automatically merge results from multiple idea_agent nodes into state.ideas via operator.add
    """
    question = input["question"]
    persona = input["persona"]
    idx = input["persona_idx"]
    mock = input["mock"]
    is_refine = input.get("is_refine", False)
    previous_idea = input.get("previous_idea")
    critique = input.get("critique")

    if is_refine and previous_idea:
        # Refine mode
        idea = await _refine_idea(persona, question, previous_idea, critique, mock)
        print(f"  - [Refine] [{persona}] v{idea.version}: {idea.title}")
    else:
        # Initial generation mode
        idea = await _generate_idea(persona, question, idx, mock)
        print(f"  - [Generate] [{persona}] {idea.title}")

    # Returned ideas is a list, will be merged by operator.add
    return {"ideas": [asdict(idea)]}


async def _generate_idea(persona: str, question: str, idx: int, mock: bool) -> Idea:
    """Generate initial solution"""
    if mock:
        await asyncio.sleep(0.02 * (idx % 3))
        templates = {
            "Performance": (
                "Parallelization and cache optimization",
                ["Load testing to locate bottlenecks", "Cache hit rate testing"],
                ["Cache consistency"],
            ),
            "Security": (
                "Security boundaries and governance",
                ["Threat modeling", "mTLS verification"],
                ["Increased latency"],
            ),
            "Reliability": (
                "Resilience solution (circuit breaker/rate limiting)",
                ["Fault injection testing", "Retry strategy verification"],
                ["Retry storms"],
            ),
            "Product": (
                "Streaming and progressive experience",
                ["TTFB optimization testing", "A/B testing"],
                ["Increased complexity"],
            ),
            "Cost": (
                "Cost-driven architecture",
                ["Cost curve analysis", "On-demand scaling"],
                ["Quality degradation"],
            ),
        }
        for key, (title, experiments, risks) in templates.items():
            if key in persona:
                break
        else:
            title, experiments, risks = (
                "Structured decomposition",
                ["Parallel validation", "Small-scale experiments"],
                ["Improper decomposition"],
            )
        return Idea(
            agent=persona,
            title=title,
            proposal=f"[{persona}] {title}",
            experiments=experiments,
            risks=risks,
        )
    else:
        llm = _get_llm()
        prompt = IDEA_PROMPT.format(persona=persona, question=question)
        response = await llm.ainvoke(prompt)
        data = _parse_json(response.content, {})
        return Idea(
            agent=persona,
            title=data.get("title", f"{persona}'s solution"),
            proposal=data.get("proposal", ""),
            experiments=data.get("experiments", []),
            risks=data.get("risks", []),
        )


async def _refine_idea(
    persona: str, question: str, previous: dict, critique: dict | None, mock: bool
) -> Idea:
    """Improve solution based on review feedback"""
    version = previous.get("version", 1) + 1

    if mock:
        await asyncio.sleep(0.02)
        new_experiments = list(previous.get("experiments", []))
        if critique and critique.get("improvements"):
            new_experiments.append(f"New: {critique['improvements'][0][:20]}...")
        return Idea(
            agent=persona,
            title=previous.get("title", "") + " (Improved)",
            proposal=previous.get("proposal", "") + "\n[Improved based on feedback]",
            experiments=new_experiments,
            risks=previous.get("risks", [])[:1],
            version=version,
        )
    else:
        llm = _get_llm()
        prompt = REFINE_PROMPT.format(
            persona=persona,
            question=question,
            previous_idea=json.dumps(previous, ensure_ascii=False),
            issues=", ".join(critique.get("issues", [])) if critique else "",
            improvements=", ".join(critique.get("improvements", []))
            if critique
            else "",
        )
        response = await llm.ainvoke(prompt)
        data = _parse_json(response.content, {})
        return Idea(
            agent=persona,
            title=data.get("title", previous.get("title", "")),
            proposal=data.get("proposal", previous.get("proposal", "")),
            experiments=data.get("experiments", previous.get("experiments", [])),
            risks=data.get("risks", previous.get("risks", [])),
            version=version,
        )


# ============================================================================
# Init Node: Initialize state
# ============================================================================


def init_state(state: AgentState) -> dict[str, Any]:
    """Initialize rounds and history"""
    max_rounds = state.get("max_rounds", 1)
    print(f"\n[Init] Maximum iteration rounds: {max_rounds}")
    return {
        "current_round": 1,
        "max_rounds": max_rounds,
        "ideas": [],  # Clear, let fan-in re-collect
        "history": [],
        "review": {"rounds": []},
    }


# ============================================================================
# Collect Node: Collect parallel results
# ============================================================================


def collect_ideas(state: AgentState) -> dict[str, Any]:
    """Fan-in: Collect results from all Idea Agents"""
    ideas = state.get("ideas", [])
    current_round = state.get("current_round", 1)

    print(f"\n{'='*60}")
    print(f"[Collect] Round {current_round} - Collected {len(ideas)} solutions")
    print(f"{'='*60}")

    for i, idea in enumerate(ideas, 1):
        agent = idea.get("agent", "unknown")
        version = idea.get("version", 1)
        title = idea.get("title", "")
        proposal = idea.get("proposal", "")
        experiments = idea.get("experiments", [])
        risks = idea.get("risks", [])

        print(
            f"\n┌─ Solution {i}: [{agent}] {'v' + str(version) if version > 1 else ''}"
        )
        print(f"│  Title: {title}")
        print(f"│  Description: {proposal[:100]}{'...' if len(proposal) > 100 else ''}")
        print("│  Experiments:")
        for j, exp in enumerate(experiments, 1):
            print(f"│    {j}. {exp}")
        print("│  Risks:")
        for j, risk in enumerate(risks, 1):
            print(f"│    {j}. {risk}")
        print(f"└{'─'*50}")

    # Record history
    history = list(state.get("history", []))
    history.append({"round": current_round, "phase": "collect", "n_ideas": len(ideas)})

    return {"history": history}


# ============================================================================
# Critic Node
# ============================================================================


async def critic(state: AgentState) -> dict[str, Any]:
    ideas = state.get("ideas", [])
    mock = state.get("mock", False)
    current_round = state.get("current_round", 1)

    print(f"\n[Critic] Reviewing {len(ideas)} solutions...")

    if mock:
        critiques = []
        for idea in ideas:
            issues, improvements = [], []
            if len(idea.get("experiments", [])) < 2:
                issues.append("Insufficient experiments")
                improvements.append("Add more validation experiments")
            if len(idea.get("risks", [])) < 1:
                issues.append("Insufficient risk consideration")
                improvements.append("Supplement risk analysis")
            score = 6 + len(idea.get("experiments", [])) - len(issues)
            critiques.append(
                Critique(
                    idea_agent=idea.get("agent", "unknown"),
                    issues=issues or ["Generally feasible"],
                    improvements=improvements or ["Refine execution plan"],
                    score=max(1, min(10, score)),
                )
            )
    else:
        llm = _get_llm()
        ideas_json = json.dumps(ideas, ensure_ascii=False, indent=2)
        response = await llm.ainvoke(CRITIC_PROMPT.format(ideas_json=ideas_json))
        critique_data = _parse_json(response.content, [])

        critiques = []
        for i, idea in enumerate(ideas):
            c = critique_data[i] if i < len(critique_data) else {}
            critiques.append(
                Critique(
                    idea_agent=c.get("idea_agent", idea.get("agent", "unknown")),
                    issues=c.get("issues", ["Need more details"]),
                    improvements=c.get("improvements", ["Supplement experiment plan"]),
                    score=c.get("score", 5),
                )
            )

    critique_dicts = [asdict(c) for c in critiques]
    avg_score = sum(c.score for c in critiques) / len(critiques) if critiques else 0

    print(f"\n{'='*60}")
    print(f"[Critic] Round {current_round} - Review Comments")
    print(f"{'='*60}")

    for i, c in enumerate(critique_dicts, 1):
        agent = c.get("idea_agent", "unknown")
        score = c.get("score", 0)
        issues = c.get("issues", [])
        improvements = c.get("improvements", [])

        print(f"\n┌─ Review {i}: [{agent}] Score: {score}/10")
        print("│  Issues:")
        for j, issue in enumerate(issues, 1):
            print(f"│    {j}. {issue}")
        print("│  Improvements:")
        for j, imp in enumerate(improvements, 1):
            print(f"│    {j}. {imp}")
        print(f"└{'─'*50}")

    print(f"\n>>> Average Score: {avg_score:.1f}/10")

    # Update review
    review = dict(state.get("review", {}))
    rounds = list(review.get("rounds", []))
    rounds.append({"round": current_round, "avg_score": avg_score})
    review["rounds"] = rounds

    return {"critiques": critique_dicts, "review": review}


# ============================================================================
# Should Continue
# ============================================================================


def should_continue(state: AgentState) -> Literal["dispatch_refine", "judge"]:
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", 1)

    critiques = state.get("critiques", [])
    if critiques:
        avg_score = sum(c.get("score", 0) for c in critiques) / len(critiques)
        if avg_score >= 9.0:
            print(f"  -> Average score {avg_score:.1f} >= 9.0, early termination")
            return "judge"

    if current_round < max_rounds:
        print(f"  -> Continue iteration ({current_round}/{max_rounds})")
        return "dispatch_refine"
    else:
        print("  -> Reached maximum rounds, proceed to judgment")
        return "judge"


# ============================================================================
# Dispatch Refine: Dispatch improvement tasks
# ============================================================================


def dispatch_refine(state: AgentState) -> list[Send]:
    """Fan-out: Create improvement task for each idea"""
    question = state.get("question", "")
    ideas = state.get("ideas", [])
    critiques = state.get("critiques", [])
    mock = state.get("mock", False)
    current_round = state.get("current_round", 1)

    print(
        f"\n[Dispatch Refine] Dispatching {len(ideas)} improvement tasks (Round {current_round + 1})..."
    )

    # Match critique
    critique_map = {c.get("idea_agent"): c for c in critiques}

    sends = []
    for idx, idea in enumerate(ideas):
        persona = idea.get("agent", "unknown")
        sends.append(
            Send(
                "idea_agent",
                IdeaAgentInput(
                    question=question,
                    persona=persona,
                    persona_idx=idx,
                    mock=mock,
                    previous_idea=idea,
                    critique=critique_map.get(persona),
                    is_refine=True,
                ),
            )
        )
    return sends


def increment_round(state: AgentState) -> dict[str, Any]:
    """Increment round, clear ideas for re-collection"""
    return {
        "current_round": state.get("current_round", 1) + 1,
        "ideas": [],  # Clear, let new fan-in collect improved ideas
    }


# ============================================================================
# Judge Node
# ============================================================================


async def judge(state: AgentState) -> dict[str, Any]:
    ideas = state.get("ideas", [])
    critiques = state.get("critiques", [])
    question = state.get("question", "")
    mock = state.get("mock", False)
    current_round = state.get("current_round", 1)

    print(
        f"\n[Judge] After {current_round} rounds of iteration, selecting the best solution..."
    )

    if mock:
        critique_map = {c.get("idea_agent"): c.get("score", 0) for c in critiques}
        best = (
            max(ideas, key=lambda x: critique_map.get(x.get("agent"), 0))
            if ideas
            else None
        )
        best_agent = best.get("agent") if best else "unknown"
        best_score = critique_map.get(best_agent, 0) if best else 0

        final_lines = [
            f"Question: {question}",
            "",
            f"After {current_round} rounds of review, recommended solution (from {best_agent}, score {best_score}/10):",
            "",
            f"Title: {best.get('title', '')}",
            "",
            best.get("proposal", "") if best else "",
            "",
            "Experiment Plan:",
        ]
        if best:
            for i, exp in enumerate(best.get("experiments", []), 1):
                final_lines.append(f"  {i}. {exp}")
        final_answer = "\n".join(final_lines)
    else:
        llm = _get_llm()
        ideas_json = json.dumps(ideas, ensure_ascii=False, indent=2)
        critiques_json = json.dumps(critiques, ensure_ascii=False, indent=2)
        response = await llm.ainvoke(
            JUDGE_PROMPT.format(
                n_rounds=current_round,
                ideas_json=ideas_json,
                critiques_json=critiques_json,
            )
        )
        data = _parse_json(response.content, {})
        best_agent = data.get(
            "selected_agent", ideas[0].get("agent") if ideas else "unknown"
        )
        reason = data.get("reason", "")
        recommendation = data.get("final_recommendation", "")
        final_answer = f"Question: {question}\n\nAfter {current_round} rounds of review\nRecommended solution (from {best_agent})\nReason: {reason}\n\n{recommendation}"

    print(f"  -> Selected: {best_agent}")

    review = dict(state.get("review", {}))
    review["selected_agent"] = best_agent
    review["total_rounds"] = current_round

    return {"final_answer": final_answer, "review": review}


# ============================================================================
# Graph Builder
# ============================================================================


def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("init", init_state)
    graph.add_node("idea_agent", idea_agent)  # Each parallel task will call this node
    graph.add_node("collect", collect_ideas)
    graph.add_node("critic", critic)
    graph.add_node("increment_round", increment_round)
    graph.add_node("judge", judge)

    # Entry point
    graph.set_entry_point("init")

    # init -> dispatch (fan-out)
    graph.add_conditional_edges("init", dispatch)

    # idea_agent -> collect (fan-in, automatically waits for all parallel tasks to complete)
    graph.add_edge("idea_agent", "collect")

    # collect -> critic
    graph.add_edge("collect", "critic")

    # critic -> should_continue
    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "dispatch_refine": "increment_round",
            "judge": "judge",
        },
    )

    # increment_round -> dispatch_refine (fan-out)
    graph.add_conditional_edges("increment_round", dispatch_refine)

    # judge -> END
    graph.add_edge("judge", END)

    return graph.compile()


# ============================================================================
# CLI
# ============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LangGraph Parallel Multi-Agent (Each Agent is an Independent Node)"
    )
    p.add_argument(
        "--question",
        default="How to implement a multi-agent workflow that supports parallel experiments?",
        help="User question",
    )
    p.add_argument(
        "--n-ideas", type=int, default=5, help="Number of parallel ideas (3-10)"
    )
    p.add_argument("--max-rounds", type=int, default=1, help="Maximum iteration rounds")
    p.add_argument("--show-review", action="store_true", help="Output review records")
    p.add_argument("--mock", action="store_true", help="Mock mode")
    p.add_argument("--model", default=None, help="Override LLM_MODEL")
    return p.parse_args()


async def main():
    args = _parse_args()

    if args.model:
        os.environ["LLM_MODEL"] = args.model

    try:
        from pulsing.langgraph import with_pulsing
    except ImportError:
        with_pulsing = None

    app = build_graph()
    if with_pulsing:
        app = with_pulsing(app)

    print("=" * 60)
    print("LangGraph Parallel Multi-Agent (Each Agent is an Independent Node)")
    print("=" * 60)
    print(f"Question: {args.question}")
    print(f"Number of Parallel Idea Agents: {args.n_ideas}")
    print(f"Maximum Iteration Rounds: {args.max_rounds}")
    print(f"Mode: {'Mock' if args.mock else 'Real LLM'}")

    result = await app.ainvoke(
        {
            "question": args.question,
            "n_ideas": args.n_ideas,
            "max_rounds": args.max_rounds,
            "mock": args.mock,
        }
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.get("final_answer", ""))

    if args.show_review:
        print("\n" + "=" * 60)
        print("REVIEW")
        print("=" * 60)
        review = result.get("review", {})
        print(f"Total Rounds: {review.get('total_rounds', 1)}")
        print(f"Selected: {review.get('selected_agent', 'unknown')}")
        print("Round Scores:")
        for r in review.get("rounds", []):
            print(f"  Round {r.get('round')}: {r.get('avg_score', 0):.1f}/10")


if __name__ == "__main__":
    asyncio.run(main())
