"""
LangGraph 并行多智能体示例（每个 Agent 独立节点）

架构：
  dispatch ──┬── idea_agent (Send) ──┐
             ├── idea_agent (Send) ──┼── collect → critic → [refine loop] → judge
             └── idea_agent (Send) ──┘

特性：
- 每个 Idea Agent 是独立的 LangGraph 节点（可被 Pulsing 分布式调度）
- 使用 Send API 实现真正的 fan-out/fan-in 并行
- 支持多轮评议改进

环境变量配置：
  export OPENAI_API_KEY="sk-xxx"
  export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可选
  export LLM_MODEL="gpt-4o-mini"                        # 可选

Usage:
  python parallel_ideas.py --mock                                    # 模拟模式
  python parallel_ideas.py --question "如何提升 API 吞吐？"           # 单轮
  python parallel_ideas.py --question "xxx" --max-rounds 3           # 多轮迭代
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
# LLM 客户端
# ============================================================================

_llm_client = None


def _get_llm():
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("请先安装 langchain-openai：pip install langchain-openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未设置 OPENAI_API_KEY，请设置或使用 --mock 模式")

    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    _llm_client = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
        max_tokens=1024,
    )
    print(f"[LLM] model={model}, base_url={base_url or '(默认)'}")
    return _llm_client


# ============================================================================
# State 定义
# ============================================================================


class AgentState(TypedDict, total=False):
    question: str
    n_ideas: int
    max_rounds: int
    current_round: int
    mock: bool
    # 使用 Annotated + operator.add 实现自动合并（fan-in）
    ideas: Annotated[list[dict[str, Any]], operator.add]
    critiques: list[dict[str, Any]]
    history: list[dict[str, Any]]
    final_answer: str
    review: dict[str, Any]


class IdeaAgentInput(TypedDict):
    """单个 Idea Agent 的输入"""

    question: str
    persona: str
    persona_idx: int
    mock: bool
    # 用于 refine 模式
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
    "性能工程师",
    "产品经理",
    "系统架构师",
    "安全专家",
    "可靠性/运维",
    "研究员",
    "数据工程师",
    "成本控制",
    "用户体验",
    "测试工程师",
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

IDEA_PROMPT = """你是一位资深的{persona}。
用户问题：{question}

请从你的专业视角，给出一个解决方案。要求：
1. 方案标题（一句话概括）
2. 方案详细描述（2-3句话）
3. 建议的实验/验证步骤（1-3 个）
4. 潜在风险（1-2 个）

请严格按以下 JSON 格式输出（不要加 markdown 代码块）：
{{"title": "...", "proposal": "...", "experiments": ["...", "..."], "risks": ["...", "..."]}}
"""

REFINE_PROMPT = """你是一位资深的{persona}。
用户问题：{question}

你之前的方案：
{previous_idea}

评审专家给出的反馈：
- 问题：{issues}
- 改进建议：{improvements}

请根据评审反馈，改进你的方案。

请严格按以下 JSON 格式输出（不要加 markdown 代码块）：
{{"title": "...", "proposal": "...", "experiments": ["...", "..."], "risks": ["...", "..."]}}
"""

CRITIC_PROMPT = """你是一位严格的技术评审专家。
请对以下方案进行评审：
1. 找出问题（1-3 个）
2. 给出改进建议（1-3 个）
3. 打分（0-10）

方案列表：
{ideas_json}

请严格按以下 JSON 格式输出：
[{{"idea_agent": "方案作者", "issues": ["问题1"], "improvements": ["改进1"], "score": 8}}, ...]
"""

JUDGE_PROMPT = """你是一位资深的技术决策者。
经过 {n_rounds} 轮评审改进后，请选出最优方案。

方案列表：
{ideas_json}

评审意见：
{critiques_json}

请输出 JSON：
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
# Dispatch Node：决定并行分发到哪些 Idea Agent
# ============================================================================


def dispatch(state: AgentState) -> list[Send]:
    """Fan-out：为每个 persona 创建一个并行任务"""
    question = state.get("question", "")
    n_ideas = max(3, min(10, int(state.get("n_ideas", 5))))
    mock = state.get("mock", False)

    personas = _get_personas(n_ideas)
    print(f"\n[Dispatch] 分发 {n_ideas} 个 Idea Agent 并行执行...")

    sends = []
    for idx, persona in enumerate(personas):
        sends.append(
            Send(
                "idea_agent",  # 目标节点
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
# Idea Agent Node：每个 Agent 独立执行
# ============================================================================


async def idea_agent(input: IdeaAgentInput) -> dict[str, Any]:
    """
    单个 Idea Agent 节点（可被 Pulsing 分布式调度）

    这是一个独立的节点，接收 IdeaAgentInput，返回单个 idea。
    LangGraph 会自动将多个 idea_agent 的结果通过 operator.add 合并到 state.ideas
    """
    question = input["question"]
    persona = input["persona"]
    idx = input["persona_idx"]
    mock = input["mock"]
    is_refine = input.get("is_refine", False)
    previous_idea = input.get("previous_idea")
    critique = input.get("critique")

    if is_refine and previous_idea:
        # Refine 模式
        idea = await _refine_idea(persona, question, previous_idea, critique, mock)
        print(f"  - [Refine] [{persona}] v{idea.version}: {idea.title}")
    else:
        # 初始生成模式
        idea = await _generate_idea(persona, question, idx, mock)
        print(f"  - [Generate] [{persona}] {idea.title}")

    # 返回的 ideas 是一个列表，会被 operator.add 合并
    return {"ideas": [asdict(idea)]}


async def _generate_idea(persona: str, question: str, idx: int, mock: bool) -> Idea:
    """生成初始方案"""
    if mock:
        await asyncio.sleep(0.02 * (idx % 3))
        templates = {
            "性能": (
                "并行化与缓存优化",
                ["压测定位瓶颈", "缓存命中率测试"],
                ["缓存一致性"],
            ),
            "安全": ("安全边界与治理", ["威胁建模", "mTLS 验证"], ["延迟增加"]),
            "可靠性": (
                "韧性方案（熔断/限流）",
                ["故障注入测试", "重试策略验证"],
                ["重试风暴"],
            ),
            "产品": ("流式与渐进式体验", ["TTFB 优化测试", "A/B 测试"], ["复杂度增加"]),
            "成本": ("成本驱动架构", ["成本曲线分析", "按需扩缩容"], ["质量下降"]),
        }
        for key, (title, experiments, risks) in templates.items():
            if key in persona:
                break
        else:
            title, experiments, risks = (
                "结构化拆解",
                ["并行验证", "小规模实验"],
                ["拆解不当"],
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
            title=data.get("title", f"{persona}的方案"),
            proposal=data.get("proposal", ""),
            experiments=data.get("experiments", []),
            risks=data.get("risks", []),
        )


async def _refine_idea(
    persona: str, question: str, previous: dict, critique: dict | None, mock: bool
) -> Idea:
    """根据评审反馈改进方案"""
    version = previous.get("version", 1) + 1

    if mock:
        await asyncio.sleep(0.02)
        new_experiments = list(previous.get("experiments", []))
        if critique and critique.get("improvements"):
            new_experiments.append(f"新增: {critique['improvements'][0][:20]}...")
        return Idea(
            agent=persona,
            title=previous.get("title", "") + " (改进版)",
            proposal=previous.get("proposal", "") + "\n[已根据反馈改进]",
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
# Init Node：初始化状态
# ============================================================================


def init_state(state: AgentState) -> dict[str, Any]:
    """初始化轮数和历史"""
    max_rounds = state.get("max_rounds", 1)
    print(f"\n[Init] 最大迭代轮数: {max_rounds}")
    return {
        "current_round": 1,
        "max_rounds": max_rounds,
        "ideas": [],  # 清空，让 fan-in 重新收集
        "history": [],
        "review": {"rounds": []},
    }


# ============================================================================
# Collect Node：收集并行结果
# ============================================================================


def collect_ideas(state: AgentState) -> dict[str, Any]:
    """Fan-in：收集所有 Idea Agent 的结果"""
    ideas = state.get("ideas", [])
    current_round = state.get("current_round", 1)

    print(f"\n{'='*60}")
    print(f"[Collect] Round {current_round} - 收集到 {len(ideas)} 个方案")
    print(f"{'='*60}")

    for i, idea in enumerate(ideas, 1):
        agent = idea.get("agent", "unknown")
        version = idea.get("version", 1)
        title = idea.get("title", "")
        proposal = idea.get("proposal", "")
        experiments = idea.get("experiments", [])
        risks = idea.get("risks", [])

        print(f"\n┌─ 方案 {i}: [{agent}] {'v' + str(version) if version > 1 else ''}")
        print(f"│  标题: {title}")
        print(f"│  描述: {proposal[:100]}{'...' if len(proposal) > 100 else ''}")
        print("│  实验计划:")
        for j, exp in enumerate(experiments, 1):
            print(f"│    {j}. {exp}")
        print("│  风险:")
        for j, risk in enumerate(risks, 1):
            print(f"│    {j}. {risk}")
        print(f"└{'─'*50}")

    # 记录历史
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

    print(f"\n[Critic] 评审 {len(ideas)} 个方案...")

    if mock:
        critiques = []
        for idea in ideas:
            issues, improvements = [], []
            if len(idea.get("experiments", [])) < 2:
                issues.append("实验不够充分")
                improvements.append("增加更多验证实验")
            if len(idea.get("risks", [])) < 1:
                issues.append("风险考虑不足")
                improvements.append("补充风险分析")
            score = 6 + len(idea.get("experiments", [])) - len(issues)
            critiques.append(
                Critique(
                    idea_agent=idea.get("agent", "unknown"),
                    issues=issues or ["总体可行"],
                    improvements=improvements or ["细化执行计划"],
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
                    issues=c.get("issues", ["需要更多细节"]),
                    improvements=c.get("improvements", ["补充实验计划"]),
                    score=c.get("score", 5),
                )
            )

    critique_dicts = [asdict(c) for c in critiques]
    avg_score = sum(c.score for c in critiques) / len(critiques) if critiques else 0

    print(f"\n{'='*60}")
    print(f"[Critic] Round {current_round} - 评审意见")
    print(f"{'='*60}")

    for i, c in enumerate(critique_dicts, 1):
        agent = c.get("idea_agent", "unknown")
        score = c.get("score", 0)
        issues = c.get("issues", [])
        improvements = c.get("improvements", [])

        print(f"\n┌─ 评审 {i}: [{agent}] 评分: {score}/10")
        print("│  问题:")
        for j, issue in enumerate(issues, 1):
            print(f"│    {j}. {issue}")
        print("│  改进建议:")
        for j, imp in enumerate(improvements, 1):
            print(f"│    {j}. {imp}")
        print(f"└{'─'*50}")

    print(f"\n>>> 平均分: {avg_score:.1f}/10")

    # 更新 review
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
            print(f"  -> 平均分 {avg_score:.1f} >= 9.0，提前结束")
            return "judge"

    if current_round < max_rounds:
        print(f"  -> 继续迭代（{current_round}/{max_rounds}）")
        return "dispatch_refine"
    else:
        print("  -> 达到最大轮数，进入评判")
        return "judge"


# ============================================================================
# Dispatch Refine：分发改进任务
# ============================================================================


def dispatch_refine(state: AgentState) -> list[Send]:
    """Fan-out：为每个 idea 创建改进任务"""
    question = state.get("question", "")
    ideas = state.get("ideas", [])
    critiques = state.get("critiques", [])
    mock = state.get("mock", False)
    current_round = state.get("current_round", 1)

    print(
        f"\n[Dispatch Refine] 分发 {len(ideas)} 个改进任务（Round {current_round + 1}）..."
    )

    # 匹配 critique
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
    """增加轮数，清空 ideas 以便重新收集"""
    return {
        "current_round": state.get("current_round", 1) + 1,
        "ideas": [],  # 清空，让新的 fan-in 收集改进后的 ideas
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

    print(f"\n[Judge] 经过 {current_round} 轮迭代，选择最优方案...")

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
            f"问题：{question}",
            "",
            f"经过 {current_round} 轮评议，推荐方案（来自 {best_agent}，评分 {best_score}/10）：",
            "",
            f"标题：{best.get('title', '')}",
            "",
            best.get("proposal", "") if best else "",
            "",
            "实验计划：",
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
        final_answer = f"问题：{question}\n\n经过 {current_round} 轮评议\n推荐方案（来自 {best_agent}）\n理由：{reason}\n\n{recommendation}"

    print(f"  -> 选中：{best_agent}")

    review = dict(state.get("review", {}))
    review["selected_agent"] = best_agent
    review["total_rounds"] = current_round

    return {"final_answer": final_answer, "review": review}


# ============================================================================
# Graph Builder
# ============================================================================


def build_graph():
    graph = StateGraph(AgentState)

    # 节点
    graph.add_node("init", init_state)
    graph.add_node("idea_agent", idea_agent)  # 每个并行任务都会调用这个节点
    graph.add_node("collect", collect_ideas)
    graph.add_node("critic", critic)
    graph.add_node("increment_round", increment_round)
    graph.add_node("judge", judge)

    # 入口
    graph.set_entry_point("init")

    # init -> dispatch (fan-out)
    graph.add_conditional_edges("init", dispatch)

    # idea_agent -> collect (fan-in，自动等待所有并行任务完成)
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
        description="LangGraph 并行多智能体（每个 Agent 独立节点）"
    )
    p.add_argument(
        "--question",
        default="如何实现一个可并行实验的多智能体工作流？",
        help="用户问题",
    )
    p.add_argument("--n-ideas", type=int, default=5, help="并行 idea 数量（3-10）")
    p.add_argument("--max-rounds", type=int, default=1, help="最大迭代轮数")
    p.add_argument("--show-review", action="store_true", help="输出评审记录")
    p.add_argument("--mock", action="store_true", help="模拟模式")
    p.add_argument("--model", default=None, help="覆盖 LLM_MODEL")
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
    print("LangGraph 并行多智能体（每个 Agent 独立节点）")
    print("=" * 60)
    print(f"问题：{args.question}")
    print(f"并行 Idea Agent 数量：{args.n_ideas}")
    print(f"最大迭代轮数：{args.max_rounds}")
    print(f"模式：{'模拟' if args.mock else '真实 LLM'}")

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
        print(f"总轮数：{review.get('total_rounds', 1)}")
        print(f"选中：{review.get('selected_agent', 'unknown')}")
        print("各轮评分：")
        for r in review.get("rounds", []):
            print(f"  Round {r.get('round')}: {r.get('avg_score', 0):.1f}/10")


if __name__ == "__main__":
    asyncio.run(main())
