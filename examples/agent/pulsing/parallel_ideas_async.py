"""
Pulsing 并行多智能体示例

架构: N 个 IdeaAgent 并行迭代，竞争提交给 Judge

Usage:
  python parallel_ideas_async.py --mock                    # 模拟模式
  python parallel_ideas_async.py --question "问题"         # 真实 LLM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pulsing.actor import remote, resolve
from pulsing.agent import runtime, llm, parse_json

# ============================================================================
# 配置
# ============================================================================

PERSONAS = [
    "性能工程师",
    "产品经理",
    "系统架构师",
    "安全专家",
    "可靠性运维",
    "研究员",
    "数据工程师",
    "成本控制",
    "用户体验",
    "测试工程师",
]

# Mock 模式的模板
MOCK_TEMPLATES = {
    "性能": ("并行化与缓存优化", ["压测定位瓶颈", "缓存命中率测试"], ["缓存一致性"]),
    "安全": ("安全边界与治理", ["威胁建模", "mTLS 验证"], ["延迟增加"]),
    "可靠性": ("韧性方案", ["故障注入测试", "重试策略验证"], ["重试风暴"]),
    "产品": ("流式体验", ["TTFB 优化", "A/B 测试"], ["复杂度增加"]),
    "成本": ("成本驱动架构", ["成本曲线分析", "按需扩缩容"], ["质量下降"]),
}

MOCK_SUGGESTIONS = {
    "性能": "建议增加缓存层和连接池优化",
    "安全": "建议增加 mTLS 和审计日志",
    "可靠性": "建议增加熔断器和重试机制",
    "产品": "建议增加流式输出和进度反馈",
    "成本": "建议评估按需扩缩容策略",
}

# ============================================================================
# LLM 工具
# ============================================================================


async def get_llm():
    return await llm(temperature=0.7)


# ============================================================================
# Judge Actor
# ============================================================================


@remote
class JudgeActor:
    def __init__(self, timeout: float, mock: bool):
        self.timeout = timeout
        self.mock = mock
        self.submissions: list[dict] = []
        self.deadline: float | None = None
        self.result: dict | None = None
        self._done = asyncio.Event()
        self._agents: list[str] = []

    def set_agents(self, names: list[str]):
        self._agents = names

    async def submit(self, idea: dict, iterations: int) -> dict:
        now = time.time()

        # 检查截止
        if self.deadline and now > self.deadline:
            print(f"  ❌ [{idea['agent']}] 超时被拒")
            return {"accepted": False}

        self.submissions.append({"idea": idea, "iterations": iterations, "time": now})

        # 第一个提交启动计时器
        if not self.deadline:
            self.deadline = now + self.timeout
            print(f"\n[Judge] 收到首个提案 [{idea['agent']}]，{self.timeout}s 后截止")
            asyncio.create_task(self._timer())

        remaining = self.deadline - now
        print(
            f"  ✓ [{idea['agent']}] v{idea.get('version',1)} "
            f"分数:{idea.get('score',0)}/10 迭代:{iterations} 剩余:{remaining:.1f}s"
        )
        return {"accepted": True, "remaining": remaining}

    async def _timer(self):
        await asyncio.sleep(self.timeout)
        print(f"\n[Judge] ⏰ 截止！共 {len(self.submissions)} 个提案")

        # 停止所有 agent
        for name in self._agents:
            try:
                agent = await resolve(name)
                await agent.stop()
            except Exception as e:
                print(f"[Judge] 停止 agent '{name}' 时出错: {e}")

        await self._decide()

    async def _decide(self):
        if not self.submissions:
            self.result = {"error": "无提案"}
            self._done.set()
            return

        # 打印提案列表
        for i, s in enumerate(self.submissions, 1):
            idea = s["idea"]
            print(
                f"  {i}. [{idea['agent']}] {idea['title']} - {idea.get('score',0)}/10"
            )

        # 选择最高分
        best = max(self.submissions, key=lambda x: x["idea"].get("score", 0))
        self.result = {
            "selected": best["idea"]["agent"],
            "reason": f"分数最高 ({best['idea'].get('score',0)}/10)",
            "proposal": best["idea"]["proposal"],
            "count": len(self.submissions),
        }
        print(f"\n[Judge] 选择: [{self.result['selected']}]")
        self._done.set()

    async def get_result(self) -> dict:
        await self._done.wait()
        return self.result or {}


# ============================================================================
# IdeaAgent Actor
# ============================================================================


@remote
class IdeaAgent:
    def __init__(
        self,
        persona: str,
        question: str,
        judge: str,
        peers: list[str],
        max_rounds: int,
        target_score: int,
        mock: bool,
        collab: bool,
    ):
        self.persona = persona
        self.question = question
        self.judge_name = judge
        self.peers = peers
        self.max_rounds = max_rounds
        self.target_score = target_score
        self.mock = mock
        self.collab = collab

        self.idea: dict | None = None
        self.prev_idea: dict | None = None
        self.prev_score = 0
        self.iterations = 0
        self.collabs = 0
        self._stopped = False
        self._task: asyncio.Task | None = None
        self._result: dict | None = None

    async def start(self):
        if self._task:
            return
        self._task = asyncio.create_task(self._run())

    async def get_result(self) -> dict:
        if not self._task:
            return {"error": "not started"}
        if self._result:
            return self._result
        self._result = await self._task
        return self._result

    def stop(self):
        self._stopped = True
        print(f"  [{self.persona}] 🛑 停止")

    def assist(self, from_agent: str, context: dict) -> dict:
        """同步协助（基于当前想法快速返回）"""
        title = context.get("idea", {}).get("title", "")
        print(f"  [{self.persona}] 📨 收到 [{from_agent}] 请求: {title[:20]}")

        # 基于专业领域给建议
        suggestion = next(
            (s for k, s in MOCK_SUGGESTIONS.items() if k in self.persona),
            "建议多角度评估",
        )
        if self.idea:
            suggestion += f"。参考我的方案: {self.idea.get('title','')}"

        print(f"  [{self.persona}] 💡 回复 [{from_agent}]: {suggestion[:40]}...")
        return {"from": self.persona, "suggestion": suggestion}

    # -------------------- 主循环 --------------------

    async def _run(self) -> dict:
        start = time.time()

        # 1. 生成初始方案
        print(f"\n  [{self.persona}] 🚀 开始")
        self.idea = await self._generate()
        self.iterations = 1

        # 初始评审
        critique = await self._critique()
        self.idea["score"] = critique["score"]
        self.prev_score = critique["score"]
        self.prev_idea = self.idea.copy()

        # 2. 迭代改进
        while self.iterations < self.max_rounds and not self._stopped:
            if self.idea["score"] >= self.target_score:
                print(f"  [{self.persona}] ✅ 达标 {self.target_score}")
                break

            # 协作
            advice = []
            if self.collab and critique.get("consult"):
                advice = await self._collaborate(critique["consult"][:2])

            if self._stopped:
                break

            # 改进
            self.iterations += 1
            self.prev_idea = self.idea.copy()
            self.prev_score = self.idea["score"]

            self.idea = await self._refine(critique, advice)
            critique = await self._critique()
            new_score = critique["score"]
            self.idea["score"] = new_score

            # 分数对比
            if new_score < self.prev_score:
                print(f"  [{self.persona}] ⚠️ {self.prev_score}→{new_score} 回退")
                self.idea = self.prev_idea.copy()
            elif new_score > self.prev_score:
                print(f"  [{self.persona}] ⬆️ {self.prev_score}→{new_score}")

        # 3. 提交
        if self._stopped:
            return {"persona": self.persona, "accepted": False, "reason": "stopped"}

        elapsed = time.time() - start
        print(f"  [{self.persona}] 📤 提交 ({elapsed:.1f}s)")

        judge = await resolve(self.judge_name)
        result = await judge.submit(self.idea, self.iterations)

        return {
            "persona": self.persona,
            "accepted": result.get("accepted", False),
            "score": self.idea.get("score", 0),
            "iterations": self.iterations,
            "elapsed": elapsed,
            "collabs": self.collabs,
        }

    # -------------------- 核心方法 --------------------

    async def _generate(self) -> dict:
        if self.mock:
            await asyncio.sleep(random.uniform(0.3, 1.0))
            for key, (title, exps, risks) in MOCK_TEMPLATES.items():
                if key in self.persona:
                    break
            else:
                title, exps, risks = "通用方案", ["验证实验"], ["风险"]
            idea = {
                "agent": self.persona,
                "title": title,
                "proposal": f"{self.persona}: {title}",
                "experiments": exps,
                "risks": risks,
                "version": 1,
                "score": 0,
            }
        else:
            llm = await get_llm()
            prompt = f'你是{self.persona}，针对问题提出方案：{self.question}\n输出JSON：{{"title":"","proposal":"","experiments":[],"risks":[]}}'
            resp = await llm.ainvoke(prompt)
            data = parse_json(resp.content, {})
            idea = {
                "agent": self.persona,
                "title": data.get("title", "方案"),
                "proposal": data.get("proposal", ""),
                "experiments": data.get("experiments", []),
                "risks": data.get("risks", []),
                "version": 1,
                "score": 0,
            }

        self._log_idea(idea, "📝 生成")
        return idea

    async def _critique(self) -> dict:
        # 可咨询的专家（排除自己）
        available_experts = [p for p in PERSONAS if p != self.persona]

        if self.mock:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            score = min(
                10,
                max(
                    1,
                    5
                    + len(self.idea.get("experiments", []))
                    + self.idea.get("version", 1)
                    - 1,
                ),
            )
            issues = (
                ["实验不足"]
                if len(self.idea.get("experiments", [])) < 3
                else ["方案可行"]
            )
            consult = [p for p in ["性能工程师", "安全专家"] if p != self.persona][:2]
            critique = {
                "score": score,
                "issues": issues,
                "improvements": ["增加实验"],
                "consult": consult,
            }
        else:
            llm = await get_llm()
            prompt = f"""评审以下方案，找出问题并给建议。

方案：{json.dumps(self.idea, ensure_ascii=False)}

可咨询的专家：{', '.join(available_experts[:5])}

输出JSON（不要markdown）：
{{"score": 1-10分, "issues": ["问题1"], "improvements": ["改进建议1"], "consult": ["建议咨询的专家名称"]}}"""
            resp = await llm.ainvoke(prompt)
            data = parse_json(resp.content, {})
            critique = {
                "score": data.get("score", 5),
                "issues": data.get("issues", []),
                "improvements": data.get("improvements", []),
                "consult": data.get("consult", []),
            }

        consult_info = (
            f" → 建议咨询: {critique['consult']}" if critique.get("consult") else ""
        )
        print(
            f"  [{self.persona}] 📋 评审: {critique['score']}/10 | {', '.join(str(i) for i in critique['issues'][:2])}{consult_info}"
        )
        return critique

    async def _refine(self, critique: dict, advice: list[dict]) -> dict:
        if self.mock:
            await asyncio.sleep(random.uniform(0.2, 0.5))
            exps = list(self.idea.get("experiments", []))
            exps.append(
                f"新增: {critique['improvements'][0][:10]}..."
                if critique.get("improvements")
                else "改进实验"
            )
            for a in advice:
                exps.append(f"协作: {a.get('suggestion', '')[:15]}...")
            refined = {
                **self.idea,
                "title": self.idea["title"] + " (改进)",
                "experiments": exps[:5],
                "version": self.idea.get("version", 1) + 1,
                "score": 0,
            }
        else:
            llm = await get_llm()
            collab_text = (
                "\n".join([f"- {a['from']}: {a['suggestion']}" for a in advice])
                if advice
                else ""
            )
            prompt = f"你是{self.persona}，改进方案：{json.dumps(self.idea, ensure_ascii=False)}\n问题：{critique['issues']}\n建议：{critique['improvements']}\n{collab_text}\n输出JSON"
            resp = await llm.ainvoke(prompt)
            data = parse_json(resp.content, {})
            refined = {
                "agent": self.persona,
                "title": data.get("title", self.idea["title"]),
                "proposal": data.get("proposal", self.idea["proposal"]),
                "experiments": data.get("experiments", []),
                "risks": data.get("risks", []),
                "version": self.idea.get("version", 1) + 1,
                "score": 0,
            }

        self._log_idea(refined, f"🔄 改进v{refined['version']}")
        return refined

    async def _collaborate(self, experts: list[str]) -> list[dict]:
        advice = []
        for expert in experts:
            if expert == self.persona:
                continue
            # 找到对应的 peer
            peer_name = next((n for n in self.peers if expert in n), None)
            if not peer_name:
                continue

            print(f"  [{self.persona}] 🤝 请求 [{expert}]")
            try:
                peer = await asyncio.wait_for(resolve(peer_name), timeout=5)
                resp = await asyncio.wait_for(
                    peer.assist(from_agent=self.persona, context={"idea": self.idea}),
                    timeout=10,
                )
                if resp.get("suggestion"):
                    advice.append(resp)
                    self.collabs += 1
                    print(
                        f"  [{self.persona}] ✓ 收到 [{resp['from']}]: {resp['suggestion'][:30]}..."
                    )
            except Exception as e:
                print(f"  [{self.persona}] ⚠ 协作失败: {e}")
        return advice

    def _log_idea(self, idea: dict, action: str):
        print(f"  [{self.persona}] {action}: {idea['title']}")
        experiments = idea.get("experiments", [])[:3]
        if experiments:
            if isinstance(experiments[0], dict):
                # 复杂实验格式
                for i, exp in enumerate(experiments, 1):
                    name = exp.get("name", exp.get("title", f"实验{i}"))
                    print(f"    {i}. {name}")
            else:
                # 简单字符串格式
                for i, exp in enumerate(experiments, 1):
                    print(f"    {i}. {exp[:50]}{'...' if len(str(exp)) > 50 else ''}")


# ============================================================================
# 入口
# ============================================================================


async def run(
    question: str,
    n: int = 3,
    rounds: int = 3,
    target: int = 8,
    timeout: float = 30,
    mock: bool = False,
    collab: bool = True,
):
    print("=" * 50)
    print(f"问题: {question}")
    print(
        f"Agent: {n} | 轮数: {rounds} | 目标: {target} | 超时: {timeout}s | 协作: {collab}"
    )
    print("=" * 50)

    async with runtime():
        # 创建 Judge
        judge = await JudgeActor.spawn(timeout=timeout, mock=mock, name="judge")

        # 创建所有 Agent
        agent_names = [f"agent_{p}" for p in PERSONAS]
        agents = []
        for i, persona in enumerate(PERSONAS):
            name = f"agent_{persona}"
            peers = [n for n in agent_names if n != name]
            agent = await IdeaAgent.spawn(
                persona=persona,
                question=question,
                judge="judge",
                peers=peers,
                max_rounds=rounds,
                target_score=target,
                mock=mock,
                collab=collab,
                name=name,
            )
            agents.append(agent)

        await judge.set_agents(agent_names)

        # 启动前 n 个 agent
        active = agents[:n]
        print(f"\n启动 {n} 个 Agent: {[PERSONAS[i] for i in range(n)]}")
        for a in active:
            await a.start()

        # 等待结果
        results = await asyncio.gather(
            *[a.get_result() for a in active], return_exceptions=True
        )
        final = await judge.get_result()

        # 输出
        print("\n" + "=" * 50)
        print(f"结果: [{final.get('selected')}] - {final.get('reason')}")
        print(f"提案: {final.get('count', 0)} 个")
        print("\nAgent 统计:")
        for r in results:
            if isinstance(r, dict):
                s = "✓" if r.get("accepted") else "✗"
                print(
                    f"  {s} [{r['persona']}] 分数:{r.get('score',0)} 迭代:{r.get('iterations')} 协作:{r.get('collabs',0)}"
                )

        return {"final": final, "agents": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="如何实现高性能多智能体工作流？")
    parser.add_argument("--n-ideas", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--target-score", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-collab", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        run(
            question=args.question,
            n=args.n_ideas,
            rounds=args.max_rounds,
            target=args.target_score,
            timeout=args.timeout,
            mock=args.mock,
            collab=not args.no_collab,
        )
    )
