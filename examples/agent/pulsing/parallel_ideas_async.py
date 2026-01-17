"""
Pulsing Parallel Multi-Agent Example

Architecture: N IdeaAgents iterate in parallel, competing to submit to Judge

Usage:
  python parallel_ideas_async.py --mock                    # Mock mode
  python parallel_ideas_async.py --question "question"     # Real LLM
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
# Configuration
# ============================================================================

PERSONAS = [
    "Performance Engineer",
    "Product Manager",
    "System Architect",
    "Security Expert",
    "Reliability Ops",
    "Researcher",
    "Data Engineer",
    "Cost Controller",
    "UX Designer",
    "Test Engineer",
]

# Mock mode templates
MOCK_TEMPLATES = {
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
        "Resilience solutions",
        ["Fault injection testing", "Retry strategy verification"],
        ["Retry storms"],
    ),
    "Product": (
        "Streaming experience",
        ["TTFB optimization", "A/B testing"],
        ["Increased complexity"],
    ),
    "Cost": (
        "Cost-driven architecture",
        ["Cost curve analysis", "On-demand scaling"],
        ["Quality degradation"],
    ),
}

MOCK_SUGGESTIONS = {
    "Performance": "Suggest adding cache layer and connection pool optimization",
    "Security": "Suggest adding mTLS and audit logs",
    "Reliability": "Suggest adding circuit breakers and retry mechanisms",
    "Product": "Suggest adding streaming output and progress feedback",
    "Cost": "Suggest evaluating on-demand scaling strategy",
}

# ============================================================================
# LLM Utilities
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

        # Check deadline
        if self.deadline and now > self.deadline:
            print(f"  ❌ [{idea['agent']}] Rejected (timeout)")
            return {"accepted": False}

        self.submissions.append({"idea": idea, "iterations": iterations, "time": now})

        # First submission starts timer
        if not self.deadline:
            self.deadline = now + self.timeout
            print(
                f"\n[Judge] Received first proposal [{idea['agent']}], deadline in {self.timeout}s"
            )
            asyncio.create_task(self._timer())

        remaining = self.deadline - now
        print(
            f"  ✓ [{idea['agent']}] v{idea.get('version',1)} "
            f"score:{idea.get('score',0)}/10 iterations:{iterations} remaining:{remaining:.1f}s"
        )
        return {"accepted": True, "remaining": remaining}

    async def _timer(self):
        await asyncio.sleep(self.timeout)
        print(f"\n[Judge] ⏰ Deadline! Total {len(self.submissions)} proposals")

        # Stop all agents
        for name in self._agents:
            try:
                agent = await resolve(name)
                await agent.stop()
            except Exception as e:
                print(f"[Judge] Error stopping agent '{name}': {e}")

        await self._decide()

    async def _decide(self):
        if not self.submissions:
            self.result = {"error": "No proposals"}
            self._done.set()
            return

        # Print proposal list
        for i, s in enumerate(self.submissions, 1):
            idea = s["idea"]
            print(
                f"  {i}. [{idea['agent']}] {idea['title']} - {idea.get('score',0)}/10"
            )

        # Select highest score
        best = max(self.submissions, key=lambda x: x["idea"].get("score", 0))
        self.result = {
            "selected": best["idea"]["agent"],
            "reason": f"Highest score ({best['idea'].get('score',0)}/10)",
            "proposal": best["idea"]["proposal"],
            "count": len(self.submissions),
        }
        print(f"\n[Judge] Selected: [{self.result['selected']}]")
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
        print(f"  [{self.persona}] 🛑 Stopped")

    def assist(self, from_agent: str, context: dict) -> dict:
        """Synchronous assistance (quick return based on current idea)"""
        title = context.get("idea", {}).get("title", "")
        print(
            f"  [{self.persona}] 📨 Received request from [{from_agent}]: {title[:20]}"
        )

        # Give suggestions based on professional field
        suggestion = next(
            (s for k, s in MOCK_SUGGESTIONS.items() if k in self.persona),
            "Suggest multi-angle evaluation",
        )
        if self.idea:
            suggestion += f". Reference my proposal: {self.idea.get('title','')}"

        print(f"  [{self.persona}] 💡 Reply to [{from_agent}]: {suggestion[:40]}...")
        return {"from": self.persona, "suggestion": suggestion}

    # -------------------- Main Loop --------------------

    async def _run(self) -> dict:
        start = time.time()

        # 1. Generate initial proposal
        print(f"\n  [{self.persona}] 🚀 Started")
        self.idea = await self._generate()
        self.iterations = 1

        # Initial critique
        critique = await self._critique()
        self.idea["score"] = critique["score"]
        self.prev_score = critique["score"]
        self.prev_idea = self.idea.copy()

        # 2. Iterative improvement
        while self.iterations < self.max_rounds and not self._stopped:
            if self.idea["score"] >= self.target_score:
                print(f"  [{self.persona}] ✅ Reached target {self.target_score}")
                break

            # Collaboration
            advice = []
            if self.collab and critique.get("consult"):
                advice = await self._collaborate(critique["consult"][:2])

            if self._stopped:
                break

            # Refine
            self.iterations += 1
            self.prev_idea = self.idea.copy()
            self.prev_score = self.idea["score"]

            self.idea = await self._refine(critique, advice)
            critique = await self._critique()
            new_score = critique["score"]
            self.idea["score"] = new_score

            # Score comparison
            if new_score < self.prev_score:
                print(f"  [{self.persona}] ⚠️ {self.prev_score}→{new_score} Reverted")
                self.idea = self.prev_idea.copy()
            elif new_score > self.prev_score:
                print(f"  [{self.persona}] ⬆️ {self.prev_score}→{new_score}")

        # 3. Submit
        if self._stopped:
            return {"persona": self.persona, "accepted": False, "reason": "stopped"}

        elapsed = time.time() - start
        print(f"  [{self.persona}] 📤 Submitted ({elapsed:.1f}s)")

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

    # -------------------- Core Methods --------------------

    async def _generate(self) -> dict:
        if self.mock:
            await asyncio.sleep(random.uniform(0.3, 1.0))
            for key, (title, exps, risks) in MOCK_TEMPLATES.items():
                if key in self.persona:
                    break
            else:
                title, exps, risks = (
                    "Generic solution",
                    ["Validation experiment"],
                    ["Risk"],
                )
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
            prompt = f'You are {self.persona}, propose a solution for: {self.question}\nOutput JSON: {{"title":"","proposal":"","experiments":[],"risks":[]}}'
            resp = await llm.ainvoke(prompt)
            data = parse_json(resp.content, {})
            idea = {
                "agent": self.persona,
                "title": data.get("title", "Solution"),
                "proposal": data.get("proposal", ""),
                "experiments": data.get("experiments", []),
                "risks": data.get("risks", []),
                "version": 1,
                "score": 0,
            }

        self._log_idea(idea, "📝 Generated")
        return idea

    async def _critique(self) -> dict:
        # Available experts to consult (exclude self)
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
                ["Insufficient experiments"]
                if len(self.idea.get("experiments", [])) < 3
                else ["Solution feasible"]
            )
            consult = [
                p
                for p in ["Performance Engineer", "Security Expert"]
                if p != self.persona
            ][:2]
            critique = {
                "score": score,
                "issues": issues,
                "improvements": ["Add experiments"],
                "consult": consult,
            }
        else:
            llm = await get_llm()
            prompt = f"""Review the following proposal, identify issues and give suggestions.

Proposal: {json.dumps(self.idea, ensure_ascii=False)}

Available experts: {', '.join(available_experts[:5])}

Output JSON (no markdown):
{{"score": 1-10, "issues": ["issue1"], "improvements": ["improvement1"], "consult": ["expert name to consult"]}}"""
            resp = await llm.ainvoke(prompt)
            data = parse_json(resp.content, {})
            critique = {
                "score": data.get("score", 5),
                "issues": data.get("issues", []),
                "improvements": data.get("improvements", []),
                "consult": data.get("consult", []),
            }

        consult_info = (
            f" → Suggest consulting: {critique['consult']}"
            if critique.get("consult")
            else ""
        )
        print(
            f"  [{self.persona}] 📋 Critique: {critique['score']}/10 | {', '.join(str(i) for i in critique['issues'][:2])}{consult_info}"
        )
        return critique

    async def _refine(self, critique: dict, advice: list[dict]) -> dict:
        if self.mock:
            await asyncio.sleep(random.uniform(0.2, 0.5))
            exps = list(self.idea.get("experiments", []))
            exps.append(
                f"New: {critique['improvements'][0][:10]}..."
                if critique.get("improvements")
                else "Improved experiment"
            )
            for a in advice:
                exps.append(f"Collaboration: {a.get('suggestion', '')[:15]}...")
            refined = {
                **self.idea,
                "title": self.idea["title"] + " (Improved)",
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
            prompt = f"You are {self.persona}, improve the proposal: {json.dumps(self.idea, ensure_ascii=False)}\nIssues: {critique['issues']}\nSuggestions: {critique['improvements']}\n{collab_text}\nOutput JSON"
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

        self._log_idea(refined, f"🔄 Refined v{refined['version']}")
        return refined

    async def _collaborate(self, experts: list[str]) -> list[dict]:
        advice = []
        for expert in experts:
            if expert == self.persona:
                continue
            # Find corresponding peer
            peer_name = next((n for n in self.peers if expert in n), None)
            if not peer_name:
                continue

            print(f"  [{self.persona}] 🤝 Requesting [{expert}]")
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
                        f"  [{self.persona}] ✓ Received from [{resp['from']}]: {resp['suggestion'][:30]}..."
                    )
            except Exception as e:
                print(f"  [{self.persona}] ⚠ Collaboration failed: {e}")
        return advice

    def _log_idea(self, idea: dict, action: str):
        print(f"  [{self.persona}] {action}: {idea['title']}")
        experiments = idea.get("experiments", [])[:3]
        if experiments:
            if isinstance(experiments[0], dict):
                # Complex experiment format
                for i, exp in enumerate(experiments, 1):
                    name = exp.get("name", exp.get("title", f"Experiment {i}"))
                    print(f"    {i}. {name}")
            else:
                # Simple string format
                for i, exp in enumerate(experiments, 1):
                    print(f"    {i}. {exp[:50]}{'...' if len(str(exp)) > 50 else ''}")


# ============================================================================
# Entry Point
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
    print(f"Question: {question}")
    print(
        f"Agents: {n} | Rounds: {rounds} | Target: {target} | Timeout: {timeout}s | Collaboration: {collab}"
    )
    print("=" * 50)

    async with runtime():
        # Create Judge
        judge = await JudgeActor.spawn(timeout=timeout, mock=mock, name="judge")

        # Create all Agents
        # Sanitize persona names for use in actor names (replace spaces with underscores)
        agent_names = [f"agent_{p.replace(' ', '_')}" for p in PERSONAS]
        agents = []
        for i, persona in enumerate(PERSONAS):
            # Use sanitized name for actor name
            name = f"agent_{persona.replace(' ', '_')}"
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

        # Start first n agents
        active = agents[:n]
        print(f"\nStarting {n} Agents: {[PERSONAS[i] for i in range(n)]}")
        for a in active:
            await a.start()

        # Wait for results
        results = await asyncio.gather(
            *[a.get_result() for a in active], return_exceptions=True
        )
        final = await judge.get_result()

        # Output
        print("\n" + "=" * 50)
        print(f"Result: [{final.get('selected')}] - {final.get('reason')}")
        print(f"Proposals: {final.get('count', 0)}")
        print("\nAgent Statistics:")
        for r in results:
            if isinstance(r, dict):
                s = "✓" if r.get("accepted") else "✗"
                print(
                    f"  {s} [{r['persona']}] score:{r.get('score',0)} iterations:{r.get('iterations')} collabs:{r.get('collabs',0)}"
                )

        return {"final": final, "agents": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question", default="How to implement high-performance multi-agent workflows?"
    )
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
