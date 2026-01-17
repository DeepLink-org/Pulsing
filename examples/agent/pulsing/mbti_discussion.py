"""
基于 MBTI 人格类型的多智能体讨论与投票示例

演示 @remote 与 @agent 的区别：
  - @remote: 基础 Actor 装饰器
  - @agent: 带元信息的 Actor（用于可视化/调试）

本示例中：
  - ModeratorActor: 使用 @remote（普通 Actor）
  - MBTIAgent: 使用 @agent（附带 MBTI 角色元信息）

Usage:
  python mbti_discussion.py --mock --topic "远程办公 vs 现场办公"
  python mbti_discussion.py --topic "AI 是否应该有情感" --group-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import random
from collections import Counter

from pulsing.actor import remote, resolve
from pulsing.agent import agent, runtime, llm, parse_json, list_agents

# ============================================================================
# MBTI 人格配置
# ============================================================================

MBTI_TYPES = {
    "INTJ": {
        "name": "建筑师",
        "traits": "战略思维、独立、追求效率",
        "persuade_rate": 0.1,
        "population": 2.1,
    },
    "INTP": {
        "name": "逻辑学家",
        "traits": "分析型、创新、追求真理",
        "persuade_rate": 0.15,
        "population": 3.3,
    },
    "ENTJ": {
        "name": "指挥官",
        "traits": "果断、领导力、高效执行",
        "persuade_rate": 0.1,
        "population": 1.8,
    },
    "ENTP": {
        "name": "辩论家",
        "traits": "善辩、创新、挑战传统",
        "persuade_rate": 0.2,
        "population": 3.2,
    },
    "INFJ": {
        "name": "提倡者",
        "traits": "理想主义、洞察力、关注意义",
        "persuade_rate": 0.25,
        "population": 1.5,
    },
    "INFP": {
        "name": "调停者",
        "traits": "理想主义、同理心、追求和谐",
        "persuade_rate": 0.3,
        "population": 4.4,
    },
    "ENFJ": {
        "name": "主人公",
        "traits": "魅力、同理心、善于激励",
        "persuade_rate": 0.35,
        "population": 2.5,
    },
    "ENFP": {
        "name": "竞选者",
        "traits": "热情、创造力、善于沟通",
        "persuade_rate": 0.3,
        "population": 8.1,
    },
    "ISTJ": {
        "name": "物流师",
        "traits": "务实、可靠、注重细节",
        "persuade_rate": 0.1,
        "population": 11.6,
    },
    "ISFJ": {
        "name": "守卫者",
        "traits": "关怀、尽责、注重传统",
        "persuade_rate": 0.2,
        "population": 13.8,
    },
    "ESTJ": {
        "name": "总经理",
        "traits": "组织能力强、务实、果断",
        "persuade_rate": 0.1,
        "population": 8.7,
    },
    "ESFJ": {
        "name": "执政官",
        "traits": "关怀、合作、注重和谐",
        "persuade_rate": 0.25,
        "population": 12.3,
    },
    "ISTP": {
        "name": "鉴赏家",
        "traits": "冷静、分析、动手能力强",
        "persuade_rate": 0.15,
        "population": 5.4,
    },
    "ISFP": {
        "name": "探险家",
        "traits": "灵活、敏感、追求美感",
        "persuade_rate": 0.25,
        "population": 8.8,
    },
    "ESTP": {
        "name": "企业家",
        "traits": "行动派、灵活、善于应变",
        "persuade_rate": 0.2,
        "population": 4.3,
    },
    "ESFP": {
        "name": "表演者",
        "traits": "活力、乐观、善于社交",
        "persuade_rate": 0.25,
        "population": 8.5,
    },
}

PERSUASION_MESSAGES = {
    "INTJ": "从长远战略来看，这是更优的选择",
    "INTP": "从逻辑分析来看，你的论点有一个漏洞",
    "ENTJ": "我们需要果断行动，犹豫会错失机会",
    "ENTP": "让我们从另一个角度思考这个问题",
    "INFJ": "考虑到对社会的长远影响，我们应该重新考虑",
    "INFP": "如果我们站在他人的角度来看，或许会有不同的理解",
    "ENFJ": "为了团队的共同目标，我建议我们寻找共识",
    "ENFP": "想象一下如果我们这样做，会有多少可能性！",
    "ISTJ": "根据过去的经验和数据，这是更可靠的选择",
    "ISFJ": "考虑到对大家的影响，我认为我们需要谨慎",
    "ESTJ": "为了提高效率，我们需要明确的计划和执行",
    "ESFJ": "为了维护团队和谐，也许我们可以各退一步",
    "ISTP": "从技术可行性来看，这是最实际的方案",
    "ISFP": "每个人的感受都很重要，但结果同样重要",
    "ESTP": "与其犹豫不决，不如先尝试看看效果",
    "ESFP": "我觉得这个选择会更有趣也更有效！",
}


def sample_mbti_group(size: int) -> list[str]:
    types = list(MBTI_TYPES.keys())
    weights = [MBTI_TYPES[t]["population"] for t in types]
    return random.choices(types, weights=weights, k=size)


# ============================================================================
# Moderator - 使用 @remote（普通 Actor，无元信息）
# ============================================================================


@remote
class ModeratorActor:
    """主持人 Actor：协调整个讨论流程（使用 @remote）"""

    def __init__(self, topic: str, rounds: int, debate_time: float, mock: bool):
        self.topic = topic
        self.rounds = rounds
        self.debate_time = debate_time
        self.mock = mock
        self.agents: list[dict] = []
        self.opinions: list[dict] = []
        self.debates: list[dict] = []
        self.votes: dict[str, list[str]] = {}

    async def register_agent(self, name: str, mbti: str) -> dict:
        self.agents.append({"name": name, "mbti": mbti})
        return {"status": "registered"}

    async def submit_opinion(
        self, name: str, mbti: str, stance: str, argument: str
    ) -> dict:
        self.opinions.append(
            {"name": name, "mbti": mbti, "stance": stance, "argument": argument}
        )
        print(f"  [{mbti}] 📝 {stance}")
        print(f"       └─ {argument}")
        return {"received": True}

    async def submit_vote(self, mbti: str, vote: str) -> dict:
        print(f"  [{mbti}] 🗳️ {vote}")
        if vote not in self.votes:
            self.votes[vote] = []
        self.votes[vote].append(mbti)
        return {"received": True}

    async def start_discussion(self) -> dict:
        for r in range(self.rounds):
            print(f"\n{'='*60}")
            print(f"第 {r+1} 轮：发表观点")
            print(f"{'='*60}")

            for agent_info in self.agents:
                proxy = await resolve(agent_info["name"])
                await proxy.form_opinion(self.opinions[-10:])

            print(f"\n{'='*60}")
            print(f"第 {r+1} 轮：自由辩论（{self.debate_time}s）")
            print(f"{'='*60}")

            for agent_info in self.agents:
                my_opinion = next(
                    (o for o in self.opinions if o["name"] == agent_info["name"]), None
                )
                if not my_opinion:
                    continue

                opponents = [
                    o
                    for o in self.opinions
                    if o["stance"] != my_opinion["stance"]
                    and o["name"] != agent_info["name"]
                ]
                if not opponents:
                    opponents = [
                        o for o in self.opinions if o["name"] != agent_info["name"]
                    ]
                if not opponents:
                    continue

                target = random.choice(opponents)
                proxy = await resolve(agent_info["name"])
                result = await proxy.debate(target)

                if result.get("success"):
                    self.debates.append(result)
                    icon = "✅" if result["changed"] else "❌"
                    print(
                        f"  [{result['from']}] 💬 → [{result['to']}]: {result['message'][:40]}..."
                    )
                    print(f"       └─ {icon} [{result['to']}]: {result['reply']}")

        print(f"\n{'='*60}")
        print("最终投票")
        print(f"{'='*60}")

        for agent_info in self.agents:
            proxy = await resolve(agent_info["name"])
            await proxy.vote()

        return self._summarize()

    def _summarize(self) -> dict:
        print(f"\n{'='*60}")
        print("投票结果")
        print(f"{'='*60}")

        total = sum(len(v) for v in self.votes.values())
        sorted_votes = sorted(self.votes.items(), key=lambda x: len(x[1]), reverse=True)

        for stance, voters in sorted_votes:
            pct = len(voters) / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {stance:10} {bar} {len(voters)}票 ({pct:.0f}%)")
            print(f"            └─ {', '.join(voters)}")

        winner = sorted_votes[0][0] if sorted_votes else "无"
        success = sum(1 for d in self.debates if d.get("changed"))

        print(f"\n辩论统计: {len(self.debates)} 次交流, {success} 次成功说服")
        print(f"\n{'='*60}")
        print(f"最终结果: {winner} 获胜")
        print(f"{'='*60}")

        return {"winner": winner, "votes": self.votes, "debates": len(self.debates)}


# ============================================================================
# MBTI Agent - 使用 @agent（附带元信息，可用于可视化）
# ============================================================================


@agent(
    role="MBTI 参与者",
    goal="基于人格特点参与讨论",
    backstory="根据 MBTI 性格类型表达观点",
)
class MBTIAgent:
    """MBTI Agent：自主参与讨论的 Actor（使用 @agent，附带元信息）"""

    def __init__(
        self, agent_name: str, mbti: str, topic: str, moderator: str, mock: bool
    ):
        self.name = agent_name
        self.mbti = mbti
        self.info = MBTI_TYPES[mbti]
        self.topic = topic
        self.moderator_name = moderator
        self.mock = mock
        self.stance: str = ""
        self.argument: str = ""

    async def form_opinion(self, others: list[dict]) -> dict:
        if self.mock:
            await asyncio.sleep(random.uniform(0.05, 0.15))
            if self.mbti in ["INTJ", "INTP", "ENTJ", "ENTP"]:
                stance = random.choice(["支持", "反对"])
            elif self.mbti in ["INFJ", "INFP", "ENFJ", "ENFP"]:
                stance = random.choice(["支持", "有条件支持", "中立"])
            else:
                stance = random.choice(["中立", "有条件支持", "反对"])

            args = {
                "支持": f"作为{self.info['name']}，我认为这有利于提高效率和创新",
                "反对": f"作为{self.info['name']}，我担心这会带来不可控的风险",
                "中立": f"作为{self.info['name']}，我认为需要更多信息才能做出判断",
                "有条件支持": f"作为{self.info['name']}，我支持但需要设立明确边界",
            }
            self.stance = stance
            self.argument = args.get(stance, "需要讨论")
        else:
            client = await llm(temperature=0.8)
            ctx = (
                "\n".join([f"- {o['mbti']}: {o['stance']}" for o in others[-5:]])
                if others
                else "暂无"
            )
            prompt = f"""你是 {self.mbti} ({self.info['name']})，性格特点：{self.info['traits']}。
议题：{self.topic}
其他人观点：
{ctx}
请基于你的性格特点，对议题发表观点。输出JSON：{{"stance": "支持/反对/中立/有条件支持", "argument": "基于议题的论点（40字内）"}}"""
            resp = await client.ainvoke(prompt)
            data = parse_json(resp.content, {})
            self.stance = data.get("stance", "中立")
            self.argument = data.get("argument", "需要讨论")

        moderator = await resolve(self.moderator_name)
        await moderator.submit_opinion(self.name, self.mbti, self.stance, self.argument)
        return {"mbti": self.mbti, "stance": self.stance}

    async def debate(self, target: dict) -> dict:
        target_mbti = target["mbti"]
        target_stance = target["stance"]

        if self.mock:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            message = PERSUASION_MESSAGES.get(self.mbti, "请考虑我的观点")
            target_rate = MBTI_TYPES[target_mbti]["persuade_rate"]
            changed = random.random() < target_rate
            if changed:
                reply = random.choice(
                    ["有道理，我同意", "你说服我了", "好的，我改变想法"]
                )
            else:
                reply = random.choice(["我不同意", "论据不足", "我坚持立场"])
        else:
            client = await llm(temperature=0.8)
            prompt1 = f"""你是{self.mbti}，议题是"{self.topic}"，你的立场是{self.stance}。
对方{target_mbti}立场是{target_stance}。用一句话说服对方。输出JSON：{{"message": "说服话术（30字内）"}}"""
            resp1 = await client.ainvoke(prompt1)
            message = parse_json(resp1.content, {}).get("message", "请考虑我的观点")

            target_info = MBTI_TYPES[target_mbti]
            prompt2 = f"""你是{target_mbti}（{target_info['name']}），{self.mbti}说："{message}"。
基于你的性格特点（{target_info['traits']}），你会被说服吗？输出JSON：{{"changed": true/false, "reply": "回复（10字内）"}}"""
            resp2 = await client.ainvoke(prompt2)
            data = parse_json(resp2.content, {})
            changed = data.get("changed", False)
            reply = data.get("reply", "我坚持立场")

        return {
            "success": True,
            "from": self.mbti,
            "to": target_mbti,
            "message": message,
            "reply": reply,
            "changed": changed,
        }

    async def vote(self) -> dict:
        if self.mock:
            await asyncio.sleep(random.uniform(0.02, 0.05))
        moderator = await resolve(self.moderator_name)
        await moderator.submit_vote(self.mbti, self.stance or "弃权")
        return {"mbti": self.mbti, "vote": self.stance}


# ============================================================================
# 主流程
# ============================================================================


async def run(
    topic: str,
    group_size: int = 6,
    rounds: int = 2,
    debate_time: float = 5.0,
    mock: bool = False,
):
    print("=" * 60)
    print("MBTI 人格类型讨论与投票")
    print("=" * 60)
    print(f"议题: {topic}")
    print(f"人数: {group_size} | 轮数: {rounds} | 辩论: {debate_time}s/轮")
    print(f"模式: {'模拟' if mock else 'LLM'}")

    async with runtime():
        mbti_group = sample_mbti_group(group_size)
        dist = Counter(mbti_group)
        print("\n小组:")
        for mbti, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {mbti} ({MBTI_TYPES[mbti]['name']}): {count}人")

        # 创建主持人（@remote）
        moderator = await ModeratorActor.spawn(
            topic=topic,
            rounds=rounds,
            debate_time=debate_time,
            mock=mock,
            name="moderator",
        )

        # 创建参与者（@agent，附带元信息）
        for i, mbti in enumerate(mbti_group):
            agent_name = f"agent_{i}_{mbti}"
            await MBTIAgent.spawn(
                agent_name=agent_name,
                mbti=mbti,
                topic=topic,
                moderator="moderator",
                mock=mock,
                name=agent_name,
            )
            await moderator.register_agent(agent_name, mbti)

        # 展示 @agent 的元信息功能
        print("\n已注册的 Agent（通过元信息）:")
        for name, meta in list_agents().items():
            print(f"  {name}: {meta.role}")

        # 启动讨论
        result = await moderator.start_discussion()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="远程办公是否应该成为主流工作方式？")
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--debate-time", type=float, default=5.0)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        run(
            topic=args.topic,
            group_size=args.group_size,
            rounds=args.rounds,
            debate_time=args.debate_time,
            mock=args.mock,
        )
    )
