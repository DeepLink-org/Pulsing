"""
Multi-Agent Discussion and Voting Example Based on MBTI Personality Types

Demonstrates the difference between @remote and @agent:
  - @remote: Basic Actor decorator
  - @agent: Actor with metadata (for visualization/debugging)

In this example:
  - ModeratorActor: Uses @remote (regular Actor)
  - MBTIAgent: Uses @agent (with MBTI role metadata)

Usage:
  python mbti_discussion.py --mock --topic "Remote work vs On-site work"
  python mbti_discussion.py --topic "Should AI have emotions" --group-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import random
from collections import Counter

from pulsing.actor import remote, resolve
from pulsing.agent import agent, runtime, llm, parse_json, list_agents

# ============================================================================
# MBTI Personality Configuration
# ============================================================================

MBTI_TYPES = {
    "INTJ": {
        "name": "Architect",
        "traits": "Strategic thinking, independent, efficiency-oriented",
        "persuade_rate": 0.1,
        "population": 2.1,
    },
    "INTP": {
        "name": "Logician",
        "traits": "Analytical, innovative, truth-seeking",
        "persuade_rate": 0.15,
        "population": 3.3,
    },
    "ENTJ": {
        "name": "Commander",
        "traits": "Decisive, leadership, efficient execution",
        "persuade_rate": 0.1,
        "population": 1.8,
    },
    "ENTP": {
        "name": "Debater",
        "traits": "Argumentative, innovative, challenges tradition",
        "persuade_rate": 0.2,
        "population": 3.2,
    },
    "INFJ": {
        "name": "Advocate",
        "traits": "Idealistic, insightful, meaning-focused",
        "persuade_rate": 0.25,
        "population": 1.5,
    },
    "INFP": {
        "name": "Mediator",
        "traits": "Idealistic, empathetic, harmony-seeking",
        "persuade_rate": 0.3,
        "population": 4.4,
    },
    "ENFJ": {
        "name": "Protagonist",
        "traits": "Charismatic, empathetic, inspiring",
        "persuade_rate": 0.35,
        "population": 2.5,
    },
    "ENFP": {
        "name": "Campaigner",
        "traits": "Enthusiastic, creative, communicative",
        "persuade_rate": 0.3,
        "population": 8.1,
    },
    "ISTJ": {
        "name": "Logistician",
        "traits": "Practical, reliable, detail-oriented",
        "persuade_rate": 0.1,
        "population": 11.6,
    },
    "ISFJ": {
        "name": "Defender",
        "traits": "Caring, responsible, tradition-focused",
        "persuade_rate": 0.2,
        "population": 13.8,
    },
    "ESTJ": {
        "name": "Executive",
        "traits": "Organized, practical, decisive",
        "persuade_rate": 0.1,
        "population": 8.7,
    },
    "ESFJ": {
        "name": "Consul",
        "traits": "Caring, cooperative, harmony-focused",
        "persuade_rate": 0.25,
        "population": 12.3,
    },
    "ISTP": {
        "name": "Virtuoso",
        "traits": "Calm, analytical, hands-on",
        "persuade_rate": 0.15,
        "population": 5.4,
    },
    "ISFP": {
        "name": "Adventurer",
        "traits": "Flexible, sensitive, aesthetic-seeking",
        "persuade_rate": 0.25,
        "population": 8.8,
    },
    "ESTP": {
        "name": "Entrepreneur",
        "traits": "Action-oriented, flexible, adaptable",
        "persuade_rate": 0.2,
        "population": 4.3,
    },
    "ESFP": {
        "name": "Entertainer",
        "traits": "Energetic, optimistic, social",
        "persuade_rate": 0.25,
        "population": 8.5,
    },
}

PERSUASION_MESSAGES = {
    "INTJ": "From a long-term strategic perspective, this is a better choice",
    "INTP": "From a logical analysis, there's a flaw in your argument",
    "ENTJ": "We need decisive action, hesitation will miss opportunities",
    "ENTP": "Let's think about this from another angle",
    "INFJ": "Considering the long-term impact on society, we should reconsider",
    "INFP": "If we look at it from others' perspectives, we might understand differently",
    "ENFJ": "For the team's common goal, I suggest we find consensus",
    "ENFP": "Imagine how many possibilities there would be if we did this!",
    "ISTJ": "Based on past experience and data, this is a more reliable choice",
    "ISFJ": "Considering the impact on everyone, I think we need to be cautious",
    "ESTJ": "To improve efficiency, we need clear plans and execution",
    "ESFJ": "To maintain team harmony, perhaps we can both compromise",
    "ISTP": "From a technical feasibility perspective, this is the most practical solution",
    "ISFP": "Everyone's feelings matter, but results are equally important",
    "ESTP": "Rather than hesitating, let's try it and see the results",
    "ESFP": "I think this choice would be more interesting and effective!",
}


def sample_mbti_group(size: int) -> list[str]:
    types = list(MBTI_TYPES.keys())
    weights = [MBTI_TYPES[t]["population"] for t in types]
    return random.choices(types, weights=weights, k=size)


# ============================================================================
# Moderator - Uses @remote (Regular Actor, no metadata)
# ============================================================================


@remote
class ModeratorActor:
    """Moderator Actor: Coordinates the entire discussion process (uses @remote)"""

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
            print(f"\n{'=' * 60}")
            print(f"Round {r + 1}: Express Opinions")
            print(f"{'=' * 60}")

            for agent_info in self.agents:
                proxy = await resolve(agent_info["name"])
                await proxy.form_opinion(self.opinions[-10:])

            print(f"\n{'=' * 60}")
            print(f"Round {r + 1}: Free Debate ({self.debate_time}s)")
            print(f"{'=' * 60}")

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

        print(f"\n{'=' * 60}")
        print("Final Voting")
        print(f"{'=' * 60}")

        for agent_info in self.agents:
            proxy = await resolve(agent_info["name"])
            await proxy.vote()

        return self._summarize()

    def _summarize(self) -> dict:
        print(f"\n{'=' * 60}")
        print("Voting Results")
        print(f"{'=' * 60}")

        total = sum(len(v) for v in self.votes.values())
        sorted_votes = sorted(self.votes.items(), key=lambda x: len(x[1]), reverse=True)

        for stance, voters in sorted_votes:
            pct = len(voters) / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {stance:10} {bar} {len(voters)} votes ({pct:.0f}%)")
            print(f"            └─ {', '.join(voters)}")

        winner = sorted_votes[0][0] if sorted_votes else "None"
        success = sum(1 for d in self.debates if d.get("changed"))

        print(
            f"\nDebate Statistics: {len(self.debates)} exchanges, {success} successful persuasions"
        )
        print(f"\n{'=' * 60}")
        print(f"Final Result: {winner} wins")
        print(f"{'=' * 60}")

        return {"winner": winner, "votes": self.votes, "debates": len(self.debates)}


# ============================================================================
# MBTI Agent - Uses @agent (with metadata, can be used for visualization)
# ============================================================================


@agent(
    role="MBTI Participant",
    goal="Participate in discussion based on personality traits",
    backstory="Express views according to MBTI personality type",
)
class MBTIAgent:
    """MBTI Agent: Autonomous Actor participating in discussion (uses @agent, with metadata)"""

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
                stance = random.choice(["Support", "Oppose"])
            elif self.mbti in ["INFJ", "INFP", "ENFJ", "ENFP"]:
                stance = random.choice(["Support", "Conditional Support", "Neutral"])
            else:
                stance = random.choice(["Neutral", "Conditional Support", "Oppose"])

            args = {
                "Support": f"As {self.info['name']}, I think this helps improve efficiency and innovation",
                "Oppose": f"As {self.info['name']}, I'm concerned this will bring uncontrollable risks",
                "Neutral": f"As {self.info['name']}, I think we need more information to make a judgment",
                "Conditional Support": f"As {self.info['name']}, I support but need clear boundaries",
            }
            self.stance = stance
            self.argument = args.get(stance, "Needs discussion")
        else:
            client = await llm(temperature=0.8)
            ctx = (
                "\n".join([f"- {o['mbti']}: {o['stance']}" for o in others[-5:]])
                if others
                else "None"
            )
            prompt = f"""You are {self.mbti} ({self.info["name"]}), personality traits: {self.info["traits"]}.
Topic: {self.topic}
Others' views:
{ctx}
Please express your view on the topic based on your personality traits. Output JSON: {{"stance": "Support/Oppose/Neutral/Conditional Support", "argument": "Argument based on topic (within 40 characters)"}}"""
            resp = await client.ainvoke(prompt)
            data = parse_json(resp.content, {})
            self.stance = data.get("stance", "Neutral")
            self.argument = data.get("argument", "Needs discussion")

        moderator = await resolve(self.moderator_name)
        await moderator.submit_opinion(self.name, self.mbti, self.stance, self.argument)
        return {"mbti": self.mbti, "stance": self.stance}

    async def debate(self, target: dict) -> dict:
        target_mbti = target["mbti"]
        target_stance = target["stance"]

        if self.mock:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            message = PERSUASION_MESSAGES.get(self.mbti, "Please consider my view")
            target_rate = MBTI_TYPES[target_mbti]["persuade_rate"]
            changed = random.random() < target_rate
            if changed:
                reply = random.choice(
                    [
                        "That makes sense, I agree",
                        "You convinced me",
                        "OK, I changed my mind",
                    ]
                )
            else:
                reply = random.choice(
                    ["I disagree", "Insufficient evidence", "I stand firm"]
                )
        else:
            client = await llm(temperature=0.8)
            prompt1 = f"""You are {self.mbti}, the topic is "{self.topic}", your stance is {self.stance}.
The other party {target_mbti}'s stance is {target_stance}. Persuade them in one sentence. Output JSON: {{"message": "Persuasion message (within 30 characters)"}}"""
            resp1 = await client.ainvoke(prompt1)
            message = parse_json(resp1.content, {}).get(
                "message", "Please consider my view"
            )

            target_info = MBTI_TYPES[target_mbti]
            prompt2 = f"""You are {target_mbti} ({target_info["name"]}), {self.mbti} says: "{message}".
Based on your personality traits ({target_info["traits"]}), would you be persuaded? Output JSON: {{"changed": true/false, "reply": "Reply (within 10 characters)"}}"""
            resp2 = await client.ainvoke(prompt2)
            data = parse_json(resp2.content, {})
            changed = data.get("changed", False)
            reply = data.get("reply", "I stand firm")

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
        await moderator.submit_vote(self.mbti, self.stance or "Abstain")
        return {"mbti": self.mbti, "vote": self.stance}


# ============================================================================
# Main Flow
# ============================================================================


async def run(
    topic: str,
    group_size: int = 6,
    rounds: int = 2,
    debate_time: float = 5.0,
    mock: bool = False,
):
    print("=" * 60)
    print("MBTI Personality Type Discussion and Voting")
    print("=" * 60)
    print(f"Topic: {topic}")
    print(
        f"Participants: {group_size} | Rounds: {rounds} | Debate: {debate_time}s/round"
    )
    print(f"Mode: {'Mock' if mock else 'LLM'}")

    async with runtime():
        mbti_group = sample_mbti_group(group_size)
        dist = Counter(mbti_group)
        print("\nGroup:")
        for mbti, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {mbti} ({MBTI_TYPES[mbti]['name']}): {count}")

        # Create moderator (@remote)
        moderator = await ModeratorActor.spawn(
            topic=topic,
            rounds=rounds,
            debate_time=debate_time,
            mock=mock,
            name="moderator",
        )

        # Create participants (@agent, with metadata)
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

        # Show @agent metadata functionality
        print("\nRegistered Agents (via metadata):")
        for name, meta in list_agents().items():
            print(f"  {name}: {meta.role}")

        # Start discussion
        result = await moderator.start_discussion()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic", default="Should remote work become the mainstream work style?"
    )
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
