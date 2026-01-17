"""
🗣️ AI Chat Room - Multiple AI Agents Free Conversation

Run: python examples/quickstart/ai_chat_room.py

Watch AIs with different personalities discuss a topic!

Optional arguments:
  --topic "your topic"  Set discussion topic
  --rounds 5            Set number of discussion rounds
"""

import argparse
import asyncio
import random
from pulsing.actor import remote, resolve
from pulsing.agent import runtime

# AI persona configuration
AI_PERSONAS = {
    "Optimist": {
        "emoji": "😊",
        "style": "Always sees the positive side, speaks with encouragement",
        "phrases": [
            "I think this is great!",
            "Looking on the bright side...",
            "This is a good opportunity!",
        ],
    },
    "Rationalist": {
        "emoji": "🤔",
        "style": "Speaks with data and logic, likes to analyze pros and cons",
        "phrases": [
            "From the data...",
            "Logically speaking...",
            "We need to consider...",
        ],
    },
    "Creative": {
        "emoji": "💡",
        "style": "Likes jumping thinking, proposes novel ideas",
        "phrases": [
            "If we change perspective...",
            "Here's a crazy idea...",
            "Imagine if...",
        ],
    },
    "Pragmatist": {
        "emoji": "🔧",
        "style": "Focuses on feasibility and execution details",
        "phrases": [
            "How do we do this specifically?",
            "In practice...",
            "For implementation...",
        ],
    },
}


@remote
class ChatAgent:
    """AI agent in the chat room"""

    def __init__(self, agent_name: str, persona: str, topic: str):
        self.agent_name = agent_name
        self.persona = persona
        self.topic = topic
        self.config = AI_PERSONAS[persona]
        self.history: list[str] = []

    def receive_message(self, from_agent: str, message: str) -> str:
        """Receive messages from other agents"""
        self.history.append(f"{from_agent}: {message}")
        return "Received"

    def generate_response(self) -> str:
        """Generate response (mock mode)"""
        phrase = random.choice(self.config["phrases"])

        responses = [
            f"{phrase} Regarding '{self.topic}', I think this is a topic worth exploring in depth.",
            f"{phrase} Speaking of this topic, I'd like to add my own perspective.",
            f"{phrase} After hearing everyone's discussion, I have some new ideas.",
            f"{phrase} This topic is very interesting, it reminds me of some things.",
        ]
        return random.choice(responses)

    async def speak(self, room_name: str) -> dict:
        """Speak in the chat room"""
        response = self.generate_response()

        # Notify the chat room
        room = await resolve(room_name)
        await room.broadcast(self.agent_name, response)

        return {"agent": self.agent_name, "message": response}


@remote
class ChatRoom:
    """Chat room - coordinates agent conversations"""

    def __init__(self, topic: str):
        self.topic = topic
        self.agents: list[str] = []
        self.messages: list[dict] = []

    def join(self, agent_name: str, persona: str) -> str:
        """Agent joins the chat room"""
        self.agents.append(agent_name)
        emoji = AI_PERSONAS[persona]["emoji"]
        print(f"  {emoji} [{agent_name}] ({persona}) joined the chat room")
        return "Welcome!"

    def broadcast(self, from_agent: str, message: str) -> None:
        """Broadcast message"""
        persona = None
        for name in self.agents:
            if name == from_agent:
                # Find the speaker's persona
                for p, config in AI_PERSONAS.items():
                    if name.startswith(p):
                        persona = p
                        break
                break

        emoji = AI_PERSONAS.get(persona, {}).get("emoji", "💬") if persona else "💬"
        self.messages.append({"from": from_agent, "message": message})
        print(f"\n  {emoji} [{from_agent}]: {message}")

    def get_history(self) -> list[dict]:
        """Get chat history"""
        return self.messages


async def main(topic: str, rounds: int):
    print("=" * 60)
    print("🗣️  AI Chat Room")
    print("=" * 60)
    print(f"\n📋 Discussion topic: {topic}")
    print(f"🔄 Discussion rounds: {rounds}")
    print("\n--- Participants entering ---\n")

    async with runtime():
        # Create chat room
        room = await ChatRoom.spawn(topic=topic, name="chat_room")

        # Create AI agents with different personalities
        agents = []
        for persona in AI_PERSONAS.keys():
            agent_name = persona + "_AI"
            agent = await ChatAgent.spawn(
                agent_name=agent_name,
                persona=persona,
                topic=topic,
                name=agent_name,
            )
            await room.join(agent_name, persona)
            agents.append((agent_name, agent))

        print("\n--- Discussion begins ---")

        # Conduct multiple rounds of discussion
        for round_num in range(1, rounds + 1):
            print(f"\n{'─' * 40}")
            print(f"Round {round_num}")
            print(f"{'─' * 40}")

            # Randomize speaking order
            random.shuffle(agents)

            for agent_name, agent in agents:
                await agent.speak("chat_room")
                await asyncio.sleep(0.1)  # Simulate thinking time

        # Display statistics
        history = await room.get_history()
        print("\n" + "=" * 60)
        print("📊 Discussion Statistics")
        print("=" * 60)
        print(f"  Total messages: {len(history)}")
        print(f"  Participants: {len(agents)} AIs")
        print(f"  Discussion rounds: {rounds}")

    print("\n" + "=" * 60)
    print("✅ Chat ended!")
    print("=" * 60)
    print("\n💡 Tip: Modify AI_PERSONAS to customize AI personalities")
    print("💡 Tip: Set OPENAI_API_KEY to connect to real LLM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Chat Room")
    parser.add_argument(
        "--topic", default="Will AI replace human jobs?", help="Discussion topic"
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="Number of discussion rounds"
    )
    args = parser.parse_args()

    asyncio.run(main(args.topic, args.rounds))
