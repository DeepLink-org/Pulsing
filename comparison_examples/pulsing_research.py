"""
Pulsing Multi-Agent Research Workflow using @agent decorator
Same functionality as AutoGen version
"""

from pulsing.actor import resolve
from pulsing.agent import agent, runtime


# Researcher Agent
@agent(role="Researcher", goal="Research topics")
class ResearcherAgent:
    async def research(self, topic: str) -> list[str]:
        return [
            f"Research point 1 about {topic}",
            f"Research point 2 about {topic}",
        ]


# Analyst Agent
@agent(role="Analyst", goal="Analyze research results")
class AnalystAgent:
    async def analyze(self, points: list[str]) -> str:
        combined = " ".join(points)
        return f"Analysis: {combined[:50]}..."


# Reporter Agent
@agent(role="Reporter", goal="Write final report")
class ReporterAgent:
    async def write(self, summary: str) -> str:
        return f"Final Report:\n{summary}"


# Workflow
async def run_workflow():
    async with runtime():
        # Spawn agents with names
        researcher = await ResearcherAgent.spawn(name="researcher")
        analyst = await AnalystAgent.spawn(name="analyst")
        reporter = await ReporterAgent.spawn(name="reporter")

        # Resolve by name (automatic load balancing)
        researcher = await resolve("researcher")
        analyst = await resolve("analyst")
        reporter = await resolve("reporter")

        # Execute workflow
        research = await researcher.research("Quantum Computing")
        analysis = await analyst.analyze(research)
        report = await reporter.write(analysis)

        print(report)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_workflow())
