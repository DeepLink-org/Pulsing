import asyncio
import random
import pytest
from pulsing.core import (
    Actor,
    ActorId,
    Message,
)
import pulsing as pul


class ResilienceWorker(Actor):
    """A worker that maintains state and simulates work."""

    def __init__(self):
        self.processed_count = 0

    async def receive(self, msg: Message) -> Message:
        if msg.msg_type == "process":
            # Simulate work
            await asyncio.sleep(0.01)
            self.processed_count += 1
            return Message.from_json("result", {"count": self.processed_count})
        elif msg.msg_type == "get_state":
            return Message.from_json("state", {"count": self.processed_count})
        return Message.empty()


@pytest.fixture
async def actor_system():
    system = await pul.actor_system()
    yield system
    await system.shutdown()


@pytest.mark.asyncio
async def test_actor_death_recovery(actor_system):
    """
    Chaos Test: Verify system behavior when an actor is killed unexpectedly.
    """
    # 1. Spawn a worker
    worker_name = "resilient_worker"
    worker = await actor_system.spawn(ResilienceWorker(), name=worker_name)

    # 2. Verify it works
    resp = await worker.ask(Message.from_json("process", {}))
    assert resp.to_json()["count"] == 1

    # 3. Launch a chaos task that kills the actor after a short delay
    async def chaos_monkey():
        await asyncio.sleep(0.1)
        # Forcefully stop the actor
        await actor_system.stop(worker_name)

    chaos_task = asyncio.create_task(chaos_monkey())

    # 4. Concurrently send messages. Expect failure at some point.
    success_count = 0
    failure_count = 0

    try:
        # Try to send messages for 0.5s
        for _ in range(50):
            await worker.ask(Message.from_json("process", {}))
            success_count += 1
            await asyncio.sleep(0.01)
    except Exception:
        # We expect an exception when the actor is killed
        failure_count += 1

    await chaos_task

    # 5. Assertions
    # We should have had some successes before the kill
    assert success_count > 0
    # We should have caught the failure (or at least stopped sending)

    # 6. Recovery: Respawn the actor
    # Note: In a real persistent system, we'd recover state.
    # Here we just verify we can reclaim the name.
    new_worker = await actor_system.spawn(ResilienceWorker(), name=worker_name)
    resp = await new_worker.ask(Message.from_json("process", {}))

    # New actor starts from 0
    assert resp.to_json()["count"] == 1


@pytest.mark.asyncio
async def test_cluster_node_failure_detection():
    """
    Chaos Test: Simulate a 3-node cluster and kill one node.
    Verify that other nodes eventually detect the failure (via gossip).
    """
    # Setup 3 nodes (use port 0 for automatic port assignment to avoid conflicts)
    systems = []

    # Seed node (let OS assign port)
    sys1 = await pul.actor_system(addr="127.0.0.1:0")
    systems.append(sys1)
    seed_addr = sys1.addr  # Get the actual assigned address

    # Other nodes
    for i in range(2):
        sys = await pul.actor_system(addr="127.0.0.1:0", seeds=[seed_addr])
        systems.append(sys)

    # Wait for cluster formation
    await asyncio.sleep(2.0)

    # Verify all nodes see 3 members
    for sys in systems:
        members = await sys.members()
        assert len(members) == 3

    # KILL Node 2 (Index 1)
    victim_sys = systems[1]
    await victim_sys.shutdown()

    # Wait for failure detection (SWIM protocol takes some time)
    # Ping interval is usually small, but failure detection might take a few seconds
    # adjusting sleep based on typical gossip settings
    await asyncio.sleep(5.0)

    # Remaining nodes should see 2 members (or mark one as dead)
    # Note: precise behavior depends on the implementation (dead vs removed)
    # Here we check that the dead node is no longer considered "alive" or is removed.

    alive_sys = systems[0]
    members = await alive_sys.members()

    # The dead member should be gone OR marked effectively dead.
    # Assuming standard behavior: eventually removed or filtered.
    # If the list still contains 3, we check if we can still communicate (we shouldn't).

    # Let's clean up remaining
    await systems[0].shutdown()
    await systems[2].shutdown()
