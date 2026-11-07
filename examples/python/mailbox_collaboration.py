"""Router → worker → subagent/reviewer → router, using real actor mailboxes.

Run: python examples/python/mailbox_collaboration.py
No model API, agent framework, or MCP dependency is required for this demo.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid

import pulsing as pul
from pulsing.mailbox import DIRECTORY_NAME, MailboxClient, MailboxDirectory


async def main(state_dir: str) -> None:
    await pul.init()
    try:
        directory = await MailboxDirectory.spawn(state_dir, name=DIRECTORY_NAME)
        await directory.register("router", "router")
        router = await MailboxClient.connect("router")
        await router.create_role("worker", "worker")
        await router.create_role("reviewer", "reviewer")
        worker = await MailboxClient.connect("worker")
        reviewer = await MailboxClient.connect("reviewer")
        await worker.create_role("helper", "subagent")
        helper = await MailboxClient.connect("helper")

        task = await router.send(
            "worker",
            "Add two numbers",
            "Compute 20 + 22 and obtain a review.",
            uuid.uuid4().hex,
        )
        states = [(await router.status("worker", task["id"]))["state"]]
        await worker.read(task["id"])
        states.append((await router.status("worker", task["id"]))["state"])
        claim = await worker.claim(task["id"])
        states.append((await router.status("worker", task["id"]))["state"])

        delegated = await worker.send(
            "helper", "Compute", "20 + 22", uuid.uuid4().hex, task["thread_id"]
        )
        child_claim = await helper.claim(delegated["id"])
        answer = 20 + 22
        await helper.reply(delegated["id"], str(answer), uuid.uuid4().hex)
        await helper.finish(
            delegated["id"], child_claim["claim_token"], result=str(answer)
        )

        review = await worker.send(
            "reviewer",
            "Review",
            f"Check result: {answer}",
            uuid.uuid4().hex,
            task["thread_id"],
        )
        review_claim = await reviewer.claim(review["id"])
        assert answer == 42
        await reviewer.finish(
            review["id"], review_claim["claim_token"], result="42 verified"
        )
        reply = await worker.reply(
            task["id"], "42; independent review passed", uuid.uuid4().hex
        )
        await worker.finish(task["id"], claim["claim_token"], result="42 verified")
        states.append((await router.status("worker", task["id"]))["state"])
        result = await router.read(reply["id"])
        print(
            json.dumps(
                {
                    "states": states,
                    "roles": await router.roles(),
                    "thread_id": task["thread_id"],
                    "answer": result["body"],
                },
                indent=2,
            )
        )
    finally:
        await pul.shutdown()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pulsing-mail-demo-") as state_dir:
        asyncio.run(main(state_dir))
