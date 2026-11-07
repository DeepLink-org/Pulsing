"""Real MCP SDK sessions, separate processes, and networked Pulsing actors."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


@asynccontextmanager
async def host(state_dir):
    env = os.environ | {"PULSING_TRACING": "0", "PULSING_PROBING_AUTO_TRACING": "0"}
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pulsing.mailbox",
        "serve",
        "--addr",
        "127.0.0.1:0",
        "--state-dir",
        str(state_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
        if not line.startswith(b"Mailbox host ready at "):
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            pytest.fail(f"host did not start: {line!r}\n{stderr.decode()}")
        yield line.decode().strip().removeprefix("Mailbox host ready at ")
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=15)
            except TimeoutError:
                process.kill()
                await process.wait()


@asynccontextmanager
async def session(identity, seed):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "pulsing.mailbox", "mcp", "--identity", identity, "--seed", seed],
        # Exercise protocol isolation even when the caller enabled Rust tracing.
        env=os.environ | {"PULSING_TRACING": "1", "PULSING_PROBING_AUTO_TRACING": "0"},
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as client,
    ):
        await client.initialize()
        yield client


async def tool(client, name, **arguments):
    response = await client.call_tool(name, arguments)
    assert not response.isError, response.content
    return json.loads(response.content[0].text)


async def test_mcp_router_worker_review_and_host_restart(tmp_path):
    async with host(tmp_path) as seed, AsyncExitStack() as stack:
        router = await stack.enter_async_context(session("router", seed))
        tools = (await router.list_tools()).tools
        assert len(tools) == 9
        send = next(t for t in tools if t.name == "mail_send")
        assert "sender" not in send.inputSchema["properties"]
        assert "idempotency_key" in send.inputSchema["required"]
        prompt = await router.get_prompt("mailbox_collaboration")
        assert "untrusted" in prompt.messages[0].content.text
        resource = await router.read_resource("mailbox://local/inbox")
        assert json.loads(resource.contents[0].text)["identity"] == "router"
        await tool(router, "mail_create_role", identity="worker", kind="worker")
        await tool(router, "mail_create_role", identity="reviewer", kind="reviewer")
        worker = await stack.enter_async_context(session("worker", seed))
        reviewer = await stack.enter_async_context(session("reviewer", seed))
        task = await tool(
            router,
            "mail_send",
            to="worker",
            subject="Implement",
            body="Verify the mailbox protocol",
            idempotency_key="assignment",
        )
        duplicate = await tool(
            router,
            "mail_send",
            to="worker",
            subject="Implement",
            body="Verify the mailbox protocol",
            idempotency_key="assignment",
        )
        assert duplicate["id"] == task["id"]
        assert (await tool(router, "mail_status", to="worker", message_id=task["id"]))[
            "state"
        ] == "unread"
        assert (await tool(worker, "mail_list"))["counts"]["unread"] == 1
        assert (await tool(worker, "mail_read", message_id=task["id"]))[
            "state"
        ] == "read"
        claimed = await tool(worker, "mail_claim", message_id=task["id"])
        assert (await tool(router, "mail_status", to="worker", message_id=task["id"]))[
            "state"
        ] == "processing"
        review = await tool(
            worker,
            "mail_send",
            to="reviewer",
            subject="Review",
            body="Check the result",
            idempotency_key="review",
            thread_id=task["thread_id"],
        )
        review_claim = await tool(reviewer, "mail_claim", message_id=review["id"])
        await tool(
            reviewer,
            "mail_finish",
            message_id=review["id"],
            claim_token=review_claim["claim_token"],
            result="checked",
        )
        reply = await tool(
            worker,
            "mail_reply",
            message_id=task["id"],
            body="Reviewed and ready",
            idempotency_key="answer",
        )
        assert reply["thread_id"] == task["thread_id"]
        await tool(
            worker,
            "mail_finish",
            message_id=task["id"],
            claim_token=claimed["claim_token"],
            result="reviewed",
        )
        assert (await tool(router, "mail_list"))["counts"]["unread"] == 1
        forbidden = await reviewer.call_tool(
            "mail_status", {"to": "worker", "message_id": task["id"]}
        )
        assert forbidden.isError
        wrong_claim = await worker.call_tool(
            "mail_finish", {"message_id": task["id"], "claim_token": "stale"}
        )
        assert wrong_claim.isError
    # Fresh processes and new transport connections, same SQLite state directory.
    async with host(tmp_path) as seed, session("router", seed) as router:
        receipt = await tool(router, "mail_status", to="worker", message_id=task["id"])
        assert receipt["state"] == "completed"
        assert receipt["result"] == "reviewed"
        assert (await tool(router, "mail_read", message_id=reply["id"]))[
            "body"
        ] == "Reviewed and ready"
