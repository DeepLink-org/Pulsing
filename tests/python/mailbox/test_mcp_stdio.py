"""Real MCP SDK sessions, separate processes, and networked Pulsing actors."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from unittest.mock import AsyncMock

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
async def session(
    identity,
    seed,
    plugin_root=None,
    *,
    state_dir=None,
    kind="worker",
    addr=None,
    env=None,
):
    if plugin_root is None:
        server = {
            "command": sys.executable,
            "args": [
                "-m",
                "pulsing.mailbox",
                "mcp",
                "--identity",
                identity,
                "--seed",
                seed,
            ],
        }
    else:
        from pulsing.mailbox.codex import generate_plugin

        root = generate_plugin(
            plugin_root,
            identity,
            [seed] if seed else [],
            addr=addr,
            state_dir=state_dir,
            kind=kind,
        )
        server = json.loads((root / "plugins/pulsing/.mcp.json").read_text())[
            "mcpServers"
        ]["pulsing-mailbox"]
    environment = os.environ.copy()
    environment.pop("SEED", None)
    environment.pop("PULSING_ADDR", None)
    params = StdioServerParameters(
        **server,
        # Exercise protocol isolation even when the caller enabled Rust tracing.
        env=environment
        | {"PULSING_TRACING": "1", "PULSING_PROBING_AUTO_TRACING": "0"}
        | (env or {}),
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
    if name == "mail_roles":
        return response.structuredContent["result"]
    return json.loads(response.content[0].text)


async def test_mcp_uses_configured_private_address(monkeypatch):
    from pulsing.mailbox import mcp

    init = AsyncMock()
    shutdown = AsyncMock()
    connect = AsyncMock()
    monkeypatch.setattr(mcp.pul, "init", init)
    monkeypatch.setattr(mcp.pul, "shutdown", shutdown)
    monkeypatch.setattr(mcp.MailboxClient, "connect", connect)
    server = mcp.create_server("worker", ["10.0.0.10:8100"], "10.0.0.11:8201")
    async with server.settings.lifespan(server) as context:
        assert context["mail"] is connect.return_value
        init.assert_awaited_once_with(addr="10.0.0.11:8201", seeds=["10.0.0.10:8100"])
        connect.assert_awaited_once_with("worker")
    shutdown.assert_awaited_once()


@pytest.mark.parametrize("generated_plugin", [False, True])
async def test_mcp_router_worker_review_and_host_restart(tmp_path, generated_plugin):
    def package(name):
        return tmp_path / f"plugin-{name}" if generated_plugin else None

    async with host(tmp_path) as seed, AsyncExitStack() as stack:
        router = await stack.enter_async_context(
            session("router", seed, package("router"))
        )
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
        worker = await stack.enter_async_context(
            session("worker", seed, package("worker"))
        )
        reviewer = await stack.enter_async_context(
            session("reviewer", seed, package("reviewer"))
        )
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
    # A new host has empty RAM even when rendezvous and log paths are reused.
    async with (
        host(tmp_path) as seed,
        session("router", seed, package("router-restarted")) as router,
    ):
        assert (await tool(router, "mail_list"))["counts"]["unread"] == 0
        assert {role["identity"] for role in await tool(router, "mail_roles")} == {
            "router"
        }
        missing = await router.call_tool("mail_read", {"message_id": reply["id"]})
        assert missing.isError
    from pulsing.mailbox.replay import replay_logs

    history = replay_logs([tmp_path / "mail-events.jsonl"])
    assert (
        next(m for m in history["messages"] if m["id"] == task["id"])["result"]
        == "reviewed"
    )
    assert not list(tmp_path.glob("*.sqlite3"))


async def test_host_rejects_second_owner_of_state_directory(tmp_path):
    async with host(tmp_path):
        second = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pulsing.mailbox",
            "serve",
            "--addr",
            "127.0.0.1:0",
            "--state-dir",
            str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(second.communicate(), timeout=15)
            assert second.returncode != 0
            assert b"BlockingIOError" in stderr
        finally:
            if second.returncode is None:
                second.kill()
                await second.wait()


async def test_plugin_starts_network_without_service_and_registers_peers(tmp_path):
    from pulsing.mailbox.managed import ENDPOINT_FILE
    from test_managed import stop_host

    state = tmp_path / "state"
    try:
        # No host fixture, serve command, or mail_create_role call.
        async with (
            session(
                "router", None, tmp_path / "router", state_dir=state, kind="router"
            ) as router,
            AsyncExitStack() as peers,
        ):
            endpoint = json.loads((state / ENDPOINT_FILE).read_text())
            # A second local Codex reuses the same automatically started host.
            worker = await peers.enter_async_context(
                session("worker", None, tmp_path / "worker", state_dir=state)
            )
            assert json.loads((state / ENDPOINT_FILE).read_text()) == endpoint
            # Another machine's equivalent: join via seed, no pre-registration.
            reviewer = await peers.enter_async_context(
                session(
                    "reviewer",
                    endpoint["addr"],
                    tmp_path / "reviewer",
                    state_dir=tmp_path / "unused",
                    kind="reviewer",
                )
            )
            assert not (tmp_path / "unused").exists()
            roles = await tool(router, "mail_roles")
            assert {role["identity"] for role in roles} == {
                "router",
                "worker",
                "reviewer",
            }
            task = await tool(
                router,
                "mail_send",
                to="worker",
                subject="Automatic bootstrap",
                body="Verify implicit mailbox lifecycle",
                idempotency_key="auto",
            )
            claim = await tool(worker, "mail_claim", message_id=task["id"])
            assert (await tool(reviewer, "mail_list"))["counts"]["unread"] == 0
        # Closing all Codex sessions must not stop the shared host.
        async with session(
            "worker", None, tmp_path / "worker-again", state_dir=state
        ) as worker:
            assert json.loads((state / ENDPOINT_FILE).read_text()) == endpoint
            await tool(
                worker,
                "mail_finish",
                message_id=task["id"],
                claim_token=claim["claim_token"],
                result="done after router exit",
            )

        # Restarting the sidecar loses live messages; the actor log remains replayable.
        await stop_host(state, signal.SIGKILL)
        async with session(
            "router", None, tmp_path / "restarted", state_dir=state, kind="router"
        ) as router:
            assert (await tool(router, "mail_list"))["counts"]["unread"] == 0
            assert {r["identity"] for r in await tool(router, "mail_roles")} == {
                "router"
            }
        from pulsing.mailbox.replay import replay_logs

        history = replay_logs([state / "mail-events.jsonl"])
        assert (
            next(m for m in history["messages"] if m["id"] == task["id"])["result"]
            == "done after router exit"
        )
    finally:
        await stop_host(state)


@pytest.mark.parametrize("env_mode", [False, True])
async def test_seed_one_hosts_in_mcp_process_without_restoring_mail(tmp_path, env_mode):
    from pulsing.mailbox.managed import ENDPOINT_FILE, _running

    state = tmp_path / "seed-state"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        seed_addr = f"127.0.0.1:{listener.getsockname()[1]}"
    seed_setting = None if env_mode else "1"
    seed_env = {"SEED": "1", "PULSING_ADDR": seed_addr} if env_mode else {}
    async with session(
        "router",
        seed_setting,
        tmp_path / "seed",
        state_dir=state,
        kind="router",
        addr=None if env_mode else seed_addr,
        env=seed_env,
    ) as seed:
        resource = await seed.read_resource("mailbox://local/inbox")
        info = json.loads(resource.contents[0].text)
        endpoint = json.loads((state / ENDPOINT_FILE).read_text())
        assert info["seed"] is True
        assert info["pid"] == endpoint["pid"]  # Not a different sidecar process.
        assert info["addr"] == endpoint["addr"] == seed_addr
        peer_env = {"SEED": info["addr"]} if env_mode else {}
        async with session(
            "worker",
            None if env_mode else info["addr"],
            tmp_path / "peer",
            state_dir=tmp_path / "unused",
            env=peer_env,
        ) as worker:
            peer_info = json.loads(
                (await worker.read_resource("mailbox://local/inbox")).contents[0].text
            )
            assert peer_info["seed"] is False
            assert peer_info["pid"] != endpoint["pid"]
            assert not (tmp_path / "unused").exists()
            task = await tool(
                seed,
                "mail_send",
                to="worker",
                subject="In-process seed",
                body="Verify seed address and memory ownership",
                idempotency_key="seed-one",
            )
            claim = await tool(worker, "mail_claim", message_id=task["id"])
            await tool(
                worker,
                "mail_finish",
                message_id=task["id"],
                claim_token=claim["claim_token"],
                result="verified",
            )
    assert not _running(state)
    assert not (state / ENDPOINT_FILE).exists()

    # Reloading SEED=1 creates a fresh empty directory, not recovered mail.
    async with session(
        "router",
        seed_setting,
        tmp_path / "seed-restarted",
        state_dir=state,
        kind="router",
        addr=None if env_mode else seed_addr,
        env=seed_env,
    ) as seed:
        assert (await tool(seed, "mail_list"))["counts"]["completed"] == 0
        assert {r["identity"] for r in await tool(seed, "mail_roles")} == {"router"}
