"""Actual launcher processes and MCP HTTP calls; no paid model sessions."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("mcp")
httpx = pytest.importorskip("httpx")

PROBE = Path(__file__).with_name("launcher_probe.py")

from pulsing.mailbox.replay import replay_logs  # noqa: E402


@asynccontextmanager
async def harness(tmp_path, instance, seed="1", addr="127.0.0.1:0"):
    state = tmp_path / instance
    environment = os.environ | {"SEED": seed}
    environment.pop("PULSING_ADDR", None)
    if addr is not None:
        environment["PULSING_ADDR"] = addr
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pulsing.cli",
        "--instance",
        instance,
        "--state-dir",
        str(state),
        "--",
        sys.executable,
        str(PROBE),
        "--literal-argument",
        "two words",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    try:
        line = await asyncio.wait_for(process.stdout.readline(), 30)
        if not line:
            _, stderr = await asyncio.wait_for(process.communicate(), 10)
            pytest.fail(stderr.decode())
        ready = json.loads(line)
        assert ready["args"] == ["--literal-argument", "two words"]
        assert json.loads((state / "endpoint.json").read_text())["pid"] == process.pid
        yield process, ready["ready"], state
    finally:
        if process.returncode is None:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                process.kill()
                await process.wait()
        assert process.returncode == 0, (await process.stderr.read()).decode()
        assert not (state / "endpoint.json").exists()


async def request(process, name, arguments=None, thread="main"):
    process.stdin.write(
        (
            json.dumps(
                {
                    "tool": name,
                    "arguments": arguments or {},
                    "thread": thread,
                }
            )
            + "\n"
        ).encode()
    )
    await process.stdin.drain()
    line = await asyncio.wait_for(process.stdout.readline(), 20)
    assert line, "harness exited"
    response = json.loads(line)
    assert "error" not in response, response
    return response.get("result", response)


async def test_per_agent_mailboxes_and_missing_metadata(tmp_path):
    async with harness(tmp_path, "local") as (process, parent, state):
        child = await request(process, "mail_identity", thread="subagent")
        assert parent["identity"] != child["identity"]
        assert parent["addr"] == child["addr"]
        assert (await request(process, "missing-meta"))["isError"] is True
        task = await request(
            process,
            "mail_send",
            {
                "to": child["identity"],
                "subject": "Task",
                "body": "Check isolation",
                "idempotency_key": "one",
            },
        )
        assert (await request(process, "mail_list"))["counts"]["unread"] == 0
        assert (await request(process, "mail_list", thread="subagent"))["counts"][
            "unread"
        ] == 1
        forbidden = await request(process, "mail_identity", thread="other")
        assert forbidden["identity"] != child["identity"]
        process.stdin.write(
            (
                json.dumps(
                    {
                        "tool": "mail_read",
                        "arguments": {"message_id": task["id"]},
                        "thread": "other",
                    }
                )
                + "\n"
            ).encode()
        )
        await process.stdin.drain()
        response = json.loads(await asyncio.wait_for(process.stdout.readline(), 10))
        assert "error" in response
    history = replay_logs([state / "mail-events.jsonl"])
    assert (
        len(
            [
                s
                for s in history["streams"].values()
                if s["actor"].startswith("mail/boxes/")
            ]
        )
        == 3
    )
    assert not list(state.glob("*.sqlite3"))


@pytest.mark.parametrize("automatic_address", [False, True])
async def test_cross_companion_delivery_is_owned_by_recipient(
    tmp_path, automatic_address
):
    first_seed = "1"
    addr = "127.0.0.1:0"
    if automatic_address:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            first_seed = f"127.0.0.1:{probe.getsockname()[1]}"
        addr = None
    async with harness(tmp_path, "a", first_seed, addr=addr) as (
        first,
        sender,
        first_state,
    ):
        if automatic_address:
            assert sender["addr"] == first_seed
        # In automatic mode both invocations have exactly the same SEED and no
        # PULSING_ADDR. The second joins the occupied local seed address.
        second_seed = first_seed if automatic_address else sender["addr"]
        async with harness(tmp_path, "b", second_seed, addr=addr) as (
            second,
            recipient,
            second_state,
        ):
            # Resolve is already exercised by mail_send, independently of
            # eventual directory enumeration.
            task = await request(
                first,
                "mail_send",
                {
                    "to": recipient["identity"],
                    "subject": "Remote task",
                    "body": "Do work",
                    "idempotency_key": "remote",
                },
            )
            claim = await request(second, "mail_claim", {"message_id": task["id"]})
            await request(
                second,
                "mail_finish",
                {
                    "message_id": task["id"],
                    "claim_token": claim["claim_token"],
                    "result": "remote complete",
                },
            )
            receipt = await request(
                first,
                "mail_status",
                {
                    "to": recipient["identity"],
                    "message_id": task["id"],
                },
            )
            assert receipt["result"] == "remote complete"
            roles = await request(second, "mail_roles")
            assert {role["instance"] for role in roles} == {"a", "b"}
            sent_history = replay_logs([first_state / "mail-events.jsonl"])
            received_history = replay_logs([second_state / "mail-events.jsonl"])
            assert not any(m["id"] == task["id"] for m in sent_history["messages"])
            assert (
                next(m for m in received_history["messages"] if m["id"] == task["id"])[
                    "state"
                ]
                == "completed"
            )
            first.stdin.close()
            await asyncio.wait_for(first.wait(), 10)
            # The seed is not the owner of B's in-memory mailboxes.
            own_inbox = await request(second, "mail_list")
            assert own_inbox["counts"]["completed"] == 1
    assert not list(first_state.glob("*.sqlite3"))
    assert not list(second_state.glob("*.sqlite3"))


async def test_exit_status_and_launch_failure_cleanup(tmp_path):
    for command, expected in [
        ([sys.executable, "-c", "raise SystemExit(7)"], 7),
        (["/nonexistent/pulsing-agent"], 127),
    ]:
        state = tmp_path / str(expected)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pulsing.cli",
            "--state-dir",
            str(state),
            "--",
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ | {"SEED": "1", "PULSING_ADDR": "127.0.0.1:0"},
        )
        await asyncio.wait_for(process.communicate(), 20)
        assert process.returncode == expected
        assert not (state / "endpoint.json").exists()


async def test_local_mcp_rejects_requests_without_token(tmp_path):
    # Obtain the local URL from the companion's normal readiness-independent
    # child environment, without printing or persisting the bearer token.
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pulsing.cli",
        "--state-dir",
        str(tmp_path),
        "--",
        sys.executable,
        "-c",
        "import os; print(os.environ['PULSING_MCP_URL'], flush=True); input()",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ | {"SEED": "1", "PULSING_ADDR": "127.0.0.1:0"},
    )
    try:
        url = (await asyncio.wait_for(process.stdout.readline(), 20)).decode().strip()
        async with httpx.AsyncClient() as client:
            assert (await client.post(url, json={})).status_code == 401
    finally:
        process.stdin.write(b"\n")
        await process.stdin.drain()
        await asyncio.wait_for(process.communicate(), 15)
    assert process.returncode == 0


async def test_same_instance_has_stable_identity_but_fresh_mail_after_restart(tmp_path):
    async with harness(tmp_path, "stable") as (process, identity, _):
        task = await request(
            process,
            "mail_send",
            {
                "to": identity["identity"],
                "subject": "Observe",
                "body": "Keep this in the replay log, not live storage",
                "idempotency_key": "observe",
            },
        )
    async with harness(tmp_path, "stable") as (process, identity_after, _):
        assert identity_after["identity"] == identity["identity"]
        assert (await request(process, "mail_list"))["counts"]["unread"] == 0
    history = replay_logs([tmp_path / "stable/mail-events.jsonl"])
    assert next(m for m in history["messages"] if m["id"] == task["id"])[
        "body"
    ].startswith("Keep this")


async def test_sigterm_is_forwarded_and_companion_stops(tmp_path):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pulsing.cli",
        "--state-dir",
        str(tmp_path),
        "--",
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, lambda *_: exit(23)); "
        "print('ready', flush=True); time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ | {"SEED": "1", "PULSING_ADDR": "127.0.0.1:0"},
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), 20) == b"ready\n"
        process.terminate()
        await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 143
        assert not (tmp_path / "endpoint.json").exists()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_real_runtime_replans_a_lost_local_seed_bind(tmp_path):
    import pulsing as pul
    from pulsing.cli.launch import companion_runtime
    from pulsing.cli.network import Network

    async with harness(tmp_path, "seed-owner") as (_, seed, _):
        # Simulate the seed port being taken between auto-detection and bind.
        selected = Network(seed["addr"], [], True)
        async with companion_runtime(selected, tmp_path / "racer", "racer", True) as (
            _,
            joined,
        ):
            assert not joined.is_seed
            assert joined.peers == [seed["addr"]]
            assert pul.get_system().addr != seed["addr"]
