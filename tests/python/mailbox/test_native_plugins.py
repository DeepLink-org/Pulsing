"""Native registration contracts plus actual companion/MCP processes; no models."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from pulsing.mailbox.native import dsh_command, generate_plugin, pi_command

NODE = shutil.which("node")
PROBE = Path(__file__).with_name("native_probe.mjs")


def test_native_packages_are_self_contained_and_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSING_MCP_TOKEN", "must-not-save-this")
    monkeypatch.setenv("SEED", "must-not-save-this-either")
    for harness in ("pi", "dsh"):
        root = generate_plugin(
            tmp_path / harness,
            harness,
            seeds=["127.0.0.1:8100"],
            runtime_dir=tmp_path / "runtime",
        )
        manifest = json.loads((root / "package.json").read_text())
        assert manifest["name"] == "pulsing"
        assert "dependencies" not in manifest
        if harness == "pi":
            assert manifest["pi"]["extensions"] == ["./pi.mjs"]
        else:
            assert manifest["dsh"]["bundle"]["patch"] == "./cordis.patch.yml"
            assert "name: pulsing" in (root / "cordis.patch.yml").read_text()
        for path in root.iterdir():
            assert "must-not-save-this" not in path.read_text()
        assert (root / "PROTOCOL.md").read_text() == (
            Path(__file__).parents[3] / "python/pulsing/mailbox/COMPANION.md"
        ).read_text()
        with pytest.raises(FileExistsError):
            generate_plugin(root, harness)
        assert (root / "bridge.mjs").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"harness": "other"},
        {"python": "relative"},
        {"instance": "has.dot"},
        {"instance": "x" * 25},
        {"addr": " "},
        {"harness": "dsh", "dispatch": True},
        {"dispatch_max_turns": 0},
        {"dispatch_max_turns": 65},
    ],
)
def test_invalid_native_package_options_do_not_create_output(tmp_path, options):
    harness = options.pop("harness", "pi")
    root = tmp_path / "plugin"
    with pytest.raises(ValueError):
        generate_plugin(root, harness, **options)
    assert not root.exists()


def test_native_command_injection_preserves_user_arguments(tmp_path):
    pi = ["pi", "--", "a prompt with spaces"]
    selected = pi_command(pi)
    assert selected[0] == "pi"
    assert selected[1] == "--extension"
    assert selected[3:] == pi[1:]
    assert Path(selected[2]).is_file()
    dsh = ["dsh", "--profile", "web", "--", "literal"]
    selected = dsh_command(dsh, tmp_path)
    assert selected[0:2] == ["dsh", "--patch"]
    assert selected[3:] == dsh[1:]
    patch = json.loads(Path(selected[2]).read_text())
    assert patch[0]["insert"][0]["name"].startswith("file:///")
    assert "TOKEN" not in Path(selected[2]).read_text()
    selected = dsh_command(["dsh", "web", "--", "literal"], tmp_path)
    assert selected[3:] == ["--profile", "web", "--", "literal"]
    assert pi_command(["pi", "install", "./plugin"]) == ["pi", "install", "./plugin"]


def test_pi_package_carries_explicit_dispatch_settings(tmp_path):
    root = generate_plugin(tmp_path / "pi", "pi", dispatch=True, dispatch_max_turns=3)
    config = json.loads((root / "runtime.json").read_text())
    assert config["dispatch"] is True
    assert config["dispatchMaxTurns"] == 3
    assert "--dispatch" not in config["launchArgs"]  # infrastructure child is generic
    assert (root / "dispatch.mjs").is_file()


@asynccontextmanager
async def native_harness(tmp_path, harness, *, seed="1", standalone=False):
    if not NODE:
        pytest.skip("Node.js is required for native plugins")
    pytest.importorskip("mcp")
    version = await asyncio.create_subprocess_exec(
        NODE, "--version", stdout=asyncio.subprocess.PIPE
    )
    output, _ = await version.communicate()
    if int(output.decode().split(".")[0].removeprefix("v")) < 22:
        pytest.skip("native plugins require Node.js 22.19+")
    runtime = tmp_path / harness
    environment = os.environ | {"SEED": seed, "PULSING_DISPATCH": "0"}
    for name in ("PULSING_MCP_URL", "PULSING_MCP_TOKEN", "PULSING_PROTOCOL"):
        environment.pop(name, None)
    environment.pop("PULSING_ADDR", None)
    environment["PULSING_PYTHON"] = sys.executable
    if standalone:
        package = generate_plugin(
            tmp_path / f"plugin-{harness}",
            harness,
            runtime_dir=runtime,
            instance=harness,
            addr="127.0.0.1:0",
        )
        command = [NODE, str(PROBE), harness, str(package)]
    else:
        command = [
            sys.executable,
            "-m",
            "pulsing.cli",
            "--instance",
            harness,
            "--runtime-dir",
            str(runtime),
            "--addr",
            "127.0.0.1:0",
            "--adapter",
            "generic",
            "--",
            NODE,
            str(PROBE),
            harness,
        ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    try:
        line = await asyncio.wait_for(process.stdout.readline(), 50)
        if not line:
            _, error = await process.communicate()
            pytest.fail(error.decode())
        ready = json.loads(line)
        assert len(ready["tools"]) == 9
        yield process, ready["ready"], runtime
    finally:
        if process.returncode is None:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 20)
            except TimeoutError:
                process.kill()
                await process.wait()
        assert process.returncode == 0, (await process.stderr.read()).decode()
        assert not (runtime / "endpoint.json").exists()


async def native_call(process, tool, arguments=None, *, session="root", **extra):
    process.stdin.write(
        (
            json.dumps(
                {
                    "tool": tool,
                    "arguments": arguments or {},
                    "session": session,
                    **extra,
                }
            )
            + "\n"
        ).encode()
    )
    await process.stdin.drain()
    line = await asyncio.wait_for(process.stdout.readline(), 20)
    assert line
    return json.loads(line)


@pytest.mark.parametrize("harness", ["pi", "dsh"])
async def test_standalone_plugins_bootstrap_and_keep_session_mailboxes_isolated(
    tmp_path, harness
):
    async with native_harness(tmp_path, harness, standalone=True) as (
        process,
        root,
        runtime,
    ):
        companion_pid = json.loads((runtime / "endpoint.json").read_text())["pid"]
        assert os.getpgid(companion_pid) == companion_pid
        child = (await native_call(process, "mail_identity", session="child"))["result"]
        assert root["identity"] != child["identity"]
        task = (
            await native_call(
                process,
                "mail_send",
                {
                    "to": child["identity"],
                    "subject": "Task",
                    "body": "Native delivery",
                    "idempotency_key": "one",
                },
            )
        )["result"]
        missing = await native_call(process, "mail_identity", session=None)
        assert "authoritative" in missing["error"]
        assert (await native_call(process, "mail_list"))["result"]["counts"][
            "unread"
        ] == 0
        assert (
            await native_call(
                process, "mail_read", {"message_id": task["id"]}, session="child"
            )
        )["result"]["body"] == "Native delivery"
        claim = (
            await native_call(
                process, "mail_claim", {"message_id": task["id"]}, session="child"
            )
        )["result"]
        await native_call(
            process,
            "mail_finish",
            {
                "message_id": task["id"],
                "claim_token": claim["claim_token"],
                "result": "Checked",
            },
            session="child",
        )
        assert (
            await native_call(
                process,
                "mail_status",
                {
                    "to": child["identity"],
                    "message_id": task["id"],
                },
            )
        )["result"]["state"] == "completed"
        if harness == "pi":
            switched = await native_call(
                process, "mail_identity", session="new-session", action="switch"
            )
            assert switched["result"]["identity"] not in (
                root["identity"],
                child["identity"],
            )
            assert (await native_call(process, "mail_identity"))["result"] == root
        assert not list(runtime.glob("*.sqlite3"))


async def test_pi_and_dsh_exchange_mail_across_companions(tmp_path):
    async with native_harness(tmp_path, "pi") as (pi, parent, _):
        async with native_harness(tmp_path, "dsh", seed=parent["addr"]) as (
            dsh,
            reviewer,
            _,
        ):
            message = (
                await native_call(
                    pi,
                    "mail_send",
                    {
                        "to": reviewer["identity"],
                        "subject": "Review",
                        "body": "Review from Pi",
                        "idempotency_key": "review-one",
                    },
                )
            )["result"]
            assert (
                await native_call(
                    dsh,
                    "mail_read",
                    {
                        "message_id": message["id"],
                    },
                )
            )["result"]["sender"] == parent["identity"]
            reply = (
                await native_call(
                    dsh,
                    "mail_reply",
                    {
                        "message_id": message["id"],
                        "body": "Reviewed by DSH",
                        "idempotency_key": "reply-one",
                    },
                )
            )["result"]
            assert (
                await native_call(
                    pi,
                    "mail_read",
                    {
                        "message_id": reply["id"],
                    },
                )
            )["result"]["in_reply_to"] == message["id"]


def test_native_bridge_contracts():
    if not NODE:
        pytest.skip("Node.js is required for native plugins")
    import subprocess

    subprocess.run(
        [NODE, "--test", str(PROBE.with_name("native_bridge.test.mjs"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("harness", ["pi", "dsh"])
async def test_upstream_native_sdk_registration_without_inference(tmp_path, harness):
    sdk_root = os.environ.get("PULSING_NATIVE_SDK_ROOT")
    if not NODE or not sdk_root:
        pytest.skip(
            "set PULSING_NATIVE_SDK_ROOT to opt into upstream SDK compatibility"
        )
    environment = os.environ | {
        "SEED": "1",
        "PI_OFFLINE": "1",
        "PULSING_DISPATCH": "0",
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pulsing.cli",
        "--addr",
        "127.0.0.1:0",
        "--runtime-dir",
        str(tmp_path / "runtime"),
        "--adapter",
        "generic",
        "--",
        NODE,
        str(PROBE.with_name("native_sdk_probe.mjs")),
        str(Path(sdk_root).resolve()),
        str(tmp_path / "sdk-state"),
        harness,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    try:
        output, errors = await asyncio.wait_for(process.communicate(), 60)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, errors.decode()
    result = json.loads(output)
    assert result["sdkLoaded"] and result["tools"] == 9


async def test_real_pi_dispatch_with_deterministic_model(tmp_path):
    sdk_root = os.environ.get("PULSING_NATIVE_SDK_ROOT")
    if not NODE or not sdk_root:
        pytest.skip("set PULSING_NATIVE_SDK_ROOT for the native Pi dispatch test")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pulsing.cli",
        "--addr",
        "127.0.0.1:0",
        "--runtime-dir",
        str(tmp_path / "runtime"),
        "--adapter",
        "generic",
        "--",
        NODE,
        str(PROBE.with_name("dispatch_sdk_probe.mjs")),
        str(Path(sdk_root).resolve()),
        str(tmp_path / "sdk-state"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ | {"SEED": "1", "PI_OFFLINE": "1", "PULSING_DISPATCH": "0"},
    )
    try:
        output, errors = await asyncio.wait_for(process.communicate(), 60)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, errors.decode()
    assert json.loads(output)["completed"] == 2
