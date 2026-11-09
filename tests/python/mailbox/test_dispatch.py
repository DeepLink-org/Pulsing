"""Delivery edges are volatile, bounded, and never block the mailbox actor."""

import asyncio
import threading
import shutil
import subprocess
import sys
import os
from pathlib import Path

import pytest

from pulsing.mailbox import Mailbox
from pulsing.mailbox.replay import replay_logs
from pulsing.mailbox.signals import DeliverySignals, delivery_signals


def test_javascript_dispatch_state_machine():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for native dispatch")
    subprocess.run(
        [node, "--test", str(Path(__file__).with_name("dispatch.test.mjs"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (["--dispatch", "--", "dsh", "web"], "requires --adapter pi"),
        (["--dispatch-max-turns", "0", "--", "pi"], "must be 1"),
    ],
)
def test_dispatch_cli_fails_before_starting_unsupported_or_unbounded_work(
    tmp_path, arguments, error
):
    runtime = tmp_path / "runtime"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pulsing.cli",
            "--runtime-dir",
            str(runtime),
            "--addr",
            "127.0.0.1:0",
            *arguments,
        ],
        env=os.environ | {"SEED": "1", "PULSING_DISPATCH": "0"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert error in result.stderr
    assert not runtime.exists()


async def test_signal_subscription_race_and_threadsafe_delivery():
    signals = DeliverySignals()
    signals.start("worker", "one")
    signals.publish("worker", "one", 1)
    await asyncio.wait_for(signals.wait("worker", "one", 0, 60), 0.1)
    pending = asyncio.create_task(signals.wait("worker", "one", 1, 60))
    await asyncio.sleep(0)
    publisher = threading.Thread(target=signals.publish, args=("worker", "one", 2))
    publisher.start()
    await asyncio.wait_for(pending, 1)
    publisher.join()
    assert signals._count == 0
    assert not signals._waiters


async def test_wait_timeout_cancel_limits_and_new_incarnation():
    signals = DeliverySignals(max_waiters=1)
    signals.start("worker", "old")
    pending = asyncio.create_task(signals.wait("worker", "old", 0, 60))
    await asyncio.sleep(0)
    with pytest.raises(ValueError, match="too many"):
        await signals.wait("worker", "old", 0, 60)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert signals._count == 0
    await signals.wait("worker", "old", 0, 0.001)
    assert signals._count == 0
    pending = asyncio.create_task(signals.wait("worker", "old", 0, 60))
    await asyncio.sleep(0)
    signals.start("worker", "new")
    await asyncio.wait_for(pending, 1)
    signals.stop("worker", "old")
    assert signals._versions["worker"] == ("new", 0)
    signals.stop("worker", "new")
    assert "worker" not in signals._versions


def test_changes_page_only_unclaimed_nonterminal_work_and_do_not_mark_read():
    box = Mailbox("dispatch-pages")
    one = box.deliver("sender", "one", "", "one")
    two = box.deliver("sender", "two", "", "two")
    three = box.deliver("sender", "three", "", "three")
    box.claim(one["id"])
    box.read_message(two["id"])
    first = box.changes(limit=1)
    assert first["items"] == [{"id": two["id"], "state": "read"}]
    assert first["cursor"] == 2 and first["more"]
    second = box.changes(first["epoch"], first["cursor"])
    assert second["items"] == [{"id": three["id"], "state": "unread"}]
    assert second["cursor"] == 3 and not second["more"]
    assert box.inbox()["counts"]["unread"] == 1
    box.deliver("sender", "three", "", "three")
    assert box.changes(second["epoch"], second["cursor"])["items"] == []
    assert box.changes("different-epoch", 999)["cursor"] == 3
    with pytest.raises(ValueError, match="ahead"):
        box.changes(first["epoch"], 999)
    box.on_stop()


def test_failed_log_append_does_not_publish_delivery(tmp_path, monkeypatch):
    box = Mailbox("dispatch-log-failure", event_log=str(tmp_path / "events.jsonl"))
    epoch = box.changes()["epoch"]

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(box._events, "emit", fail)
        with pytest.raises(OSError):
            box.deliver("sender", "not accepted", "", "one")
    assert delivery_signals._versions[box._identity] == (epoch, 0)
    box.on_stop()


def test_dispatch_observations_replay_without_claiming_work(tmp_path):
    path = tmp_path / "events.jsonl"
    box = Mailbox("dispatch-log", event_log=str(path))
    message = box.deliver("sender", "one", "", "one")
    assert box.observe_dispatch("requested", "run-one", [message["id"]])["recorded"]
    box.observe_dispatch("paused", reason="cancelled")
    replay = replay_logs([path])
    assert replay["messages"][0]["state"] == "unread"
    assert [event["phase"] for event in replay["dispatches"]] == ["requested", "paused"]
    with pytest.raises(ValueError, match="phase"):
        box.observe_dispatch("completed")
    with pytest.raises(ValueError, match="not found"):
        box.observe_dispatch("started", "run-one", ["unknown"])
    box.on_stop()


async def test_real_remote_delivery_wakes_wait_without_blocking_actor(tmp_path):
    pytest.importorskip("mcp")
    from test_launcher import harness, request

    async with harness(tmp_path, "receiver") as (receiver, recipient, _):
        initial = await request(receiver, "mail_wait", {"timeout_seconds": 1})
        waiter = asyncio.create_task(
            request(
                receiver,
                "mail_wait",
                {
                    "epoch": initial["epoch"],
                    "after": initial["cursor"],
                    "timeout_seconds": 20,
                },
            )
        )
        async with harness(tmp_path, "sender", seed=recipient["addr"]) as (
            sender,
            _,
            _,
        ):
            message = await request(
                sender,
                "mail_send",
                {
                    "to": recipient["identity"],
                    "subject": "Wake",
                    "body": "This must not block behind mail_wait",
                    "idempotency_key": "wake-one",
                },
            )
            delivered = await asyncio.wait_for(waiter, 5)
            assert delivered["items"] == [{"id": message["id"], "state": "unread"}]
            assert (await request(receiver, "mail_list"))["counts"]["unread"] == 1
