"""Actor-owned observations reconstruct history, never live mailbox recovery."""

from __future__ import annotations

import json

import pytest

from pulsing.mailbox import Mailbox
from pulsing.mailbox.replay import render, replay_logs


def test_actor_history_replays_bodies_states_retries_and_new_lifetimes(tmp_path):
    path = tmp_path / "actors.jsonl"
    box = Mailbox("worker", event_log=str(path), instance="node")
    message = box.deliver("router", "Proof", "Explain the lemma", "send-one")
    assert (
        box.deliver("router", "Proof", "Explain the lemma", "send-one")["id"]
        == message["id"]
    )
    box.read_message(message["id"])
    claim = box.claim(message["id"])
    result = box.finish(message["id"], claim["claim_token"], result="Lemma checked")
    box.finish(message["id"], claim["claim_token"], result="Lemma checked")
    box.receipt(message["id"], "router")
    box.on_stop()
    fresh = Mailbox("worker", event_log=str(path), instance="node")
    assert fresh.inbox()["counts"]["completed"] == 0
    with pytest.raises(ValueError, match="not found"):
        fresh.read_message(message["id"])
    fresh.on_stop()
    history = replay_logs([path])
    assert len(history["messages"]) == 1
    for key, value in result.items():
        assert history["messages"][0][key] == value
    assert len(history["streams"]) == 2
    assert all(stream["stopped"] for stream in history["streams"].values())
    assert history["rejected"][0]["message_id"] == message["id"]
    text = path.read_text()
    assert claim["claim_token"] not in text
    assert "delivery.retried" in text and "finish.retried" in text
    assert "Explain the lemma" in render(history, "markdown")
    assert not list(tmp_path.glob("*.sqlite3"))


def test_actor_log_failure_does_not_accept_a_message(tmp_path, monkeypatch):
    box = Mailbox("worker", event_log=str(tmp_path / "events.jsonl"))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(box._events, "emit", fail)
        with pytest.raises(OSError, match="disk full"):
            box.deliver("router", "Task", "body", "one")
    assert sum(box.inbox()["counts"].values()) == 0
    assert box.deliver("router", "Task", "body", "one")["state"] == "unread"
    assert len(replay_logs([tmp_path / "events.jsonl"])["messages"]) == 1


def test_library_mailboxes_need_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    box = Mailbox("worker")
    box.deliver("router", "Task", "body", "one")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("corruption", ["gap", "invalid_json", "conflicting_duplicate"])
def test_replay_rejects_incomplete_or_conflicting_history(tmp_path, corruption):
    path = tmp_path / "events.jsonl"
    box = Mailbox("worker", event_log=str(path))
    message = box.deliver("router", "Task", "body", "one")
    box.read_message(message["id"])
    lines = path.read_text().splitlines()
    if corruption == "gap":
        lines.pop(1)
    elif corruption == "invalid_json":
        lines.append('{"truncated":')
    else:
        changed = json.loads(lines[1])
        changed["message"]["body"] = "different"
        lines.append(json.dumps(changed))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=str(path)):
        replay_logs([path])


def test_repeated_log_input_is_idempotent(tmp_path):
    path = tmp_path / "events.jsonl"
    box = Mailbox("worker", event_log=str(path))
    box.deliver("router", "Task", "body", "one")
    assert replay_logs([path, path]) == replay_logs([path])


def test_replay_cli_only_writes_an_offline_report(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "events.jsonl"
    box = Mailbox("worker", event_log=str(path))
    message = box.deliver("router", "Task", "body", "one")
    output = tmp_path / "replay.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pulsing.mailbox",
            "replay",
            str(path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["messages"][0]["id"] == message["id"]
    fresh = Mailbox("worker")
    assert sum(fresh.inbox()["counts"].values()) == 0
