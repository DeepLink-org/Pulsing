"""State, durability, and actual actor serialization contracts."""

from __future__ import annotations

import asyncio

import pytest

import pulsing as pul
from pulsing.mailbox import DIRECTORY_NAME, Mailbox, MailboxClient, MailboxDirectory
from pulsing.mailbox.actors import validate_identity


@pytest.fixture
def box(tmp_path):
    mailbox = Mailbox("worker", str(tmp_path))
    try:
        yield mailbox
    finally:
        mailbox.on_stop()


def deliver(box, key="task-1", **kwargs):
    return box.deliver("router", "Task", "Do work", key, **kwargs)


@pytest.mark.parametrize(
    "identity", ["", "../worker", "a/b", "a b", ".", "-bad", "x" * 65]
)
def test_invalid_identity(identity):
    with pytest.raises(ValueError):
        validate_identity(identity)


def test_read_and_processing_are_different(box):
    mail = deliver(box)
    assert mail["state"] == "unread"
    assert box.inbox()["counts"]["unread"] == 1
    assert box.inbox()["items"][0]["id"] == mail["id"]
    assert "body" not in box.inbox()["items"][0]
    assert box.receipt(mail["id"], "router")["state"] == "unread"
    assert box.read_message(mail["id"])["state"] == "read"
    claim = box.claim(mail["id"])
    assert claim["state"] == "processing"
    assert claim["attempt"] == 1
    assert "claim_token" not in box.read_message(mail["id"])
    assert box.read_message(mail["id"])["state"] == "processing"
    finished = box.finish(mail["id"], claim["claim_token"], result="verified")
    assert finished["state"] == "completed"
    assert box.receipt(mail["id"], "router")["result"] == "verified"
    assert box.read_message(mail["id"])["state"] == "completed"


def test_idempotent_delivery_and_conflict(box):
    mail = deliver(box)
    assert deliver(box)["id"] == mail["id"]
    assert box.inbox()["counts"]["unread"] == 1
    with pytest.raises(ValueError, match="different content"):
        box.deliver("router", "Task", "Changed", "task-1")
    with pytest.raises(ValueError, match="different content"):
        box.deliver("router", "Task", "Do work", "task-1", thread_id="another")
    # Keys are scoped to a sender, not shared across independent senders.
    assert box.deliver("reviewer", "Task", "Do work", "task-1")["id"] != mail["id"]


def test_live_claim_has_one_owner_and_finish_is_idempotent(box):
    mail = deliver(box)
    claim = box.claim(mail["id"])
    with pytest.raises(ValueError, match="already claimed"):
        box.claim(mail["id"])
    with pytest.raises(ValueError, match="token"):
        box.finish(mail["id"], "wrong")
    first = box.finish(mail["id"], claim["claim_token"], "failed", "blocked")
    assert box.finish(mail["id"], claim["claim_token"], "failed", "blocked") == first
    with pytest.raises(ValueError, match="terminal"):
        box.finish(mail["id"], claim["claim_token"], result="changed")
    with pytest.raises(ValueError, match="terminal"):
        box.claim(mail["id"])


def test_expired_claim_can_be_recovered_but_not_completed(box, monkeypatch):
    now = 1000.0
    monkeypatch.setattr("pulsing.mailbox.actors.time.time", lambda: now)
    mail = deliver(box)
    old = box.claim(mail["id"], lease_seconds=10)
    now += 11
    with pytest.raises(ValueError, match="expired"):
        box.finish(mail["id"], old["claim_token"])
    new = box.claim(mail["id"])
    assert new["attempt"] == 2
    assert new["claim_token"] != old["claim_token"]
    with pytest.raises(ValueError, match="token"):
        box.finish(mail["id"], old["claim_token"])
    assert box.finish(mail["id"], new["claim_token"])["state"] == "completed"


def test_pagination_validation_and_sender_receipt(box):
    messages = [deliver(box, str(i)) for i in range(3)]
    first = box.inbox(limit=2)
    second = box.inbox(limit=2, offset=first["next_offset"])
    assert len(first["items"]) == 2
    assert len(second["items"]) == 1
    assert second["next_offset"] is None
    assert {m["id"] for m in first["items"] + second["items"]} == {
        m["id"] for m in messages
    }
    with pytest.raises(ValueError, match="another sender"):
        box.receipt(messages[0]["id"], "stranger")
    for kwargs in ({"state": "bogus"}, {"limit": 0}, {"limit": 101}, {"offset": -1}):
        with pytest.raises(ValueError):
            box.inbox(**kwargs)
    with pytest.raises(ValueError, match="not found"):
        box.read_message("missing")
    with pytest.raises(ValueError):
        box.claim(messages[0]["id"], lease_seconds=0)


def test_bounded_storage(tmp_path):
    box = Mailbox("limited", str(tmp_path), capacity=1)
    try:
        mail = deliver(box)
        assert deliver(box)["id"] == mail["id"]
        with pytest.raises(ValueError, match="capacity"):
            deliver(box, key="second")
        with pytest.raises(ValueError, match="body"):
            box.deliver("router", "Task", "x" * 65537, "large")
    finally:
        box.on_stop()


def test_restart_retains_mail_claim_and_deduplication(tmp_path):
    first = Mailbox("worker", str(tmp_path))
    mail = deliver(first)
    claim = first.claim(mail["id"])
    first.on_stop()
    second = Mailbox("worker", str(tmp_path))
    try:
        assert deliver(second)["id"] == mail["id"]
        assert second.read_message(mail["id"])["attempt"] == 1
        assert second.finish(mail["id"], claim["claim_token"])["state"] == "completed"
    finally:
        second.on_stop()


@pytest.fixture
async def actors(tmp_path):
    await pul.init()
    directory = await MailboxDirectory.spawn(str(tmp_path), name=DIRECTORY_NAME)
    await directory.register("router", "router")
    router = await MailboxClient.connect("router")
    await router.create_role("worker", "worker")
    await router.create_role("reviewer", "reviewer")
    yield {
        name: await MailboxClient.connect(name)
        for name in ("router", "worker", "reviewer")
    }
    await pul.shutdown()


async def test_actor_delivery_and_concurrent_claims(actors):
    router, worker = actors["router"], actors["worker"]
    deliveries = await asyncio.gather(
        *(router.send("worker", "Task", "Do work", "same-send") for _ in range(10))
    )
    assert len({m["id"] for m in deliveries}) == 1
    message_id = deliveries[0]["id"]
    attempts = await asyncio.gather(
        worker.claim(message_id), worker.claim(message_id), return_exceptions=True
    )
    assert sum(isinstance(result, dict) for result in attempts) == 1
    successful = next(result for result in attempts if isinstance(result, dict))
    await worker.finish(message_id, successful["claim_token"], result="done")
    assert (await router.status("worker", message_id))["state"] == "completed"


async def test_router_worker_subagent_and_reviewer(actors):
    router, worker, reviewer = (actors[n] for n in ("router", "worker", "reviewer"))
    await worker.create_role("helper")
    helper = await MailboxClient.connect("helper")
    assert (
        next(r for r in await router.roles() if r["identity"] == "helper")["parent"]
        == "worker"
    )
    task = await router.send(
        "worker", "Implement", "Scope: prototype only", "implement"
    )
    work = await worker.claim(task["id"])
    child = await worker.send(
        "helper", "Check", "Verify the invariant", "check", task["thread_id"]
    )
    child_work = await helper.claim(child["id"])
    await helper.reply(child["id"], "Invariant checked", "child-result")
    await helper.finish(child["id"], child_work["claim_token"], result="verified")
    review = await worker.send(
        "reviewer", "Review", "Inspect results", "review", task["thread_id"]
    )
    reviewed = await reviewer.claim(review["id"])
    await reviewer.finish(review["id"], reviewed["claim_token"], result="approved")
    response = await worker.reply(task["id"], "Done; review passed", "final-reply")
    await worker.finish(
        task["id"], work["claim_token"], result="delivered and reviewed"
    )
    assert response["in_reply_to"] == task["id"]
    assert response["thread_id"] == task["thread_id"]
    assert (await router.inbox())["counts"]["unread"] == 1
    assert (await router.status("worker", task["id"]))["state"] == "completed"
    assert (await router.read(response["id"]))["body"] == "Done; review passed"


async def test_client_identity_and_role_validation(actors):
    router = actors["router"]
    with pytest.raises(ValueError, match="unknown mailbox"):
        await MailboxClient.connect("unknown")
    with pytest.raises(ValueError, match="unknown mailbox"):
        await router.send("unknown", "Task", "body", "key")
    await router.create_role("child")
    assert (await router.create_role("child"))["parent"] == "router"
    with pytest.raises(Exception, match="different role metadata"):
        await actors["worker"].create_role("child")
    message = await router.send("worker", "Task", "body", "key")
    with pytest.raises(Exception, match="another sender"):
        await actors["reviewer"].status("worker", message["id"])


async def test_reply_to_maximum_unicode_subject(actors):
    task = await actors["router"].send("worker", "界" * 85, "body", "unicode")
    reply = await actors["worker"].reply(task["id"], "result", "unicode-reply")
    assert reply["subject"].startswith("Re: ")
    assert len(reply["subject"].encode("utf-8")) <= 256
