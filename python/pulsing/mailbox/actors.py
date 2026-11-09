"""Bounded in-memory agent mailboxes, serially owned by ordinary Pulsing actors."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from functools import wraps
from itertools import islice

import pulsing as pul

from .events import ActorEvents
from .signals import delivery_signals

DIRECTORY_NAME = "mail/directory"
STATES = ("unread", "read", "processing", "completed", "failed")
ROLE_KINDS = ("router", "worker", "subagent", "reviewer", "observer")


def validate_identity(identity: str) -> str:
    if not isinstance(identity, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", identity
    ):
        raise ValueError("identity must be 1–64 ASCII letters, digits, _, . or -")
    return identity


def mailbox_name(identity: str) -> str:
    return f"mail/boxes/{validate_identity(identity)}"


def _text(value: str, name: str, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be text of at most {maximum} UTF-8 bytes")
    if not empty and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _observed(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (ValueError, TypeError) as exc:
            message_id = kwargs.get("message_id")
            if args and method.__name__ in (
                "read_message",
                "claim",
                "finish",
                "receipt",
            ):
                message_id = args[0]
            self._events.emit(
                "operation.rejected",
                operation=method.__name__,
                error_type=type(exc).__name__,
                error=str(exc),
                message_id=message_id
                if isinstance(message_id, str) and len(message_id) <= 128
                else None,
            )
            raise

    return call


@pul.remote
class Mailbox:
    """The actor owns RAM state; append-only observations are never recovery input.

    Trusted-cluster prototype, not an authentication boundary. Limits reject new
    data rather than silently evicting unread work or forgetting idempotency keys.
    """

    def __init__(
        self,
        identity: str,
        capacity: int = 10000,
        max_bytes: int = 8 * 1024 * 1024,
        event_log: str | None = None,
        instance: str | None = None,
    ):
        self._identity = validate_identity(identity)
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self._capacity = capacity
        self._max_bytes = max_bytes
        self._used_bytes = 0
        self._messages: dict[str, dict] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._counts = dict.fromkeys(STATES, 0)
        self._events = ActorEvents(event_log, mailbox_name(identity), instance)
        self._events.emit("mailbox.started", capacity=capacity, max_bytes=max_bytes)
        delivery_signals.start(identity, self._events.stream_id)

    def on_stop(self):
        try:
            self._events.emit("mailbox.stopped")
        finally:
            delivery_signals.stop(self._identity, self._events.stream_id)

    def _row(self, message_id: str) -> dict:
        _text(message_id, "message_id", 128)
        try:
            return self._messages[message_id]
        except KeyError:
            raise ValueError("message not found") from None

    def _public(self, row: dict) -> dict:
        return {
            key: value
            for key, value in row.items()
            if key not in ("claim_token", "idempotency_key")
        } | {"recipient": self._identity}

    @staticmethod
    def _weight(row: dict) -> int:
        # Logical UTF-8 payload budget plus fixed metadata allowance, not an RSS
        # estimate. Claim/read metadata changes never consume extra payload quota.
        return 1024 + sum(
            len((row.get(key) or "").encode("utf-8"))
            for key in (
                "sender",
                "subject",
                "body",
                "result",
                "thread_id",
                "in_reply_to",
                "idempotency_key",
            )
        )

    def _store(self, row: dict, event: str) -> None:
        previous = self._messages.get(row["id"])
        # Message payloads are immutable; only finish() can grow the result.
        delta = (
            self._weight(row)
            if previous is None
            else (
                len((row["result"] or "").encode("utf-8"))
                - len((previous["result"] or "").encode("utf-8"))
            )
        )
        if self._used_bytes + delta > self._max_bytes:
            raise ValueError("mailbox byte capacity reached")
        fields = (
            {"message": self._public(row), "idempotency_key": row["idempotency_key"]}
            if previous is None
            else {
                "message_id": row["id"],
                "previous_state": previous["state"],
                "changes": {
                    key: value
                    for key, value in self._public(row).items()
                    if key != "recipient" and previous.get(key) != value
                },
            }
        )
        # A failed log append is visible and does not publish a RAM transition.
        # This is not a distributed transaction or a crash-recovery protocol.
        self._events.emit(event, **fields)
        if previous is not None:
            self._counts[previous["state"]] -= 1
        self._messages[row["id"]] = row
        self._counts[row["state"]] += 1
        self._used_bytes += delta
        if previous is None:
            delivery_signals.publish(
                self._identity, self._events.stream_id, len(self._messages)
            )

    @_observed
    def changes(
        self, epoch: str | None = None, after: int = 0, limit: int = 20
    ) -> dict:
        """Read delivery edges without reading/claiming work or waiting in this actor."""
        if not isinstance(after, int) or after < 0:
            raise ValueError("after must be a nonnegative integer")
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be 1–100")
        if epoch != self._events.stream_id:
            after = 0
        if after > len(self._messages):
            raise ValueError("delivery cursor is ahead of this mailbox")
        items = []
        cursor = after
        for cursor, row in enumerate(
            islice(self._messages.values(), after, None), start=after + 1
        ):
            if row["state"] in ("unread", "read"):
                items.append({"id": row["id"], "state": row["state"]})
                if len(items) == limit:
                    break
        return {
            "epoch": self._events.stream_id,
            "cursor": cursor,
            "items": items,
            "more": cursor < len(self._messages),
        }

    @_observed
    def observe_dispatch(
        self,
        phase: str,
        dispatch_id: str = "",
        message_ids: list[str] | None = None,
        reason: str = "",
    ) -> dict:
        """An adapter observation, not a claim, completion, or execution authority."""
        if phase not in (
            "requested",
            "submitted",
            "started",
            "settled",
            "paused",
            "resumed",
            "stopped",
        ):
            raise ValueError("invalid dispatch observation phase")
        if reason not in (
            "",
            "delivery",
            "cancelled",
            "budget",
            "operator",
            "transport",
            "harness",
            "session",
        ):
            raise ValueError("invalid dispatch observation reason")
        _text(dispatch_id, "dispatch_id", 128, empty=True)
        message_ids = message_ids or []
        if not isinstance(message_ids, list) or len(message_ids) > 20:
            raise ValueError("message_ids must contain at most 20 IDs")
        for message_id in message_ids:
            self._row(message_id)
        self._events.emit(
            "dispatch.observed",
            phase=phase,
            dispatch_id=dispatch_id,
            message_ids=message_ids,
            reason=reason,
        )
        return {"recorded": True}

    @_observed
    def deliver(
        self,
        sender: str,
        subject: str,
        body: str,
        idempotency_key: str,
        thread_id: str = "",
        in_reply_to: str | None = None,
    ) -> dict:
        """Accept in RAM before acknowledging; retries within this lifetime dedup."""
        validate_identity(sender)
        _text(subject, "subject", 256)
        _text(body, "body", 65536, empty=True)
        _text(idempotency_key, "idempotency_key", 128)
        _text(thread_id, "thread_id", 128, empty=True)
        if in_reply_to is not None:
            _text(in_reply_to, "in_reply_to", 128)
        key = (sender, idempotency_key)
        if key in self._keys:
            existing = self._row(self._keys[key])
            if not (
                existing["subject"] == subject
                and existing["body"] == body
                and existing["in_reply_to"] == in_reply_to
                and (not thread_id or existing["thread_id"] == thread_id)
            ):
                raise ValueError("idempotency key already used for different content")
            self._events.emit("delivery.retried", message_id=existing["id"])
            return self._public(existing)
        if len(self._messages) >= self._capacity:
            raise ValueError("mailbox capacity reached")
        message_id = uuid.uuid4().hex
        now = time.time()
        row = {
            "id": message_id,
            "sender": sender,
            "subject": subject,
            "body": body,
            "thread_id": thread_id or message_id,
            "in_reply_to": in_reply_to,
            "idempotency_key": idempotency_key,
            "state": "unread",
            "created_at": now,
            "updated_at": now,
            "attempt": 0,
            "claim_token": None,
            "lease_until": None,
            "result": None,
        }
        self._store(row, "message.delivered")
        self._keys[key] = message_id
        return self._public(row)

    @_observed
    def inbox(self, state: str | None = None, limit: int = 20, offset: int = 0) -> dict:
        if state is not None and state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1–100 and offset must be nonnegative")
        rows = list(
            islice(
                (
                    row
                    for row in self._messages.values()
                    if state is None or row["state"] == state
                ),
                offset,
                offset + limit + 1,
            )
        )
        result = {
            "items": [
                {
                    key: value
                    for key, value in self._public(row).items()
                    if key not in ("body", "result")
                }
                for row in rows[:limit]
            ],
            "counts": dict(self._counts),
            "next_offset": offset + limit if len(rows) > limit else None,
        }
        self._events.emit(
            "mailbox.listed",
            state=state,
            limit=limit,
            offset=offset,
            message_ids=[row["id"] for row in rows[:limit]],
            counts=result["counts"],
        )
        return result

    @_observed
    def read_message(self, message_id: str) -> dict:
        row = self._row(message_id)
        updated = (
            row | {"state": "read", "updated_at": time.time()}
            if row["state"] == "unread"
            else dict(row)
        )
        self._store(updated, "message.read")
        return self._public(updated)

    @_observed
    def claim(self, message_id: str, lease_seconds: int = 300) -> dict:
        if not 1 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be 1–3600")
        row = self._row(message_id)
        now = time.time()
        if not (
            row["state"] in ("unread", "read")
            or (row["state"] == "processing" and row["lease_until"] <= now)
        ):
            raise ValueError("message is already claimed or terminal")
        token = uuid.uuid4().hex
        updated = row | {
            "state": "processing",
            "claim_token": token,
            "lease_until": now + lease_seconds,
            "attempt": row["attempt"] + 1,
            "updated_at": now,
        }
        self._store(updated, "message.claimed")
        return self._public(updated) | {"claim_token": token}

    @_observed
    def finish(
        self,
        message_id: str,
        claim_token: str,
        outcome: str = "completed",
        result: str = "",
    ) -> dict:
        if outcome not in ("completed", "failed"):
            raise ValueError("outcome must be completed or failed")
        _text(result, "result", 65536, empty=True)
        row = self._row(message_id)
        if row["claim_token"] != claim_token:
            raise ValueError("claim token does not match")
        if row["state"] in ("completed", "failed"):
            if row["state"] != outcome or row["result"] != result:
                raise ValueError("terminal outcome cannot be changed")
            self._events.emit("finish.retried", message_id=message_id)
            return self._public(row)
        if row["state"] != "processing" or row["lease_until"] <= time.time():
            raise ValueError("claim has expired; claim again before finishing")
        updated = row | {
            "state": outcome,
            "result": result,
            "lease_until": None,
            "updated_at": time.time(),
        }
        self._store(updated, "message.finished")
        return self._public(updated)

    @_observed
    def receipt(self, message_id: str, requester: str) -> dict:
        row = self._row(message_id)
        if requester not in (row["sender"], self._identity):
            raise ValueError("receipt belongs to another sender")
        self._events.emit(
            "receipt.checked",
            message_id=message_id,
            state=row["state"],
            requester=requester,
        )
        return {
            key: value
            for key, value in self._public(row).items()
            if key not in ("body", "subject")
        }


@pul.remote
class MailboxDirectory:
    """Volatile role/agent registry; all owned mailboxes live on this node."""

    def __init__(self, namespace: str | None = None, event_log: str | None = None):
        if namespace is not None:
            validate_identity(namespace)
            if len(namespace) > 24 or "." in namespace:
                raise ValueError(
                    "instance identity must be at most 24 letters, digits, _ or -"
                )
        self._namespace = namespace
        self._event_log = event_log
        self._roles: dict[str, dict] = {}
        self._agents: dict[str, str] = {}
        name = f"mail/instances/{namespace}" if namespace else DIRECTORY_NAME
        self._events = ActorEvents(event_log, name, namespace)
        self._events.emit("directory.started")

    def on_stop(self):
        self._events.emit("directory.stopped")

    async def register(
        self, identity: str, kind: str = "worker", parent: str | None = None
    ) -> dict:
        validate_identity(identity)
        if self._namespace and not identity.startswith(self._namespace + "-"):
            raise ValueError("mailbox must belong to this instance")
        if kind not in ROLE_KINDS:
            raise ValueError(f"kind must be one of {ROLE_KINDS}")
        if parent is not None and validate_identity(parent) not in self._roles:
            raise ValueError("parent role does not exist")
        role = {"identity": identity, "kind": kind, "parent": parent}
        if identity in self._roles:
            if self._roles[identity] != role:
                raise ValueError(
                    "identity already registered with different role metadata"
                )
            return dict(role)
        await Mailbox.spawn(
            identity,
            event_log=self._event_log,
            instance=self._namespace,
            name=mailbox_name(identity),
        )
        self._events.emit("role.registered", role=role)
        self._roles[identity] = role
        return dict(role)

    async def ensure_role(self, identity: str, kind: str = "worker") -> dict:
        validate_identity(identity)
        if kind not in ROLE_KINDS:
            raise ValueError(f"kind must be one of {ROLE_KINDS}")
        return (
            dict(self._roles[identity])
            if identity in self._roles
            else await self.register(identity, kind)
        )

    async def ensure_agent(self, thread_id: str) -> dict:
        if self._namespace is None:
            raise ValueError("agent registration requires a companion directory")
        _text(thread_id, "thread_id", 256)
        identity = (
            f"{self._namespace}-{hashlib.sha256(thread_id.encode()).hexdigest()[:32]}"
        )
        role = await self.ensure_role(identity)
        if identity not in self._agents:
            self._events.emit("agent.bound", identity=identity, thread_id=thread_id)
            self._agents[identity] = thread_id
        return {**role, "instance": self._namespace, "agent_id": thread_id}

    def roles(self) -> list[dict]:
        return [
            dict(role)
            | (
                {"instance": self._namespace, "agent_id": self._agents.get(identity)}
                if self._namespace
                else {}
            )
            for identity, role in sorted(self._roles.items())
        ]
