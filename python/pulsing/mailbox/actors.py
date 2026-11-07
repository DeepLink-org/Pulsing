"""Durable, serially owned agent mailboxes built from ordinary Pulsing actors."""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from pathlib import Path

import pulsing as pul

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


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db


@pul.remote
class Mailbox:
    """One recipient's messages; all database access goes through this actor.

    This is a trusted-cluster prototype, not an authentication boundary. Public
    actor calls must not be exposed to untrusted nodes.
    """

    def __init__(self, identity: str, state_dir: str, capacity: int = 10000):
        self._identity = validate_identity(identity)
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        root = Path(state_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._db = _connect(root / f"mailbox-{identity}.sqlite3")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, sender TEXT NOT NULL, subject TEXT NOT NULL,
                body TEXT NOT NULL, thread_id TEXT NOT NULL, in_reply_to TEXT,
                idempotency_key TEXT NOT NULL, state TEXT NOT NULL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0, claim_token TEXT,
                lease_until REAL, result TEXT,
                UNIQUE(sender, idempotency_key)
            );
            CREATE INDEX IF NOT EXISTS inbox_state ON messages(state, created_at);
            """
        )

    def on_stop(self):
        self._db.close()

    def _row(self, message_id: str) -> sqlite3.Row:
        row = self._db.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise ValueError("message not found")
        return row

    def _public(self, row: sqlite3.Row) -> dict:
        return {
            key: row[key]
            for key in (
                "id",
                "sender",
                "subject",
                "body",
                "thread_id",
                "in_reply_to",
                "state",
                "created_at",
                "updated_at",
                "attempt",
                "lease_until",
                "result",
            )
        } | {"recipient": self._identity}

    def deliver(
        self,
        sender: str,
        subject: str,
        body: str,
        idempotency_key: str,
        thread_id: str = "",
        in_reply_to: str | None = None,
    ) -> dict:
        """Commit a delivery before acknowledging it; retries require the same key."""
        validate_identity(sender)
        _text(subject, "subject", 256)
        _text(body, "body", 65536, empty=True)
        _text(idempotency_key, "idempotency_key", 128)
        _text(thread_id, "thread_id", 128, empty=True)
        if in_reply_to is not None:
            _text(in_reply_to, "in_reply_to", 128)
        existing = self._db.execute(
            "SELECT * FROM messages WHERE sender = ? AND idempotency_key = ?",
            (sender, idempotency_key),
        ).fetchone()
        if existing is not None:
            same = (
                existing["subject"] == subject
                and existing["body"] == body
                and existing["in_reply_to"] == in_reply_to
                and (not thread_id or existing["thread_id"] == thread_id)
            )
            if not same:
                raise ValueError("idempotency key already used for different content")
            return self._public(existing)
        count = self._db.execute("SELECT count(*) FROM messages").fetchone()[0]
        if count >= self._capacity:
            raise ValueError("mailbox capacity reached")
        message_id = uuid.uuid4().hex
        now = time.time()
        with self._db:
            self._db.execute(
                """INSERT INTO messages
                (id, sender, subject, body, thread_id, in_reply_to, idempotency_key,
                 state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'unread', ?, ?)""",
                (
                    message_id,
                    sender,
                    subject,
                    body,
                    thread_id or message_id,
                    in_reply_to,
                    idempotency_key,
                    now,
                    now,
                ),
            )
        return self._public(self._row(message_id))

    def inbox(self, state: str | None = None, limit: int = 20, offset: int = 0) -> dict:
        """Listing is read-only; only read_message marks unread mail as read."""
        if state is not None and state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1–100 and offset must be nonnegative")
        rows = self._db.execute(
            """SELECT * FROM messages WHERE (? IS NULL OR state = ?)
            ORDER BY created_at, id LIMIT ? OFFSET ?""",
            (state, state, limit + 1, offset),
        ).fetchall()
        counts = {state: 0 for state in STATES}
        for row in self._db.execute(
            "SELECT state, count(*) n FROM messages GROUP BY state"
        ):
            counts[row["state"]] = row["n"]
        # Previews avoid filling an agent's context with all message bodies.
        items = [
            {k: v for k, v in self._public(row).items() if k not in ("body", "result")}
            for row in rows[:limit]
        ]
        return {
            "items": items,
            "counts": counts,
            "next_offset": offset + limit if len(rows) > limit else None,
        }

    def read_message(self, message_id: str) -> dict:
        self._row(message_id)
        with self._db:
            self._db.execute(
                """UPDATE messages SET state = 'read', updated_at = ?
                WHERE id = ? AND state = 'unread'""",
                (time.time(), message_id),
            )
        return self._public(self._row(message_id))

    def claim(self, message_id: str, lease_seconds: int = 300) -> dict:
        """Atomically claim work; expired attempts may be claimed again."""
        if not 1 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be 1–3600")
        row = self._row(message_id)
        now = time.time()
        eligible = row["state"] in ("unread", "read") or (
            row["state"] == "processing" and row["lease_until"] <= now
        )
        if not eligible:
            raise ValueError("message is already claimed or terminal")
        token = uuid.uuid4().hex
        with self._db:
            self._db.execute(
                """UPDATE messages SET state = 'processing', claim_token = ?,
                lease_until = ?, attempt = attempt + 1, updated_at = ? WHERE id = ?""",
                (token, now + lease_seconds, now, message_id),
            )
        return self._public(self._row(message_id)) | {"claim_token": token}

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
            if row["state"] == outcome and row["result"] == result:
                return self._public(row)  # An acknowledgement may have been lost.
            raise ValueError("terminal outcome cannot be changed")
        if row["state"] != "processing" or row["lease_until"] <= time.time():
            raise ValueError("claim has expired; claim again before finishing")
        with self._db:
            self._db.execute(
                """UPDATE messages SET state = ?, result = ?, lease_until = NULL,
                updated_at = ? WHERE id = ?""",
                (outcome, result, time.time(), message_id),
            )
        return self._public(self._row(message_id))

    def receipt(self, message_id: str, requester: str) -> dict:
        """Only the sender or recipient can query a receipt via the client."""
        row = self._row(message_id)
        if requester not in (row["sender"], self._identity):
            raise ValueError("receipt belongs to another sender")
        return {
            key: value
            for key, value in self._public(row).items()
            if key not in ("body", "subject")
        }


@pul.remote
class MailboxDirectory:
    """Durable role directory; creates a distinct actor for every role."""

    def __init__(self, state_dir: str):
        self._root = Path(state_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._db = _connect(self._root / "directory.sqlite3")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS roles
            (identity TEXT PRIMARY KEY, kind TEXT NOT NULL, parent TEXT)"""
        )
        self._started: set[str] = set()

    def on_stop(self):
        self._db.close()

    async def restore(self):
        for role in self._db.execute("SELECT * FROM roles").fetchall():
            if role["identity"] not in self._started:
                await Mailbox.spawn(
                    role["identity"],
                    str(self._root),
                    name=mailbox_name(role["identity"]),
                )
                self._started.add(role["identity"])

    async def register(
        self, identity: str, kind: str = "worker", parent: str | None = None
    ) -> dict:
        validate_identity(identity)
        if kind not in ROLE_KINDS:
            raise ValueError(f"kind must be one of {ROLE_KINDS}")
        if parent is not None:
            validate_identity(parent)
            if not self._db.execute(
                "SELECT 1 FROM roles WHERE identity = ?", (parent,)
            ).fetchone():
                raise ValueError("parent role does not exist")
        row = self._db.execute(
            "SELECT * FROM roles WHERE identity = ?", (identity,)
        ).fetchone()
        if row is not None:
            if row["kind"] != kind or row["parent"] != parent:
                raise ValueError(
                    "identity already registered with different role metadata"
                )
            return dict(row)
        await Mailbox.spawn(identity, str(self._root), name=mailbox_name(identity))
        self._started.add(identity)
        with self._db:
            self._db.execute(
                "INSERT INTO roles VALUES (?, ?, ?)", (identity, kind, parent)
            )
        return {"identity": identity, "kind": kind, "parent": parent}

    def roles(self) -> list[dict]:
        return [
            dict(row)
            for row in self._db.execute("SELECT * FROM roles ORDER BY identity")
        ]
