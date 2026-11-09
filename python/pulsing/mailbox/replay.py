"""Read actor observations offline. This module cannot spawn or restore actors."""

from __future__ import annotations

import json
from pathlib import Path

KINDS = {
    "directory.started",
    "directory.stopped",
    "mailbox.started",
    "mailbox.stopped",
    "message.delivered",
    "message.read",
    "message.claimed",
    "message.finished",
    "mailbox.listed",
    "receipt.checked",
    "operation.rejected",
    "delivery.retried",
    "finish.retried",
    "role.registered",
    "agent.bound",
    "dispatch.observed",
}


def replay_logs(paths: list[str | Path]) -> dict:
    streams = {}
    seen = {}
    messages = {}
    roles = []
    rejected = []
    dispatches = []
    events = 0
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    event = json.loads(line)
                    if event["version"] != 1:
                        raise ValueError("unsupported actor event version")
                    event_id = event["event_id"]
                    if event_id in seen:
                        if seen[event_id] != event:
                            raise ValueError("conflicting duplicate event")
                        continue
                    stream_id = event["stream_id"]
                    stream = streams.setdefault(
                        stream_id,
                        {
                            "actor": event["actor"],
                            "instance": event.get("instance"),
                            "sequence": 0,
                            "stopped": False,
                        },
                    )
                    if (
                        stream["actor"] != event["actor"]
                        or event["sequence"] != stream["sequence"] + 1
                    ):
                        raise ValueError("actor event sequence gap or reordering")
                    kind = event["event"]
                    if kind not in KINDS or stream["stopped"]:
                        raise ValueError("unknown event or event after actor stopped")
                    if kind.endswith(".started") and stream["sequence"] != 0:
                        raise ValueError("stream restarted without a new stream_id")
                    if stream["sequence"] == 0 and kind not in (
                        "mailbox.started",
                        "directory.started",
                    ):
                        raise ValueError("stream must begin with a started event")
                    stream["sequence"] = event["sequence"]
                    if kind.endswith(".stopped"):
                        stream["stopped"] = True
                    if kind == "message.delivered":
                        message = event["message"]
                        key = (stream_id, message["id"])
                        if key in messages:
                            raise ValueError("duplicate message delivery")
                        messages[key] = dict(message) | {
                            "stream_id": stream_id,
                            "actor": event["actor"],
                            "instance": event.get("instance"),
                        }
                    elif kind in (
                        "message.read",
                        "message.claimed",
                        "message.finished",
                    ):
                        message = messages[(stream_id, event["message_id"])]
                        if message["state"] != event["previous_state"]:
                            raise ValueError(
                                "message state does not match event history"
                            )
                        if not set(event["changes"]) <= {
                            "state",
                            "updated_at",
                            "attempt",
                            "lease_until",
                            "result",
                        }:
                            raise ValueError("event changes immutable message fields")
                        message.update(event["changes"])
                    elif kind in ("role.registered", "agent.bound"):
                        roles.append(event)
                    elif kind == "operation.rejected":
                        rejected.append(event)
                    elif kind == "dispatch.observed":
                        dispatches.append(event)
                    seen[event_id] = event
                    events += 1
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return {
        "events": events,
        "streams": streams,
        "roles": roles,
        "messages": sorted(
            messages.values(), key=lambda row: (row["created_at"], row["id"])
        ),
        "rejected": rejected,
        "dispatches": dispatches,
    }


def render(replay: dict, format: str = "json") -> str:
    if format == "json":
        return json.dumps(replay, ensure_ascii=False, indent=2) + "\n"
    if format != "markdown":
        raise ValueError("format must be json or markdown")
    lines = ["# Mailbox actor replay (offline, not live recovery)", ""]
    for message in replay["messages"]:
        lines.extend(
            [
                f"## {message['subject']}",
                "",
                f"- ID: `{message['id']}`",
                f"- Stream: `{message['stream_id']}`",
                f"- From: `{message['sender']}`",
                f"- To: `{message['recipient']}`",
                f"- State: `{message['state']}`",
                f"- In reply to: `{message['in_reply_to']}`",
                "",
                message["body"],
                "",
                f"Result: {message['result']}",
                "",
            ]
        )
    if replay.get("dispatches"):
        lines.extend(
            [
                "## Dispatch observations",
                "",
                "```json",
                json.dumps(replay["dispatches"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)
