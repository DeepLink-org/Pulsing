"""Opt-in, token-redacted JSONL audit at the authenticated MCP HTTP boundary."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

SENSITIVE = {
    "claim_token",
    "token",
    "access_token",
    "api_key",
    "authorization",
    "PULSING_MCP_TOKEN",
    "encrypted_content",
    "encryptedContent",
}


def sanitized(value):
    if isinstance(value, dict):
        return {
            key: (
                "sha256:" + hashlib.sha256(str(item).encode()).hexdigest()
                if key in SENSITIVE
                else sanitized(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    if isinstance(value, str):
        try:
            nested = json.loads(value)
        except ValueError:
            pattern = r"""(['"]?(?:claim_token|access_token|api_key|PULSING_MCP_TOKEN)['"]?\s*[:=]\s*['"])([^'"]+)(['"])"""
            return re.sub(
                pattern,
                lambda match: (
                    match[1]
                    + "sha256:"
                    + hashlib.sha256(match[2].encode()).hexdigest()
                    + match[3]
                ),
                value,
            )
        if isinstance(nested, (dict, list)):
            return json.dumps(sanitized(nested), ensure_ascii=False)
    return value


class MCPAudit:
    def __init__(self, path: Path, instance: str):
        self.path = path.expanduser().absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.instance = instance
        self.write({"event": "audit.started"})

    def write(self, value):
        event = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
            "instance": self.instance,
            **value,
        }
        encoded = (json.dumps(sanitized(event), ensure_ascii=False) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(fd, encoded[offset:])
                if not written:
                    raise OSError("audit log write made no progress")
                offset += written
        finally:
            os.close(fd)

    def wrap(self, app):
        async def audited(scope, receive, send):
            if scope["type"] != "http" or scope["method"] != "POST":
                await app(scope, receive, send)
                return
            chunks = []
            length = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                length += len(chunk)
                if length > 4 * 1024 * 1024:
                    from starlette.responses import Response

                    await Response(status_code=413)(scope, receive, send)
                    return
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            try:
                request = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                request = {"invalid_json": True, "bytes": len(body)}
            params = request.get("params", {}) if isinstance(request, dict) else {}
            if not isinstance(params, dict):
                params = {}
            meta = params.get("_meta") or {}
            context = {
                "event_id": uuid.uuid4().hex,
                "method": request.get("method") if isinstance(request, dict) else None,
                "tool": params.get("name"),
                "thread_id": meta.get("threadId") if isinstance(meta, dict) else None,
            }
            self.write({**context, "event": "mcp.request", "rpc": request})
            sent_body = False
            response = bytearray()
            status = None
            started = time.monotonic()
            error = None

            async def replay():
                nonlocal sent_body
                if not sent_body:
                    sent_body = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            async def capture(message):
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                elif message["type"] == "http.response.body":
                    response.extend(message.get("body", b""))
                await send(message)

            try:
                await app(scope, replay, capture)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                try:
                    result = json.loads(response) if response else None
                except (ValueError, UnicodeDecodeError):
                    result = {"raw_response": response.decode(errors="replace")}
                self.write(
                    {
                        **context,
                        "event": "mcp.response",
                        "status": status,
                        "duration_ms": round((time.monotonic() - started) * 1000, 3),
                        "rpc": result,
                        "error": error,
                    }
                )

        return audited
