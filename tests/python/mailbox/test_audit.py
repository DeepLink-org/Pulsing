from __future__ import annotations

import json

import pytest

from pulsing.mailbox.audit import MCPAudit, sanitized


def test_redacts_tokens_in_structured_and_nested_json():
    value = {
        "claim_token": "private-claim",
        "content": [
            {"text": json.dumps({"claim_token": "private-claim", "body": "数学论点"})}
        ],
    }
    encoded = json.dumps(sanitized(value), ensure_ascii=False)
    assert "private-claim" not in encoded
    assert "数学论点" in encoded
    assert "sha256:" in encoded


async def test_records_full_request_response_without_headers(tmp_path):
    audit = MCPAudit(tmp_path / "events.jsonl", "one")
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "mail_finish",
                "_meta": {"threadId": "child"},
                "arguments": {"message_id": "m1", "claim_token": "private-claim"},
            },
        }
    ).encode()
    sent = []

    async def receive():
        return {"type": "http.request", "body": body}

    async def send(message):
        sent.append(message)

    async def app(scope, receive, send):
        assert (await receive())["body"] == body
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(
                    {"jsonrpc": "2.0", "id": 7, "result": {"state": "completed"}}
                ).encode(),
            }
        )

    await audit.wrap(app)(
        {
            "type": "http",
            "method": "POST",
            "headers": [(b"authorization", b"Bearer private-auth")],
        },
        receive,
        send,
    )
    text = audit.path.read_text()
    events = [json.loads(line) for line in text.splitlines()]
    assert len(events) == 3
    assert events[1]["event"] == "mcp.request"
    assert events[1]["thread_id"] == "child"
    assert events[1]["event_id"] == events[2]["event_id"]
    assert events[2]["rpc"]["result"]["state"] == "completed"
    assert "private-claim" not in text
    assert "private-auth" not in text
    assert (audit.path.stat().st_mode & 0o777) == 0o600


async def test_records_server_error_as_incomplete_response(tmp_path):
    audit = MCPAudit(tmp_path / "events.jsonl", "one")

    async def receive():
        return {"type": "http.request", "body": b'{"method":"tools/call"}'}

    async def app(*args):
        raise RuntimeError("failed")

    async def send(message):
        pass

    with pytest.raises(RuntimeError):
        await audit.wrap(app)({"type": "http", "method": "POST"}, receive, send)
    events = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert events[-1]["error"] == "RuntimeError"


async def test_audit_with_actual_mcp_http_client(tmp_path):
    pytest.importorskip("mcp")
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.fastmcp import FastMCP

    from pulsing.cli.launch import mcp_endpoint

    server = FastMCP("audit-test")

    @server.tool()
    async def claim() -> dict:
        return {"id": "one", "state": "processing", "claim_token": "private-claim"}

    audit = MCPAudit(tmp_path / "wire.jsonl", "companion")
    async with mcp_endpoint(server, "private-auth", audit) as (url, _):
        async with httpx.AsyncClient(
            headers={"Authorization": "Bearer private-auth"}
        ) as http:
            async with streamable_http_client(url, http_client=http) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    result = await client.call_tool(
                        "claim", {}, meta={"threadId": "child"}
                    )
                    assert not result.isError
                    assert "private-claim" in result.content[0].text
    text = audit.path.read_text()
    events = [json.loads(line) for line in text.splitlines()]
    tool = [event for event in events if event.get("tool") == "claim"]
    assert len(tool) == 2
    assert tool[0]["thread_id"] == "child"
    assert tool[0]["event_id"] == tool[1]["event_id"]
    payload = json.loads(tool[1]["rpc"]["result"]["content"][0]["text"])
    assert payload["state"] == "processing"
    assert "private-auth" not in text
    assert "private-claim" not in text
