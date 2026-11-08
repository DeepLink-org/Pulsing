"""A deterministic harness used to test the launcher without model inference."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    url = os.environ["PULSING_MCP_URL"]
    headers = {"Authorization": f"Bearer {os.environ['PULSING_MCP_TOKEN']}"}
    async with httpx.AsyncClient(headers=headers) as http:
        async with streamable_http_client(url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as client:
                initialized = await client.initialize()
                assert "threadId" in initialized.instructions

                async def call(name, arguments=None, thread="main"):
                    response = await client.call_tool(
                        name, arguments or {}, meta={"threadId": thread}
                    )
                    assert not response.isError, response.content
                    if name == "mail_roles":
                        return response.structuredContent["result"]
                    return json.loads(response.content[0].text)

                identity = await call("mail_identity")
                print(json.dumps({"ready": identity, "args": sys.argv[1:]}), flush=True)
                while True:
                    line = await asyncio.to_thread(sys.stdin.readline)
                    if not line:
                        return
                    request = json.loads(line)
                    if request["tool"] == "missing-meta":
                        result = await client.call_tool("mail_list", {})
                        print(json.dumps({"isError": result.isError}), flush=True)
                        continue
                    try:
                        result = await call(
                            request["tool"],
                            request.get("arguments"),
                            request.get("thread", "main"),
                        )
                        print(json.dumps({"result": result}), flush=True)
                    except AssertionError as exc:
                        print(json.dumps({"error": str(exc)}), flush=True)


asyncio.run(main())
