# SPDX-License-Identifier: Apache-2.0
"""Tests for pulsing.forge.tool_calls."""

from __future__ import annotations

import pytest

from pulsing.forge.result import ToolResult
from pulsing.forge.tool_calls import (
    OpenAIToolCallAccumulator,
    anthropic_tool_result_block,
    extract_tool_calls,
    forge_tool_definitions,
    openai_tool_message,
    parse_tool_arguments,
    to_anthropic_tools,
    to_openai_tools,
)

pytestmark = pytest.mark.forge


def test_parse_tool_arguments_json_string() -> None:
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}


def test_parse_tool_arguments_invalid_json() -> None:
    assert parse_tool_arguments("{bad") == {}


def test_extract_openai_tool_calls() -> None:
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "Glob",
                    "arguments": '{"pattern": "*.md"}',
                },
            }
        ],
    }
    calls = extract_tool_calls(msg, provider="openai")
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "Glob"
    assert calls[0].arguments == {"pattern": "*.md"}


def test_extract_anthropic_tool_calls() -> None:
    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "checking"},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "Read",
                "input": {"file_path": "a.txt"},
            },
        ],
    }
    calls = extract_tool_calls(msg, provider="anthropic")
    assert len(calls) == 1
    assert calls[0].name == "Read"
    assert calls[0].arguments["file_path"] == "a.txt"


def test_openai_tool_message_and_anthropic_block() -> None:
    result = ToolResult(content="ok", is_error=False)
    om = openai_tool_message("id1", result)
    assert om["role"] == "tool"
    assert om["tool_call_id"] == "id1"

    block = anthropic_tool_result_block("id1", ToolResult(content="err", is_error=True))
    assert block["type"] == "tool_result"
    assert block["is_error"] is True


def test_schema_conversion_roundtrip() -> None:
    anthropic = forge_tool_definitions(["Glob", "Read"])
    openai = to_openai_tools(anthropic)
    names = {t["function"]["name"] for t in openai}
    assert names == {"Glob", "Read"}
    back = to_anthropic_tools(openai)
    assert {t["name"] for t in back} == {"Glob", "Read"}
    assert "input_schema" in back[0]


def test_openai_stream_accumulator() -> None:
    acc = OpenAIToolCallAccumulator()
    acc.feed_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "Glob", "arguments": '{"pat'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    acc.feed_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": 'tern": "*.py"}'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    calls = acc.finish()
    assert calls[0].name == "Glob"
    assert calls[0].arguments == {"pattern": "*.py"}
