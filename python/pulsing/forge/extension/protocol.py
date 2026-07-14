# SPDX-License-Identifier: Apache-2.0
"""Codex Extension tool names and shared wire constants."""

from __future__ import annotations

# --- web.run (client search / browse; ext/web-search) ---
WEB_NAMESPACE = "web"
WEB_RUN_TOOL = "web.run"

# --- skills (ext/skills + ~/.agents/skills) ---
SKILLS_NAMESPACE = "skills"
SKILLS_LIST_TOOL = "skills.list"
SKILLS_READ_TOOL = "skills.read"

# --- memories (ext/memories; $CODEX_HOME/memories) ---
MEMORIES_NAMESPACE = "memories"
MEMORIES_LIST_TOOL = "memories.list"
MEMORIES_READ_TOOL = "memories.read"
MEMORIES_SEARCH_TOOL = "memories.search"
MEMORIES_ADD_AD_HOC_NOTE_TOOL = "memories.add_ad_hoc_note"

# --- hosted Provider tool (Responses API; not client sandbox) ---
WEB_SEARCH_TOOL = "web_search"

WEB_RUN_TOOLS: frozenset[str] = frozenset({WEB_RUN_TOOL})
SKILLS_TOOLS: frozenset[str] = frozenset({SKILLS_LIST_TOOL, SKILLS_READ_TOOL})
MEMORIES_TOOLS: frozenset[str] = frozenset(
    {
        MEMORIES_LIST_TOOL,
        MEMORIES_READ_TOOL,
        MEMORIES_SEARCH_TOOL,
        MEMORIES_ADD_AD_HOC_NOTE_TOOL,
    }
)
WEB_SEARCH_TOOLS: frozenset[str] = frozenset({WEB_SEARCH_TOOL})

EXTENSION_TOOL_NAMES: frozenset[str] = (
    WEB_RUN_TOOLS | SKILLS_TOOLS | MEMORIES_TOOLS | WEB_SEARCH_TOOLS
)
