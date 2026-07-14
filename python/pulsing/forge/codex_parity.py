# SPDX-License-Identifier: Apache-2.0
"""Internal tool-surface conformance tracker (registry / callable / wire / behavior).

Used by CI and ``tests/python/forge/test_gates.py``. Not linked from public product docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pulsing.forge.integrated import (
    FORGE_HOST_TOOL_NAMES,
    FORGE_ISOLATED_TOOL_NAMES,
    FORGE_TOOL_NAMES,
)

Gate = Literal["pass", "partial", "fail", "na", "equiv"]

# Host tools with no Rust handler today (maturin default path → Unknown tool).
PYTHON_ONLY_HOST: frozenset[str] = frozenset(
    {
        "exec",
        "wait",
        "web.run",
        "skills.list",
        "skills.read",
        "memories.list",
        "memories.read",
        "memories.search",
        "memories.add_ad_hoc_note",
        "web_search",
    }
)

RUST_BUILTIN: frozenset[str] = frozenset(
    {
        "shell_command",
        "exec_command",
        "write_stdin",
        "apply_patch",
        "view_image",
        "update_plan",
        "new_context",
        "get_context_remaining",
        "request_user_input",
        "request_permissions",
        "tool_search",
        "list_available_plugins_to_install",
        "request_plugin_install",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
        "Read",
        "Glob",
        "Grep",
        "Edit",
        "Write",
        "Bash",
    }
)


class Scope(str, Enum):
    """Whether a Codex capability counts toward the CCRP claim."""

    CCRP = "ccrp"  # must parity for "Codex base" marketing
    FORGE_EXTRA = "forge_extra"  # Forge adds (Claude interop)
    EQUIVALENT = "equivalent"  # covered elsewhere (Craft), not same tool name
    OUT_OF_SCOPE = "out_of_scope"  # intentionally not Forge (TUI, hosted-only, etc.)


@dataclass(frozen=True)
class ParityEntry:
    codex_id: str
    forge_tool: str | None
    scope: Scope
    domain: str
    wire: Gate = "partial"
    behavior: Gate = "partial"
    integration: Gate = "partial"
    equivalent: str | None = None
    notes: str = ""


# Codex Client Runtime Profile — standard coding-agent tool surface (spec_plan + ext/*).
# Update wire/behavior/integration when tests land; registry/callable are computed.
CCRP_MANIFEST: tuple[ParityEntry, ...] = (
    # Execution
    ParityEntry(
        "shell_command",
        "shell_command",
        Scope.CCRP,
        "exec",
        "partial",
        "partial",
        "partial",
    ),
    ParityEntry(
        "exec_command",
        "exec_command",
        Scope.CCRP,
        "exec",
        "partial",
        "partial",
        "partial",
    ),
    ParityEntry(
        "write_stdin",
        "write_stdin",
        Scope.CCRP,
        "exec",
        "partial",
        "partial",
        "partial",
    ),
    # Files / patch
    ParityEntry(
        "apply_patch",
        "apply_patch",
        Scope.CCRP,
        "fs",
        "partial",
        "partial",
        "fail",
        notes="Freeform LLM exposure pending",
    ),
    ParityEntry(
        "view_image", "view_image", Scope.CCRP, "fs", "partial", "partial", "partial"
    ),
    ParityEntry("Read", "Read", Scope.FORGE_EXTRA, "fs", "na", "na", "pass"),
    ParityEntry("Glob", "Glob", Scope.FORGE_EXTRA, "fs", "na", "na", "pass"),
    ParityEntry("Grep", "Grep", Scope.FORGE_EXTRA, "fs", "na", "na", "pass"),
    ParityEntry("Edit", "Edit", Scope.FORGE_EXTRA, "fs", "na", "na", "pass"),
    ParityEntry("Write", "Write", Scope.FORGE_EXTRA, "fs", "na", "na", "pass"),
    ParityEntry("Bash", "Bash", Scope.FORGE_EXTRA, "exec", "na", "na", "pass"),
    # Session
    ParityEntry(
        "update_plan", "update_plan", Scope.CCRP, "session", "pass", "pass", "partial"
    ),
    ParityEntry(
        "request_user_input",
        "request_user_input",
        Scope.CCRP,
        "session",
        "pass",
        "partial",
        "partial",
        notes="No Codex TUI; autoResolutionMs ok",
    ),
    ParityEntry(
        "request_permissions",
        "request_permissions",
        Scope.CCRP,
        "session",
        "pass",
        "partial",
        "partial",
    ),
    ParityEntry(
        "new_context", "new_context", Scope.CCRP, "session", "pass", "pass", "partial"
    ),
    ParityEntry(
        "get_context_remaining",
        "get_context_remaining",
        Scope.CCRP,
        "session",
        "pass",
        "partial",
        "partial",
        notes="Token estimate only",
    ),
    # Discovery / plugins
    ParityEntry(
        "tool_search",
        "tool_search",
        Scope.CCRP,
        "discovery",
        "partial",
        "partial",
        "partial",
    ),
    ParityEntry(
        "list_available_plugins_to_install",
        "list_available_plugins_to_install",
        Scope.CCRP,
        "discovery",
        "pass",
        "partial",
        "partial",
    ),
    ParityEntry(
        "request_plugin_install",
        "request_plugin_install",
        Scope.CCRP,
        "discovery",
        "pass",
        "partial",
        "partial",
    ),
    # MCP
    ParityEntry(
        "list_mcp_resources",
        "list_mcp_resources",
        Scope.CCRP,
        "mcp",
        "pass",
        "partial",
        "partial",
        notes="Hybrid Rust path wired; Python-only LocalToolRuntime has no MCP",
    ),
    ParityEntry(
        "list_mcp_resource_templates",
        "list_mcp_resource_templates",
        Scope.CCRP,
        "mcp",
        "pass",
        "partial",
        "partial",
    ),
    ParityEntry(
        "read_mcp_resource",
        "read_mcp_resource",
        Scope.CCRP,
        "mcp",
        "pass",
        "partial",
        "partial",
    ),
    ParityEntry(
        "mcp_dynamic_tools",
        None,
        Scope.CCRP,
        "mcp",
        "pass",
        "partial",
        "partial",
        notes="Per-server function tools via Rust MCP runtime",
    ),
    # Code mode
    ParityEntry(
        "exec",
        "exec",
        Scope.CCRP,
        "code_mode",
        "partial",
        "partial",
        "fail",
        notes="L2 yield resume; no OS sandbox",
    ),
    ParityEntry("wait", "wait", Scope.CCRP, "code_mode", "partial", "partial", "fail"),
    # Extension
    ParityEntry(
        "web.run", "web.run", Scope.CCRP, "extension", "partial", "partial", "fail"
    ),
    ParityEntry(
        "skills.list",
        "skills.list",
        Scope.CCRP,
        "extension",
        "partial",
        "partial",
        "fail",
    ),
    ParityEntry(
        "skills.read",
        "skills.read",
        Scope.CCRP,
        "extension",
        "partial",
        "partial",
        "fail",
    ),
    ParityEntry(
        "memories.list",
        "memories.list",
        Scope.CCRP,
        "extension",
        "pass",
        "pass",
        "partial",
    ),
    ParityEntry(
        "memories.read",
        "memories.read",
        Scope.CCRP,
        "extension",
        "pass",
        "pass",
        "partial",
    ),
    ParityEntry(
        "memories.search",
        "memories.search",
        Scope.CCRP,
        "extension",
        "pass",
        "pass",
        "partial",
    ),
    ParityEntry(
        "memories.add_ad_hoc_note",
        "memories.add_ad_hoc_note",
        Scope.CCRP,
        "extension",
        "pass",
        "pass",
        "partial",
    ),
    ParityEntry(
        "web_search",
        "web_search",
        Scope.CCRP,
        "extension",
        "partial",
        "na",
        "fail",
        notes="Hosted; Craft model config",
    ),
    # Equivalents (not counted in CCRP score denominator as failures)
    ParityEntry(
        "spawn_agent",
        None,
        Scope.EQUIVALENT,
        "multi_agent",
        "na",
        "na",
        "partial",
        equivalent="Craft Summon",
    ),
    ParityEntry(
        "multi_agent_v2",
        None,
        Scope.EQUIVALENT,
        "multi_agent",
        "na",
        "na",
        "partial",
        equivalent="Craft cluster tools",
    ),
    # Out of scope
    ParityEntry(
        "image_gen.imagegen", None, Scope.OUT_OF_SCOPE, "extension", "na", "na", "na"
    ),
    ParityEntry("goals", None, Scope.OUT_OF_SCOPE, "extension", "na", "na", "na"),
    ParityEntry(
        "spawn_agents_on_csv", None, Scope.OUT_OF_SCOPE, "agent_jobs", "na", "na", "na"
    ),
    ParityEntry(
        "codex_cli_tui_login", None, Scope.OUT_OF_SCOPE, "product", "na", "na", "na"
    ),
)


@dataclass
class GateScore:
    name: str
    passed: int
    total: int
    pct: float
    blockers: list[str] = field(default_factory=list)


@dataclass
class ParityReport:
    certification: str
    gates: dict[str, GateScore]
    ccrp_entries: list[ParityEntry]
    summary: str


def _registry_status(entry: ParityEntry) -> Gate:
    if entry.scope != Scope.CCRP:
        return "na"
    if entry.codex_id == "mcp_dynamic_tools":
        # Runtime-registered mcp__* tools; not a static FORGE_TOOL_NAMES entry.
        return "pass"
    name = entry.forge_tool
    if not name:
        return "fail"
    return "pass" if name in FORGE_TOOL_NAMES else "fail"


def _callable_status(entry: ParityEntry, *, rust_available: bool) -> Gate:
    if entry.scope != Scope.CCRP:
        return "na"
    if entry.codex_id == "mcp_dynamic_tools":
        return "pass" if rust_available else "partial"
    name = entry.forge_tool
    if not name or name not in FORGE_TOOL_NAMES:
        return "fail"
    if rust_available and name in PYTHON_ONLY_HOST:
        return "pass"
    if name in FORGE_ISOLATED_TOOL_NAMES or name in FORGE_HOST_TOOL_NAMES:
        if (
            rust_available
            and name not in RUST_BUILTIN
            and name not in FORGE_ISOLATED_TOOL_NAMES
        ):
            return "fail"
        return "pass"
    return "fail"


def _score_gate(
    entries: list[ParityEntry], gate: str, *, rust_available: bool = True
) -> GateScore:
    ccrp = [e for e in entries if e.scope == Scope.CCRP]
    blockers: list[str] = []
    passed = 0
    for e in ccrp:
        if gate == "registry":
            st = _registry_status(e)
        elif gate == "callable":
            st = _callable_status(e, rust_available=rust_available)
        else:
            st = getattr(e, gate, "partial")
        if st == "pass":
            passed += 1
        elif st in ("fail", "partial"):
            blockers.append(f"{e.codex_id}: {st}")
    total = len(ccrp)
    pct = (100.0 * passed / total) if total else 100.0
    return GateScore(gate, passed, total, pct, blockers)


def parity_report(*, rust_available: bool = True) -> ParityReport:
    """Compute CCRP gate scores. Used in CI and docs."""
    gates = {
        "registry": _score_gate(
            list(CCRP_MANIFEST), "registry", rust_available=rust_available
        ),
        "callable": _score_gate(
            list(CCRP_MANIFEST), "callable", rust_available=rust_available
        ),
        "wire": _score_gate(list(CCRP_MANIFEST), "wire", rust_available=rust_available),
        "behavior": _score_gate(
            list(CCRP_MANIFEST), "behavior", rust_available=rust_available
        ),
        "integration": _score_gate(
            list(CCRP_MANIFEST), "integration", rust_available=rust_available
        ),
    }
    cert = _certification_level(gates)
    ccrp = [e for e in CCRP_MANIFEST if e.scope == Scope.CCRP]
    summary = (
        f"CCRP certification: {cert} | "
        f"registry {gates['registry'].pct:.0f}% | "
        f"callable {gates['callable'].pct:.0f}% | "
        f"wire {gates['wire'].pct:.0f}% | "
        f"behavior {gates['behavior'].pct:.0f}% | "
        f"integration {gates['integration'].pct:.0f}%"
    )
    return ParityReport(cert, gates, ccrp, summary)


def _certification_level(gates: dict[str, GateScore]) -> str:
    if gates["integration"].pct >= 95:
        return "Platinum"
    if gates["behavior"].pct >= 90:
        return "Gold"
    if gates["wire"].pct >= 90:
        return "Silver"
    if gates["registry"].pct >= 100 and gates["callable"].pct >= 100:
        return "Bronze+"
    if gates["registry"].pct >= 100:
        return "Bronze"
    return "Incomplete"


def assert_registry_gate() -> None:
    """CI: every CCRP tool must appear in FORGE_TOOL_NAMES."""
    report = parity_report()
    g = report.gates["registry"]
    assert g.pct == 100.0, f"Registry gate failed ({g.passed}/{g.total}): {g.blockers}"


def format_report_text(report: ParityReport | None = None) -> str:
    report = report or parity_report()
    lines = [report.summary, ""]
    for name, g in report.gates.items():
        lines.append(f"  {name}: {g.passed}/{g.total} ({g.pct:.0f}%)")
        for b in g.blockers[:8]:
            lines.append(f"    - {b}")
        if len(g.blockers) > 8:
            lines.append(f"    ... +{len(g.blockers) - 8} more")
    return "\n".join(lines)
