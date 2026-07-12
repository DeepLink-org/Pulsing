# SPDX-License-Identifier: Apache-2.0
"""Hub-side sandbox view (policy + static Bash risk hints — MVP sandbox summary only)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pulsing.agent.loop.sandbox import normalize_policy


@dataclass
class SandboxManager:
    """Aggregates effective Bash policy for status lines and lightweight heuristics."""

    policy: str
    dangerously_disable_sandbox: bool = False

    def effective_policy(self) -> str:
        if self.dangerously_disable_sandbox:
            return "off"
        return normalize_policy(self.policy)

    def describe(self) -> str:
        pol = self.effective_policy()
        bits = [
            f"bash_policy={pol}",
            "exec=argv list (no shell=True)",
        ]
        if pol == "bwrap":
            bits.append("bubblewrap minimal profile (Linux)")
        elif pol == "restricted":
            bits.append("env -i + bash --norc/--noprofile")
        return "; ".join(bits)

    def bash_warnings(self, command: str) -> list[str]:
        """Static heuristics only (does not parse shell robustly)."""
        c = (command or "").strip()
        out: list[str] = []
        if re.search(r"\brm\s+(-[rfRF]*\s*)*[/~]", c):
            out.append("possible broad rm — review paths")
        if "curl" in c and re.search(r"\|\s*sh", c):
            out.append("curl|sh pattern — high risk")
        if ">/" in c or ">>/" in c:
            out.append("redirect to absolute path — confirm target")
        return out
