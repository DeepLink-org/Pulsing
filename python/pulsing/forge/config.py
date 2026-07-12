# SPDX-License-Identifier: Apache-2.0
"""Tool worker configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolWorkerConfig:
    cwd: str = "."
    sandbox_policy: str = "off"
    dangerously_disable_sandbox: bool = False
    auto_approve: bool = False
    # Gossip name of the ForgeEventInbox (or host) that receives Forge tell events.
    event_sink_name: str | None = None
    # Host agent gossip name for exec approval / permissions ask RPC.
    host_name: str | None = None

    def approval_sink(self) -> str | None:
        return (self.host_name or self.event_sink_name or "").strip() or None
