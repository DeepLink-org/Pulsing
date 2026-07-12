# SPDX-License-Identifier: Apache-2.0
"""Codex plugin id: `{plugin_name}@{marketplace_name}`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PluginId:
    plugin_name: str
    marketplace_name: str

    @property
    def id(self) -> str:
        return f"{self.plugin_name}@{self.marketplace_name}"

    @classmethod
    def parse(cls, raw: str) -> PluginId:
        text = (raw or "").strip()
        if "@" not in text:
            raise ValueError(f"invalid plugin id {raw!r}: expected name@marketplace")
        name, marketplace = text.rsplit("@", 1)
        name = name.strip()
        marketplace = marketplace.strip()
        if not name or not marketplace:
            raise ValueError(f"invalid plugin id {raw!r}")
        return cls(plugin_name=name, marketplace_name=marketplace)
