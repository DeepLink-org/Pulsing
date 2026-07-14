# SPDX-License-Identifier: Apache-2.0
"""In-memory agent session id (no disk persistence)."""

from __future__ import annotations

import uuid


class AgentSession:
    def __init__(self, *, model: str, session_id: str | None = None) -> None:
        self.model = model
        self.session_id = session_id or uuid.uuid4().hex[:16]

    def append_message(self, message: dict) -> None:
        _ = message
