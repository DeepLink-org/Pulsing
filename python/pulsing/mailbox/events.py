"""Actor-owned replay observations, separate from volatile mailbox state."""

from __future__ import annotations

import datetime
import os
import time
import uuid
from pathlib import Path

from .audit import append_event


class ActorEvents:
    def __init__(self, path: str | None, actor: str, instance: str | None = None):
        self.path = Path(path).expanduser().absolute() if path else None
        self.actor = actor
        self.instance = instance
        self.stream_id = uuid.uuid4().hex
        self.sequence = 0

    def emit(self, event: str, **fields) -> None:
        if self.path is None:
            return
        sequence = self.sequence + 1
        append_event(
            self.path,
            {
                "version": 1,
                "event_id": uuid.uuid4().hex,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "time_ns": time.time_ns(),
                "pid": os.getpid(),
                "actor": self.actor,
                "instance": self.instance,
                "stream_id": self.stream_id,
                "sequence": sequence,
                "event": event,
                **fields,
            },
        )
        self.sequence = sequence
