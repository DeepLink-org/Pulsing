"""Per-harness mailbox ownership; Pulsing discovery routes between companions."""

from __future__ import annotations

import asyncio

import pulsing as pul

from .actors import Mailbox, MailboxDirectory, mailbox_name, validate_identity
from .client import MailboxClient

DIRECTORY_PREFIX = "mail/instances/"


class CompanionClient(MailboxClient):
    async def _mailbox(self, identity: str):
        validate_identity(identity)
        # The destination's owner creates and holds its in-memory actor. Do not require
        # remote recipients to appear in this companion's local role directory.
        return await Mailbox.resolve(mailbox_name(identity), timeout=15)

    async def roles(self) -> list[dict]:
        named = await pul.get_system().all_named_actors()
        names = {
            entry["path"]
            for entry in named
            if entry["path"].startswith(DIRECTORY_PREFIX)
        }

        async def read(name):
            directory = await MailboxDirectory.resolve(name, timeout=3)
            return await directory.roles()

        # Discovery is eventual. Errors are surfaced rather than reporting a
        # complete collaborator list while silently dropping a failing owner.
        groups = await asyncio.gather(*(read(name) for name in sorted(names)))
        return [role for group in groups for role in group]


class AgentClients:
    """No caller-supplied sender argument and no shared fallback identity."""

    def __init__(self, directory):
        self.directory = directory

    async def __call__(self, ctx) -> CompanionClient:
        meta = ctx.request_context.meta
        data = meta.model_dump() if meta is not None else {}
        thread_id = data.get("threadId")
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError(
                "MCP caller must supply authoritative _meta.threadId; "
                "upgrade the harness rather than sharing another agent's identity"
            )
        role = await self.directory.ensure_agent(thread_id)
        return CompanionClient(role["identity"], self.directory)
