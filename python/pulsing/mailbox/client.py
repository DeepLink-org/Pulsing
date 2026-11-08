"""Identity-bound mailbox client shared by Python users and the MCP adapter."""

from __future__ import annotations

from .actors import (
    DIRECTORY_NAME,
    Mailbox,
    MailboxDirectory,
    mailbox_name,
    validate_identity,
)


class MailboxClient:
    """No sender parameter: outgoing mail always uses this client's identity.

    The binding prevents accidental impersonation through MCP. It is not cluster
    authentication; only trusted local processes may choose an identity.
    """

    def __init__(self, identity: str, directory):
        self.identity = validate_identity(identity)
        self._directory = directory

    @classmethod
    async def connect(
        cls, identity: str, timeout: float = 15, *, register_kind: str | None = None
    ) -> MailboxClient:
        directory = await MailboxDirectory.resolve(DIRECTORY_NAME, timeout=timeout)
        client = cls(identity, directory)
        if register_kind is not None:
            # Existing role metadata (including parent) remains canonical.
            await directory.ensure_role(identity, register_kind)
        await client._mailbox(identity)  # Require host-side registration.
        return client

    async def _mailbox(self, identity: str):
        validate_identity(identity)
        roles = await self._directory.roles()
        if identity not in {role["identity"] for role in roles}:
            raise ValueError(f"unknown mailbox: {identity}")
        return await Mailbox.resolve(mailbox_name(identity), timeout=15)

    async def roles(self) -> list[dict]:
        return await self._directory.roles()

    async def create_role(self, identity: str, kind: str = "subagent") -> dict:
        return await self._directory.register(identity, kind, parent=self.identity)

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        idempotency_key: str,
        thread_id: str = "",
        in_reply_to: str | None = None,
    ) -> dict:
        mailbox = await self._mailbox(to)
        return await mailbox.deliver(
            self.identity, subject, body, idempotency_key, thread_id, in_reply_to
        )

    async def inbox(
        self, state: str | None = None, limit: int = 20, offset: int = 0
    ) -> dict:
        mailbox = await self._mailbox(self.identity)
        return await mailbox.inbox(state, limit, offset)

    async def read(self, message_id: str) -> dict:
        mailbox = await self._mailbox(self.identity)
        return await mailbox.read_message(message_id)

    async def claim(self, message_id: str, lease_seconds: int = 300) -> dict:
        mailbox = await self._mailbox(self.identity)
        return await mailbox.claim(message_id, lease_seconds)

    async def finish(
        self,
        message_id: str,
        claim_token: str,
        outcome: str = "completed",
        result: str = "",
    ) -> dict:
        mailbox = await self._mailbox(self.identity)
        return await mailbox.finish(message_id, claim_token, outcome, result)

    async def status(self, to: str, message_id: str) -> dict:
        mailbox = await self._mailbox(to)
        return await mailbox.receipt(message_id, self.identity)

    async def reply(self, message_id: str, body: str, idempotency_key: str) -> dict:
        original = await self.read(message_id)
        subject = f"Re: {original['subject']}".encode("utf-8")[:256].decode(
            "utf-8", errors="ignore"
        )
        return await self.send(
            original["sender"],
            subject,
            body,
            idempotency_key,
            original["thread_id"],
            original["id"],
        )
