"""Experimental actor-backed mail for agent coordination, without an LLM runtime."""

from .actors import DIRECTORY_NAME, Mailbox, MailboxDirectory
from .client import MailboxClient

__all__ = ["DIRECTORY_NAME", "Mailbox", "MailboxDirectory", "MailboxClient"]
