"""AutoGenAgentWrapper - Wraps AutoGen Agent as Pulsing Actor"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pulsing.actor import Actor, ActorId

if TYPE_CHECKING:
    from .runtime import PulsingRuntime

logger = logging.getLogger("pulsing.autogen")


class AutoGenAgentWrapper(Actor):
    """Wraps AutoGen Agent as Pulsing Actor"""

    def __init__(self, agent: Any, runtime: "PulsingRuntime"):
        """
        Args:
            agent: AutoGen Agent instance
            runtime: PulsingRuntime instance
        """
        self._agent = agent
        self._runtime = runtime
        self._actor_id: ActorId | None = None

    def on_start(self, actor_id: ActorId) -> None:
        """Actor startup callback"""
        self._actor_id = actor_id
        logger.debug(f"AutoGenAgentWrapper started: {actor_id}")

    def on_stop(self) -> None:
        """Actor stop callback"""
        # Call Agent's close method
        if hasattr(self._agent, "close"):
            try:
                close_result = self._agent.close()
                if asyncio.iscoroutine(close_result):
                    # Cannot await in sync context, ignore
                    pass
            except Exception as e:
                logger.warning(f"Error closing agent: {e}")

    def metadata(self) -> dict[str, str]:
        """Returns metadata"""
        meta = {"wrapper": "AutoGenAgentWrapper"}
        if hasattr(self._agent, "metadata"):
            agent_meta = self._agent.metadata
            if callable(agent_meta):
                agent_meta = agent_meta()
            if hasattr(agent_meta, "type"):
                meta["agent_type"] = agent_meta.type
            if hasattr(agent_meta, "key"):
                meta["agent_key"] = agent_meta.key
            if hasattr(agent_meta, "description"):
                meta["description"] = agent_meta.description
        return meta

    async def receive(self, msg: Any) -> Any:
        """Handle received message"""
        # Check if it's an AutoGen format message
        if isinstance(msg, dict) and msg.get("__autogen_msg__"):
            return await self._handle_autogen_message(msg)

        # Directly pass through to Agent (compatible with native Pulsing messages)
        if hasattr(self._agent, "on_message"):
            ctx = self._create_default_context()
            result = await self._agent.on_message(msg, ctx=ctx)
            return result

        return {"__error__": f"Agent does not support message type: {type(msg)}"}

    async def _handle_autogen_message(self, envelope: dict) -> dict:
        """Handle AutoGen format message"""
        try:
            # Parse message
            payload = envelope.get("payload")
            sender_info = envelope.get("sender")
            topic_info = envelope.get("topic_id")
            is_rpc = envelope.get("is_rpc", True)
            message_id = envelope.get("message_id", "")

            # Construct MessageContext
            ctx = self._create_message_context(
                sender_info=sender_info,
                topic_info=topic_info,
                is_rpc=is_rpc,
                message_id=message_id,
            )

            # Call Agent.on_message
            if hasattr(self._agent, "on_message"):
                result = await self._agent.on_message(payload, ctx=ctx)
                return {
                    "__autogen_response__": True,
                    "result": result,
                }
            else:
                return {
                    "__autogen_response__": True,
                    "__error__": "Agent does not have on_message method",
                }

        except Exception as e:
            logger.exception(f"Error handling AutoGen message: {e}")
            return {
                "__autogen_response__": True,
                "__error__": str(e),
            }

    def _create_message_context(
        self,
        sender_info: dict | None,
        topic_info: dict | None,
        is_rpc: bool,
        message_id: str,
    ) -> Any:
        """Create AutoGen MessageContext"""
        try:
            from autogen_core import CancellationToken, MessageContext, TopicId, AgentId

            # Construct sender
            sender = None
            if sender_info and sender_info.get("type"):
                sender = AgentId(sender_info["type"], sender_info.get("key", "default"))

            # Construct topic_id
            topic_id = None
            if topic_info:
                topic_id = TopicId(
                    topic_info["type"], topic_info.get("source", "default")
                )

            return MessageContext(
                sender=sender,
                topic_id=topic_id,
                is_rpc=is_rpc,
                cancellation_token=CancellationToken(),
                message_id=message_id,
            )
        except ImportError:
            # If autogen_core is not available, return a simple dict
            return {
                "sender": sender_info,
                "topic_id": topic_info,
                "is_rpc": is_rpc,
                "message_id": message_id,
            }

    def _create_default_context(self) -> Any:
        """Create default MessageContext"""
        return self._create_message_context(
            sender_info=None,
            topic_info=None,
            is_rpc=True,
            message_id="",
        )
