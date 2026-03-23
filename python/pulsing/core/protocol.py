"""Wire protocol — message serialization helpers for the actor call/response format."""

import asyncio
import logging
import uuid
from typing import Any

from pulsing._core import Message
from pulsing.exceptions import PulsingActorError

logger = logging.getLogger(__name__)


def _consume_task_exception(task: asyncio.Task) -> None:
    """Consume exception from background task to avoid 'Task exception was never retrieved'."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except (RuntimeError, OSError, ConnectionError) as e:
        if "closed" in str(e).lower() or "stream" in str(e).lower():
            logger.debug("Stream closed before response: %s", e)
        else:
            logger.exception("Stream task failed: %s", e)
    except Exception:
        logger.exception("Stream task failed")


def _wrap_call(method: str, args: tuple, kwargs: dict, is_async: bool) -> dict:
    """Build a flat call message.

    Format: {"__call__": method, "__async__": bool, "args": (...), "kwargs": {...}}
    """
    return {"__call__": method, "__async__": is_async, "args": args, "kwargs": kwargs}


def _unwrap_call(msg: dict) -> tuple[str, tuple, dict, bool]:
    """Parse a flat call message. Returns (method, args, kwargs, is_async)."""
    return (
        msg.get("__call__", ""),
        tuple(msg.get("args", ())),
        dict(msg.get("kwargs", {})),
        msg.get("__async__", False),
    )


def _wrap_response(result: Any = None, error: str | None = None) -> dict:
    """Build a flat response message."""
    if error:
        return {"__error__": error}
    return {"__result__": result}


def _unwrap_response(resp: dict) -> tuple[Any, str | None]:
    """Parse a response. Returns (result, error) — one will be None.

    Accepts flat format and Rust actor JSON ({"result": ...} / {"error": ...}).
    """
    if "__error__" in resp:
        return (None, resp["__error__"])
    if "__result__" in resp:
        return (resp["__result__"], None)
    if "error" in resp:
        return (None, resp["error"])
    if "result" in resp:
        return (resp["result"], None)
    return (None, None)


def _check_response(resp, ref) -> Any:
    """Unwrap response, raise PulsingActorError on errors, return result.

    Handles dict responses from Python actors and Message responses (streaming)
    from the Rust runtime transparently.
    """
    if isinstance(resp, Message):
        if resp.is_stream:
            return resp
        try:
            resp = resp.to_json()
        except ValueError:
            # HTTP/2 transport doesn't preserve msg_type in responses,
            # so pickle-encoded Python actor responses arrive with empty msg_type.
            # Fall back to pickle deserialization.
            import pickle

            resp = pickle.loads(resp.payload)
    if isinstance(resp, dict):
        result, error = _unwrap_response(resp)
        if error:
            raise PulsingActorError(error, actor_name=str(ref.actor_id.id))
        return result
    return resp


def _normalize_actor_name(cls_name: str, name: str | None) -> str:
    """Build actor path from optional name and class name."""
    if name and "/" in name:
        return name
    if name:
        return f"actors/{name}"
    return f"actors/{cls_name}_{uuid.uuid4().hex[:8]}"
