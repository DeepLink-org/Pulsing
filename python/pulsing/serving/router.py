"""OpenAI-compatible HTTP API router."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass

from aiohttp import web

from pulsing.core import ActorId, ActorSystem, get_system, remote
from pulsing.serving.scheduler import Scheduler  # noqa: F401 (used in type annotation)


@dataclass
class ChatCompletionRequest:
    model: str
    messages: list[dict[str, str]]
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    max_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ChatCompletionRequest":
        return cls(
            model=data.get("model", ""),
            messages=data.get("messages", []),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p", 1.0),
            stream=data.get("stream", False),
            max_tokens=data.get("max_tokens"),
        )


@dataclass
class CompletionRequest:
    model: str
    prompt: str | list[str]
    max_tokens: int = 16
    temperature: float = 1.0
    stream: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "CompletionRequest":
        return cls(
            model=data.get("model", ""),
            prompt=data.get("prompt", ""),
            max_tokens=data.get("max_tokens", 16),
            temperature=data.get("temperature", 1.0),
            stream=data.get("stream", False),
        )


class _OpenAIHandler:
    """OpenAI-compatible HTTP request handler."""

    def __init__(
        self, actor_system: ActorSystem, model_name: str, scheduler: Scheduler
    ):
        self._actor_system = actor_system
        self.model_name = model_name
        self._request_count = 0
        self._scheduler = scheduler

    async def index(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "service": "Pulsing OpenAI-Compatible API",
                "model": self.model_name,
            }
        )

    async def health_check(self, request: web.Request) -> web.Response:
        total_workers = await self._scheduler.get_worker_count()
        healthy_workers = await self._scheduler.get_healthy_worker_count()

        return web.json_response(
            {
                "status": "healthy" if healthy_workers > 0 else "degraded",
                "model": self.model_name,
                "total_workers": total_workers,
                "healthy_workers": healthy_workers,
                "request_count": self._request_count,
            }
        )

    async def list_models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "pulsing",
                    }
                ],
            }
        )

    async def chat_completions(self, request: web.Request) -> web.Response:
        self._request_count += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

        req = ChatCompletionRequest.from_dict(data)
        worker_ref = await self._scheduler.select_worker()
        if not worker_ref:
            return web.json_response(
                {"error": {"message": "No available workers"}}, status=503
            )

        prompt = self._build_chat_prompt(req.messages)
        if req.stream:
            return await self._stream_generate(
                request,
                request_id,
                req.model,
                prompt,
                worker_ref,
                req.max_tokens or 512,
                is_chat=True,
            )
        else:
            return await self._sync_generate(
                request_id,
                req.model,
                prompt,
                worker_ref,
                req.max_tokens or 512,
                is_chat=True,
            )

    async def completions(self, request: web.Request) -> web.Response:
        self._request_count += 1
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

        req = CompletionRequest.from_dict(data)
        worker_ref = await self._scheduler.select_worker()
        if not worker_ref:
            return web.json_response(
                {"error": {"message": "No available workers"}}, status=503
            )

        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        if req.stream:
            return await self._stream_generate(
                request,
                request_id,
                req.model,
                prompt,
                worker_ref,
                req.max_tokens,
                is_chat=False,
            )
        else:
            return await self._sync_generate(
                request_id, req.model, prompt, worker_ref, req.max_tokens, is_chat=False
            )

    def _build_chat_prompt(self, messages: list[dict]) -> str:
        parts = [
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in messages
        ]
        return "\n".join(parts + ["Assistant:"])

    async def _sync_generate(
        self,
        request_id: str,
        model: str,
        prompt: str,
        worker_ref,
        max_tokens: int,
        is_chat: bool,
    ) -> web.Response:
        created = int(time.time())

        try:
            worker = worker_ref.as_any()
            result = await worker.generate(prompt=prompt, max_new_tokens=max_tokens)
            if isinstance(result, dict) and "error" in result:
                text = f"[Error: {result['error']}]"
                prompt_tokens = completion_tokens = 0
            else:
                text = result.get("text", "")
                prompt_tokens = result.get("prompt_tokens", 0)
                completion_tokens = result.get("completion_tokens", 0)
        except Exception as e:
            text = f"[Error: {e}]"
            prompt_tokens = completion_tokens = 0

        res_data = {
            "id": request_id,
            "object": "chat.completion" if is_chat else "text_completion",
            "created": created,
            "model": model or self.model_name,
            "choices": [{"index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if is_chat:
            res_data["choices"][0]["message"] = {"role": "assistant", "content": text}
        else:
            res_data["choices"][0]["text"] = text
        return web.json_response(res_data)

    def _build_chunk(
        self, request_id, obj_type, created, model, text, finish_reason, is_chat
    ) -> dict:
        choice: dict = {"index": 0, "finish_reason": finish_reason}
        if is_chat:
            choice["delta"] = {"content": text} if text else {}
        else:
            choice["text"] = text or ""
        return {
            "id": request_id,
            "object": obj_type,
            "created": created,
            "model": model or self.model_name,
            "choices": [choice],
        }

    async def _stream_generate(
        self,
        request: web.Request,
        request_id: str,
        model: str,
        prompt: str,
        worker_ref,
        max_tokens: int,
        is_chat: bool,
    ) -> web.StreamResponse:
        created = int(time.time())
        stream_response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await stream_response.prepare(request)

        obj_type = "chat.completion.chunk" if is_chat else "text_completion"

        async def _send(data: dict):
            await stream_response.write(f"data: {json.dumps(data)}\n\n".encode())

        try:
            worker = worker_ref.as_any()
            stream = worker.generate_stream(prompt=prompt, max_new_tokens=max_tokens)

            async for chunk in stream:
                if isinstance(chunk, dict) and chunk.get("error"):
                    await _send({"error": chunk["error"]})
                    await stream_response.write(b"data: [DONE]\n\n")
                    return stream_response
                try:
                    finish_reason = chunk.get("finish_reason")
                    text = chunk.get("text", "")

                    if finish_reason:
                        if text:
                            await _send(
                                self._build_chunk(
                                    request_id,
                                    obj_type,
                                    created,
                                    model,
                                    text,
                                    finish_reason,
                                    is_chat,
                                )
                            )
                        break

                    if text:
                        await _send(
                            self._build_chunk(
                                request_id,
                                obj_type,
                                created,
                                model,
                                text,
                                None,
                                is_chat,
                            )
                        )
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            await _send({"error": str(e)})

        await _send(
            self._build_chunk(
                request_id,
                obj_type,
                created,
                model,
                "",
                "stop",
                is_chat,
            )
        )
        await stream_response.write(b"data: [DONE]\n\n")
        return stream_response


def _create_app(system: ActorSystem, model_name: str, scheduler) -> web.Application:
    """Build the aiohttp Application with OpenAI-compatible routes."""
    handler = _OpenAIHandler(system, model_name, scheduler)
    app = web.Application()
    app.router.add_get("/", handler.index)
    app.router.add_get("/health", handler.health_check)
    app.router.add_get("/v1/models", handler.list_models)
    app.router.add_post("/v1/chat/completions", handler.chat_completions)
    app.router.add_post("/v1/completions", handler.completions)
    app["scheduler"] = scheduler
    return app


def _build_scheduler(system: ActorSystem, worker_name: str, scheduler_type: str):
    """Create a Scheduler instance from scheduler_type string."""
    from .load_stream import StreamLoadScheduler
    from .scheduler import get_scheduler

    if scheduler_type == "stream_load":
        return StreamLoadScheduler(system, worker_name)
    return get_scheduler(scheduler_type, system, worker_name)


@remote
class Router:
    """OpenAI-compatible HTTP API router Actor.

    Starts an HTTP server on ``on_start`` and stops it on ``on_stop``.
    Exposes ``health_check`` and ``get_config`` as remote-callable methods.

    CLI usage::

        pulsing actor pulsing.serving.Router --addr 0.0.0.0:8000 -- \\
            --http_port 8080 --model_name my-model

    Args:
        http_host: HTTP listen address (default: "0.0.0.0")
        http_port: HTTP listen port (default: 8080)
        model_name: Model name returned in API responses (default: "pulsing-model")
        worker_name: Worker actor name to route to (default: "worker")
        scheduler_type: Scheduling policy — stream_load (default) / random /
            round_robin / power_of_two / cache_aware
    """

    def __init__(
        self,
        http_host: str = "0.0.0.0",
        http_port: int = 8080,
        model_name: str = "pulsing-model",
        worker_name: str = "worker",
        scheduler_type: str = "stream_load",
    ):
        self.http_host = http_host
        self.http_port = http_port
        self.model_name = model_name
        self.worker_name = worker_name
        self.scheduler_type = scheduler_type

        self._runner: web.AppRunner | None = None
        self._scheduler: Scheduler | None = None

    async def on_start(self, actor_id: ActorId) -> None:
        system = get_system()
        self._scheduler = _build_scheduler(
            system, self.worker_name, self.scheduler_type
        )
        await self._scheduler.start()

        app = _create_app(system, self.model_name, self._scheduler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.http_host, self.http_port).start()

        print(
            f"[Router] HTTP server started at http://{self.http_host}:{self.http_port}"
        )
        print(f"[Router] Actor started: {actor_id}")

    def on_stop(self) -> None:
        if self._runner:
            asyncio.create_task(self._shutdown())

    async def _shutdown(self) -> None:
        if self._scheduler:
            await self._scheduler.stop()
            self._scheduler = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            print("[Router] HTTP server stopped")

    def metadata(self) -> dict[str, str]:
        return {
            "type": "router",
            "http_host": self.http_host,
            "http_port": str(self.http_port),
            "model_name": self.model_name,
            "worker_name": self.worker_name,
            "scheduler_type": self.scheduler_type,
        }

    def health_check(self) -> dict:
        return {
            "status": "healthy",
            "http_port": self.http_port,
            "model_name": self.model_name,
        }

    def get_config(self) -> dict:
        return {
            "http_host": self.http_host,
            "http_port": self.http_port,
            "model_name": self.model_name,
            "worker_name": self.worker_name,
            "scheduler_type": self.scheduler_type,
        }


async def start_router(
    system: ActorSystem,
    http_host: str = "0.0.0.0",
    http_port: int = 8080,
    model_name: str = "pulsing-model",
    worker_name: str = "worker",
    scheduler_type: str = "stream_load",
    scheduler=None,
) -> web.AppRunner:
    """Start an OpenAI-compatible HTTP server without creating a Router actor.

    Useful for embedding the router in an existing asyncio application.
    For CLI / Actor-lifecycle usage prefer the ``Router`` actor class instead.

    Returns:
        ``web.AppRunner`` — pass to ``stop_router()`` for cleanup.
    """
    if scheduler is None:
        scheduler = _build_scheduler(system, worker_name, scheduler_type)

    await scheduler.start()

    app = _create_app(system, model_name, scheduler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, http_host, http_port).start()

    print(f"[Router] HTTP server started at http://{http_host}:{http_port}")
    return runner


async def stop_router(runner: web.AppRunner) -> None:
    """Stop a router started via ``start_router()``."""
    if not runner:
        return
    scheduler = runner.app.get("scheduler")
    if scheduler:
        await scheduler.stop()
    await runner.cleanup()
    print("[Router] HTTP server stopped")
