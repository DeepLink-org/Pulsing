"""OpenAI-compatible HTTP API router."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass

from aiohttp import web

from pulsing.actor import Actor, ActorId, ActorSystem, Message, get_system


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

    def __init__(self, actor_system: ActorSystem, model_name: str, scheduler):
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
        if hasattr(self._scheduler, "get_worker_count"):
            count = self._scheduler.get_worker_count()
            if hasattr(count, "__await__"):
                total_workers = await count
            else:
                total_workers = count
        else:
            total_workers = 0

        if hasattr(self._scheduler, "get_healthy_worker_count"):
            healthy_workers = await self._scheduler.get_healthy_worker_count()
        elif hasattr(self._scheduler, "get_all_loads"):
            healthy_workers = len(self._scheduler.get_all_loads())
        else:
            healthy_workers = total_workers

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
            msg = Message.from_json(
                "GenerateRequest", {"prompt": prompt, "max_new_tokens": max_tokens}
            )
            result = (await worker_ref.ask(msg)).to_json()
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

        try:
            req_msg = Message.from_json(
                "GenerateStreamRequest",
                {"prompt": prompt, "max_new_tokens": max_tokens},
            )
            stream_message = await worker_ref.ask(req_msg)

            # Check if returned message is a stream message
            if not stream_message.is_stream:
                # If not stream message, might be error message
                error_data = stream_message.to_json()
                error_msg = error_data.get("error", "Unknown error")
                await stream_response.write(
                    f"data: {json.dumps({'error': error_msg})}\n\n".encode()
                )
                await stream_response.write(b"data: [DONE]\n\n")
                return stream_response

            reader = stream_message.stream_reader()

            async for chunk in reader:
                try:
                    finish_reason = chunk.get("finish_reason")
                    text = chunk.get("text", "")

                    # Check if finished
                    if finish_reason:
                        # Send final chunk (if has text)
                        if text:
                            data = {
                                "id": request_id,
                                "object": obj_type,
                                "created": created,
                                "model": model or self.model_name,
                                "choices": [
                                    {"index": 0, "finish_reason": finish_reason}
                                ],
                            }
                            if is_chat:
                                data["choices"][0]["delta"] = {"content": text}
                            else:
                                data["choices"][0]["text"] = text
                            await stream_response.write(
                                f"data: {json.dumps(data)}\n\n".encode()
                            )
                        break

                    # Only send non-empty text
                    if text:
                        data = {
                            "id": request_id,
                            "object": obj_type,
                            "created": created,
                            "model": model or self.model_name,
                            "choices": [{"index": 0, "finish_reason": None}],
                        }
                        if is_chat:
                            data["choices"][0]["delta"] = {"content": text}
                        else:
                            data["choices"][0]["text"] = text
                        await stream_response.write(
                            f"data: {json.dumps(data)}\n\n".encode()
                        )
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            await stream_response.write(
                f"data: {json.dumps({'error': str(e)})}\n\n".encode()
            )

        final = {
            "id": request_id,
            "object": obj_type,
            "created": created,
            "model": model or self.model_name,
            "choices": [{"index": 0, "finish_reason": "stop"}],
        }
        if is_chat:
            final["choices"][0]["delta"] = {}
        else:
            final["choices"][0]["text"] = ""
        await stream_response.write(f"data: {json.dumps(final)}\n\n".encode())
        await stream_response.write(b"data: [DONE]\n\n")
        return stream_response


async def start_router(
    system: ActorSystem,
    http_host: str = "0.0.0.0",
    http_port: int = 8080,
    model_name: str = "pulsing-model",
    worker_name: str = "worker",
    scheduler_type: str = "stream_load",
    scheduler=None,
    scheduler_class=None,  # Backward compatibility
) -> web.AppRunner:
    """Start Router HTTP server, returns AppRunner

    Args:
        system: ActorSystem instance
        http_host: HTTP listen address
        http_port: HTTP listen port
        model_name: Model name
        worker_name: Worker actor name
        scheduler_type: Scheduler type, supports:
            - "stream_load": Stream load-aware (default, recommended)
            - "random": Random
            - "round_robin": Round robin
            - "power_of_two": Power-of-Two Choices
            - "cache_aware": Cache-aware
        scheduler: Custom scheduler instance (takes priority)
        scheduler_class: [Deprecated] Use scheduler parameter instead

    Returns:
        AppRunner instance
    """
    from .load_stream import StreamLoadScheduler
    from .scheduler import (
        RUST_POLICIES_AVAILABLE,
        RandomScheduler,
        RoundRobinScheduler,
        RustCacheAwareScheduler,
        RustPowerOfTwoScheduler,
    )

    # Backward compatibility: scheduler_class -> scheduler
    if scheduler_class is not None and scheduler is None:
        scheduler = scheduler_class(system, worker_name)

    # Create scheduler
    if scheduler is None:
        scheduler_map = {
            "stream_load": StreamLoadScheduler,
            "random": RandomScheduler,
            "round_robin": RoundRobinScheduler,
        }

        # Rust high-performance schedulers (requires compilation)
        if RUST_POLICIES_AVAILABLE:
            scheduler_map["power_of_two"] = RustPowerOfTwoScheduler
            scheduler_map["cache_aware"] = RustCacheAwareScheduler

        scheduler_class = scheduler_map.get(scheduler_type, StreamLoadScheduler)
        scheduler = scheduler_class(system, worker_name)

    # Start scheduler (if has start method)
    if hasattr(scheduler, "start"):
        await scheduler.start()

    handler = _OpenAIHandler(system, model_name, scheduler)

    app = web.Application()
    app.router.add_get("/", handler.index)
    app.router.add_get("/health", handler.health_check)
    app.router.add_get("/v1/models", handler.list_models)
    app.router.add_post("/v1/chat/completions", handler.chat_completions)
    app.router.add_post("/v1/completions", handler.completions)

    # Save scheduler reference for cleanup
    app["scheduler"] = scheduler

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, http_host, http_port)
    await site.start()

    print(f"[Router] HTTP server started at http://{http_host}:{http_port}")
    print(f"[Router] Using scheduler: {scheduler_type}")
    return runner


async def stop_router(runner: web.AppRunner):
    """Stop Router HTTP server"""
    if runner:
        # Stop scheduler (if has stop method)
        app = runner.app
        scheduler = app.get("scheduler")
        if scheduler and hasattr(scheduler, "stop"):
            await scheduler.stop()

        await runner.cleanup()
        print("[Router] HTTP server stopped")


class Router(Actor):
    """Router Actor - OpenAI-compatible HTTP API router as an Actor

    This actor wraps the start_router/stop_router functions to provide
    a CLI-compatible entry point via `pulsing actor pulsing.actors.Router`.

    Args:
        http_host: HTTP listen address (default: "0.0.0.0")
        http_port: HTTP listen port (default: 8080)
        model_name: Model name for API responses (default: "pulsing-model")
        worker_name: Worker actor name to route requests to (default: "worker")
        scheduler_type: Scheduler type, supports:
            - "stream_load": Stream load-aware (default, recommended)
            - "random": Random
            - "round_robin": Round robin
            - "power_of_two": Power-of-Two Choices
            - "cache_aware": Cache-aware

    Example:
        # Start via CLI
        pulsing actor pulsing.actors.Router \\
            --http_host 0.0.0.0 \\
            --http_port 8080 \\
            --model_name my-llm \\
            --worker_name worker

        # Or programmatically
        router = Router(http_port=8080, model_name="my-llm")
        await system.spawn(router, name="router", public=True)
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
        self._actor_id: ActorId | None = None

    async def on_start(self, actor_id: ActorId) -> None:
        """Start the HTTP server when actor starts"""
        self._actor_id = actor_id

        # Get global system (set by CLI via init())
        system = get_system()

        # Start HTTP server
        self._runner = await start_router(
            system=system,
            http_host=self.http_host,
            http_port=self.http_port,
            model_name=self.model_name,
            worker_name=self.worker_name,
            scheduler_type=self.scheduler_type,
        )

        print(f"[Router] Actor started: {actor_id}")

    def on_stop(self) -> None:
        """Stop the HTTP server when actor stops"""
        if self._runner:
            # Schedule cleanup in background (on_stop is sync)
            asyncio.create_task(self._cleanup())

    async def _cleanup(self):
        """Async cleanup helper"""
        if self._runner:
            await stop_router(self._runner)
            self._runner = None

    def metadata(self) -> dict[str, str]:
        """Return router metadata for diagnostics"""
        return {
            "type": "router",
            "http_host": self.http_host,
            "http_port": str(self.http_port),
            "model_name": self.model_name,
            "worker_name": self.worker_name,
            "scheduler_type": self.scheduler_type,
        }

    async def receive(self, msg: Message) -> Message | None:
        """Handle diagnostic messages"""
        if msg.msg_type == "HealthCheck":
            return Message.from_json(
                "Ok",
                {
                    "status": "healthy",
                    "http_port": self.http_port,
                    "model_name": self.model_name,
                },
            )
        elif msg.msg_type == "GetConfig":
            return Message.from_json(
                "Config",
                {
                    "http_host": self.http_host,
                    "http_port": self.http_port,
                    "model_name": self.model_name,
                    "worker_name": self.worker_name,
                    "scheduler_type": self.scheduler_type,
                },
            )
        else:
            return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})
