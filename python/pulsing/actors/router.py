"""Router - OpenAI 兼容 HTTP API 路由器"""

import json
import time
import uuid
from dataclasses import dataclass

from aiohttp import web

from pulsing.actor import ActorSystem, Message


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
    """OpenAI 兼容 HTTP 请求处理器"""

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
        # 兼容不同类型的 scheduler
        if hasattr(self._scheduler, "get_worker_count"):
            count = self._scheduler.get_worker_count()
            # 如果是协程则 await
            if hasattr(count, "__await__"):
                total_workers = await count
            else:
                total_workers = count
        else:
            total_workers = 0

        if hasattr(self._scheduler, "get_healthy_worker_count"):
            healthy_workers = await self._scheduler.get_healthy_worker_count()
        elif hasattr(self._scheduler, "get_all_loads"):
            # StreamLoadScheduler: 使用 get_all_loads 计算健康数
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

            # 检查返回的是否是流式消息
            if not stream_message.is_stream:
                # 如果不是流式消息，可能是错误消息
                error_data = stream_message.to_json()
                error_msg = error_data.get("error", "Unknown error")
                await stream_response.write(
                    f"data: {json.dumps({'error': error_msg})}\n\n".encode()
                )
                await stream_response.write(b"data: [DONE]\n\n")
                return stream_response

            reader = stream_message.stream_reader()

            async for chunk_bytes in reader:
                try:
                    chunk = json.loads(chunk_bytes)
                    finish_reason = chunk.get("finish_reason")
                    text = chunk.get("text", "")

                    # 检查是否结束
                    if finish_reason:
                        # 发送最后的 chunk（如果有文本）
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

                    # 只发送非空文本
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
    scheduler_class=None,  # 向后兼容
) -> web.AppRunner:
    """启动 Router HTTP 服务器，返回 AppRunner

    Args:
        system: ActorSystem 实例
        http_host: HTTP 监听地址
        http_port: HTTP 监听端口
        model_name: 模型名称
        worker_name: Worker actor 名称
        scheduler_type: 调度器类型，支持:
            - "stream_load": 流式负载感知 (默认，推荐)
            - "random": 随机
            - "round_robin": 轮询
            - "power_of_two": Power-of-Two Choices
            - "cache_aware": 缓存感知
        scheduler: 自定义 scheduler 实例 (优先使用)
        scheduler_class: [已废弃] 使用 scheduler 参数代替

    Returns:
        AppRunner 实例
    """
    from .load_stream import StreamLoadScheduler
    from .scheduler import (
        RUST_POLICIES_AVAILABLE,
        RandomScheduler,
        RoundRobinScheduler,
        RustCacheAwareScheduler,
        RustPowerOfTwoScheduler,
    )

    # 向后兼容: scheduler_class -> scheduler
    if scheduler_class is not None and scheduler is None:
        scheduler = scheduler_class(system, worker_name)

    # 创建 scheduler
    if scheduler is None:
        scheduler_map = {
            "stream_load": StreamLoadScheduler,
            "random": RandomScheduler,
            "round_robin": RoundRobinScheduler,
        }

        # Rust 高性能调度器 (需要编译)
        if RUST_POLICIES_AVAILABLE:
            scheduler_map["power_of_two"] = RustPowerOfTwoScheduler
            scheduler_map["cache_aware"] = RustCacheAwareScheduler

        scheduler_class = scheduler_map.get(scheduler_type, StreamLoadScheduler)
        scheduler = scheduler_class(system, worker_name)

    # 启动 scheduler (如果有 start 方法)
    if hasattr(scheduler, "start"):
        await scheduler.start()

    handler = _OpenAIHandler(system, model_name, scheduler)

    app = web.Application()
    app.router.add_get("/", handler.index)
    app.router.add_get("/health", handler.health_check)
    app.router.add_get("/v1/models", handler.list_models)
    app.router.add_post("/v1/chat/completions", handler.chat_completions)
    app.router.add_post("/v1/completions", handler.completions)

    # 保存 scheduler 引用用于清理
    app["scheduler"] = scheduler

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, http_host, http_port)
    await site.start()

    print(f"[Router] HTTP server started at http://{http_host}:{http_port}")
    print(f"[Router] Using scheduler: {scheduler_type}")
    return runner


async def stop_router(runner: web.AppRunner):
    """停止 Router HTTP 服务器"""
    if runner:
        # 停止 scheduler (如果有 stop 方法)
        app = runner.app
        scheduler = app.get("scheduler")
        if scheduler and hasattr(scheduler, "stop"):
            await scheduler.stop()

        await runner.cleanup()
        print("[Router] HTTP server stopped")
