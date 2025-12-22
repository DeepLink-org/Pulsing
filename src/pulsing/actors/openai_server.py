"""
OpenAI Compatible HTTP Server

提供兼容 OpenAI API 的 HTTP 服务：
- /v1/chat/completions - 聊天补全
- /v1/completions - 文本补全
- /v1/models - 模型列表
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, List, Union

from aiohttp import web


@dataclass
class ChatCompletionRequest:
    """聊天补全请求"""
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    max_tokens: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ChatCompletionRequest":
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
    """文本补全请求"""
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = 16
    temperature: float = 1.0
    stream: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CompletionRequest":
        return cls(
            model=data.get("model", ""),
            prompt=data.get("prompt", ""),
            max_tokens=data.get("max_tokens", 16),
            temperature=data.get("temperature", 1.0),
            stream=data.get("stream", False),
        )


class OpenAIServer:
    """OpenAI 兼容的 HTTP 服务器"""
    
    def __init__(self, router_scheduler, model_name: str = "pulsing-model"):
        self.scheduler = router_scheduler
        self.model_name = model_name
        self._request_count = 0
    
    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/health", self.health_check)
        app.router.add_get("/v1/models", self.list_models)
        app.router.add_get("/v1/workers", self.list_workers)
        app.router.add_post("/v1/chat/completions", self.chat_completions)
        app.router.add_post("/v1/completions", self.completions)
        return app
    
    async def index(self, request: web.Request) -> web.Response:
        return web.json_response({
            "service": "Pulsing OpenAI-Compatible API",
            "model": self.model_name,
        })
    
    async def health_check(self, request: web.Request) -> web.Response:
        workers = await self.scheduler.get_workers()
        healthy = await self.scheduler.get_healthy_worker_count()
        return web.json_response({
            "status": "healthy" if healthy > 0 else "degraded",
            "model": self.model_name,
            "total_workers": len(workers),
            "healthy_workers": healthy,
            "request_count": self._request_count,
        })
    
    async def list_models(self, request: web.Request) -> web.Response:
        return web.json_response({
            "object": "list",
            "data": [{
                "id": self.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "pulsing",
            }]
        })
    
    async def list_workers(self, request: web.Request) -> web.Response:
        workers = await self.scheduler.get_workers()
        return web.json_response({
            "workers": [
                {
                    "worker_id": w.worker_id[:8] + "...",
                    "endpoint": w.endpoint,
                    "is_healthy": w.is_healthy,
                    "request_count": w.request_count,
                }
                for w in workers
            ],
            "total": len(workers),
        })
    
    async def chat_completions(self, request: web.Request) -> web.Response:
        self._request_count += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "Invalid JSON"}}, status=400
            )
        
        req = ChatCompletionRequest.from_dict(data)
        worker = await self.scheduler.get_next_worker()
        if not worker:
            return web.json_response(
                {"error": {"message": "No available workers"}}, status=503
            )
        
        prompt = self._build_chat_prompt(req.messages)
        
        if req.stream:
            return await self._stream_response(request, request_id, req.model, prompt, worker)
        else:
            return self._sync_response(request_id, req.model, prompt, worker, is_chat=True)
    
    async def completions(self, request: web.Request) -> web.Response:
        self._request_count += 1
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "Invalid JSON"}}, status=400
            )
        
        req = CompletionRequest.from_dict(data)
        worker = await self.scheduler.get_next_worker()
        if not worker:
            return web.json_response(
                {"error": {"message": "No available workers"}}, status=503
            )
        
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        
        if req.stream:
            return await self._stream_completion(request, request_id, req.model, prompt, worker)
        else:
            return self._sync_response(request_id, req.model, prompt, worker, is_chat=False)
    
    def _build_chat_prompt(self, messages: List[Dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)
    
    def _sync_response(self, request_id: str, model: str, prompt: str, worker, is_chat: bool) -> web.Response:
        created = int(time.time())
        text = f"[Worker: {worker.worker_id[:8]}...] Echo: {prompt[-100:]}"
        
        if is_chat:
            return web.json_response({
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model or self.model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        else:
            return web.json_response({
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model or self.model_name,
                "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
    
    async def _stream_response(self, request: web.Request, request_id: str, model: str, prompt: str, worker) -> web.StreamResponse:
        created = int(time.time())
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)
        
        text = f"[Worker: {worker.worker_id[:8]}...] Echo: {prompt[-50:]}"
        for char in text:
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model or self.model_name,
                "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(data)}\n\n".encode())
            await asyncio.sleep(0.01)
        
        # 结束
        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model or self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        await response.write(f"data: {json.dumps(final)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response
    
    async def _stream_completion(self, request: web.Request, request_id: str, model: str, prompt: str, worker) -> web.StreamResponse:
        created = int(time.time())
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)
        
        text = f"[Worker: {worker.worker_id[:8]}...] {prompt[-50:]}"
        for char in text:
            data = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model or self.model_name,
                "choices": [{"text": char, "index": 0, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(data)}\n\n".encode())
            await asyncio.sleep(0.01)
        
        final = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model or self.model_name,
            "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
        }
        await response.write(f"data: {json.dumps(final)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response
