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
from pulsing.actor import Message, ActorSystem


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
    
    def __init__(self, actor_system: ActorSystem, model_name: str = "pulsing-model"):
        self._actor_system = actor_system
        self.model_name = model_name
        self._request_count = 0
    
    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/health", self.health_check)
        app.router.add_get("/v1/models", self.list_models)
        app.router.add_post("/v1/chat/completions", self.chat_completions)
        app.router.add_post("/v1/completions", self.completions)
        return app
    
    async def index(self, request: web.Request) -> web.Response:
        return web.json_response({
            "service": "Pulsing OpenAI-Compatible API",
            "model": self.model_name,
        })
    
    async def health_check(self, request: web.Request) -> web.Response:
        # 获取当前可用的 worker 数量
        try:
            workers = await self._actor_system.get_named_instances("worker")
            healthy_workers = sum(1 for w in workers if w.get("status") == "Alive")
        except Exception:
            healthy_workers = 0
        
        return web.json_response({
            "status": "healthy" if healthy_workers > 0 else "degraded",
            "model": self.model_name,
            "healthy_workers": healthy_workers,
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
    
    async def chat_completions(self, request: web.Request) -> web.Response:
        self._request_count += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)
        
        req = ChatCompletionRequest.from_dict(data)
        
        # 直接通过 Named Actor 解析获取 worker
        try:
            worker_ref = await self._actor_system.resolve_named("worker")
        except Exception as e:
            return web.json_response({"error": {"message": f"No available workers: {e}"}}, status=503)
        
        prompt = self._build_chat_prompt(req.messages)
        if req.stream:
            return await self._stream_generate(request, request_id, req.model, prompt, worker_ref, req.max_tokens or 512, is_chat=True)
        else:
            return await self._sync_generate(request_id, req.model, prompt, worker_ref, req.max_tokens or 512, is_chat=True)
    
    async def completions(self, request: web.Request) -> web.Response:
        self._request_count += 1
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)
        
        req = CompletionRequest.from_dict(data)
        
        # 直接通过 Named Actor 解析获取 worker
        try:
            worker_ref = await self._actor_system.resolve_named("worker")
        except Exception as e:
            return web.json_response({"error": {"message": f"No available workers: {e}"}}, status=503)
        
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        if req.stream:
            return await self._stream_generate(request, request_id, req.model, prompt, worker_ref, req.max_tokens, is_chat=False)
        else:
            return await self._sync_generate(request_id, req.model, prompt, worker_ref, req.max_tokens, is_chat=False)
    
    def _build_chat_prompt(self, messages: List[Dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)
    
    async def _sync_generate(self, request_id: str, model: str, prompt: str, worker_ref, max_tokens: int, is_chat: bool) -> web.Response:
        created = int(time.time())
        
        try:
            result = await worker_ref.ask_json("GenerateRequest", {"prompt": prompt, "max_new_tokens": max_tokens})
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
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
        }
        if is_chat:
            res_data["choices"][0]["message"] = {"role": "assistant", "content": text}
        else:
            res_data["choices"][0]["text"] = text
        return web.json_response(res_data)
    
    async def _stream_generate(self, request: web.Request, request_id: str, model: str, prompt: str, worker_ref, max_tokens: int, is_chat: bool) -> web.StreamResponse:
        created = int(time.time())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)
        
        obj_type = "chat.completion.chunk" if is_chat else "text_completion"
        
        try:
            req_msg = Message.from_json("GenerateStreamRequest", {"prompt": prompt, "max_new_tokens": max_tokens})
            reader = await worker_ref.ask_stream(req_msg)
            
            async for chunk_bytes in reader:
                try:
                    chunk = json.loads(chunk_bytes)
                    text = chunk.get("text", "")
                    if text:
                        data = {
                            "id": request_id, "object": obj_type, "created": created, "model": model or self.model_name,
                            "choices": [{"index": 0, "finish_reason": None}]
                        }
                        if is_chat:
                            data["choices"][0]["delta"] = {"content": text}
                        else:
                            data["choices"][0]["text"] = text
                        await response.write(f"data: {json.dumps(data)}\n\n".encode())
                    if chunk.get("finish_reason"):
                        break
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            await response.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
        
        # 结束标记
        final = {"id": request_id, "object": obj_type, "created": created, "model": model or self.model_name, "choices": [{"index": 0, "finish_reason": "stop"}]}
        if is_chat:
            final["choices"][0]["delta"] = {}
        else:
            final["choices"][0]["text"] = ""
        await response.write(f"data: {json.dumps(final)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response
