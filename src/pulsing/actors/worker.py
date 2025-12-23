"""
Transformers Worker Actor - 基于 Transformers 的推理 Worker
"""

import asyncio
import uuid
import json
import sys
from dataclasses import dataclass
from typing import Optional, Dict, List, Union

# # 关键：从底层核心导入，并强制检查
# from dynamo._core import actor as _core_actor
# StreamMessage = _core_actor.StreamMessage
# RawMessage = _core_actor.Message
# Message = _core_actor.UnifiedMessage
# ActorRef = _core_actor.ActorRef
# ActorId = _core_actor.ActorId

# 从高层导入接口
from pulsing.actor import Actor
from pulsing.actor import StreamMessage
from pulsing.actor import Message
from pulsing.actor import ActorRef
from pulsing.actor import ActorId
from pulsing.actor import RawMessage
from .base import BaseServiceActor


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False


class TransformersWorkerHandler(Actor):
    """Transformers Worker 消息处理器"""
    
    def __init__(self, model_name: str, device: str = "cuda", gen_config: Optional[GenerationConfig] = None):
        self.model_name = model_name
        self.device = device
        self.gen_config = gen_config or GenerationConfig()
        self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        
        self._actor_id: Optional[ActorId] = None
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._request_count = 0
    
    def on_start(self, actor_id: ActorId) -> None:
        self._actor_id = actor_id
        print(f"[Worker] ID: {self.worker_id}")
    
    def on_stop(self) -> None:
        self._model = None
        self._tokenizer = None
    
    def metadata(self) -> Dict[str, str]:
        return {"type": "worker", "model": self.model_name, "device": self.device}
    
    async def load_model(self):
        """加载模型"""
        if self._is_loaded:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("需要安装 transformers 和 torch") from e
        
        print(f"[Worker] Loading {self.model_name}...")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        torch_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        model_kwargs = {"device_map": "auto"} if self.device == "cuda" else {}
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype, **model_kwargs
        )
        
        if self.device != "cuda":
            self._model.to(self.device)
        
        self._model.eval()
        self._is_loaded = True
        print(f"[Worker] Model loaded on {self.device}")
    
    async def receive(self, msg: Message) -> Union[Message, StreamMessage]:
        """处理请求"""
        msg_type = msg.msg_type
        print(f"[Worker] Received: {msg_type}")
        
        try:
            if msg_type == "GenerateRequest":
                return await self._handle_generate(msg)
            elif msg_type == "GenerateStreamRequest":
                # 显式返回 StreamMessage
                return await self._handle_generate_stream(msg)
            elif msg_type == "HealthCheck":
                return Message.from_json("Ok", {
                    "status": "healthy",
                    "worker_id": self.worker_id,
                    "is_loaded": self._is_loaded,
                })
            else:
                return Message.from_json("Error", {"error": f"Unknown: {msg_type}"})
        except Exception as e:
            import traceback
            print(f"[Worker] Error handling {msg_type}: {e}")
            traceback.print_exc()
            return Message.from_json("Error", {"error": str(e)})
    
    async def _handle_generate(self, msg: Message) -> Message:
        """同步生成"""
        if not self._is_loaded:
            await self.load_model()
        
        data = msg.to_json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", self.gen_config.max_new_tokens)
        
        self._request_count += 1
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        # 同步生成
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=self.gen_config.do_sample,
        )
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return Message.from_json("GenerateResponse", {
            "text": text,
            "worker_id": self.worker_id,
            "prompt_tokens": input_len,
            "completion_tokens": len(new_tokens),
        })
    
    async def _handle_generate_stream(self, msg: Message) -> StreamMessage:
        """流式生成"""
        from threading import Thread
        
        if not self._is_loaded:
            await self.load_model()
        
        data = msg.to_json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", self.gen_config.max_new_tokens)
        
        self._request_count += 1
        
        # 显式使用核心库创建
        stream_msg, writer = StreamMessage.create("GenerateStream")
        
        async def produce():
            try:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
                input_len = inputs["input_ids"].shape[1]
                
                from transformers import TextIteratorStreamer
                streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self._tokenizer.eos_token_id,
                    "do_sample": self.gen_config.do_sample,
                    "streamer": streamer,
                }
                
                thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
                thread.start()
                
                token_count = 0
                for text in streamer:
                    if text:
                        token_count += 1
                        await writer.write_json({
                            "text": text,
                            "worker_id": self.worker_id,
                        })
                thread.join()
                
                await writer.write_json({
                    "text": "",
                    "finish_reason": "stop",
                    "prompt_tokens": input_len,
                    "completion_tokens": token_count,
                })
            except Exception as e:
                print(f"[Worker] produce error: {e}")
                try:
                    await writer.error(str(e))
                except:
                    pass
            finally:
                writer.close()
        
        asyncio.create_task(produce())
        return stream_msg


class TransformersWorkerActor(BaseServiceActor):
    """Transformers Worker 服务"""
    
    def __init__(
        self,
        model: str,
        namespace: str = "dynamo",
        addr: Optional[str] = None,
        seeds: Optional[List[str]] = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
        preload_model: bool = False,
    ):
        super().__init__(namespace=namespace, addr=addr, seeds=seeds, public=True)
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.preload_model = preload_model
        self._handler: Optional[TransformersWorkerHandler] = None
    
    @property
    def service_name(self) -> str:
        return "worker"
    
    def _create_actor(self) -> Actor:
        gen_config = GenerationConfig(max_new_tokens=self.max_new_tokens)
        self._handler = TransformersWorkerHandler(
            model_name=self.model,
            device=self.device,
            gen_config=gen_config,
        )
        return self._handler
    
    async def start(self) -> ActorRef:
        actor_ref = await super().start()
        if self.preload_model and self._handler:
            await self._handler.load_model()
        print(f"[Worker] Ready (model={self.model})")
        return actor_ref
    
    @property
    def worker_id(self) -> Optional[str]:
        return self._handler.worker_id if self._handler else None
