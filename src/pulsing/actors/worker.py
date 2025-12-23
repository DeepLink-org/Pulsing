"""Transformers Worker Actor - LLM 推理 Worker"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Union

from pulsing.actor import Actor, StreamMessage, Message, ActorId


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False


class TransformersWorker(Actor):
    """Transformers LLM 推理 Worker，支持同步和流式生成"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        gen_config: Optional[GenerationConfig] = None,
        preload: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.gen_config = gen_config or GenerationConfig()
        self.preload = preload
        self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        
        self._actor_id: Optional[ActorId] = None
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._request_count = 0
    
    async def on_start(self, actor_id: ActorId) -> None:
        self._actor_id = actor_id
        print(f"[Worker] {self.worker_id} - {self.model_name}")
        if self.preload:
            await self.load_model()
    
    def on_stop(self) -> None:
        self._model = None
        self._tokenizer = None
    
    def metadata(self) -> Dict[str, str]:
        return {"type": "worker", "model": self.model_name, "device": self.device}
    
    async def load_model(self):
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
        print(f"[Worker] Model ready on {self.device}")
    
    async def receive(self, msg: Message) -> Union[Message, StreamMessage]:
        try:
            if msg.msg_type == "GenerateRequest":
                return await self._handle_generate(msg)
            elif msg.msg_type == "GenerateStreamRequest":
                return await self._handle_generate_stream(msg)
            elif msg.msg_type == "HealthCheck":
                return Message.from_json("Ok", {
                    "status": "healthy",
                    "worker_id": self.worker_id,
                    "is_loaded": self._is_loaded,
                })
            else:
                return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})
        except Exception as e:
            print(f"[Worker] Error: {e}")
            return Message.from_json("Error", {"error": str(e)})
    
    async def _handle_generate(self, msg: Message) -> Message:
        if not self._is_loaded:
            await self.load_model()
        
        data = msg.to_json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", self.gen_config.max_new_tokens)
        self._request_count += 1
        
        loop = asyncio.get_running_loop()
        
        def _generate_sync():
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
                do_sample=self.gen_config.do_sample,
            )
            
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text, input_len, len(new_tokens)
            
        text, prompt_tokens, completion_tokens = await loop.run_in_executor(None, _generate_sync)
        
        return Message.from_json("GenerateResponse", {
            "text": text,
            "worker_id": self.worker_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })
    
    async def _handle_generate_stream(self, msg: Message) -> StreamMessage:
        from threading import Thread
        
        if not self._is_loaded:
            await self.load_model()
        
        data = msg.to_json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", self.gen_config.max_new_tokens)
        self._request_count += 1
        
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
