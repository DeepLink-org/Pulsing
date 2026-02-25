"""Transformers Worker - LLM Inference Worker (pulsing.remote)"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from threading import Thread

from pulsing.core import ActorId, StreamMessage, remote


@dataclass
class GenerationConfig:
    """Generation configuration"""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False


@remote
class TransformersWorker:
    """Transformers LLM Inference Worker，支持同步/流式生成与负载订阅。

    通过 pulsing.remote 暴露方法：generate、generate_stream、subscribe_load、health_check、get_load。
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        gen_config: GenerationConfig | None = None,
        preload: bool = False,
        capacity: int = 100,
    ):
        self.model_name = model_name
        self.device = device
        self.gen_config = gen_config or GenerationConfig()
        self.preload = preload
        self.capacity = capacity
        self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"

        self._actor_id: ActorId | None = None
        self._node_id: str | None = None
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        self._current_load = 0
        self._request_count = 0
        self._load_subscribers: list = []

    async def on_start(self, actor_id: ActorId) -> None:
        self._actor_id = actor_id
        self._node_id = str(actor_id)
        print(f"[Worker] {self.worker_id} - {self.model_name}")
        if self.preload:
            await self.load_model()

    def on_stop(self) -> None:
        self._model = None
        self._tokenizer = None
        for writer in self._load_subscribers:
            try:
                writer.close()
            except Exception:
                pass
        self._load_subscribers.clear()

    def metadata(self) -> dict[str, str]:
        return {
            "type": "worker",
            "model": self.model_name,
            "device": self.device,
            "worker_id": self.worker_id,
            "load": str(self._current_load),
            "processed": str(self._request_count),
            "capacity": str(self.capacity),
            "is_loaded": str(self._is_loaded).lower(),
        }

    @property
    def current_load(self) -> int:
        return self._current_load

    @property
    def load_ratio(self) -> float:
        return self._current_load / max(1, self.capacity)

    def _get_load_snapshot(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "node_id": self._node_id or self.worker_id,
            "load": self._current_load,
            "capacity": self.capacity,
            "processed": self._request_count,
            "timestamp": time.time(),
        }

    async def _push_load_update(self):
        if not self._load_subscribers:
            return
        snapshot = self._get_load_snapshot()
        dead = []
        for writer in self._load_subscribers:
            try:
                await writer.write(snapshot)
            except Exception:
                dead.append(writer)
        for w in dead:
            self._load_subscribers.remove(w)

    async def load_model(self):
        if self._is_loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("Please install transformers and torch") from e

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

    # -------------------------------------------------------------------------
    # 对外方法（替代原 receive 的消息类型分支）
    # -------------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> dict:
        """同步生成，返回 {text, prompt_tokens, completion_tokens} 或 {error}。"""
        if max_new_tokens is None:
            max_new_tokens = self.gen_config.max_new_tokens
        if not self._is_loaded:
            await self.load_model()

        self._current_load += 1
        self._request_count += 1
        asyncio.create_task(self._push_load_update())

        try:
            loop = asyncio.get_running_loop()

            def _run():
                inputs = self._tokenizer(prompt, return_tensors="pt").to(
                    self._model.device
                )
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

            text, prompt_tokens, completion_tokens = await loop.run_in_executor(
                None, _run
            )
            return {
                "text": text,
                "worker_id": self.worker_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        except Exception as e:
            print(f"[Worker] Error: {e}")
            return {"error": str(e)}
        finally:
            self._current_load -= 1
            asyncio.create_task(self._push_load_update())

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ):
        """流式生成，async generator 逐条 yield {text, worker_id} 或最终 {finish_reason, prompt_tokens, completion_tokens}。"""
        if max_new_tokens is None:
            max_new_tokens = self.gen_config.max_new_tokens
        if not self._is_loaded:
            await self.load_model()

        self._current_load += 1
        self._request_count += 1
        asyncio.create_task(self._push_load_update())

        stream_msg, writer = StreamMessage.create("GenerateStream")
        worker = self

        async def produce():
            try:
                from transformers import TextIteratorStreamer

                inputs = worker._tokenizer(prompt, return_tensors="pt").to(
                    worker._model.device
                )
                input_len = inputs["input_ids"].shape[1]
                streamer = TextIteratorStreamer(
                    worker._tokenizer, skip_prompt=True, skip_special_tokens=True
                )
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": worker._tokenizer.eos_token_id,
                    "do_sample": worker.gen_config.do_sample,
                    "streamer": streamer,
                }
                thread = Thread(target=worker._model.generate, kwargs=gen_kwargs)
                thread.start()
                token_count = 0
                for text in streamer:
                    if text:
                        token_count += 1
                        await writer.write(
                            {"text": text, "worker_id": worker.worker_id}
                        )
                thread.join()
                await writer.write(
                    {
                        "text": "",
                        "finish_reason": "stop",
                        "prompt_tokens": input_len,
                        "completion_tokens": token_count,
                    }
                )
            except Exception as e:
                print(f"[Worker] produce error: {e}")
                try:
                    await writer.error(str(e))
                except Exception:
                    pass
            finally:
                worker._current_load -= 1
                asyncio.create_task(worker._push_load_update())
                writer.close()

        asyncio.create_task(produce())
        async for chunk in stream_msg.stream_reader():
            yield chunk

    async def subscribe_load(self):
        """订阅负载更新，async generator 每秒 yield 一次负载快照。"""
        stream_msg, writer = StreamMessage.create("LoadStream")
        self._load_subscribers.append(writer)
        worker = self

        async def produce():
            try:
                await writer.write(worker._get_load_snapshot())
                while True:
                    await asyncio.sleep(1.0)
                    await writer.write(worker._get_load_snapshot())
            except Exception:
                pass
            finally:
                if writer in worker._load_subscribers:
                    worker._load_subscribers.remove(writer)
                writer.close()

        asyncio.create_task(produce())
        async for snapshot in stream_msg.stream_reader():
            yield snapshot

    def health_check(self) -> dict:
        """健康检查。"""
        return {
            "status": "healthy",
            "worker_id": self.worker_id,
            "is_loaded": self._is_loaded,
        }

    def get_load(self) -> dict:
        """当前负载快照。"""
        return self._get_load_snapshot()
