"""Transformers Worker Actor - LLM Inference Worker"""

import asyncio
import time
import uuid
from dataclasses import dataclass

from pulsing.actor import Actor, ActorId, Message, StreamMessage


@dataclass
class GenerationConfig:
    """Generation configuration"""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False


class TransformersWorker(Actor):
    """Transformers LLM Inference Worker, supports synchronous and streaming generation

    Supports streaming load subscription (SubscribeLoad), Router can subscribe and receive load updates in real-time.
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

        # Load tracking
        self._current_load = 0
        self._request_count = 0

        # Load subscribers (streaming push)
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
        # Close all subscription streams
        for writer in self._load_subscribers:
            try:
                writer.close()
            except Exception:
                pass
        self._load_subscribers.clear()

    def metadata(self) -> dict[str, str]:
        """Returns worker metadata"""
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
        """Get load snapshot"""
        return {
            "worker_id": self.worker_id,
            "node_id": self._node_id or self.worker_id,
            "load": self._current_load,
            "capacity": self.capacity,
            "processed": self._request_count,
            "timestamp": time.time(),
        }

    async def _push_load_update(self):
        """Push load update to all subscribers"""
        if not self._load_subscribers:
            return

        snapshot = self._get_load_snapshot()
        dead_writers = []

        for writer in self._load_subscribers:
            try:
                await writer.write(snapshot)
            except Exception:
                dead_writers.append(writer)

        # Clean up disconnected connections
        for w in dead_writers:
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

    async def receive(self, msg: Message) -> Message | StreamMessage:
        try:
            if msg.msg_type == "GenerateRequest":
                return await self._handle_generate(msg)
            elif msg.msg_type == "GenerateStreamRequest":
                return await self._handle_generate_stream(msg)
            elif msg.msg_type == "SubscribeLoad":
                return self._handle_subscribe_load()
            elif msg.msg_type == "HealthCheck":
                return Message.from_json(
                    "Ok",
                    {
                        "status": "healthy",
                        "worker_id": self.worker_id,
                        "is_loaded": self._is_loaded,
                    },
                )
            elif msg.msg_type == "GetLoad":
                return Message.from_json("LoadInfo", self._get_load_snapshot())
            else:
                return Message.from_json("Error", {"error": f"Unknown: {msg.msg_type}"})
        except Exception as e:
            print(f"[Worker] Error: {e}")
            return Message.from_json("Error", {"error": str(e)})

    def _handle_subscribe_load(self) -> StreamMessage:
        """Handle load subscription request, returns a stream that continuously pushes load updates"""
        stream_msg, writer = StreamMessage.create("LoadStream")
        self._load_subscribers.append(writer)

        worker = self

        async def produce():
            try:
                # Immediately send current state
                await writer.write(worker._get_load_snapshot())

                # Periodic push (every second)
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
        return stream_msg

    async def _handle_generate(self, msg: Message) -> Message:
        if not self._is_loaded:
            await self.load_model()

        data = msg.to_json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", self.gen_config.max_new_tokens)

        # Start request - increase load
        self._current_load += 1
        self._request_count += 1
        asyncio.create_task(self._push_load_update())

        try:
            loop = asyncio.get_running_loop()

            def _generate_sync():
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
                None, _generate_sync
            )

            return Message.from_json(
                "GenerateResponse",
                {
                    "text": text,
                    "worker_id": self.worker_id,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )
        finally:
            # Request completed - decrease load
            self._current_load -= 1
            asyncio.create_task(self._push_load_update())

    async def _handle_generate_stream(self, msg: Message) -> StreamMessage:
        from threading import Thread

        if not self._is_loaded:
            await self.load_model()

        data = msg.to_json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", self.gen_config.max_new_tokens)

        # Start request - increase load
        self._current_load += 1
        self._request_count += 1
        asyncio.create_task(self._push_load_update())

        stream_msg, writer = StreamMessage.create("GenerateStream")

        # Save reference for decreasing load in produce
        worker = self

        async def produce():
            try:
                inputs = worker._tokenizer(prompt, return_tensors="pt").to(
                    worker._model.device
                )
                input_len = inputs["input_ids"].shape[1]

                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    worker._tokenizer, skip_prompt=True, skip_special_tokens=True
                )
                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": worker._tokenizer.eos_token_id,
                    "do_sample": worker.gen_config.do_sample,
                    "streamer": streamer,
                }

                thread = Thread(target=worker._model.generate, kwargs=generation_kwargs)
                thread.start()

                token_count = 0
                for text in streamer:
                    if text:
                        token_count += 1
                        await writer.write(
                            {
                                "text": text,
                                "worker_id": worker.worker_id,
                            }
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
                # Request completed - decrease load
                worker._current_load -= 1
                asyncio.create_task(worker._push_load_update())
                writer.close()

        asyncio.create_task(produce())
        return stream_msg
