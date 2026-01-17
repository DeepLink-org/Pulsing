"""vLLM Worker Actor - High-Performance Inference Worker Based on vLLM V1 Engine

Referencing Dynamo implementation, supports:
1. Prefill/Decode separation (PD Disaggregation)
2. Multimodal input processing (images)
3. KV Cache management and cleanup
4. LoRA dynamic loading/unloading
5. OpenAI-compatible text input/output mode
6. Engine monitoring and health checks
"""

# VllmWorker Actor - Main Actor Class

import asyncio
import logging
import os
import uuid
from typing import Any

from pulsing.actor import Actor, ActorId, Message, StreamMessage

from .vllm_handlers import BaseWorkerHandler, DecodeWorkerHandler, PrefillWorkerHandler
from .vllm_utils import _is_macos, _setup_macos_metal_env

try:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class VllmWorker(Actor):
    """vLLM Inference Worker Actor

    Supports vLLM V1 engine, features aligned with Dynamo:
    1. Supports PD separation (Prefill / Decode roles)
    2. Supports multimodal input (Image)
    3. Supports KV Cache cross-node transmission parameters
    4. Supports LoRA dynamic loading/unloading
    5. Supports OpenAI-compatible text input/output
    """

    def __init__(
        self,
        model: str,
        role: str = "aggregated",  # Options: aggregated, prefill, decode
        engine_args: dict[str, Any] | None = None,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        max_new_tokens: int = 512,
        enable_multimodal: bool = False,
        use_vllm_tokenizer: bool = False,
        # macOS Metal/MLX support parameters
        mlx_device: str
        | None = None,  # 'gpu' or 'cpu', default read from environment variable
        metal_memory_fraction: float | None = None,  # 0.0-1.0, default 0.8
        **kwargs,
    ):
        self.model = model
        self.role = role.lower()
        self.default_max_new_tokens = max_new_tokens
        self.enable_multimodal = enable_multimodal
        self.use_vllm_tokenizer = use_vllm_tokenizer

        # Set Metal/MLX environment variables on macOS
        _setup_macos_metal_env(mlx_device, metal_memory_fraction)

        self.engine_args_dict = engine_args or {}
        self.engine_args_dict.update(
            {
                "model": model,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": trust_remote_code,
                # Critical fix: use vllm's generation config instead of loading from HuggingFace
                # This avoids AssertionError caused by default repetition_penalty=1.1
                # Reference warning: Default sampling parameters have been overridden by the model's Hugging Face generation config
                "generation_config": "vllm",
            }
        )

        # Cleanup kwargs for AsyncEngineArgs
        kwargs.pop("max_new_tokens", None)
        kwargs.pop("enable_multimodal", None)
        kwargs.pop("use_vllm_tokenizer", None)
        self.engine_args_dict.update(kwargs)

        self.worker_id = f"vllm-{self.role}-{uuid.uuid4().hex[:8]}"
        self._engine: AsyncLLM | None = None
        self._handler: BaseWorkerHandler | None = None
        self._is_ready = False
        self._actor_id: ActorId | None = None
        self._init_task: asyncio.Task | None = None

    async def on_start(self, actor_id: ActorId) -> None:
        """Return quickly, initialize engine in background"""
        self._actor_id = actor_id
        if not VLLM_AVAILABLE:
            logger.error("vLLM not installed or version incompatible.")
            self._is_ready = False
            return

        # Initialize engine in background task
        async def init_background():
            try:
                logger.info(
                    f"Starting vLLM engine initialization for model: {self.model}"
                )
                await self._init_engine()
                if self._is_ready:
                    logger.info(
                        f"vLLM engine initialized successfully for {self.worker_id}"
                    )
                else:
                    logger.error(
                        "vLLM engine initialization completed but engine is not ready"
                    )
            except Exception as e:
                logger.exception(f"Failed to initialize vLLM engine: {e}")
                self._is_ready = False

        self._init_task = asyncio.create_task(init_background())
        logger.info(
            f"vLLM Worker {self.worker_id} started, engine initializing in background..."
        )

    async def _init_engine(self):
        if self._is_ready:
            return

        logger.info(f"Initializing vLLM ({self.role}) for model: {self.model}")

        # Detect platform and log information
        if _is_macos():
            mlx_device = os.environ.get("VLLM_MLX_DEVICE", "not set")
            metal_memory = os.environ.get("VLLM_METAL_MEMORY_FRACTION", "not set")
            logger.info(
                f"Running on macOS with Metal support: "
                f"VLLM_MLX_DEVICE={mlx_device}, "
                f"VLLM_METAL_MEMORY_FRACTION={metal_memory}"
            )

        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Move entire initialization process to thread pool
        def init_sync():
            """Synchronous initialization function, executed in thread pool"""
            args = AsyncEngineArgs(**self.engine_args_dict)
            usage_context = UsageContext.OPENAI_API_SERVER
            engine_config = args.create_engine_config(usage_context=usage_context)

            return AsyncLLM.from_vllm_config(
                vllm_config=engine_config,
                usage_context=usage_context,
                disable_log_requests=True,
            )

        loop = asyncio.get_event_loop()
        logger.info("Starting vLLM engine initialization in thread pool...")
        self._engine = await loop.run_in_executor(None, init_sync)

        # Create corresponding handler
        model_max_len = getattr(
            getattr(self._engine.vllm_config, "model_config", None),
            "max_model_len",
            None,
        )

        # Get default sampling parameters
        default_sampling_params = {}
        try:
            default_sampling_params = (
                AsyncEngineArgs(**self.engine_args_dict)
                .create_model_config()
                .get_diff_sampling_param()
            )
            logger.debug(f"Default sampling params: {default_sampling_params}")

            # If using text mode but penalties detected, ensure tokenizer is available
            # Otherwise clear these penalties to avoid AssertionError
            if self.use_vllm_tokenizer:
                has_penalties = (
                    default_sampling_params.get("repetition_penalty", 1.0) != 1.0
                    or default_sampling_params.get("presence_penalty", 0.0) != 0.0
                    or default_sampling_params.get("frequency_penalty", 0.0) != 0.0
                )
                if has_penalties:
                    tokenizer_available = (
                        hasattr(self._engine, "tokenizer")
                        and self._engine.tokenizer is not None
                    )
                    logger.info(
                        f"Default params have penalties, tokenizer_available={tokenizer_available}"
                    )
                    if not tokenizer_available:
                        logger.warning(
                            "Default sampling params contain penalties but tokenizer is not available, "
                            "resetting penalties to avoid AssertionError"
                        )
                        default_sampling_params["repetition_penalty"] = 1.0
                        default_sampling_params["presence_penalty"] = 0.0
                        default_sampling_params["frequency_penalty"] = 0.0
        except Exception as e:
            logger.warning(f"Could not get default sampling params: {e}")

        if self.role == "prefill":
            self._handler = PrefillWorkerHandler(
                engine=self._engine,
                default_sampling_params=default_sampling_params,
                model_max_len=model_max_len,
                enable_multimodal=self.enable_multimodal,
                use_vllm_tokenizer=self.use_vllm_tokenizer,
            )
        else:
            # Both aggregated and decode roles use DecodeWorkerHandler
            self._handler = DecodeWorkerHandler(
                engine=self._engine,
                default_sampling_params=default_sampling_params,
                model_max_len=model_max_len,
                enable_multimodal=self.enable_multimodal,
                use_vllm_tokenizer=self.use_vllm_tokenizer,
            )

        self._is_ready = True
        logger.info(f"vLLM Worker {self.worker_id} ready")

    def on_stop(self) -> None:
        # Cancel initialization task
        if (
            hasattr(self, "_init_task")
            and self._init_task
            and not self._init_task.done()
        ):
            self._init_task.cancel()

        # Clean up handler resources
        if self._handler:
            self._handler.cleanup()

        self._engine = None
        self._handler = None
        self._is_ready = False

    def metadata(self) -> dict[str, str]:
        meta = {
            "type": "vllm_worker",
            "role": self.role,
            "model": self.model,
            "worker_id": self.worker_id,
            "ready": str(self._is_ready),
            "multimodal_enabled": str(self.enable_multimodal),
            "text_mode": str(self.use_vllm_tokenizer),
        }

        if self._is_ready and self._engine:
            # Try to get vLLM engine runtime statistics
            try:
                config = self._engine.vllm_config
                meta.update(
                    {
                        "total_kv_blocks": str(config.cache_config.num_gpu_blocks),
                        "max_num_seqs": str(config.scheduler_config.max_num_seqs),
                        "max_num_batched_tokens": str(
                            config.scheduler_config.max_num_batched_tokens
                        ),
                        "block_size": str(config.cache_config.block_size),
                    }
                )
            except Exception:
                pass

        return meta

    async def receive(self, msg: Message) -> Message | StreamMessage:
        # If engine is not ready, wait for initialization to complete
        if not self._is_ready:
            if not VLLM_AVAILABLE:
                error_msg = "vLLM not installed or version incompatible"
                if msg.msg_type.endswith("StreamRequest"):
                    stream_msg, writer = StreamMessage.create("Error")
                    asyncio.create_task(writer.error(error_msg))
                    writer.close()
                    return stream_msg
                return Message.from_json("Error", {"error": error_msg})

            # Wait for engine initialization to complete
            max_wait = 60.0
            wait_interval = 0.5
            waited = 0.0

            while not self._is_ready and waited < max_wait:
                await asyncio.sleep(wait_interval)
                waited += wait_interval

            if not self._is_ready:
                error_msg = f"vLLM engine initialization timeout after {max_wait}s"
                logger.error(error_msg)
                if msg.msg_type.endswith("StreamRequest"):
                    stream_msg, writer = StreamMessage.create("Error")
                    asyncio.create_task(writer.error(error_msg))
                    writer.close()
                    return stream_msg
                return Message.from_json("Error", {"error": error_msg})

        try:
            if msg.msg_type in ("GenerateRequest", "ChatCompletionRequest"):
                return await self._handle_generate(msg)
            elif msg.msg_type in (
                "GenerateStreamRequest",
                "ChatCompletionStreamRequest",
            ):
                return await self._handle_generate_stream(msg)
            elif msg.msg_type == "HealthCheck":
                # Detailed health check
                health_status = self._handler.engine_monitor.get_health_status()
                health_status["role"] = self.role
                health_status["worker_id"] = self.worker_id
                return Message.from_json("Ok", health_status)
            elif msg.msg_type == "ClearKVCache":
                result = await self._handler.clear_kv_cache()
                return Message.from_json("Ok", result)
            elif msg.msg_type == "LoadLoRA":
                # LoRA loading support
                data = msg.to_json()
                lora_name = data.get("lora_name")
                lora_path = data.get("lora_path")
                if not lora_name or not lora_path:
                    return Message.from_json(
                        "Error",
                        {"error": "Missing required fields: lora_name and lora_path"},
                    )
                result = await self._handler.load_lora(lora_name, lora_path)
                return Message.from_json("Ok", result)
            elif msg.msg_type == "UnloadLoRA":
                # LoRA unloading support
                data = msg.to_json()
                lora_name = data.get("lora_name")
                if not lora_name:
                    return Message.from_json(
                        "Error", {"error": "Missing required field: lora_name"}
                    )
                result = await self._handler.unload_lora(lora_name)
                return Message.from_json("Ok", result)
            elif msg.msg_type == "ListLoRAs":
                # LoRA list support
                result = await self._handler.list_loras()
                return Message.from_json("Ok", result)
            else:
                if msg.msg_type.endswith("StreamRequest"):
                    stream_msg, writer = StreamMessage.create("Error")
                    asyncio.create_task(
                        writer.error(f"Unsupported type: {msg.msg_type}")
                    )
                    writer.close()
                    return stream_msg
                return Message.from_json(
                    "Error", {"error": f"Unsupported type: {msg.msg_type}"}
                )
        except Exception as e:
            logger.exception(f"Error handling {msg.msg_type}: {e}")
            if msg.msg_type.endswith("StreamRequest"):
                stream_msg, writer = StreamMessage.create("Error")
                asyncio.create_task(writer.error(str(e)))
                writer.close()
                return stream_msg
            return Message.from_json("Error", {"error": str(e)})

    async def _handle_generate(self, msg: Message) -> Message:
        data = msg.to_json()

        try:
            # Use handler to generate results
            # Accumulate complete text and information
            accumulated_text = ""
            finish_reason = None
            result_count = 0

            async for result in self._handler.generate(data):
                result_count += 1

                # Extract text content (supports different formats)
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]

                    # Streaming format: extract from delta.content
                    if "delta" in choice and "content" in choice["delta"]:
                        accumulated_text += choice["delta"]["content"]
                    # Non-streaming format: extract from message.content
                    elif "message" in choice and "content" in choice["message"]:
                        accumulated_text = choice["message"]["content"]
                    # Or extract directly from text
                    elif "text" in choice:
                        accumulated_text = choice["text"]

                    # Extract finish_reason
                    if "finish_reason" in choice and choice["finish_reason"]:
                        finish_reason = choice["finish_reason"]

            # Return complete response (OpenAI format)
            if accumulated_text or result_count > 0:
                response = {
                    "text": accumulated_text,
                    "finish_reason": finish_reason or "stop",
                    "completion_tokens": (
                        len(accumulated_text.split()) if accumulated_text else 0
                    ),
                    "prompt_tokens": 0,  # TODO: Calculate actual prompt tokens
                }
                return Message.from_json("GenerateResponse", response)
            return Message.from_json("Error", {"error": "No output"})
        except Exception as e:
            logger.exception(f"Error in generate: {e}")
            return Message.from_json("Error", {"error": str(e)})

    async def _handle_generate_stream(self, msg: Message) -> StreamMessage:
        stream_msg, writer = StreamMessage.create("GenerateStream")

        async def produce():
            try:
                data = msg.to_json()
                async for chunk in self._handler.generate(data):
                    await writer.write(chunk)
                    if chunk.get("finish_reason"):
                        break
            except Exception as e:
                logger.exception(f"Error in stream generation: {e}")
                await writer.error(str(e))
            finally:
                writer.close()

        asyncio.create_task(produce())
        return stream_msg
