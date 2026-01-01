"""vLLM Worker Actor - 基于 vLLM V1 引擎的高性能推理 Worker

参考 Dynamo 实现，支持：
1. Prefill/Decode 分离 (PD Disaggregation)
2. 多模态输入处理（图片）
3. KV Cache 管理和清理
4. LoRA 动态加载/卸载
5. OpenAI 兼容的文本输入输出模式
6. 引擎监控和健康检查
"""

# VllmWorker Actor - 主 Actor 类

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
    """vLLM 推理 Worker Actor

    支持 vLLM V1 引擎，功能对齐 Dynamo：
    1. 支持 PD 分离 (Prefill / Decode 角色)
    2. 支持多模态输入 (Image)
    3. 支持 KV Cache 跨节点传输参数
    4. 支持 LoRA 动态加载/卸载
    5. 支持 OpenAI 兼容的文本输入输出
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
        # macOS Metal/MLX 支持参数
        mlx_device: str | None = None,  # 'gpu' 或 'cpu'，默认从环境变量读取
        metal_memory_fraction: float | None = None,  # 0.0-1.0，默认 0.8
        **kwargs,
    ):
        self.model = model
        self.role = role.lower()
        self.default_max_new_tokens = max_new_tokens
        self.enable_multimodal = enable_multimodal
        self.use_vllm_tokenizer = use_vllm_tokenizer

        # 在 macOS 上设置 Metal/MLX 环境变量
        _setup_macos_metal_env(mlx_device, metal_memory_fraction)

        self.engine_args_dict = engine_args or {}
        self.engine_args_dict.update(
            {
                "model": model,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": trust_remote_code,
                # 关键修复：使用 vllm 的生成配置而不是从 HuggingFace 加载
                # 这可以避免默认的 repetition_penalty=1.1 导致 AssertionError
                # 参考警告信息：Default sampling parameters have been overridden by the model's Hugging Face generation config
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
        """快速返回，在后台初始化引擎"""
        self._actor_id = actor_id
        if not VLLM_AVAILABLE:
            logger.error("vLLM not installed or version incompatible.")
            self._is_ready = False
            return

        # 在后台任务中初始化引擎
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

        # 检测平台并记录信息
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

        # 将整个初始化过程移到线程池中执行
        def init_sync():
            """同步初始化函数，在线程池中执行"""
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

        # 创建对应的处理器
        model_max_len = getattr(
            getattr(self._engine.vllm_config, "model_config", None),
            "max_model_len",
            None,
        )

        # 获取默认采样参数
        default_sampling_params = {}
        try:
            default_sampling_params = (
                AsyncEngineArgs(**self.engine_args_dict)
                .create_model_config()
                .get_diff_sampling_param()
            )
            logger.debug(f"Default sampling params: {default_sampling_params}")

            # 如果使用文本模式但检测到 penalties，需要确保 tokenizer 可用
            # 否则清除这些 penalties 以避免 AssertionError
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
            # aggregated 或 decode 角色都使用 DecodeWorkerHandler
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
        # 取消初始化任务
        if (
            hasattr(self, "_init_task")
            and self._init_task
            and not self._init_task.done()
        ):
            self._init_task.cancel()

        # 清理处理器资源
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
            # 尝试获取 vLLM 引擎的运行时统计
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
        # 如果引擎未就绪，等待初始化完成
        if not self._is_ready:
            if not VLLM_AVAILABLE:
                error_msg = "vLLM not installed or version incompatible"
                if msg.msg_type.endswith("StreamRequest"):
                    stream_msg, writer = StreamMessage.create("Error")
                    asyncio.create_task(writer.error(error_msg))
                    writer.close()
                    return stream_msg
                return Message.from_json("Error", {"error": error_msg})

            # 等待引擎初始化完成
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
                # 详细的健康检查
                health_status = self._handler.engine_monitor.get_health_status()
                health_status["role"] = self.role
                health_status["worker_id"] = self.worker_id
                return Message.from_json("Ok", health_status)
            elif msg.msg_type == "ClearKVCache":
                result = await self._handler.clear_kv_cache()
                return Message.from_json("Ok", result)
            elif msg.msg_type == "LoadLoRA":
                # LoRA 加载支持
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
                # LoRA 卸载支持
                data = msg.to_json()
                lora_name = data.get("lora_name")
                if not lora_name:
                    return Message.from_json(
                        "Error", {"error": "Missing required field: lora_name"}
                    )
                result = await self._handler.unload_lora(lora_name)
                return Message.from_json("Ok", result)
            elif msg.msg_type == "ListLoRAs":
                # LoRA 列表支持
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
            # 使用处理器生成结果
            # 累积完整的文本和信息
            accumulated_text = ""
            finish_reason = None
            result_count = 0

            async for result in self._handler.generate(data):
                result_count += 1

                # 提取文本内容（支持不同的格式）
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]

                    # 流式格式：从 delta.content 提取
                    if "delta" in choice and "content" in choice["delta"]:
                        accumulated_text += choice["delta"]["content"]
                    # 非流式格式：从 message.content 提取
                    elif "message" in choice and "content" in choice["message"]:
                        accumulated_text = choice["message"]["content"]
                    # 或者直接从 text 提取
                    elif "text" in choice:
                        accumulated_text = choice["text"]

                    # 提取 finish_reason
                    if "finish_reason" in choice and choice["finish_reason"]:
                        finish_reason = choice["finish_reason"]

            # 返回完整的响应（OpenAI 格式）
            if accumulated_text or result_count > 0:
                response = {
                    "text": accumulated_text,
                    "finish_reason": finish_reason or "stop",
                    "completion_tokens": (
                        len(accumulated_text.split()) if accumulated_text else 0
                    ),
                    "prompt_tokens": 0,  # TODO: 计算实际的 prompt tokens
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
