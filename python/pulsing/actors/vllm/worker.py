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

from .handlers import BaseWorkerHandler, DecodeWorkerHandler, PrefillWorkerHandler
from .utils import _is_macos, _setup_macos_metal_env

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
        # 分布式推理参数 (参考 Dynamo 实现)
        tensor_parallel_size: int = 1,  # TP: 张量并行，切分模型到多个 GPU
        pipeline_parallel_size: int = 1,  # PP: 流水线并行，分布模型层
        data_parallel_size: int = 1,  # DP: 数据并行
        data_parallel_rank: int | None = None,  # DP rank，用于多副本部署
        enable_expert_parallel: bool = False,  # EP: MoE 模型专家并行
        distributed_executor_backend: str | None = None,  # "mp" 或 "ray"
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

        # 分布式配置
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.data_parallel_rank = data_parallel_rank or 0
        self.enable_expert_parallel = enable_expert_parallel
        self._is_primary_rank = data_parallel_rank is None or data_parallel_rank == 0

        # macOS Metal/MLX 环境
        _setup_macos_metal_env(mlx_device, metal_memory_fraction)

        # 构建引擎参数
        self.engine_args_dict = self._build_engine_args(
            engine_args=engine_args,
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            enable_expert_parallel=enable_expert_parallel,
            distributed_executor_backend=distributed_executor_backend,
            **kwargs,
        )

        self.worker_id = f"vllm-{self.role}-{uuid.uuid4().hex[:8]}"
        self._engine: AsyncLLM | None = None
        self._handler: BaseWorkerHandler | None = None
        self._is_ready = False
        self._engine_is_dead = False
        self._actor_id: ActorId | None = None
        self._init_task: asyncio.Task | None = None

    @staticmethod
    def _build_engine_args(
        engine_args: dict[str, Any] | None,
        model: str,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        data_parallel_rank: int | None,
        enable_expert_parallel: bool,
        distributed_executor_backend: str | None,
        **kwargs,
    ) -> dict[str, Any]:
        """构建 vLLM 引擎参数

        整合基础参数和分布式参数的处理逻辑。
        """
        args = engine_args.copy() if engine_args else {}

        # 基础参数
        args.update(
            {
                "model": model,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": trust_remote_code,
                "generation_config": "vllm",  # 使用 vllm 默认配置，避免 HF 的 repetition_penalty
            }
        )

        # 分布式参数 (仅在非默认值时添加)
        if tensor_parallel_size > 1:
            args["tensor_parallel_size"] = tensor_parallel_size
        if pipeline_parallel_size > 1:
            args["pipeline_parallel_size"] = pipeline_parallel_size
        if data_parallel_size > 1:
            args["data_parallel_size"] = data_parallel_size
        if data_parallel_rank is not None:
            args["data_parallel_rank"] = data_parallel_rank
        if enable_expert_parallel:
            args["enable_expert_parallel"] = enable_expert_parallel

        # 分布式执行后端 (TP=1 时默认用 mp 避免 GIL 问题)
        if distributed_executor_backend:
            args["distributed_executor_backend"] = distributed_executor_backend
        elif tensor_parallel_size == 1:
            args["distributed_executor_backend"] = "mp"

        # 过滤掉自定义参数，只保留 vLLM 支持的参数
        excluded_keys = {
            "max_new_tokens",
            "enable_multimodal",
            "use_vllm_tokenizer",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "data_parallel_rank",
            "enable_expert_parallel",
            "distributed_executor_backend",
        }
        for key, value in kwargs.items():
            if key not in excluded_keys:
                args[key] = value

        return args

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
                on_engine_dead=self._on_engine_dead,
            )
        else:
            # aggregated 或 decode 角色都使用 DecodeWorkerHandler
            self._handler = DecodeWorkerHandler(
                engine=self._engine,
                default_sampling_params=default_sampling_params,
                model_max_len=model_max_len,
                enable_multimodal=self.enable_multimodal,
                use_vllm_tokenizer=self.use_vllm_tokenizer,
                on_engine_dead=self._on_engine_dead,
            )

        # 启动引擎健康监控 (参考 Dynamo engine_monitor.py)
        self._handler.engine_monitor.start_monitoring(
            on_engine_dead=self._on_engine_dead
        )
        logger.info(f"Engine health monitoring started for {self.worker_id}")

        self._is_ready = True

        # 记录 DP rank 条件注册状态 (参考 Dynamo main.py)
        if self.data_parallel_size > 1:
            if self._is_primary_rank:
                logger.info(
                    f"vLLM Worker {self.worker_id} ready (primary rank, will register model)"
                )
            else:
                logger.info(
                    f"vLLM Worker {self.worker_id} ready (rank {self.data_parallel_rank}, "
                    f"will NOT register model - only primary rank registers)"
                )
        else:
            logger.info(f"vLLM Worker {self.worker_id} ready")

    def _on_engine_dead(self) -> None:
        """引擎死亡回调函数 (参考 Dynamo handlers.py)

        当 vLLM 引擎检测到 EngineDeadError 时调用此方法。
        这通常意味着引擎已经崩溃，需要进行清理和可能的重启。
        """
        logger.error(f"Engine dead callback triggered for {self.worker_id}")
        self._engine_is_dead = True
        self._is_ready = False

        # 注意：在 Actor 模式下，我们不直接调用 os._exit(1)
        # 而是设置状态，让上层 Actor 系统处理
        # 如果需要强制退出，可以在这里添加逻辑

    @property
    def engine_is_dead(self) -> bool:
        """返回引擎是否已死亡"""
        return self._engine_is_dead

    def on_stop(self) -> None:
        # 停止引擎健康监控
        if self._handler and hasattr(self._handler, "engine_monitor"):
            self._handler.engine_monitor.stop_monitoring()

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

    @property
    def is_primary_rank(self) -> bool:
        """判断是否是主 rank (用于模型注册)

        在 DP 部署中，只有 rank 0 需要注册模型到服务发现。
        """
        return self._is_primary_rank

    def metadata(self) -> dict[str, str]:
        meta = {
            "type": "vllm_worker",
            "role": self.role,
            "model": self.model,
            "worker_id": self.worker_id,
            "ready": str(self._is_ready),
            "multimodal_enabled": str(self.enable_multimodal),
            "text_mode": str(self.use_vllm_tokenizer),
            # 分布式配置
            "tp_size": str(self.tensor_parallel_size),
            "pp_size": str(self.pipeline_parallel_size),
            "dp_size": str(self.data_parallel_size),
            "dp_rank": str(self.data_parallel_rank),
            "ep_enabled": str(self.enable_expert_parallel),
            "is_primary_rank": str(self._is_primary_rank),
            # 健康状态
            "engine_healthy": str(not self._engine_is_dead),
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
                # 从 vllm_config 获取实际的并行配置
                parallel_config = getattr(config, "parallel_config", None)
                if parallel_config:
                    meta["actual_tp_size"] = str(
                        getattr(parallel_config, "tensor_parallel_size", 1)
                    )
                    meta["actual_pp_size"] = str(
                        getattr(parallel_config, "pipeline_parallel_size", 1)
                    )
                    meta["actual_dp_size"] = str(
                        getattr(parallel_config, "data_parallel_size", 1)
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
                    await writer.write_json(chunk)
                    if chunk.get("finish_reason"):
                        break
            except Exception as e:
                logger.exception(f"Error in stream generation: {e}")
                await writer.error(str(e))
            finally:
                writer.close()

        asyncio.create_task(produce())
        return stream_msg
