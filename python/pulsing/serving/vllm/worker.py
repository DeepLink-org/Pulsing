"""vLLM Worker - High-performance inference Worker based on vLLM V1 engine (pulsing.remote)

Referencing Dynamo implementation, supports:
1. Prefill/Decode separation (PD Disaggregation)
2. Multimodal input processing (images)
3. KV Cache management and cleanup
4. LoRA dynamic loading/unloading
5. OpenAI-compatible text input/output mode
6. Engine monitoring and health checks
"""

import asyncio
import logging
import os
import uuid
from typing import Any

from pulsing.core import ActorId, StreamMessage, remote

from .handlers import BaseWorkerHandler, DecodeWorkerHandler, PrefillWorkerHandler
from .utils import _is_macos, _setup_macos_metal_env

try:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


@remote
class VllmWorker:
    """vLLM inference Worker，通过 pulsing.remote 暴露方法。

    支持 vLLM V1 engine，功能对齐 Dynamo：
    1. PD 分离（Prefill / Decode 角色）
    2. 多模态输入（Image）
    3. KV Cache 管理与清理
    4. LoRA 动态加载/卸载
    5. OpenAI 兼容文本输入/输出
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
        # Distributed inference parameters (referencing Dynamo implementation)
        tensor_parallel_size: int = 1,  # TP: Tensor parallelism, split model across multiple GPUs
        pipeline_parallel_size: int = 1,  # PP: Pipeline parallelism, distribute model layers
        data_parallel_size: int = 1,  # DP: Data parallelism
        data_parallel_rank: int | None = None,  # DP rank, for multi-replica deployment
        enable_expert_parallel: bool = False,  # EP: MoE model expert parallelism
        distributed_executor_backend: str | None = None,  # "mp" or "ray"
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

        # Distributed configuration
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.data_parallel_rank = data_parallel_rank or 0
        self.enable_expert_parallel = enable_expert_parallel
        self._is_primary_rank = data_parallel_rank is None or data_parallel_rank == 0

        # macOS Metal/MLX environment
        _setup_macos_metal_env(mlx_device, metal_memory_fraction)

        # Build engine arguments
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
        """Build vLLM engine arguments

        Integrates basic parameters and distributed parameters processing logic.
        """
        args = engine_args.copy() if engine_args else {}

        # Basic parameters
        args.update(
            {
                "model": model,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": trust_remote_code,
                "generation_config": "vllm",  # Use vllm default config to avoid HF's repetition_penalty
            }
        )

        # Distributed parameters (only add when not default values)
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

        # Distributed executor backend (default to mp when TP=1 to avoid GIL issues)
        if distributed_executor_backend:
            args["distributed_executor_backend"] = distributed_executor_backend
        elif tensor_parallel_size == 1:
            args["distributed_executor_backend"] = "mp"

        # Filter out custom parameters, only keep vLLM-supported parameters
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

            # If using text mode but penalties are detected, ensure tokenizer is available
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
                on_engine_dead=self._on_engine_dead,
            )
        else:
            # Both aggregated and decode roles use DecodeWorkerHandler
            self._handler = DecodeWorkerHandler(
                engine=self._engine,
                default_sampling_params=default_sampling_params,
                model_max_len=model_max_len,
                enable_multimodal=self.enable_multimodal,
                use_vllm_tokenizer=self.use_vllm_tokenizer,
                on_engine_dead=self._on_engine_dead,
            )

        # Start engine health monitoring (referencing Dynamo engine_monitor.py)
        self._handler.engine_monitor.start_monitoring(
            on_engine_dead=self._on_engine_dead
        )
        logger.info(f"Engine health monitoring started for {self.worker_id}")

        self._is_ready = True

        # Log DP rank conditional registration status (referencing Dynamo main.py)
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
        """Engine dead callback function (referencing Dynamo handlers.py)

        Called when vLLM engine detects EngineDeadError.
        This usually means the engine has crashed and needs cleanup and possibly restart.
        """
        logger.error(f"Engine dead callback triggered for {self.worker_id}")
        self._engine_is_dead = True
        self._is_ready = False

        # Note: In Actor mode, we don't directly call os._exit(1)
        # Instead, set state and let upper-level Actor system handle it
        # If forced exit is needed, add logic here

    @property
    def engine_is_dead(self) -> bool:
        """Return whether the engine is dead"""
        return self._engine_is_dead

    def on_stop(self) -> None:
        # Stop engine health monitoring
        if self._handler and hasattr(self._handler, "engine_monitor"):
            self._handler.engine_monitor.stop_monitoring()

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

    @property
    def is_primary_rank(self) -> bool:
        """Check if this is the primary rank (for model registration)

        In DP deployment, only rank 0 needs to register model to service discovery.
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
            # Distributed configuration
            "tp_size": str(self.tensor_parallel_size),
            "pp_size": str(self.pipeline_parallel_size),
            "dp_size": str(self.data_parallel_size),
            "dp_rank": str(self.data_parallel_rank),
            "ep_enabled": str(self.enable_expert_parallel),
            "is_primary_rank": str(self._is_primary_rank),
            # Health status
            "engine_healthy": str(not self._engine_is_dead),
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
                # Get actual parallel configuration from vllm_config
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

    async def _ensure_ready(self) -> None:
        """等待 engine 就绪，超时则抛出异常。"""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed or version incompatible")
        max_wait = 60.0
        wait_interval = 0.5
        waited = 0.0
        while not self._is_ready and waited < max_wait:
            await asyncio.sleep(wait_interval)
            waited += wait_interval
        if not self._is_ready:
            raise RuntimeError(f"vLLM engine initialization timeout after {max_wait}s")

    def _build_request(
        self, prompt: str = "", max_new_tokens: int | None = None, **kwargs
    ) -> dict:
        data = kwargs.copy()
        data.setdefault("prompt", prompt)
        data.setdefault("max_new_tokens", max_new_tokens or self.default_max_new_tokens)
        return data

    async def _collect_generate_result(self, data: dict) -> dict:
        """从 handler.generate 的迭代结果聚合成单次响应 dict。"""
        accumulated_text = ""
        finish_reason = None
        result_count = 0
        async for result in self._handler.generate(data):
            result_count += 1
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    accumulated_text += choice["delta"]["content"]
                elif "message" in choice and "content" in choice["message"]:
                    accumulated_text = choice["message"]["content"]
                elif "text" in choice:
                    accumulated_text = choice["text"]
                if "finish_reason" in choice and choice["finish_reason"]:
                    finish_reason = choice["finish_reason"]
        if accumulated_text or result_count > 0:
            return {
                "text": accumulated_text,
                "finish_reason": finish_reason or "stop",
                "completion_tokens": (
                    len(accumulated_text.split()) if accumulated_text else 0
                ),
                "prompt_tokens": 0,
            }
        return {"error": "No output"}

    # -------------------------------------------------------------------------
    # 对外方法（替代原 receive 的消息类型分支）
    # -------------------------------------------------------------------------

    async def generate(
        self,
        prompt: str = "",
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> dict:
        """同步生成，返回 {text, finish_reason, prompt_tokens, completion_tokens} 或 {error}。"""
        await self._ensure_ready()
        try:
            data = self._build_request(
                prompt=prompt, max_new_tokens=max_new_tokens, **kwargs
            )
            return await self._collect_generate_result(data)
        except Exception as e:
            logger.exception("Error in generate: %s", e)
            return {"error": str(e)}

    async def generate_stream(
        self,
        prompt: str = "",
        max_new_tokens: int | None = None,
        **kwargs,
    ):
        """流式生成，async generator 逐条 yield chunk。"""
        await self._ensure_ready()
        stream_msg, writer = StreamMessage.create("GenerateStream")
        data = self._build_request(
            prompt=prompt, max_new_tokens=max_new_tokens, **kwargs
        )

        async def produce():
            try:
                async for chunk in self._handler.generate(data):
                    await writer.write(chunk)
                    if chunk.get("finish_reason"):
                        break
            except Exception as e:
                logger.exception("Error in stream generation: %s", e)
                await writer.error(str(e))
            finally:
                writer.close()

        asyncio.create_task(produce())
        async for chunk in stream_msg.stream_reader():
            yield chunk

    def health_check(self) -> dict:
        """健康检查，含 engine 状态。"""
        if not self._is_ready or not self._handler:
            return {
                "role": self.role,
                "worker_id": self.worker_id,
                "ready": False,
                "error": "Engine not ready",
            }
        status = self._handler.engine_monitor.get_health_status()
        status["role"] = self.role
        status["worker_id"] = self.worker_id
        return status

    async def clear_kv_cache(self) -> dict:
        """清理 KV Cache。"""
        await self._ensure_ready()
        result = await self._handler.clear_kv_cache()
        return result if isinstance(result, dict) else {"ok": True}

    async def load_lora(self, lora_name: str, lora_path: str) -> dict:
        """加载 LoRA。"""
        await self._ensure_ready()
        if not lora_name or not lora_path:
            return {"error": "Missing required fields: lora_name and lora_path"}
        result = await self._handler.load_lora(lora_name, lora_path)
        return result if isinstance(result, dict) else {"ok": True}

    async def unload_lora(self, lora_name: str) -> dict:
        """卸载 LoRA。"""
        await self._ensure_ready()
        if not lora_name:
            return {"error": "Missing required field: lora_name"}
        result = await self._handler.unload_lora(lora_name)
        return result if isinstance(result, dict) else {"ok": True}

    async def list_loras(self) -> Any:
        """列出已加载的 LoRA。"""
        await self._ensure_ready()
        return await self._handler.list_loras()
