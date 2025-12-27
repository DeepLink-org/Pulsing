"""vLLM Worker Actor - 基于 vLLM V1 引擎的高性能推理 Worker (功能增强版)"""

import asyncio
import base64
import logging
import os
import platform
import uuid
from io import BytesIO
from typing import Any

from pulsing.actor import Actor, ActorId, Message, StreamMessage

try:
    from PIL import Image
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


def _is_macos() -> bool:
    """检测是否在 macOS 上运行"""
    return platform.system() == "Darwin"


def _setup_macos_metal_env(
    mlx_device: str | None = None,
    metal_memory_fraction: float | None = None,
) -> None:
    """设置 macOS Metal/MLX 环境变量
    
    Args:
        mlx_device: MLX 设备类型 ('gpu' 或 'cpu')，默认 'gpu'
        metal_memory_fraction: Metal 内存使用比例 (0.0-1.0)，默认 0.8
    """
    if not _is_macos():
        return
    
    # 设置 MLX 设备
    if mlx_device is None:
        mlx_device = os.environ.get("VLLM_MLX_DEVICE", "gpu")
    
    if "VLLM_MLX_DEVICE" not in os.environ:
        os.environ["VLLM_MLX_DEVICE"] = mlx_device
        logger.info(f"Set VLLM_MLX_DEVICE={mlx_device} for macOS Metal support")
    
    # 设置 Metal 内存使用比例
    if metal_memory_fraction is None:
        metal_memory_fraction_str = os.environ.get("VLLM_METAL_MEMORY_FRACTION")
        if metal_memory_fraction_str:
            metal_memory_fraction = float(metal_memory_fraction_str)
        else:
            metal_memory_fraction = 0.8  # 默认 80%
    
    if "VLLM_METAL_MEMORY_FRACTION" not in os.environ:
        os.environ["VLLM_METAL_MEMORY_FRACTION"] = str(metal_memory_fraction)
        logger.info(
            f"Set VLLM_METAL_MEMORY_FRACTION={metal_memory_fraction} for macOS Metal support"
        )


class VllmWorker(Actor):
    """vLLM 推理 Worker Actor

    支持 vLLM V1 引擎，功能对齐 Dynamo：
    1. 支持 PD 分离 (Prefill / Decode 角色)
    2. 支持多模态输入 (Image)
    3. 支持 KV Cache 跨节点传输参数
    """

    def __init__(
        self,
        model: str,
        role: str = "aggregated",  # Options: aggregated, prefill, decode
        engine_args: dict[str, Any] | None = None,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        max_new_tokens: int = 512,
        # macOS Metal/MLX 支持参数
        mlx_device: str | None = None,  # 'gpu' 或 'cpu'，默认从环境变量读取
        metal_memory_fraction: float | None = None,  # 0.0-1.0，默认 0.8
        **kwargs,
    ):
        self.model = model
        self.role = role.lower()
        self.default_max_new_tokens = max_new_tokens

        # 在 macOS 上设置 Metal/MLX 环境变量
        _setup_macos_metal_env(mlx_device, metal_memory_fraction)

        self.engine_args_dict = engine_args or {}
        self.engine_args_dict.update(
            {
                "model": model,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": trust_remote_code,
            }
        )

        # Cleanup kwargs for AsyncEngineArgs
        kwargs.pop("max_new_tokens", None)
        self.engine_args_dict.update(kwargs)

        self.worker_id = f"vllm-{self.role}-{uuid.uuid4().hex[:8]}"
        self._engine: AsyncLLM | None = None
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

        # 在后台任务中初始化引擎，避免阻塞 on_start
        async def init_background():
            try:
                logger.info(f"Starting vLLM engine initialization for model: {self.model}")
                await self._init_engine()
                if self._is_ready:
                    logger.info(f"vLLM engine initialized successfully for {self.worker_id}")
                else:
                    logger.error(f"vLLM engine initialization completed but engine is not ready")
            except Exception as e:
                logger.exception(f"Failed to initialize vLLM engine: {e}")
                self._is_ready = False

        self._init_task = asyncio.create_task(init_background())
        logger.info(f"vLLM Worker {self.worker_id} started, engine initializing in background...")

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

        # 将整个初始化过程移到线程池中执行，避免阻塞事件循环
        # vLLM 的初始化可能包含同步阻塞操作（如模型加载）
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
        logger.info(f"Starting vLLM engine initialization in thread pool...")
        self._engine = await loop.run_in_executor(None, init_sync)
        
        self._is_ready = True
        logger.info(f"vLLM Worker {self.worker_id} ready")

    def on_stop(self) -> None:
        # 取消初始化任务（如果还在运行）
        if hasattr(self, '_init_task') and self._init_task and not self._init_task.done():
            self._init_task.cancel()
        
        self._engine = None
        self._is_ready = False

    def metadata(self) -> dict[str, str]:
        meta = {
            "type": "vllm_worker",
            "role": self.role,
            "model": self.model,
            "worker_id": self.worker_id,
            "ready": str(self._is_ready),
        }

        if self._is_ready and self._engine:
            # 尝试获取 vLLM 引擎的运行时统计，对齐 Dynamo
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
        # 如果引擎未就绪，等待初始化完成（最多等待 60 秒）
        if not self._is_ready:
            if not VLLM_AVAILABLE:
                error_msg = "vLLM not installed or version incompatible"
                if msg.msg_type in ("GenerateStreamRequest", "ChatCompletionStreamRequest"):
                    stream_msg, writer = StreamMessage.create("Error")
                    asyncio.create_task(writer.error(error_msg))
                    writer.close()
                    return stream_msg
                return Message.from_json("Error", {"error": error_msg})
            
            # 等待引擎初始化完成
            max_wait = 60.0  # 最多等待 60 秒
            wait_interval = 0.5  # 每 0.5 秒检查一次
            waited = 0.0
            
            while not self._is_ready and waited < max_wait:
                await asyncio.sleep(wait_interval)
                waited += wait_interval
            
            if not self._is_ready:
                error_msg = f"vLLM engine initialization timeout after {max_wait}s"
                logger.error(error_msg)
                if msg.msg_type in ("GenerateStreamRequest", "ChatCompletionStreamRequest"):
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
                return Message.from_json("Ok", {"status": "healthy", "role": self.role})
            elif msg.msg_type == "ClearKVCache":
                await self._engine.reset_prefix_cache()
                return Message.from_json("Ok", {"message": "KV cache cleared"})
            else:
                # 对于流式请求，返回流式错误消息
                if msg.msg_type.endswith("StreamRequest"):
                    stream_msg, writer = StreamMessage.create("Error")
                    asyncio.create_task(writer.error(f"Unsupported type: {msg.msg_type}"))
                    writer.close()
                    return stream_msg
                return Message.from_json(
                    "Error", {"error": f"Unsupported type: {msg.msg_type}"}
                )
        except Exception as e:
            logger.exception(f"Error handling {msg.msg_type}: {e}")
            # 对于流式请求，返回流式错误消息
            if msg.msg_type in ("GenerateStreamRequest", "ChatCompletionStreamRequest"):
                stream_msg, writer = StreamMessage.create("Error")
                asyncio.create_task(writer.error(str(e)))
                writer.close()
                return stream_msg
            return Message.from_json("Error", {"error": str(e)})

    async def _build_prompt(self, data: dict[str, Any]) -> TokensPrompt | TextPrompt:
        """构建 vLLM 输入 Prompt，支持多模态"""
        prompt_text = data.get("prompt")
        token_ids = data.get("token_ids")

        # 处理多模态数据
        mm_data = data.get("multi_modal_data")
        if mm_data and "image" in mm_data:
            images = mm_data["image"]
            if isinstance(images, str):  # URL or base64
                mm_data["image"] = await self._load_image(images)
            elif isinstance(images, list):
                mm_data["image"] = [await self._load_image(img) for img in images]

        if token_ids:
            return TokensPrompt(prompt_token_ids=token_ids, multi_modal_data=mm_data)
        return TextPrompt(prompt=prompt_text, multi_modal_data=mm_data)

    async def _load_image(self, image_source: str) -> "Image.Image":
        """加载图片，支持 Data URL (Base64)"""
        if image_source.startswith("data:image"):
            try:
                # data:image/png;base64,xxxx
                header, data = image_source.split(",", 1)
                image_bytes = base64.b64decode(data)
                return await asyncio.to_thread(Image.open, BytesIO(image_bytes))
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image: {e}") from e

        # 暂时不支持 HTTP URL 下载，建议由前端/Processor 转换成 Base64
        raise ValueError(f"Unsupported image source: {image_source[:20]}...")

    def _build_sampling_params(self, data: dict[str, Any]) -> SamplingParams:
        """解析采样参数，支持 PD 分离相关参数"""
        sampling_dict = {
            "n": data.get("n", 1),
            "temperature": data.get("temperature", 1.0),
            "top_p": data.get("top_p", 1.0),
            "top_k": data.get("top_k", -1),
            "presence_penalty": data.get("presence_penalty", 0.0),
            "frequency_penalty": data.get("frequency_penalty", 0.0),
            "repetition_penalty": data.get("repetition_penalty", 1.0),
            "stop": data.get("stop"),
            "max_tokens": data.get(
                "max_new_tokens", data.get("max_tokens", self.default_max_new_tokens)
            ),
        }

        sampling_params = SamplingParams(**sampling_dict)

        # --- PD Disaggregation 逻辑 ---
        if self.role == "prefill":
            # Prefill 角色：强制只生成 1 个 token，并开启远程解码标志
            sampling_params.max_tokens = 1
            sampling_params.min_tokens = 1
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = {
                "do_remote_decode": True
            }

        elif self.role == "decode":
            # Decode 角色：从 prefill_result 中提取 KV 传输参数
            prefill_result = data.get("prefill_result")
            if prefill_result:
                kv_params = prefill_result.get("disaggregated_params", {}).get(
                    "kv_transfer_params"
                )
                if kv_params:
                    if sampling_params.extra_args is None:
                        sampling_params.extra_args = {}
                    sampling_params.extra_args["kv_transfer_params"] = kv_params

        return sampling_params

    async def _handle_generate(self, msg: Message) -> Message:
        data = msg.to_json()
        prompt = await self._build_prompt(data)
        sampling_params = self._build_sampling_params(data)
        request_id = f"req-{uuid.uuid4().hex[:8]}"

        results_generator = self._engine.generate(prompt, sampling_params, request_id)

        final_res = None
        async for res in results_generator:
            final_res = res

        if final_res:
            return self._format_response(final_res)
        return Message.from_json("Error", {"error": "No output"})

    async def _handle_generate_stream(self, msg: Message) -> StreamMessage:
        # 先创建 StreamMessage，确保即使出错也能返回流式消息
        stream_msg, writer = StreamMessage.create("GenerateStream")
        
        async def produce():
            try:
                data = msg.to_json()
                prompt = await self._build_prompt(data)
                sampling_params = self._build_sampling_params(data)
                request_id = f"req-stream-{uuid.uuid4().hex[:8]}"

                results_generator = self._engine.generate(
                    prompt, sampling_params, request_id
                )
                last_pos = 0
                last_text = ""  # 用于检测重复
                repeat_count = 0
                max_repeats = 10  # 最多允许重复次数
                
                async for res in results_generator:
                    if res.outputs:
                        output = res.outputs[0]
                        current_text = output.text
                        text_delta = current_text[last_pos:]
                        last_pos = len(current_text)
                        
                        # 检测重复：如果新文本和上次一样，增加计数
                        if text_delta == last_text and text_delta:
                            repeat_count += 1
                            if repeat_count >= max_repeats:
                                logger.warning(f"Detected repetition, stopping generation. Text: {text_delta[:50]}...")
                                chunk = {
                                    "text": "",
                                    "finish_reason": "length",  # 使用 length 作为重复结束的原因
                                }
                                await writer.write_json(chunk)
                                break
                        else:
                            repeat_count = 0
                            last_text = text_delta

                        chunk = {
                            "text": text_delta,
                            "finish_reason": output.finish_reason,
                        }

                        # 如果是 Prefill 角色，带上 disaggregated_params
                        if self.role == "prefill" and hasattr(
                            res, "kv_transfer_params"
                        ):
                            chunk["disaggregated_params"] = {
                                "kv_transfer_params": res.kv_transfer_params
                            }

                        await writer.write_json(chunk)
                        if output.finish_reason:
                            break
            except Exception as e:
                logger.exception(f"Error in stream generation: {e}")
                await writer.error(str(e))
            finally:
                writer.close()

        asyncio.create_task(produce())
        return stream_msg

    def _format_response(self, res) -> Message:
        """统一格式化响应，支持 PD 传输参数"""
        output = res.outputs[0]
        resp_data = {
            "text": output.text,
            "worker_id": self.worker_id,
            "prompt_tokens": len(res.prompt_token_ids),
            "completion_tokens": len(output.token_ids),
            "finish_reason": output.finish_reason,
        }

        # Prefill 角色特有的返回参数
        if self.role == "prefill" and hasattr(res, "kv_transfer_params"):
            resp_data["disaggregated_params"] = {
                "kv_transfer_params": res.kv_transfer_params
            }

        return Message.from_json("GenerateResponse", resp_data)
