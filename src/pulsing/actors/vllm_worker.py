"""vLLM Worker Actor - 基于 vLLM V1 引擎的高性能推理 Worker (功能增强版)"""

import asyncio
import base64
import logging
import os
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

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
        engine_args: Optional[Dict[str, Any]] = None,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        self.model = model
        self.role = role.lower()
        self.default_max_new_tokens = max_new_tokens

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
        self._engine: Optional[AsyncLLM] = None
        self._is_ready = False
        self._actor_id: Optional[ActorId] = None

    async def on_start(self, actor_id: ActorId) -> None:
        self._actor_id = actor_id
        if not VLLM_AVAILABLE:
            logger.error("vLLM not installed or version incompatible.")
            return

        try:
            await self._init_engine()
        except Exception as e:
            logger.exception(f"Failed to initialize vLLM engine: {e}")

    async def _init_engine(self):
        if self._is_ready:
            return

        logger.info(f"Initializing vLLM ({self.role}) for model: {self.model}")

        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        args = AsyncEngineArgs(**self.engine_args_dict)
        usage_context = UsageContext.OPENAI_API_SERVER
        engine_config = args.create_engine_config(usage_context=usage_context)

        self._engine = AsyncLLM.from_vllm_config(
            vllm_config=engine_config,
            usage_context=usage_context,
            disable_log_requests=True,
        )

        self._is_ready = True
        logger.info(f"vLLM Worker {self.worker_id} ready")

    def on_stop(self) -> None:
        self._engine = None
        self._is_ready = False

    def metadata(self) -> Dict[str, str]:
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

    async def receive(self, msg: Message) -> Union[Message, StreamMessage]:
        if not self._is_ready:
            return Message.from_json("Error", {"error": "vLLM engine not ready"})

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
                return Message.from_json(
                    "Error", {"error": f"Unsupported type: {msg.msg_type}"}
                )
        except Exception as e:
            logger.exception(f"Error handling {msg.msg_type}: {e}")
            return Message.from_json("Error", {"error": str(e)})

    async def _build_prompt(
        self, data: Dict[str, Any]
    ) -> Union[TokensPrompt, TextPrompt]:
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
                raise ValueError(f"Failed to decode base64 image: {e}")

        # 暂时不支持 HTTP URL 下载，建议由前端/Processor 转换成 Base64
        raise ValueError(f"Unsupported image source: {image_source[:20]}...")

    def _build_sampling_params(self, data: Dict[str, Any]) -> SamplingParams:
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
        data = msg.to_json()
        prompt = await self._build_prompt(data)
        sampling_params = self._build_sampling_params(data)
        request_id = f"req-stream-{uuid.uuid4().hex[:8]}"

        stream_msg, writer = StreamMessage.create("GenerateStream")

        async def produce():
            try:
                results_generator = self._engine.generate(
                    prompt, sampling_params, request_id
                )
                last_pos = 0
                async for res in results_generator:
                    if res.outputs:
                        output = res.outputs[0]
                        text_delta = output.text[last_pos:]
                        last_pos = len(output.text)

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
