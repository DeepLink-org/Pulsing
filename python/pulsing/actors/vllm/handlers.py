"""vLLM Worker Actor - 基于 vLLM V1 引擎的高性能推理 Worker

参考 Dynamo 实现，支持：
1. Prefill/Decode 分离 (PD Disaggregation)
2. 多模态输入处理（图片）
3. KV Cache 管理和清理
4. LoRA 动态加载/卸载
5. OpenAI 兼容的文本输入输出模式
6. 引擎监控和健康检查
"""

# Worker 处理器：BaseWorkerHandler, PrefillWorkerHandler, DecodeWorkerHandler

import asyncio
import logging
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from typing import Any

from .sampling import build_sampling_params, build_sampling_params_openai
from .utils import (
    IMAGE_URL_KEY,
    URL_VARIANT_KEY,
    VIDEO_URL_KEY,
    ImageLoader,
    VllmEngineMonitor,
    handle_engine_dead_error,
    lora_name_to_id,
)

try:
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.exceptions import EngineDeadError

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    EngineDeadError = Exception  # Fallback for type hints

logger = logging.getLogger(__name__)


class BaseWorkerHandler(ABC):
    """Worker 处理器基类，提供通用功能"""

    def __init__(
        self,
        engine: AsyncLLM,
        default_sampling_params: dict[str, Any],
        model_max_len: int | None = None,
        enable_multimodal: bool = False,
        use_vllm_tokenizer: bool = False,
        on_engine_dead: Callable[[], None] | None = None,
    ):
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.model_max_len = model_max_len
        self.enable_multimodal = enable_multimodal
        self.use_vllm_tokenizer = use_vllm_tokenizer
        self.image_loader = ImageLoader()
        self.temp_dirs: list[tempfile.TemporaryDirectory] = []
        # LoRA 跟踪
        self.lora_id_for_name: dict[str, int] = {}
        self.lora_name_to_path: dict[str, str] = {}
        # 引擎监控 (统一管理回调和健康状态)
        self._on_engine_dead = on_engine_dead
        self.engine_monitor = VllmEngineMonitor(engine, on_engine_dead=on_engine_dead)

    @abstractmethod
    async def generate(self, request: dict[str, Any]) -> AsyncGenerator[dict, None]:
        raise NotImplementedError

    async def _extract_multimodal_data(
        self, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        """提取和解码多模态数据"""
        if "multi_modal_data" not in request or request["multi_modal_data"] is None:
            return None

        # 安全检查：如果未启用多模态则拒绝
        if not self.enable_multimodal:
            raise ValueError(
                "Received multimodal data but multimodal processing is not enabled"
            )

        mm_map = request["multi_modal_data"]
        vllm_mm_data = {}

        # 处理图片
        images = []
        for item in mm_map.get(IMAGE_URL_KEY, []):
            if isinstance(item, dict) and URL_VARIANT_KEY in item:
                url = item[URL_VARIANT_KEY]
                try:
                    image = await self.image_loader.load_image(url)
                    images.append(image)
                    logger.debug(f"Loaded image from URL: {url[:80]}...")
                except Exception:
                    logger.exception(f"Failed to load image from {url[:80]}...")
                    raise

        if images:
            vllm_mm_data["image"] = images[0] if len(images) == 1 else images
            logger.debug(f"Extracted {len(images)} image(s) for multimodal processing")

        # 处理视频（未来扩展）
        if VIDEO_URL_KEY in mm_map:
            logger.warning("Video multimodal data not yet supported")

        return vllm_mm_data if vllm_mm_data else None

    @staticmethod
    def _build_completion_usage(request_output: RequestOutput) -> dict[str, Any]:
        """构建使用统计信息"""
        return {
            "prompt_tokens": (
                len(request_output.prompt_token_ids)
                if request_output.prompt_token_ids
                else None
            ),
            "completion_tokens": len(request_output.outputs[0].token_ids),
            "total_tokens": (
                len(request_output.prompt_token_ids)
                + len(request_output.outputs[0].token_ids)
                if request_output.prompt_token_ids
                else None
            ),
            "prompt_tokens_details": (
                {"cached_tokens": request_output.num_cached_tokens}
                if request_output.num_cached_tokens
                else None
            ),
        }

    @staticmethod
    def _extract_logprobs(
        output, num_output_tokens_so_far: int
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        """提取 logprobs 信息"""
        if output.logprobs is None:
            return None, None

        new_logprobs = output.logprobs[num_output_tokens_so_far:]
        if not new_logprobs:
            return None, None

        log_probs = []
        top_logprobs = []

        for token_idx, token_logprobs_dict in enumerate(new_logprobs):
            if token_logprobs_dict is None:
                continue

            actual_token_id = output.token_ids[num_output_tokens_so_far + token_idx]
            selected_logprob = token_logprobs_dict[actual_token_id]
            log_probs.append(float(selected_logprob.logprob))

            token_top_logprobs = []
            for tok_id, logprob_info in token_logprobs_dict.items():
                token_top_logprobs.append(
                    {
                        "rank": (
                            logprob_info.rank if hasattr(logprob_info, "rank") else 0
                        ),
                        "token_id": tok_id,
                        "token": (
                            logprob_info.decoded_token
                            if hasattr(logprob_info, "decoded_token")
                            else None
                        ),
                        "logprob": float(logprob_info.logprob),
                    }
                )
            top_logprobs.append(token_top_logprobs)

        return log_probs if log_probs else None, top_logprobs if top_logprobs else None

    async def generate_tokens(
        self,
        prompt,
        sampling_params,
        request_id,
        lora_request=None,
        data_parallel_rank: int | None = None,
    ):
        """生成 tokens

        Args:
            prompt: 输入 prompt (TokensPrompt 或 TextPrompt)
            sampling_params: 采样参数
            request_id: 请求 ID
            lora_request: LoRA 请求 (可选)
            data_parallel_rank: 数据并行 rank (可选，用于 DP 部署)
        """
        try:
            if lora_request:
                logger.debug(
                    f"Starting token generation for request {request_id} with LoRA: "
                    f"{lora_request.lora_name} (ID: {lora_request.lora_int_id})"
                    f"{f', dp_rank={data_parallel_rank}' if data_parallel_rank is not None else ''}"
                )
            else:
                logger.debug(
                    f"Starting token generation for request {request_id} (no LoRA)"
                    f"{f', dp_rank={data_parallel_rank}' if data_parallel_rank is not None else ''}"
                )

            gen = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                data_parallel_rank=data_parallel_rank,
            )

            num_output_tokens_so_far = 0
            async for res in gen:
                if not res.outputs:
                    if lora_request:
                        logger.debug(
                            f"Request {request_id} with LoRA {lora_request.lora_name} "
                            "returned no outputs"
                        )
                    yield {"finish_reason": "error", "token_ids": []}
                    break

                output = res.outputs[0]
                next_total_toks = len(output.token_ids)
                out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

                # 提取 logprobs
                log_probs, top_logprobs = self._extract_logprobs(
                    output, num_output_tokens_so_far
                )
                if log_probs is not None:
                    out["log_probs"] = log_probs
                if top_logprobs is not None:
                    out["top_logprobs"] = top_logprobs

                if output.finish_reason:
                    out["finish_reason"] = output.finish_reason
                    out["completion_usage"] = BaseWorkerHandler._build_completion_usage(
                        request_output=res
                    )
                    if lora_request:
                        logger.debug(
                            f"Completed token generation for request {request_id} with LoRA "
                            f"{lora_request.lora_name}: {next_total_toks} output tokens, "
                            f"finish_reason={output.finish_reason}"
                        )
                    else:
                        logger.debug(
                            f"Completed token generation for request {request_id}: "
                            f"{next_total_toks} output tokens, finish_reason={output.finish_reason}"
                        )
                if output.stop_reason:
                    out["stop_reason"] = output.stop_reason
                yield out
                num_output_tokens_so_far = next_total_toks

        except asyncio.CancelledError:
            raise GeneratorExit(
                "Engine was shut down during token generation"
            ) from None
        except EngineDeadError as e:
            handle_engine_dead_error(e, "token generation", self._on_engine_dead)
            raise
        except Exception as e:
            logger.exception(f"Error in token generation: {e}")
            raise

    async def clear_kv_cache(self) -> dict[str, Any]:
        """清理 KV Cache"""
        try:
            await self.engine_client.reset_prefix_cache()
            return {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            logger.exception("Failed to clear KV cache")
            return {"status": "error", "message": str(e)}

    async def load_lora(self, lora_name: str, lora_path: str) -> dict[str, Any]:
        """动态加载 LoRA 适配器

        Args:
            lora_name: LoRA 适配器名称
            lora_path: LoRA 适配器路径（本地文件系统路径）

        Returns:
            包含状态的字典
        """
        try:
            if lora_name in self.lora_id_for_name:
                return {
                    "status": "error",
                    "message": f"LoRA adapter '{lora_name}' is already loaded",
                }

            # 生成确定性 ID
            lora_id = lora_name_to_id(lora_name)

            # 添加 LoRA 到引擎
            await self.engine_client.add_lora(
                LoRARequest(
                    lora_name=lora_name, lora_int_id=lora_id, lora_path=lora_path
                )
            )

            # 跟踪 LoRA
            self.lora_id_for_name[lora_name] = lora_id
            self.lora_name_to_path[lora_name] = lora_path
            logger.info(
                f"Successfully loaded LoRA adapter: {lora_name} with ID {lora_id}"
            )

            return {
                "status": "success",
                "message": f"LoRA adapter '{lora_name}' loaded successfully",
                "lora_name": lora_name,
                "lora_id": lora_id,
            }
        except Exception as e:
            logger.exception(f"Failed to load LoRA adapter: {e}")
            return {"status": "error", "message": str(e)}

    async def unload_lora(self, lora_name: str) -> dict[str, Any]:
        """卸载 LoRA 适配器

        Args:
            lora_name: LoRA 适配器名称

        Returns:
            包含状态的字典
        """
        try:
            if lora_name not in self.lora_id_for_name:
                return {
                    "status": "error",
                    "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.lora_id_for_name.keys())}",
                }

            logger.debug(f"Unloading LoRA adapter: {lora_name}")
            lora_id = self.lora_id_for_name[lora_name]

            await self.engine_client.remove_lora(lora_id)

            # 从跟踪字典中移除
            del self.lora_id_for_name[lora_name]
            if lora_name in self.lora_name_to_path:
                del self.lora_name_to_path[lora_name]

            logger.info(
                f"Successfully unloaded LoRA adapter: {lora_name} with ID {lora_id}"
            )
            return {
                "status": "success",
                "message": f"LoRA adapter '{lora_name}' unloaded successfully",
                "lora_name": lora_name,
                "lora_id": lora_id,
            }
        except Exception as e:
            logger.exception(f"Failed to unload LoRA adapter: {e}")
            return {"status": "error", "message": str(e)}

    async def list_loras(self) -> dict[str, Any]:
        """列出所有已加载的 LoRA 适配器

        Returns:
            包含 LoRA 列表的字典
        """
        try:
            loras = dict(self.lora_id_for_name)
            return {
                "status": "success",
                "loras": loras,
                "count": len(loras),
            }
        except Exception as e:
            logger.exception(f"Failed to list LoRA adapters: {e}")
            return {"status": "error", "message": str(e)}

    def add_temp_dir(self, temp_dir: tempfile.TemporaryDirectory) -> None:
        """添加临时目录以便稍后清理"""
        if temp_dir is not None:
            self.temp_dirs.append(temp_dir)

    def cleanup(self):
        """清理资源"""
        for temp_dir in self.temp_dirs:
            try:
                temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")


class PrefillWorkerHandler(BaseWorkerHandler):
    """Prefill Worker 处理器 - 只执行 prefill 阶段"""

    async def generate(self, request: dict[str, Any]) -> AsyncGenerator[dict, None]:
        """生成 prefill 结果"""
        request_id = f"prefill-{uuid.uuid4().hex[:8]}"
        logger.debug(f"Prefill Request ID: {request_id}")

        # 提取多模态数据
        multi_modal_data = await self._extract_multimodal_data(request)

        token_ids = request.get("token_ids", [])
        prompt = TokensPrompt(
            prompt_token_ids=token_ids, multi_modal_data=multi_modal_data
        )

        # 构建采样参数
        sampling_params = build_sampling_params(
            request, self.default_sampling_params, self.model_max_len
        )

        # 配置 prefill 模式：只生成 1 个 token，开启远程解码
        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        sampling_params.extra_args["kv_transfer_params"] = {
            "do_remote_decode": True,
        }
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1

        # LoRA 支持
        lora_request = None
        model_name = request.get("model")
        if model_name and model_name in self.lora_id_for_name:
            lora_id = self.lora_id_for_name[model_name]
            lora_request = LoRARequest(
                lora_name=model_name,
                lora_int_id=lora_id,
                lora_path=self.lora_name_to_path[model_name],
            )
            logger.info(
                f"Prefill request {request_id} will use LoRA adapter: {model_name} (ID: {lora_id})"
            )

        # 获取 data_parallel_rank (参考 Dynamo handlers.py)
        dp_rank = request.get("dp_rank", None)

        try:
            gen = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                data_parallel_rank=dp_rank,
            )

            async for res in gen:
                logger.debug(f"kv transfer params: {res.kv_transfer_params}")

                token_ids = res.outputs[0].token_ids if res.outputs else []

                output: dict[str, Any] = {
                    "token_ids": list(token_ids),
                    "disaggregated_params": (
                        {"kv_transfer_params": res.kv_transfer_params}
                        if res.kv_transfer_params
                        else None
                    ),
                    "completion_usage": BaseWorkerHandler._build_completion_usage(
                        request_output=res
                    ),
                }

                if lora_request:
                    logger.info(
                        f"Prefill completed for request {request_id} with LoRA {lora_request.lora_name}: "
                        f"generated {len(token_ids)} token(s), "
                        f"has_kv_params={res.kv_transfer_params is not None}"
                    )

                yield output
        except asyncio.CancelledError:
            raise GeneratorExit("Prefill engine was shut down") from None
        except EngineDeadError as e:
            handle_engine_dead_error(e, "prefill", self._on_engine_dead)
            raise


class DecodeWorkerHandler(BaseWorkerHandler):
    """Decode Worker 处理器 - 执行 decode 阶段或完整推理"""

    async def generate(self, request: dict[str, Any]) -> AsyncGenerator[dict, None]:
        """生成 decode 结果"""
        request_id = f"decode-{uuid.uuid4().hex[:8]}"
        logger.debug(
            f"Decode Request ID: {request_id}, request keys: {list(request.keys())}"
        )

        # 自动检测输入类型
        has_token_ids = "token_ids" in request
        has_text_input = "prompt" in request or "messages" in request

        if self.use_vllm_tokenizer or (has_text_input and not has_token_ids):
            # 文本输入输出模式
            async for chunk in self._generate_text_mode(request, request_id):
                yield chunk
        elif has_token_ids:
            # Token 输入输出模式
            async for chunk in self._generate_token_mode(request, request_id):
                yield chunk
        else:
            # 既没有 token_ids 也没有文本输入
            raise ValueError(
                "Request must contain either 'token_ids' or 'prompt'/'messages' field"
            )

    async def _generate_token_mode(
        self, request: dict[str, Any], request_id: str
    ) -> AsyncGenerator[dict, None]:
        """Token 输入输出模式生成"""
        # 提取多模态数据
        multi_modal_data = await self._extract_multimodal_data(request)

        # 检查是否有 token_ids
        if "token_ids" not in request:
            raise ValueError(
                "Request must contain 'token_ids' field for token mode generation. "
                "Use 'use_vllm_tokenizer=True' for text mode."
            )

        prompt = TokensPrompt(
            prompt_token_ids=request["token_ids"], multi_modal_data=multi_modal_data
        )

        # 构建采样参数
        sampling_params = build_sampling_params(
            request, self.default_sampling_params, self.model_max_len
        )

        # 处理 prefill 结果（如果是 PD 分离模式）
        prefill_result = request.get("prefill_result")
        if prefill_result and isinstance(prefill_result, dict):
            kv_params = prefill_result.get("disaggregated_params", {}).get(
                "kv_transfer_params"
            )
            if kv_params is not None:
                if sampling_params.extra_args is None:
                    sampling_params.extra_args = {}
                sampling_params.extra_args["kv_transfer_params"] = kv_params
                logger.debug(
                    f"Using disaggregated params from prefill for request {request_id}"
                )

        prefill_prompt_tokens_details = (
            prefill_result.get("prompt_tokens_details") if prefill_result else None
        )

        # LoRA 支持
        lora_request = None
        model_name = request.get("model")
        if model_name and model_name in self.lora_id_for_name:
            lora_id = self.lora_id_for_name[model_name]
            lora_request = LoRARequest(
                lora_name=model_name,
                lora_int_id=lora_id,
                lora_path=self.lora_name_to_path[model_name],
            )
            logger.info(
                f"Decode request {request_id} will use LoRA adapter: {model_name} (ID: {lora_id})"
            )

        # 获取 data_parallel_rank (参考 Dynamo handlers.py)
        dp_rank = request.get("dp_rank", None)

        try:
            async for tok in self.generate_tokens(
                prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                data_parallel_rank=dp_rank,
            ):
                if prefill_result is not None and "completion_usage" in tok:
                    tok["completion_usage"]["prompt_tokens_details"] = (
                        prefill_prompt_tokens_details
                    )
                yield tok
        except Exception as e:
            logger.exception(f"Error in decode generation: {e}")
            raise

    async def _generate_text_mode(
        self, request: dict[str, Any], request_id: str
    ) -> AsyncGenerator[dict, None]:
        """文本输入输出模式生成（OpenAI 兼容）"""
        # 获取文本输入
        prompt_text = None

        # 尝试不同的输入格式
        if "prompt" in request:
            prompt_text = request["prompt"]
        elif "messages" in request:
            messages = request["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                # 如果是消息列表，取最后一条消息内容
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    prompt_text = last_message.get("content", "")
                else:
                    prompt_text = str(last_message)
            else:
                prompt_text = str(messages)
        elif "text" in request:
            prompt_text = request["text"]

        if not prompt_text:
            raise ValueError(
                "Request must contain 'prompt', 'messages', or 'text' field for text mode generation"
            )

        # 关键修复：在文本模式下，必须先使用 tokenizer 将文本转换为 token_ids
        # 这是因为 vLLM 在应用 penalties 时需要 prompt_token_ids
        # 参考 Dynamo 的 InputParamManager 实现

        # 获取 tokenizer
        tokenizer = getattr(self.engine_client, "tokenizer", None)
        if not tokenizer:
            raise ValueError(
                "Tokenizer not available. Text mode requires tokenizer to convert text to token_ids. "
                "This is necessary for applying sampling penalties."
            )

        # 使用 tokenizer 将文本转换为 token_ids
        try:
            token_ids = tokenizer.encode(prompt_text)
            prompt = TokensPrompt(prompt_token_ids=token_ids)
        except Exception as e:
            logger.exception(f"Failed to tokenize prompt: {e}")
            raise ValueError(f"Failed to tokenize prompt: {e}") from e

        # 构建采样参数
        sampling_params = build_sampling_params_openai(
            request, self.default_sampling_params
        )

        # 获取 data_parallel_rank (参考 Dynamo handlers.py)
        dp_rank = request.get("dp_rank", None)

        openai_request_id = request.get("id") or request.get("request_id", request_id)
        previous_text = ""

        try:
            gen = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
                data_parallel_rank=dp_rank,
            )

            async for res in gen:
                if not res.outputs:
                    yield {
                        "id": openai_request_id,
                        "created": int(time.time()),
                        "object": "chat.completion.chunk",
                        "model": "unknown",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": "error",
                            }
                        ],
                    }
                    break

                output = res.outputs[0]
                delta_text = output.text[len(previous_text) :]
                previous_text = output.text

                choice_data = {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": delta_text,
                    },
                    "finish_reason": output.finish_reason,
                }

                chunk = {
                    "id": openai_request_id,
                    "created": int(time.time()),
                    "object": "chat.completion.chunk",
                    "model": "unknown",
                    "choices": [choice_data],
                }

                yield chunk

        except EngineDeadError as e:
            handle_engine_dead_error(e, "text mode generation", self._on_engine_dead)
            raise
        except Exception as e:
            logger.exception(f"Error in text mode generation: {e}")
            raise
