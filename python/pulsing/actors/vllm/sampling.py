"""采样参数构建函数

提供两种模式的采样参数构建：
- build_sampling_params: Token 模式
- build_sampling_params_openai: OpenAI 兼容的文本模式
"""

import logging
from typing import Any

try:
    from PIL import Image
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.exceptions import EngineDeadError

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


logger = logging.getLogger(__name__)


def build_sampling_params(
    request: dict[str, Any],
    default_sampling_params: dict[str, Any],
    model_max_len: int | None = None,
) -> SamplingParams:
    """从请求构建 SamplingParams

    Args:
        request: 请求字典
        default_sampling_params: 默认采样参数
        model_max_len: 模型最大长度

    Returns:
        配置好的 SamplingParams
    """
    sampling_params = SamplingParams(**default_sampling_params)

    # 应用采样选项
    sampling_options = request.get("sampling_options", {})
    for key, value in sampling_options.items():
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # 应用停止条件
    stop_conditions = request.get("stop_conditions", {})
    for key, value in stop_conditions.items():
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # 处理 max_tokens
    provided_max_tokens = stop_conditions.get("max_tokens")
    token_ids = request.get("token_ids", [])
    input_length = len(token_ids)
    if model_max_len is not None and provided_max_tokens is None:
        # 确保默认至少生成 1 个 token
        dynamic_default = max(1, model_max_len - input_length)
        sampling_params.max_tokens = dynamic_default

    return sampling_params


def build_sampling_params_openai(
    request: dict[str, Any],
    default_sampling_params: dict[str, Any],
) -> SamplingParams:
    """从 OpenAI 兼容请求构建 SamplingParams

    Args:
        request: OpenAI 风格的请求字典
        default_sampling_params: 默认采样参数

    Returns:
        配置好的 SamplingParams
    """
    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = True

    # 映射 OpenAI 参数到 SamplingParams
    openai_mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "seed": "seed",
        "top_k": "top_k",
        "repetition_penalty": "repetition_penalty",
        "min_p": "min_p",
        "length_penalty": "length_penalty",
        "use_beam_search": "use_beam_search",
    }

    for req_key, param_key in openai_mapping.items():
        if req_key in request and request[req_key] is not None:
            if hasattr(sampling_params, param_key):
                setattr(sampling_params, param_key, request[req_key])

    # 处理 max_tokens（同时支持 max_tokens 和 max_new_tokens）
    max_tokens_value = request.get("max_tokens") or request.get("max_new_tokens")
    if max_tokens_value is not None:
        sampling_params.max_tokens = max_tokens_value

    # 处理停止序列
    if "stop" in request and request["stop"] is not None:
        sampling_params.stop = request["stop"]

    # 处理 ignore_eos
    if "ignore_eos" in request and request["ignore_eos"] is not None:
        sampling_params.ignore_eos = request["ignore_eos"]

    # 处理 min_tokens
    if "min_tokens" in request and request["min_tokens"] is not None:
        sampling_params.min_tokens = request["min_tokens"]

    return sampling_params
