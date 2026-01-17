"""Sampling parameter construction functions

Provides two modes of sampling parameter construction:
- build_sampling_params: Token mode
- build_sampling_params_openai: OpenAI-compatible text mode
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
    """Build SamplingParams from request

    Args:
        request: Request dictionary
        default_sampling_params: Default sampling parameters
        model_max_len: Model maximum length

    Returns:
        Configured SamplingParams
    """
    sampling_params = SamplingParams(**default_sampling_params)

    # Apply sampling options
    sampling_options = request.get("sampling_options", {})
    for key, value in sampling_options.items():
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # Apply stop conditions
    stop_conditions = request.get("stop_conditions", {})
    for key, value in stop_conditions.items():
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # Handle max_tokens
    provided_max_tokens = stop_conditions.get("max_tokens")
    token_ids = request.get("token_ids", [])
    input_length = len(token_ids)
    if model_max_len is not None and provided_max_tokens is None:
        # Ensure default generates at least 1 token
        dynamic_default = max(1, model_max_len - input_length)
        sampling_params.max_tokens = dynamic_default

    return sampling_params


def build_sampling_params_openai(
    request: dict[str, Any],
    default_sampling_params: dict[str, Any],
) -> SamplingParams:
    """Build SamplingParams from OpenAI-compatible request

    Args:
        request: OpenAI-style request dictionary
        default_sampling_params: Default sampling parameters

    Returns:
        Configured SamplingParams
    """
    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = True

    # Map OpenAI parameters to SamplingParams
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

    # Handle max_tokens (supports both max_tokens and max_new_tokens)
    max_tokens_value = request.get("max_tokens") or request.get("max_new_tokens")
    if max_tokens_value is not None:
        sampling_params.max_tokens = max_tokens_value

    # Handle stop sequences
    if "stop" in request and request["stop"] is not None:
        sampling_params.stop = request["stop"]

    # Handle ignore_eos
    if "ignore_eos" in request and request["ignore_eos"] is not None:
        sampling_params.ignore_eos = request["ignore_eos"]

    # Handle min_tokens
    if "min_tokens" in request and request["min_tokens"] is not None:
        sampling_params.min_tokens = request["min_tokens"]

    return sampling_params
